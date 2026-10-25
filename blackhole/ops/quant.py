"""quant.*: storing the weights in fewer bits."""

from dataclasses import dataclass

from blackhole.core.ops import Needs, Op, p, register


@register
class Fp8Experts(Op):
    """Routed experts to fp8 with 128x128 block scales; attention, routers, shared experts
    and the head stay bf16. For vLLM serving on a GPU box."""
    name, major = "quant.fp8_experts", 1
    consumes = {"model": "model"}
    produces = {"model": "model"}
    ported_from = "REAPer stages/quant.py, quant.py"

    @dataclass
    class Params:
        model: str = p("")


@register
class GgufMap(Op):
    """Plan a per-tensor GGUF type map under a size budget: expert bit widths from REAP
    layer importance x imatrix output energy, fixed types for attention, embeddings, head.
    Writes the map and its predicted file size; does not quantise."""
    name, major = "quant.gguf_map", 1
    consumes = {"model": "model", "data": "dataset", "saliency": "file"}
    produces = {"type_map": "file", "imatrix": "file", "bf16_gguf": "gguf"}
    needs = Needs(tools=("llama.cpp",))
    ported_from = "LobBot stages/quantize.py + bits.py; BH event_horizon.py PTQ map + preflight"

    @dataclass
    class Params:
        model: str = p("")
        data: str = p("", "imatrix text, chat-formatted")
        saliency: str = p("", "optional REAP scores, e.g. ${score.saliency}")
        max_gb: float = p(0.0, "file budget; 0 = footprint.max_gb minus KV and buffers")
        floor: str = p("q2_k")
        ceiling: str = p("q6_k")
        head: str = p("q8_0", "output layer type (BH keeps an 8-bit head)")


@register
class Gguf(Op):
    """Write the GGUF with llama-quantize from a type map (or a single type), then the
    parity checks: every tensor as mapped, size within 0.5% of plan, KLD and top-1 vs
    a bf16 GGUF of the same weights, llama-tokenize == HF.

    Never plain round-to-nearest: either the weights come from quant.gptq already snapped
    to this GGUF grid (packed exactly, no imatrix), or llama-quantize gets an imatrix.
    On Qwen3-8B (B200, 2026-10-23) GPTQ int4 had 2.5x less KL than round-to-nearest."""
    name, major = "quant.gguf", 1
    consumes = {"model": "model", "type_map": "file", "imatrix": "file"}
    produces = {"gguf": "gguf", "parity": "file"}
    needs = Needs(tools=("llama.cpp",))
    ported_from = "LobBot stages/quantize.py; BH event_horizon.py export + parity()"

    @dataclass
    class Params:
        model: str = p("")
        type_map: str = p("", "from quant.gguf_map; empty uses `type`")
        imatrix: str = p("", "required unless the model came from quant.gptq on this type's grid")
        type: str = p("Q4_K_M")
        pre_snapped: bool = p(False, "the weights are already on the grid (after quant.gptq): pack them exactly")


@register
class Gptq(Op):
    """GPTQ: quantise each layer's weights column by column, pushing every rounding error
    onto the columns not yet quantised using the layer's input Hessian from calibration
    text, so the layer's output changes as little as possible. Writes a model whose weights
    already sit on the target grid:
      gguf:<type>   llama.cpp's block grids (Q4_K, Q3_K, Q5_K, Q6_K; scales per super-block),
                    so quant.gguf with pre_snapped packs them bit-exactly (llmap kquant.py
                    has the grid code)
      int<b>-g<n>   plain group-wise integers for vLLM (GPTQ/Marlin kernels, compressed-tensors)

    Qwen3-8B, B200, 2026-10-23: int4 6.1 GB KL 0.028 (round-to-nearest 0.070), int3 5.3 GB
    KL 0.15, better than any pruned recipe of similar size."""
    name, major = "quant.gptq", 1
    consumes = {"model": "model", "data": "dataset"}
    produces = {"model": "model", "report": "file"}
    needs = Needs(gpu=True)
    ported_from = "GPU performance thread's B200 quantiser; llmap stages/kquant.py (GGUF grids); w4a16.py"

    @dataclass
    class Params:
        model: str = p("")
        data: str = p("", "calibration text; the data so far")
        grid: str = p("gguf:Q4_K_M", "gguf:<type> | int4-g128 | int3-g128 | typemap:<path> (per-tensor grids, e.g. from quant.reaq)")
        samples: int = p(128, "calibration sequences for the Hessians")
        damp: float = p(0.01, "Hessian dampening, as a share of its mean diagonal")
        act_order: bool = p(True, "quantise columns in order of decreasing Hessian diagonal")
        keep: list = p(["lm_head", "embed_tokens"], "tensors left at 8/16 bits")


@register
class W4A16(Op):
    """4-bit weights, 16-bit activations for vLLM on Ada/Hopper: quant.gptq with an int4
    grid, written as compressed-tensors with an fp8 head."""
    name, major = "quant.w4a16", 1
    consumes = {"model": "model", "data": "dataset"}
    produces = {"model": "model"}
    needs = Needs(gpu=True)
    ported_from = "new (BH Large plan: W4A16 + FP8 head)"

    @dataclass
    class Params:
        model: str = p("")
        data: str = p("")
        head_fp8: bool = p(True)


@register
class Mlx(Op):
    """MLX folders for Apple silicon: Q<bits> group quantisation, or DWQ<bits> that trains
    the scales so next-token distributions match the full model on the data so far."""
    name, major = "quant.mlx", 1
    consumes = {"model": "model", "data": "dataset"}
    produces = {"model": "model"}
    needs = Needs(tools=("mlx",))
    ported_from = "mugge quant/mlx"

    @dataclass
    class Params:
        model: str = p("")
        data: str = p("")
        quant: str = p("DWQ4", "Q4 | Q8 | DWQ4 | ...")
        group_size: int = p(64)


@register
class Reaq(Op):
    """REAQ, Router-weighted Expert Activation Quantization: a BlackHole-Harness original
    (2026-10-23). EXPERIMENTAL until the GPU thread's interval test reports.

    REAP's router-weighted activation score (saliency x routing frequency per expert) sets
    each expert's bit width inside an interval [min_bits, max_bits]: experts the purpose
    leans on get more bits, rarely used ones fewer, under a byte budget (a knapsack over
    expert x type, cost = bytes, value = score-weighted error of the type). No expert is
    removed. Attention, router, shared experts and the head keep fixed types.

    Earlier history: a first version (2026-10-23, then called expert_bits, allowing 2-bit and
    pruning the coldest experts) lost to uniform int4/int3 at equal size by 0.01-0.04 KL,
    most likely because rare experts went to 2 bits; hence the interval, which keeps every
    expert at min_bits or above. Uniform GPTQ per-layer quant is the baseline to beat.

    MoE only, floor 3 bits (B200, 2026-10-23): on a dense model MoEfied to experts, REAQ
    lost to uniform GPTQ at both budgets (int4 0.028 vs best mix 0.031; int3 0.108 vs 0.24),
    and every interval reaching down to 2-bit was bad. Params reject a Q2 floor.

    Runtimes store a layer's experts as one fused tensor of one type, so the plan is
    realised by `layout`:
      safetensors  per-expert quantisation configs for a custom torch/vLLM loader
                   (what the GPU thread's interval test measures)
      bands        sort experts by score and split each layer's fused tensors into 2-4
                   bands (hot/warm/cold), one type per band; needs the harness's
                   llama.cpp patch that loads banded expert tensors
      layer        one type per layer (stock llama.cpp and Ollama); this averages REAQ's
                   per-expert choice away, so it is only a fallback
    Writes the type map quant.gptq / quant.gguf consume, the expert permutation, and the
    predicted size and error.

    target="codebook" picks a value-table size per expert for quant.expert_codebook instead
    of a GGUF type (B200, 2026-10-23, Qwen3-30B-A3B at REAP 25%: top 50% experts by REAP
    score on 16-level tables, the rest on 8-level, KL 0.036 at 4.0 bpw vs 0.043 for a random
    split: the first per-expert allocation that beat uniform)."""
    name, major = "quant.reaq", 1
    consumes = {"model": "model", "saliency": "file", "data": "dataset"}
    produces = {"type_map": "file", "permutation": "file", "plan": "file"}
    needs = Needs(arch=("moe", "hybrid"))
    ported_from = "new (REAQ); LobBot stages/bits.py (per-layer expert bits), llmap stages/bits.py + horizon.py (knapsack)"

    @dataclass
    class Params:
        model: str = p("")
        saliency: str = p("", "REAP score table, e.g. ${score.saliency}")
        data: str = p("", "text for measuring each type's error per expert (imatrix); empty uses the score only")
        max_gb: float = p(0.0, "expert byte budget; 0 = footprint.max_gb minus the fixed parts and KV")
        min_bits: str = p("Q3_K", "the interval's floor: no expert goes below it")
        max_bits: str = p("Q6_K", "the interval's ceiling")
        score: str = p("saliency*freq", "value per expert: saliency | freq | saliency*freq | a lab op's column")
        layout: str = p("safetensors", "safetensors | bands (needs the llama.cpp patch) | layer (fallback)")
        target: str = p("bits", "bits: a GGUF type per expert in [min_bits, max_bits] | codebook: a table size per expert for quant.expert_codebook")
        levels: list = p([8, 16], "codebook target: the table sizes to choose between, smallest first")
        hot_share: float = p(0.5, "codebook target: share of experts (by score) that get the largest table; 0 = solve under max_gb")
        bands: int = p(3)

        def __post_init__(self):
            if str(self.min_bits).upper().startswith(("Q2", "IQ1", "IQ2")) or str(self.min_bits) == "int2":
                raise ValueError("quant.reaq: min_bits must be 3 bits or more (2-bit intervals measured bad)")


@register
class ExpertCodebook(Op):
    """Per-expert codebook quantisation (HEX, Header-Encoded eXperts; op name kept for now). EXPERIMENTAL,
    a BlackHole-Harness original.

    NVFP4-style layout: weights in 16-weight blocks with an FP8 scale per block, and on top
    each expert carries a header (an FP32 scale plus a learned value table of 8 or 16 levels)
    that the low-bit indices point into. The table is fitted per expert (k-means on the
    block-normalised weights, weighted by the imatrix when data is given). With a type map
    from quant.reaq (target="codebook") each expert gets its own table size.

    Rounding is GPTQ onto the codebook by default: columns are quantised to the nearest table
    level in order, and each column's error is pushed onto the not-yet-quantised columns
    through the calibration Hessian, exactly as quant.gptq does onto llama.cpp grids.

    Measured (B200, 2026-10-23, Qwen3-30B-A3B at REAP 25%, simulated quant, 32 held-out
    answers). The pieces stack:
        GPTQ int3, uniform                        3.25 bpw           KL 0.083
        8-level tables, nearest rounding          3.25 bpw (8.23 GB) KL 0.066
        8-level tables, GPTQ rounding             3.25 bpw           KL 0.054
        uniform 8-level / random split / REAQ     3.5 bpw            KL 0.049 / 0.047 / 0.041
        REAQ 16/8 levels, GPTQ rounding           9.5 GB             KL 0.033 (GPTQ int4 0.034 at 10.4 GB)
    A per-expert scale alone does nothing; at 4 bits a per-expert 16-level table with nearest
    rounding only ties NVFP4. Lab until it generalises (next: gpt-oss-20b) and a real kernel
    exists; then "REAP 25% + REAQ codebooks" replaces plan.fit's MoE int3 tiers.

    Serving: no packed format yet. It needs a custom kernel, or a llama.cpp/vLLM patch that
    reads the per-expert header and does the table lookup in the expert matmul. Until then the
    output is fake-quantised bf16 safetensors plus the headers and indices for measuring."""
    name, major = "quant.expert_codebook", 1
    consumes = {"model": "model", "data": "dataset", "type_map": "file"}
    produces = {"model": "model", "headers": "file", "report": "file"}
    needs = Needs(gpu=True, arch=("moe", "hybrid"))
    ported_from = "new (GPU thread's per-expert header bench, 2026-10-23)"

    @dataclass
    class Params:
        model: str = p("")
        data: str = p("", "calibration text for the GPTQ Hessians and imatrix weighting; the data so far by default")
        type_map: str = p("", "per-expert table sizes from quant.reaq (target=codebook), e.g. ${reaq.type_map}; empty uses `levels` for all")
        levels: int = p(8, "table size when there is no type map: 8 (3-bit indices) | 16 (4-bit)")
        block: int = p(16, "weights per block, each with its own scale")
        block_scale: str = p("fp8", "fp8 (e4m3, as NVFP4) | fp16")
        header: str = p("fp32", "per-expert header precision: scale and table")
        fit: str = p("kmeans", "kmeans | lloyd_max | nf (fixed NormalFloat table, the ablation)")
        rounding: str = p("gptq", "gptq: error-compensated onto the table, needs data | nearest (the ablation)")
        damp: float = p(0.01, "gptq: Hessian dampening")
        iters: int = p(20)
        experts_only: bool = p(True, "attention, routers, shared experts and the head keep their types")
        export: str = p("fake_quant", "fake_quant (bf16 safetensors + headers) | packed (needs the kernel; not implemented)")

        def __post_init__(self):
            if self.levels not in (4, 8, 16):
                raise ValueError("quant.expert_codebook: levels is 4, 8 or 16")
            if self.rounding not in ("gptq", "nearest"):
                raise ValueError("quant.expert_codebook: rounding is gptq or nearest")
            if self.export == "packed":
                raise ValueError("quant.expert_codebook: no packed format yet; it needs a custom kernel or llama.cpp/vLLM patch")
