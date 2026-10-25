"""eval.*: did it survive, and does it fit. Every eval op reports metrics, which
gates and the receipt read."""

from dataclasses import dataclass

from blackhole.core.ops import Needs, Op, p, register


@register
class Agreement(Op):
    """Model vs teacher on held-out teacher answers. Metrics: top1_agreement, kl,
    ppl_teacher, ppl_model."""
    name, major = "eval.agreement", 1
    consumes = {"model": "model", "teacher": "model", "heldout": "dataset"}
    produces = {"samples": "file"}
    needs = Needs(gpu=True)
    ported_from = "REAPer stages/eval.py"

    @dataclass
    class Params:
        model: str = p("")
        teacher: str = p("${base.model}")
        heldout: str = p("", "e.g. ${calib.heldout}")


@register
class Bench(Op):
    """Public suites run on the model (vLLM for safetensors, llama-server for GGUF), with
    a paired bootstrap interval against the base model's reading. Metrics per suite:
    <suite>, <suite>_lo, <suite>_hi, <suite>_kept (share of the base score)."""
    name, major = "eval.bench", 1
    consumes = {"model": "model"}
    produces = {"results": "file"}
    needs = Needs(gpu=True)
    ported_from = "LobBot stages/eval.py; BH blackhole.py share_kept, event_horizon.py gates"

    @dataclass
    class Params:
        model: str = p("")
        suites: list = p(["evalplus"], "evalplus | livecodebench | bfcl | mmlu_pro | judge:<rubric>")
        samples: int = p(1, "answers per problem (K)")
        compare_base: bool = p(True)
        thinking: bool = p(False)
        judge: str = p("", "frontier:<model> for judge suites")


@register
class Footprint(Op):
    """Measured memory of the model served the way the target will run it: GGUF bytes
    (counted once) plus all buffers of a one-slot llama-server at the schema's context
    and KV type, after a prompt of 3/4 of the context. Metrics: file_gb, footprint_gb,
    tok_s (when the target device class is known), fits."""
    name, major = "eval.footprint", 1
    consumes = {"model": "model"}
    produces = {"report": "file"}
    needs = Needs(tools=("llama.cpp",))
    ported_from = "BH BH0.1 footprint rule; LobBot eval size and tok/s estimate"

    @dataclass
    class Params:
        model: str = p("")
        context: int = p(16384)
        kv: str = p("q8_0")
        device: str = p("", "device class for the tok/s estimate, e.g. 8GB-VRAM, M-series-16GB")
