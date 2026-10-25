"""score.* and prune.*: deciding what to keep, then removing the rest.

Scoring and pruning are separate ops so a second ratio reuses the scores, which is
the expensive half (REAPer ran them in one process; its cache made `--prune` reruns cheap)."""

from dataclasses import dataclass

from blackhole.core.ops import Needs, Op, p, register


@register
class ReapScore(Op):
    """REAP saliency per expert: router-weighted expert activation norms over the
    calibration tokens, with hooks on every router of any HF MoE (moe.py).

    The saliency artifact is a shared score table other ops read (prune.reap,
    quant.reaq, quant.gguf_map, lab ops): JSON
    {"layers": [{"layer": 3, "experts": [{"id": 0, "saliency": S_j, "freq": share of tokens
    routed to j, "gate_mean": mean g_j(x) when routed, "out_norm": mean ||f_j(x)||}, ...]}],
    "tokens": n, "model": id}. S_j = mean over tokens routed to j of g_j(x)*||f_j(x)||."""
    name, major = "score.reap", 1
    consumes = {"model": "model", "data": "dataset"}
    produces = {"saliency": "file"}
    needs = Needs(gpu=True, arch=("moe", "hybrid"))
    ported_from = "REAPer stages/reap.py (score half), moe.py, hardware.py"

    @dataclass
    class Params:
        model: str = p("")
        data: str = p("", "calibration set, e.g. ${calib.data}")
        max_seq: int = p(2048)
        answers: bool = p(True, "score on prompt + answer, not prompts only")


@register
class ReapPrune(Op):
    """Keep the top experts per layer by saliency and rewrite the checkpoint shard by
    shard in its own format (fp8 blocks and MXFP4 survive). CPU only."""
    name, major = "prune.reap", 1
    consumes = {"model": "model", "saliency": "file"}
    produces = {"model": "model", "report": "file"}
    needs = Needs(arch=("moe", "hybrid"))
    ported_from = "REAPer stages/reap.py (prune half), ckpt.py"

    @dataclass
    class Params:
        model: str = p("")
        saliency: str = p("", "e.g. ${score.saliency}")
        ratio: float = p(0.5, "share of routed experts dropped in every MoE layer")
        keep_groups_even: bool = p(True, "DeepSeek-style groups keep the same count per group")


@register
class Structured(Op):
    """Dense models: keep a fraction of each FFN's channels and of the KV groups, and drop
    whole layers, chosen on the data so far. Heal afterwards."""
    name, major = "prune.structured", 1
    consumes = {"model": "model", "data": "dataset"}
    produces = {"model": "model", "report": "file"}
    needs = Needs(gpu=True, arch=("dense", "hybrid"))
    ported_from = "mugge prune/ (prune-structured)"

    @dataclass
    class Params:
        model: str = p("")
        data: str = p("")
        ffn: float = p(1.0, "fraction of FFN channels kept")
        kv_groups: float = p(1.0, "fraction of KV groups kept")
        drop_layers: int = p(0, "banned in defaults: dropping 4 layers of Qwen3-8B cost KL 0.99")
        layer_drop_samples: int = p(64)


@register
class NeuronScore(Op):
    """Activation-scored importance of every FFN neuron (intermediate channel) of a dense
    model on the calibration text: mean |activation| x norm of its down-projection column
    (Wanda-style), plus the critical set: neurons whose activations are massive outliers
    (> `massive` x the layer median) and those carrying super weights. Same JSON layout as
    REAP's table, per layer and per neuron, so prune and quant ops read either.

    B200 on Qwen3-8B (2026-10-23): 25% pruned by this score gave KL 0.24 (expert-norm 0.26,
    MoEfy+REAP 0.67); random 50% destroyed the model, so the critical set matters."""
    name, major = "score.neurons", 1
    consumes = {"model": "model", "data": "dataset"}
    produces = {"scores": "file"}
    needs = Needs(gpu=True, arch=("dense", "hybrid"))
    ported_from = "new; mugge prune/ structured scores; REAPer moe.py hooks"

    @dataclass
    class Params:
        model: str = p("")
        data: str = p("")
        max_seq: int = p(2048)
        metric: str = p("act_x_down", "act_x_down | act | down_norm")
        massive: float = p(100.0, "activation outlier factor that marks a neuron critical")


@register
class NeuronPrune(Op):
    """Remove the lowest-scored FFN neurons and rewrite the checkpoint with a smaller
    intermediate size per layer (still a standard dense checkpoint every runtime loads).
    Protect rules: the critical set from score.neurons is never removed, the top
    `protect_top` share of each layer is kept, the first and last `skip_edge` layers are
    untouched, and no layer drops more than `max_layer` of its neurons."""
    name, major = "prune.neurons", 1
    consumes = {"model": "model", "scores": "file"}
    produces = {"model": "model", "report": "file"}
    needs = Needs(arch=("dense", "hybrid"))
    ported_from = "new; mugge prune-structured (ffn) checkpoint rewrite"

    @dataclass
    class Params:
        model: str = p("")
        scores: str = p("", "e.g. ${score.scores}")
        ratio: float = p(0.0, "share of FFN neurons removed overall; the dense frontier is still being measured")
        allocation: str = p("global", "global (rank all layers together, capped by max_layer) | uniform")
        protect_critical: bool = p(True, "never remove neurons score.neurons marked critical")
        protect_top: float = p(0.01, "share of each layer's best neurons always kept")
        skip_edge: int = p(2, "first and last layers left whole")
        max_layer: float = p(0.5, "most any one layer may lose")
        multiple_of: int = p(64, "round each layer's new size for fast kernels")
