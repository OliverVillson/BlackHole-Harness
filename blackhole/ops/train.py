"""Training ops: distillation, healing, fine-tuning, QAT. All write a merged model
(llama.cpp and Ollama cannot apply LoRAs at runtime, so adapters are always merged)."""

from dataclasses import dataclass

from blackhole.core.ops import Needs, Op, p, register


@dataclass
class TrainParams:
    model: str = p("", "student; defaults to the last model in the chain")
    data: str = p("", "training rows")
    epochs: float = p(1.0)
    lr: float = p(1e-4)
    lora_r: int = p(16, "0 = full fine-tune")
    targets: list = p(["q_proj", "k_proj", "v_proj", "o_proj"])
    train_experts: bool = p(True)
    max_len: int = p(2048)
    max_minutes: float = p(60.0, "wall-clock cap; training stops, merges and saves")


@register
class HealKD(Op):
    """Heal a pruned or quantised model toward its unpruned teacher: SFT on the teacher's
    answers, plus logit KL when kd_weight > 0. NaN/inf steps are skipped.

    Always a trained distillation. A closed-form, no-training heal (refitting the next
    layer by least squares) made Qwen3-8B worse in every B200 test (2026-10-23), so there
    is no such mode: heal with training, or leave the stage out."""
    name, major = "heal.kd", 1
    consumes = {"model": "model", "teacher": "model", "data": "dataset"}
    produces = {"model": "model", "curve": "file"}
    needs = Needs(gpu=True)
    ported_from = "LobBot stages/heal.py"

    @dataclass
    class Params(TrainParams):
        teacher: str = p("${base.model}")
        kd_weight: float = p(0.5, "weight of logit KL to the teacher; 0 = plain SFT")
        fake_quant: str = p("", "heal against fake-quantised weights of this GGUF type map (final heal)")


@register
class DistilKD(Op):
    """Distil a teacher into a different (usually smaller, dense) student."""
    name, major = "distil.kd", 1
    consumes = {"model": "model", "teacher": "model", "data": "dataset"}
    produces = {"model": "model", "curve": "file"}
    needs = Needs(gpu=True)
    ported_from = "LobBot stages/heal.py dense student; BH event_horizon.py end-to-end KD"

    @dataclass
    class Params(TrainParams):
        student: str = p("", "HF id of the student when it is not the chain's model")
        teacher: str = p("${base.model}")
        kd_weight: float = p(1.0)
        temperature: float = p(1.0)


@register
class SftLora(Op):
    """Plain supervised fine-tune with a merged LoRA."""
    name, major = "sft.lora", 1
    consumes = {"model": "model", "data": "dataset"}
    produces = {"model": "model", "curve": "file"}
    needs = Needs(gpu=True)
    ported_from = "LobBot stages/sft.py"

    @dataclass
    class Params(TrainParams):
        pass


@register
class RftSelf(Op):
    """Rejection-sampling fine-tune on the model's own passing answers, in rounds. A round
    is kept only if held-out KL drops and paired pass@1 does not fall (BH round rule)."""
    name, major = "rft.self", 1
    consumes = {"model": "model", "prompts": "dataset"}
    produces = {"model": "model", "rounds": "file"}
    needs = Needs(gpu=True, tools=("vllm",))
    ported_from = "BH event_horizon.py on-policy rounds, round_rule()"

    @dataclass
    class Params(TrainParams):
        prompts: str = p("")
        rounds: int = p(4)
        stop_after_unkept: int = p(2)
        checker: str = p("tests")


@register
class BlockAP(Op):
    """Quantisation-aware training block by block: each quantised block is trained to
    reproduce the full-precision block's outputs, then the scales are frozen."""
    name, major = "qat.block_ap", 1
    consumes = {"model": "model", "data": "dataset", "type_map": "file"}
    produces = {"model": "model", "kl": "file"}
    needs = Needs(gpu=True)
    ported_from = "BH event_horizon.py Block-AP; blackhole_ternary.py layer-by-layer"

    @dataclass
    class Params:
        model: str = p("")
        data: str = p("")
        type_map: str = p("", "per-tensor GGUF types, e.g. ${map.type_map}")
        tokens: int = p(4_000_000)


@register
class MergeLora(Op):
    """Merge a LoRA adapter directory into a model."""
    name, major = "merge.lora", 1
    consumes = {"model": "model", "adapter": "dir"}
    produces = {"model": "model"}
    ported_from = "LobBot stages/heal.py merge"

    @dataclass
    class Params:
        model: str = p("")
        adapter: str = p("")


@register
class Ste(Op):
    """QAT to one GGUF type with straight-through rounding: every weight trains computing
    as llama.cpp will store it. The following quant.gguf must use that type, no imatrix."""
    name, major = "qat.ste", 1
    consumes = {"model": "model", "data": "dataset"}
    produces = {"model": "model", "curve": "file"}
    needs = Needs(gpu=True)
    ported_from = "mugge train/qat (Q4_0, TQ2_0, Q8_0); llmap stages/kquant.py for K-quants"

    @dataclass
    class Params(TrainParams):
        type: str = p("Q4_0")


@register
class RouteAnneal(Op):
    """Post-train a MoEfied model to route: start with every expert on (exactly the dense
    model), then step the active experts down along `schedule`, distilling from the dense
    teacher at each step (router straight-through gate + load-balance aux loss). LAB."""
    name, major = "heal.route_anneal", 1
    consumes = {"model": "model", "teacher": "model", "data": "dataset"}
    produces = {"model": "model", "curve": "file"}
    needs = Needs(gpu=True)
    ported_from = "reaper/gpu_perf/moefy_anneal.py"

    @dataclass
    class Params:
        model: str = p("", "MoEfied student; defaults to the last model in the chain")
        teacher: str = p("${base.model}")
        schedule: str = p("64,48,32,24,16", "experts on per token, one distillation step each")
        max_minutes: float = p(180.0, "wall-clock budget, split equally over the schedule")
        data: dict = p({}, "mix, e.g. {code = 0.5, chat = 0.3, math = 0.2}")
        lr: float = p(1e-5)
        router_lr: float = p(1e-3)
        aux_weight: float = p(0.01)
