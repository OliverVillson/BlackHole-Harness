"""data.*: the text every later stage learns from or is calibrated on."""

from dataclasses import dataclass

from blackhole.core.ops import Needs, Op, p, register


@register
class TeacherGen(Op):
    """The teacher writes prompts for a purpose and answers them. With a purpose it lists
    sub-tasks, writes varied self-contained requests for each, mixes in the closest public
    preset, and answers every prompt; held-out answers are kept back for eval.

    On a B200 the stage's time goes to vLLM startup and the prompt-writer rounds, not the
    answers, so the port keeps the written topics and prompts as their own cached files
    (keyed by teacher + purpose + writer settings, reused when only samples or answer
    settings change) and keeps one vLLM server up across the writer and answer rounds."""
    name, major = "data.teacher_gen", 1
    consumes = {"model": "model"}
    produces = {"data": "dataset", "heldout": "dataset", "topics": "file"}
    needs = Needs(gpu=True, tools=("vllm",))
    ported_from = "REAPer stages/calibrate.py + calib.py + teacher.py"

    @dataclass
    class Params:
        model: str = p("", "teacher; defaults to the base model")
        purpose: str = p("", "free text; empty uses the preset in `preset`")
        topics: list = p([], "sub-skills of the purpose (e.g. from the schema builder); empty = the teacher lists its own")
        preset: str = p("mix", "mix | chat | code | math | builtin")
        samples: int = p(64, "prompts to generate and answer; 64 calibration sequences scored as well as 256 on Qwen3-30B-A3B (B200, 2026-10-23)")
        heldout: int = p(32, "answers kept back for eval")
        purpose_mix: float = p(0.25, "share taken from the closest public preset")
        answer_max_tokens: int = p(1024)
        max_seq: int = p(2048)
        teacher: str = p("local", "local (vLLM, else transformers) | frontier:<model> (Claude/GPT, needs a key and spend cap)")
        thinking: bool = p(False)


@register
class Dataset(Op):
    """A public or local dataset, filtered to the licence allow-list in data_policy."""
    name, major = "data.dataset", 1
    produces = {"data": "dataset"}
    ported_from = "REAPer calib.py presets; BH bhdata_io.py"

    @dataclass
    class Params:
        source: str = p("", "hf:<repo>[:split] | path to .jsonl | bhdata kits dir")
        samples: int = p(0, "0 = all")
        fields: dict = p({}, "column mapping to {prompt, answer}")


@register
class Decontam(Op):
    """Drop rows that share n-grams with eval suites or carry their canaries."""
    name, major = "data.decontam", 1
    consumes = {"data": "dataset"}
    produces = {"data": "dataset", "report": "file"}
    ported_from = "BH scripts/bh/decontam.py"

    @dataclass
    class Params:
        data: str = p("")
        against: list = p(["humaneval", "mbpp", "livecodebench"], "suites and canaries to exclude")
        ngram: int = p(13)


@register
class Mix(Op):
    """Weighted mix of datasets, deduplicated and shuffled."""
    name, major = "data.mix", 1
    produces = {"data": "dataset"}
    ported_from = "BH scripts/bh/data.py build"

    @dataclass
    class Params:
        sources: list = p([], "[{data: ${a.data}, weight: 0.7}, ...]")
        samples: int = p(0)
        seed: int = p(0)


@register
class OwnRollouts(Op):
    """The current model answers prompts; answers that pass a checker (tests, a judge, an
    exact match) become training rows. llmap exp02's 12% -> 80% came from this."""
    name, major = "data.own_rollouts", 1
    consumes = {"model": "model", "prompts": "dataset"}
    produces = {"data": "dataset", "stats": "file"}
    needs = Needs(gpu=True, tools=("vllm",))
    ported_from = "BH scripts/bh/harvest.py, data.py rft/repair"

    @dataclass
    class Params:
        model: str = p("")
        prompts: str = p("")
        samples_per_prompt: int = p(4)
        checker: str = p("tests", "tests | judge:<model> | exact")
