"""The op interface: every stage of a schema runs one op.

An op is a class registered under a namespaced name and a major version:

    @register
    class ReapScore(Op):
        name, major = "score.reap", 1
        consumes = {"model": "model", "data": "dataset"}   # param name -> artifact kind
        produces = {"saliency": "file"}                     # artifact name -> kind
        needs = Needs(gpu=True, arch=("moe",))

        @dataclass
        class Params:
            data: str = ""            # filled by the runner from ${calib.data}
            model: str = ""           # defaults to the last model in the chain
            max_seq: int = 2048

        def plan(self, ctx) -> StagePlan: ...   # optional: memory, disk, time; adapt batch sizes
        def run(self, ctx) -> None: ...         # writes ctx.out(<artifact>), reports ctx.progress()

Ops run in their own process (`python -m blackhole.core.stage`), so the GPU is
free again when an op exits; the teacher and the student never share memory
unless an op chooses to. Params are a dataclass, so the catalog can describe
every op to the TUI, the validator and the frontier-model schema builder
(`describe()`), and a stage's params are checked before a GPU is touched.

Third-party ops ship in their own package and register through the
`blackhole.ops` entry point group, pointing at a module that imports and
registers them.
"""

from __future__ import annotations

import dataclasses
import json
import os
import typing
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, ClassVar

ARTIFACT_KINDS = ("model", "dataset", "file", "gguf", "dir")
BUILTIN = ("model", "lab", "plan", "data", "convert", "prune", "vocab", "train", "quant", "eval", "package", "sync")


@dataclass(frozen=True)
class Needs:
    gpu: bool = False
    arch: tuple[str, ...] = ()        # moe | dense | hybrid; empty = any
    tools: tuple[str, ...] = ()       # vllm, llama.cpp, frontier (an API key), ...


@dataclass
class StagePlan:
    """What an op expects to use on this machine, for the run plan the TUI shows
    before anything starts (and the cost estimate on a rented GPU)."""
    ok: bool = True
    gpu_mem_gb: float = 0.0
    disk_gb: float = 0.0
    hours: float = 0.0
    frontier_usd: float = 0.0
    env: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


class Op:
    name: ClassVar[str]
    major: ClassVar[int] = 1
    consumes: ClassVar[dict[str, str]] = {}
    produces: ClassVar[dict[str, str]] = {}
    needs: ClassVar[Needs] = Needs()
    Params: ClassVar[type]
    yields: ClassVar[tuple[str, ...]] = ()   # model kinds after this op (convert.moefy: dense -> moe); () = unchanged
    ported_from: ClassVar[str] = ""   # where the implementation comes from (REAPer, LobBot, llmap)

    def plan(self, ctx: "Context") -> StagePlan:
        return StagePlan()

    def run(self, ctx: "Context") -> None:
        raise NotImplementedError(f"{self.name}@{self.major}: not ported yet ({self.ported_from})")


@dataclass
class Context:
    """What a running op sees. Paths are absolute; the op never looks outside
    its stage directory except to read its inputs."""
    job: Any                          # blackhole.core.job.Job
    stage: str
    params: Any                       # the op's Params instance, resolved
    hw: dict
    stage_dir: Path

    def out(self, artifact: str) -> Path:
        """Where the op writes an artifact it declared in `produces`."""
        p = self.stage_dir / "out" / artifact
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def progress(self, pct: float | None = None, msg: str = "", **extra) -> None:
        from blackhole.core.events import emit
        emit(self.stage, "running", pct, msg, **extra)

    def metric(self, **values) -> None:
        """Record metrics; gates and the run receipt read them from metrics.json."""
        p = self.stage_dir / "metrics.json"
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur.update(values)
        p.write_text(json.dumps(cur, indent=2) + "\n")


_REGISTRY: dict[tuple[str, int], type[Op]] = {}
_loaded = False


def register(cls: type[Op]) -> type[Op]:
    key = (cls.name, cls.major)
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise ValueError(f"op {cls.name}@{cls.major} registered twice")
    _REGISTRY[key] = cls
    return cls


def _load() -> None:
    global _loaded
    if _loaded:
        return
    _loaded = True
    for mod in BUILTIN:
        import_module(f"blackhole.ops.{mod}")
    for ep in entry_points(group="blackhole.ops"):
        ep.load()
    for mod in filter(None, os.environ.get("BLACKHOLE_OPS", "").split(",")):  # dev: ops not installed as a package
        import_module(mod.strip())


def get(name: str, major: int) -> type[Op]:
    _load()
    try:
        return _REGISTRY[(name, major)]
    except KeyError:
        raise KeyError(f"unknown op {name}@{major}; `blackhole ops` lists them") from None


def catalog() -> list[type[Op]]:
    _load()
    return sorted(_REGISTRY.values(), key=lambda c: (c.name, c.major))


def make_params(cls: type[Op], values: dict) -> Any:
    known = {f.name for f in dataclasses.fields(cls.Params)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"{cls.name}@{cls.major}: unknown params {sorted(unknown)}")
    return cls.Params(**values)


def describe(cls: type[Op]) -> dict:
    """One op as JSON: what the TUI's op browser and the schema builder's prompt show."""
    hints = typing.get_type_hints(cls.Params)
    params = {}
    for f in dataclasses.fields(cls.Params):
        default = f.default if f.default is not dataclasses.MISSING else (
            f.default_factory() if f.default_factory is not dataclasses.MISSING else None)
        params[f.name] = {"type": getattr(hints[f.name], "__name__", str(hints[f.name])), "default": default,
                          "help": f.metadata.get("help", "")}
    return {"uses": f"{cls.name}@{cls.major}", "doc": (cls.__doc__ or "").strip(), "params": params,
            "consumes": cls.consumes, "produces": cls.produces, "needs": dataclasses.asdict(cls.needs),
            "ported_from": cls.ported_from}


def validate(schema) -> list[str]:
    """Problems with a parsed schema against this machine's catalog: unknown ops, unknown
    params, wrong architectures. Empty when it can run. What `blackhole validate` and the
    schema builder's repair loop call."""
    problems = []
    arch = set(schema.targets.get("architectures") or [])   # the model's kinds as the chain goes
    for st in schema.stages:
        try:
            cls = get(st.op, st.major)
        except KeyError as e:
            problems.append(f"stage {st.id}: {e.args[0]}")
            continue
        known = {f.name for f in dataclasses.fields(cls.Params)}
        for k in set(st.params) - known:
            problems.append(f"stage {st.id}: {st.uses} has no param {k!r} (has {sorted(known)})")
        if cls.needs.arch and arch and not arch & set(cls.needs.arch):
            problems.append(f"stage {st.id}: {st.uses} needs {cls.needs.arch}, the model is {sorted(arch)} here")
        if cls.yields and arch:
            arch = set(cls.yields)
    from blackhole.core.lint import lint
    return problems + lint(schema.raw)


def p(default, help: str = ""):
    """A param with help text: `ratio: float = p(0.5, "share of routed experts dropped")`."""
    if isinstance(default, (list, dict)):
        return field(default_factory=lambda: type(default)(default), metadata={"help": help})
    return field(default=default, metadata={"help": help})
