"""Schemas: declarative, shareable size-reduction and post-training recipes.

The format is mugge's recipe.toml (OliverVillson/mugge, mugge/recipe.py) with
marketplace metadata, user inputs and gates added. A schema names an ordered
list of stages, each an op from the catalog (blackhole.ops) with its params
flat in the table, and runs on any base model its [targets] allow:

    api = "blackhole/v1"
    id = "someone/reap-purpose"           # author/name
    version = "1.0.0"
    title = "Prune a MoE for one purpose"
    license = "Apache-2.0"

    [targets]
    architectures = ["moe"]

    [inputs.purpose]                     # the knobs a user sees; simple mode asks only required, non-advanced ones
    type = "text"
    required = true
    help = "What should the model be good at?"

    [[stage]]
    kind = "data.teacher_gen"            # major version 1 implied; "data.teacher_gen@2" for 2
    purpose = "${inputs.purpose}"
    samples = 512

    [[stage]]
    kind = "score.reap"                  # data defaults to the data so far (calib's), model to the model so far

Reserved keys in a stage: id (default: the kind's namespace, numbered when
taken: score, prune, eval, eval2), kind, needs, gates, when, always. Everything else is the op's params.

References in params, `when` and gate values:

    ${inputs.<name>}       a user input (defaults applied)
    ${base.model}          the base model directory; ${base.id} its Hugging Face id
    ${base.<fact>}         what model.detect found: kinds, experts, top_k, params_b, layers, vocab, ...
    ${<stage>.<artifact>}  an artifact a previous stage produced
    ${<stage>.metrics.<m>} a metric a previous stage reported
    ${hw.<field>}          the detected hardware (gpu_count, gpu_mem_gb, ram_gb, ...)

A string that is exactly one reference keeps the referenced value's type.
Like mugge, an op's `model` param that is not given is the model so far (the
last stage that produced one, else the base model) and its `data` param the
data so far (the last stage that produced a dataset, else the schema's `data`).
JSON with the same keys is an equivalent serialisation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

API_VERSION = "blackhole/v1"
REF = re.compile(r"\$\{([A-Za-z_][\w.-]*)\}")
USES = re.compile(r"^([a-z][a-z0-9_]*\.[a-z][a-z0-9_]*)@(\d+)$")
INPUT_TYPES = ("text", "int", "float", "bool", "enum", "model", "path")
GATE_OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
            ">=": lambda a, b: a >= b, "==": lambda a, b: a == b, "!=": lambda a, b: a != b}


class SchemaError(ValueError):
    pass


@dataclass
class Gate:
    metric: str                  # a metric of this stage, or <stage>.<metric> in eval.gates
    op: str
    value: Any
    on_fail: str = "stop"        # stop: skip every later stage but `always` ones | warn

    def check(self, got: Any) -> bool:
        return got is not None and GATE_OPS[self.op](got, self.value)

    def describe(self, got: Any) -> str:
        return f"{self.metric} {got} {self.op} {self.value}"


@dataclass
class Stage:
    id: str
    uses: str                    # "prune.reap@1"
    params: dict = field(default_factory=dict)       # the op's params, unresolved
    needs: dict = field(default_factory=dict)        # hardware overrides: {gpu_mem_gb, gpus, disk_gb}
    gates: list[Gate] = field(default_factory=list)
    when: Any = True             # skipped when it resolves falsy
    always: bool = False         # runs even after a stop gate or an error (uploads)

    @property
    def op(self) -> str:
        return USES.match(self.uses).group(1)

    @property
    def major(self) -> int:
        return int(USES.match(self.uses).group(2))


@dataclass
class Schema:
    id: str
    version: str
    raw: dict
    inputs: dict[str, dict]
    stages: list[Stage]
    eval_gates: list[Gate]
    sha256: str

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"

    @property
    def footprint(self) -> dict:
        return self.raw.get("footprint") or {}

    @property
    def targets(self) -> dict:
        return self.raw.get("targets") or {}

    def stage(self, sid: str) -> Stage:
        return next(s for s in self.stages if s.id == sid)


def _gates(raw: list | None, where: str) -> list[Gate]:
    out = []
    for g in raw or []:
        if set(g) - GATE_KEYS:
            raise SchemaError(f"{where}: unknown gate keys {sorted(set(g) - GATE_KEYS)} "
                              "(a top-level key written after [[...gates]] lands inside the gate in TOML)")
        if g.get("op") not in GATE_OPS:
            raise SchemaError(f"{where}: gate op must be one of {sorted(GATE_OPS)}")
        if g.get("on_fail", "stop") not in ("stop", "warn"):
            raise SchemaError(f"{where}: on_fail is stop or warn")
        out.append(Gate(g["metric"], g["op"], g["value"], g.get("on_fail", "stop")))
    return out


RESERVED = {"id", "kind", "needs", "gates", "when", "always"}
TOP_KEYS = {"api", "id", "version", "title", "description", "author", "license", "tags", "parent", "base_compat",
            "task", "base", "data", "purpose", "outputs", "targets", "footprint", "data_policy", "inputs", "stage", "eval"}
GATE_KEYS = {"metric", "op", "value", "on_fail"}


def parse(raw: dict) -> Schema:
    """Structural checks only; op names and op params are checked against the catalog
    by the runner and `blackhole validate`, so a schema parses without torch."""
    unknown = set(raw) - TOP_KEYS
    if unknown:
        raise SchemaError(f"unknown top-level keys {sorted(unknown)}")
    if raw.get("api") != API_VERSION:
        raise SchemaError(f'api must be "{API_VERSION}"')
    if not re.match(r"^[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_.-]*$", raw.get("id", "")):
        raise SchemaError("id must be author/name (lowercase)")
    if not re.match(r"^\d+\.\d+\.\d+", str(raw.get("version", ""))):
        raise SchemaError("version must be semver")
    inputs = raw.get("inputs") or {}
    for name, spec in inputs.items():
        if spec.get("type") not in INPUT_TYPES:
            raise SchemaError(f"inputs.{name}.type must be one of {INPUT_TYPES}")
        if spec.get("type") == "enum" and not spec.get("choices"):
            raise SchemaError(f"inputs.{name}: enum needs choices")
    stages, seen = [], set()
    explicit = {s.get("id") for s in raw.get("stage") or [] if s.get("id")}
    for i, s in enumerate(raw.get("stage") or []):
        kind = s.get("kind", "")
        uses = kind if "@" in kind else f"{kind}@1"
        if not USES.match(uses):
            raise SchemaError(f"stage[{i}].kind {kind!r} must look like ns.op or ns.op@2")
        sid = s.get("id")
        if not sid:
            short, n = uses.split(".")[0], 2
            sid = short
            while sid in seen or sid in explicit:
                sid, n = f"{short}{n}", n + 1
        if not re.match(r"^[a-z][a-z0-9_]*$", sid) or sid in seen or sid in ("inputs", "base", "hw", "schema"):
            raise SchemaError(f"stage[{i}].id {sid!r} must be unique snake_case and not inputs/base/hw/schema")
        params = {k: v for k, v in s.items() if k not in RESERVED}
        for ref in REF.findall(json.dumps(params)) + REF.findall(json.dumps(s.get("when", ""))):
            head = ref.split(".")[0]
            if head not in ("inputs", "base", "hw", "schema") and head not in seen:
                raise SchemaError(f"stage {sid}: ${{{ref}}} refers to a stage that does not run before it")
            if head == "inputs" and ref.split(".")[1] not in inputs:
                raise SchemaError(f"stage {sid}: ${{{ref}}} is not a declared input")
        seen.add(sid)
        stages.append(Stage(sid, uses, params, s.get("needs") or {}, _gates(s.get("gates"), f"stage {sid}"),
                            s.get("when", True), bool(s.get("always"))))
    if not stages:
        raise SchemaError("a schema needs at least one [[stage]]")
    canon = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return Schema(raw["id"], str(raw["version"]), raw, inputs, stages,
                  _gates((raw.get("eval") or {}).get("gates"), "eval"), hashlib.sha256(canon.encode()).hexdigest())


def load(path: str | Path) -> Schema:
    text = Path(path).read_text()
    if str(path).endswith(".json"):
        return parse(json.loads(text))
    import tomllib
    return parse(tomllib.loads(text))


def input_values(schema: Schema, given: dict) -> dict:
    """User values with defaults applied and types coerced; raises on a missing required input."""
    out = {}
    for name, spec in schema.inputs.items():
        v = given.get(name, spec.get("default"))
        if v is None:
            if spec.get("required"):
                raise SchemaError(f"input {name!r} is required" + (f": {spec['help']}" if spec.get("help") else ""))
            out[name] = None
            continue
        t = spec["type"]
        if t == "int":
            v = int(v)
        elif t == "float":
            v = float(v)
        elif t == "bool":
            v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
        else:
            v = str(v)
        if t in ("int", "float") and (("min" in spec and v < spec["min"]) or ("max" in spec and v > spec["max"])):
            raise SchemaError(f"input {name}={v} outside [{spec.get('min')}, {spec.get('max')}]")
        if t == "enum" and v not in spec["choices"]:
            raise SchemaError(f"input {name}={v!r} not in {spec['choices']}")
        out[name] = v
    unknown = set(given) - set(schema.inputs)
    if unknown:
        raise SchemaError(f"unknown inputs {sorted(unknown)}")
    return out


def resolve(value: Any, scope: dict) -> Any:
    """Substitute ${a.b.c} references from scope (nested dicts) in a param tree."""
    def lookup(ref: str):
        cur: Any = scope
        for part in ref.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise SchemaError(f"${{{ref}}} is not available here")
            cur = cur[part]
        return cur

    if isinstance(value, str):
        m = REF.fullmatch(value.strip())
        if m:
            return lookup(m.group(1))
        return REF.sub(lambda m: str(lookup(m.group(1))), value)
    if isinstance(value, dict):
        return {k: resolve(v, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, scope) for v in value]
    return value
