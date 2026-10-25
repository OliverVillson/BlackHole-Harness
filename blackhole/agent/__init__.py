"""Frontier models in the loop: describe an idea in words, get a schema you can run.

    from blackhole.agent import draft
    d = draft("a tiny SQL model for my 8 GB laptop that never invents columns",
              base="Qwen/Qwen3-30B-A3B-Instruct-2507", hw=detect(), provider="claude")
    d.questions        # non-empty: show them, then call draft(..., answers={q: a}) again
    d.schema_toml      # else the accepted schema
    d.summary, d.estimate, d.problems, d.usage

This is mugge's `new` + `auto` flow aimed at blackhole/v1 schemas:

1. Clarify. The model may ask up to three short questions (tool `ask`). In the TUI the
   loop stops there and the questions come back in Draft.questions; the CLI can pass an
   `answer` callback instead and keep going.
2. Ground. It reads base-model facts and fits (`list_models`, `detect_model` for any model,
   `estimate_footprint`) and
   the marketplace (`search_schemas`, `get_schema`) so it can fork a proven schema.
3. Draft and repair. It calls `propose_schema(toml)`. The harness runs schema.parse,
   ops.validate against this VM's op catalog plus core.lint's publish rules; every problem
   goes back as the tool result. After `max_repairs` rejected drafts it stops and
   returns the problems.
4. Explain. Its last message is a plain-words summary. Nothing runs until the user has
   seen the schema, the estimate and the frontier spend.

The frontier model designs; it does not write training answers. Answers come from the
open teacher on the VM (data.teacher_gen), so a draft costs cents and closed-model
outputs stay out of training data unless the user raises data_policy.frontier_spend_usd.
The `topics` it writes for data.teacher_gen are the calibration plan: the purpose broken
into concrete sub-skills, which is what REAPer's purpose string turned into on its own.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from blackhole.agent.providers import Provider, ProviderError, StopLoop, Tool, ToolError, Usage, get
from blackhole.core import ops
from blackhole.core import schema as sch
from blackhole.hw import Hardware
from blackhole.core.lint import lint
from blackhole.market import Registry
from blackhole.market import models as mm

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
EXAMPLES = ("reap-purpose.toml", "laptop-moe-gguf.toml")  # bundled fallbacks when the marketplace is offline
HIDDEN_OPS = ("lab.",)  # ops the builder never offers: lab.script runs code shipped with the schema


@dataclass
class Draft:
    schema_toml: str = ""
    questions: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    estimate: dict = field(default_factory=dict)     # {gpu_hours, frontier_usd, footprint_gb, ...}
    summary: str = ""
    usage: Usage = field(default_factory=Usage)
    cost_usd: float | None = None                     # what this draft cost in API spend
    drafts: int = 0


def _catalog() -> str:
    lines = []
    for cls in ops.catalog():
        d = ops.describe(cls)
        if d["uses"].startswith(HIDDEN_OPS):
            continue
        params = ", ".join(f"{k}={json.dumps(v['default'])}" + (f" ({v['help']})" if v["help"] else "")
                           for k, v in d["params"].items())
        arch = f" arch={list(d['needs']['arch'])}" if d["needs"]["arch"] else ""
        doc = d["doc"].splitlines()[0] if d["doc"] else ""
        lines.append(f"- {d['uses'].removesuffix('@1')}: {doc} consumes={d['consumes']} produces={d['produces']}"
                     f"{arch}\n    params: {params}")
    return "\n".join(lines)


def system_prompt(examples: list[str]) -> str:
    shown = "\n\n".join(f"```toml\n{e.strip()}\n```" for e in examples)
    return f"""\
You design post-training schemas for BlackHole-Harness, a terminal app that shrinks large \
open-weight models so people can run a model they made themselves on a mid-tier PC. The user \
describes in plain words what they want their model to be. You turn that into a schema: a TOML \
file in the blackhole/v1 format that the runner on their GPU VM executes stage by stage.

How to work:
1. If something essential is missing (what the model is for, the PC it must run on, an output \
format they need), call ask with at most three short questions. Skip anything the description \
or the context already answers; prefer a sensible default to a question.
2. Ground the design. Call list_models for the target device, detect_model on any base the user \
names that is not listed, and estimate_footprint for the recipe you have in mind. Call search_schemas; if a proven or verified schema already fits, fork \
it with get_schema and set parent = "id@version".
3. Write the schema and call propose_schema with the complete TOML. If it comes back REJECTED, \
fix every problem and propose again. You are done when it says ACCEPTED.
4. Then reply with a short plain-words summary for the user: what the model will be good at, \
the base, the main steps, the expected size on their device, and which inputs they can tune.

Format rules:
- Top-level keys first (api, id, version, title, description, author, license, tags, base_compat, \
parent, outputs), then tables. In TOML a key written after a [table] header belongs to that \
table, so outputs must come before the first table.
- Stages are [[stage]] tables with kind = "ns.op" and the op's params flat in the table. Reserved \
stage keys: id, kind, needs, gates, when, always. A stage's default id is the kind's namespace \
(score, prune, eval, eval2...).
- model and data params chain implicitly from the last stage that produced one; set them only \
to point elsewhere, e.g. "${{calib.heldout}}" or "${{distil.model}}". Other references: \
"${{inputs.name}}", "${{base.model}}", "${{hw.gpu_mem_gb}}".
- Expose the knobs a user may want as [inputs.name] (type text|int|float|bool|enum|model|path, \
default, min, max, choices, required, advanced, help). Simple mode asks only required, \
non-advanced inputs, so a required input is never advanced.
- id is "<author>/<kebab-name>" using the author from the context; a new schema starts at 0.1.0.

Design rules:
- Use the base the user named. Otherwise pick the strongest base whose recipe fits the device \
per list_models. REAP (score.reap then prune.reap) is for MoE and hybrid bases; dense bases use \
prune.structured and vocab.trim.
- Begin with data.teacher_gen. Write its purpose as one clear sentence and its topics as 8 to 30 \
concrete sub-skills and situations that cover what the user will really ask; this is the \
calibration plan. The open teacher on the VM answers; never set a frontier teacher unless the \
user asked for one, and then set data_policy.frontier_spend_usd.
- Then data.decontam. Heal with heal.kd after pruning 40% or more or before going below ~5 bits.
- If answers can be checked (code with tests, JSON, SQL against a schema, tool calls), add \
rft.self or data.own_rollouts: training on the model's own verified answers was the biggest \
measured gain in this project (12% -> 80% in the target format).
- Always add eval.agreement after pruning, eval.footprint when [footprint] is set, eval.bench \
with suites that match the purpose, [[eval.gates]] the marketplace can check, and one package \
stage per output (gguf -> package.gguf, safetensors-* -> package.hf, mlx -> quant.mlx).
- Laptops and 8-16 GB GPUs need GGUF (quant.gguf_map + quant.gguf + package.gguf); Macs can \
also take MLX; vLLM servers take safetensors (fp8 experts or w4a16).
- Use only the stage kinds below, with only their params. Default licence allow-list: MIT, \
Apache-2.0, BSD-3-Clause, CC0-1.0, CC-BY-4.0, ODC-By-1.0.

Stage kinds on this VM:
{_catalog()}

Proven schemas to learn the style from:
{shown}
"""


class Builder:
    def __init__(self, provider: Provider, *, registry: Registry | None = None, hw: Hardware | None = None,
                 device: str = "8GB-VRAM", author: str = "me",
                 answer: Callable[[list[str]], list[str]] | None = None, max_repairs: int = 3):
        self.provider, self.registry, self.hw = provider, registry, hw or Hardware()
        self.device, self.author, self.answer, self.max_repairs = device, author, answer, max_repairs
        self.accepted: dict | None = None
        self.accepted_toml = ""
        self.questions: list[str] = []
        self.problems: list[str] = []
        self.drafts = 0

    # ------------------------------------------------------------ tools

    def _ask(self, a: dict) -> str:
        qs = [q for q in a["questions"] if q.strip()][:3]
        if not qs:
            raise ToolError("ask needs at least one question")
        if self.answer is None:
            self.questions = qs
            raise StopLoop("waiting for the user")
        answers = self.answer(qs)
        return json.dumps(dict(zip(qs, answers)))

    def _list_models(self, a: dict) -> str:
        device = a.get("device") or self.device
        if device not in mm.DEVICES:
            raise ToolError(f"unknown device {device!r}; one of {list(mm.DEVICES)}")
        return json.dumps(mm.table(self.hw, device, a.get("context", 16384)))

    def _estimate(self, a: dict) -> str:
        m = mm.lookup(a["model"])
        if m is None:
            raise ToolError(f"cannot find or detect model {a['model']!r}; use a list_models key or a Hugging Face id")
        quant = a.get("quant", "Q4_K_M")
        if quant not in mm.BITS:
            raise ToolError(f"unknown quant {quant!r}; one of {list(mm.BITS)}")
        est = mm.estimate_gb(m, a.get("ratio", 0.0), quant, context=a.get("context", 16384), kv=a.get("kv", "q8_0"))
        device = a.get("device") or self.device
        budget = mm.DEVICES.get(device)
        return json.dumps({**est, "device": device, "budget_gb": budget,
                           "fits": budget is not None and est["total_gb"] <= budget})

    def _detect(self, a: dict) -> str:
        from blackhole.hw.model import detect, resolve
        try:
            rid, others = resolve(a["model"], known=[c.hf_id for c in mm.CATALOG])
            if not rid:
                raise ToolError(f"no model matches {a['model']!r}")
            return json.dumps({**detect(rid).to_dict(), "other_matches": others})
        except RuntimeError as e:
            raise ToolError(str(e)) from e

    def _search(self, a: dict) -> str:
        if self.registry is None:
            return "[] (marketplace unreachable; design from scratch)"
        rows = self.registry.search(a.get("query", ""), arch=a.get("arch", ""), device=self.device)[:8]
        return json.dumps([{k: r[k] for k in ("id", "version", "title", "description", "badge", "tested_on",
                                              "results", "fits_your_pc")} for r in rows if not r["experimental"]])

    def _get_schema(self, a: dict) -> str:
        if self.registry is None:
            raise ToolError("marketplace unreachable")
        try:
            return self.registry.text(a["ref"])
        except (KeyError, ValueError, OSError) as e:
            raise ToolError(str(e)) from e

    def _propose(self, a: dict) -> str:
        self.drafts += 1
        problems = check(a["toml"])
        if problems:
            self.problems = problems
            if self.drafts > self.max_repairs:
                raise StopLoop("too many rejected drafts")
            raise ToolError("REJECTED. Fix all of these and propose again:\n- " + "\n- ".join(problems))
        self.accepted, self.accepted_toml, self.problems = tomllib.loads(a["toml"]), a["toml"], []
        return "ACCEPTED. Now give the user the short summary."

    def tools(self) -> list[Tool]:
        def obj(props: dict, required: tuple = ()) -> dict:
            return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}

        devices = {"type": "string", "enum": list(mm.DEVICES)}
        return [
            Tool("ask", "Ask the user up to three short questions at once. Use only when the answer changes the design.",
                 obj({"questions": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}, ("questions",)),
                 self._ask),
            Tool("list_models", "Base models: size, experts, whether this VM can run a pipeline on them, and the "
                 "gentlest prune ratio + GGUF quant that fits the target device.",
                 obj({"device": devices, "context": {"type": "integer"}}), self._list_models),
            Tool("detect_model", "Facts about any model by name, Hugging Face id or local folder, read from its "
                 "config: kinds (moe/dense/hybrid/vision), experts, top-k, params, KV bytes per token, chat template.",
                 obj({"model": {"type": "string"}}, ("model",)), self._detect),
            Tool("estimate_footprint", "Estimated GB on the device after pruning and quant, KV cache included.",
                 obj({"model": {"type": "string"}, "ratio": {"type": "number", "description": "share of experts dropped"},
                      "quant": {"type": "string", "enum": list(mm.BITS)}, "context": {"type": "integer"},
                      "kv": {"type": "string", "enum": list(mm.KV_BITS)}, "device": devices}, ("model",)),
                 self._estimate),
            Tool("search_schemas", "Search the schema marketplace, best badge first, with results per base model.",
                 obj({"query": {"type": "string"}, "arch": {"type": "string", "enum": ["moe", "dense", "hybrid"]}}),
                 self._search),
            Tool("get_schema", "A marketplace schema's TOML by 'id' or 'id@version', to fork it.",
                 obj({"ref": {"type": "string"}}, ("ref",)), self._get_schema),
            Tool("propose_schema", "Validate a complete schema. Returns ACCEPTED or the problems to fix.",
                 obj({"toml": {"type": "string", "description": "the whole schema as TOML"}}, ("toml",)),
                 self._propose),
        ]

    # ------------------------------------------------------------ entry point

    def examples(self) -> list[str]:
        out = []
        if self.registry is not None:
            for row in self.registry.search(device=self.device):
                if row["badge"] in ("proven", "verified") and not row["experimental"] and len(out) < 2:
                    try:
                        out.append(self.registry.text(row["id"]))
                    except (KeyError, ValueError, OSError):
                        pass
        for name in EXAMPLES:
            if len(out) >= 2:
                break
            p = SCHEMAS_DIR / name
            if p.exists():
                out.append(p.read_text())
        return out

    def run(self, description: str, *, base: str | None = None, answers: dict[str, str] | None = None,
            on_text: Callable[[str], None] | None = None, max_turns: int = 30,
            budget_usd: float | None = None) -> Draft:
        context = {
            "author": self.author,
            "target_device": self.device,
            "device_budget_gb": mm.DEVICES.get(self.device),
            "vm": {"gpu": self.hw.gpu_name or "none", "gpus": self.hw.gpu_count, "gpu_mem_gb": self.hw.gpu_mem_gb,
                   "disk_free_gb": self.hw.disk_free_gb},
            "base_model_requested": base,
        }
        prompt = f"Context: {json.dumps(context)}\n\nWhat the user wants:\n{description}"
        if answers:
            prompt += "\n\nThe user already answered these questions:\n" + "\n".join(
                f"- {q}\n  {a}" for q, a in answers.items())
        result = self.provider.run(system_prompt(self.examples()), prompt, self.tools(),
                                   max_turns=max_turns, on_text=on_text, budget_usd=budget_usd)
        d = Draft(usage=result.usage, cost_usd=result.usage.cost_usd(self.provider.prices), drafts=self.drafts)
        if self.questions and self.accepted is None:
            d.questions = self.questions
            return d
        if self.accepted is None:
            d.problems = self.problems or [f"the model stopped without a schema: {result.text[:300]}"]
            return d
        d.schema_toml, d.summary, d.estimate = self.accepted_toml, result.text.strip(), estimate(self.accepted, base)
        return d


def check(toml_text: str) -> list[str]:
    """Every problem with a drafted schema, in words the model can act on."""
    try:
        raw = tomllib.loads(toml_text)
    except tomllib.TOMLDecodeError as e:
        return [f"TOML does not parse: {e}"]
    try:
        s = sch.parse(raw)
    except sch.SchemaError as e:
        return [str(e)]
    problems = ops.validate(s) + lint(raw, publish=True)
    problems += [f"stage {st.id}: {st.op} is not available to the builder" for st in s.stages
                 if st.op.startswith(HIDDEN_OPS)]
    return problems


def draft(description: str, base: str | None = None, hw: Hardware | None = None, provider: str | Provider = "claude",
          *, device: str = "8GB-VRAM", author: str = "me", registry: Registry | None = None,
          answers: dict[str, str] | None = None, answer: Callable[[list[str]], list[str]] | None = None,
          on_text: Callable[[str], None] | None = None, budget_usd: float | None = None) -> Draft:
    prov = get(provider) if isinstance(provider, str) else provider
    b = Builder(prov, registry=registry, hw=hw, device=device, author=author, answer=answer)
    return b.run(description, base=base, answers=answers, on_text=on_text, budget_usd=budget_usd)


# ---------------------------------------------------------------- rough estimate

# Sustained tokens/s on one GPU for a ~30B MoE: (teacher generation with vLLM, LoRA/KD training).
# Placeholders until blackhole.hw.plan_schema asks each op for its StagePlan.
GPU_SPEED = {"B200": (12000, 9000), "H100": (6000, 4500), "L40S": (2000, 1200)}


def estimate(raw: dict, base: str | None = None, gpu: str = "B200") -> dict:
    gen_tps, train_tps = GPU_SPEED.get(gpu, GPU_SPEED["H100"])
    rows, total, samples = [], 0.0, 512
    for st in raw.get("stage") or []:
        kind = st["kind"].split("@")[0]
        if kind == "data.teacher_gen":
            samples = st.get("samples") if isinstance(st.get("samples"), int) else 512
            tokens = samples * (st.get("answer_max_tokens", 1024) if isinstance(st.get("answer_max_tokens"), int)
                                else 1024)
            h = tokens / gen_tps / 3600 + 0.1  # + model load
        elif kind in ("heal.kd", "distil.kd", "sft.lora", "qat.ste", "rft.self"):
            minutes = st.get("max_minutes")
            h = minutes / 60 if isinstance(minutes, (int, float)) else max(0.25, samples * 1500 * 2 / train_tps / 3600)
        elif kind in ("score.reap", "eval.bench", "quant.gguf_map", "qat.block_ap"):
            h = 0.3
        elif kind.startswith(("eval.", "quant.", "prune.", "vocab.")):
            h = 0.1
        else:
            h = 0.0
        rows.append({"stage": st.get("id") or kind, "kind": kind, "gpu_hours": round(h, 2)})
        total += h
    out = {"gpu": gpu, "gpu_hours": round(total, 2), "stages": rows,
           "frontier_usd": (raw.get("data_policy") or {}).get("frontier_spend_usd", 0)}
    m = mm.lookup(base or raw.get("base") or (raw.get("base_compat") or [""])[0] or "", detect=False)
    if m:
        ratio = next((st.get("ratio") for st in raw["stage"] if st["kind"].startswith("prune.reap")), 0.0)
        ratio = ratio if isinstance(ratio, (int, float)) else _input_default(raw, ratio, 0.5)
        fp = raw.get("footprint") or {}
        out["footprint_gb"] = mm.estimate_gb(m, ratio, "Q4_K_M", context=fp.get("context", 16384),
                                             kv=fp.get("kv", "q8_0"))["total_gb"]
        out["base"] = m.key
    return out


def _input_default(raw: dict, ref: str, fallback: float) -> float:
    name = str(ref).removeprefix("${inputs.").removesuffix("}")
    v = (raw.get("inputs") or {}).get(name, {}).get("default", fallback)
    return v if isinstance(v, (int, float)) else fallback


__all__ = ["Draft", "Builder", "draft", "check", "estimate", "system_prompt", "ProviderError"]
