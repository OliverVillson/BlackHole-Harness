"""model.*: finding out what the base model is, and getting it onto the machine."""

import json
from dataclasses import dataclass
from pathlib import Path

from blackhole.core.ops import Op, p, register


@register
class Detect(Op):
    """Read config.json, the safetensors index and tokenizer_config.json (a few KB, never
    the weights) and record what the model is: kinds (moe, dense, hybrid, vision), experts,
    top-k, shared experts, layer split, sizes, dtype, quantisation, KV bytes per token,
    chat template, remote code. The runner adds it as the first stage of every run, puts
    the result in ${base.*}, and refuses the run if the schema's targets or any stage's
    needs do not match, before anything is downloaded."""
    name, major = "model.detect", 1
    produces = {"info": "file"}
    ported_from = "REAPer reaper_hw/model.py + reaper_tui/backend.py resolve_model"

    @dataclass
    class Params:
        model: str = p("", "Hugging Face id or local directory; defaults to the job's base")

    def run(self, ctx):
        from blackhole.hw.model import detect
        base = ctx.params.model or ctx.job.base
        info = detect(base, base if Path(base).expanduser().is_dir() else None).to_dict()
        ctx.out("info").write_text(json.dumps(info, indent=2) + "\n")
        ctx.metric(**{k: info[k] for k in ("kinds", "params_b", "active_b", "experts", "top_k", "checkpoint_gb")})
        ctx.progress(100, f"{info['model_type']} {'/'.join(info['kinds'])} {info['params_b']}B"
                     + (f", {info['experts']} experts top-{info['top_k']}" if info["experts"] else ""))


@register
class Fetch(Op):
    """Download the base model to $BLACKHOLE_HOME/models/<org>--<name>. The runner adds
    this as the second stage (`weights`) unless the base model is a local directory."""
    name, major = "model.fetch", 1
    ported_from = "REAPer stages/weights.py"

    @dataclass
    class Params:
        revision: str = p("main", "Hugging Face revision to pin")
