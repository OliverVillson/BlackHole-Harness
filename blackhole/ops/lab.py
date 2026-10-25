"""lab.*: room for ideas that are not ops yet.

lab.script runs a Python file from the schema's folder (or a path) as an op: the file
defines `run(ctx)` and reads its params from ctx.params.args, its inputs as paths,
and writes whatever it declares. It is cached, gated, logged and receipted like any
op, so an idea can be tried on real models before it becomes a proper op with typed
params. Schemas that use lab ops are marked experimental in the marketplace and are
never "proven" until the script is promoted to an op."""

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from blackhole.core.ops import Op, p, register


@register
class Script(Op):
    """Run a Python file's run(ctx). Inputs are any artifacts passed in `inputs`; the
    script writes to ctx.out(name) for each name in `outputs` and may report metrics."""
    name, major = "lab.script", 1
    consumes = {"model": "model"}
    produces = {"model": "model", "out": "dir"}
    ported_from = "new"

    @dataclass
    class Params:
        file: str = p("", "path to the .py file, relative to the schema")
        model: str = p("")
        inputs: dict = p({}, "name -> ${stage.artifact}")
        args: dict = p({}, "free-form params for the script")
        code_sha256: str = p("", "pinned hash of the file; the cache key includes it")

    def run(self, ctx):
        path = Path(ctx.params.file)
        if not path.is_absolute():
            path = ctx.job.root / path
        spec = importlib.util.spec_from_file_location("lab_script", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.run(ctx)
