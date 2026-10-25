"""Ops that need no GPU, for testing the runner: loaded through BLACKHOLE_OPS."""

import json
from dataclasses import dataclass

from blackhole.core.ops import Op, p, register


@register
class Gen(Op):
    """Writes a dataset and counts its runs."""
    name, major = "test.gen", 1
    produces = {"data": "dataset"}

    @dataclass
    class Params:
        n: int = p(3)

    def run(self, ctx):
        ctx.out("data").write_text("\n".join(json.dumps({"i": i}) for i in range(ctx.params.n)))
        ctx.metric(rows=ctx.params.n)


@register
class Shrink(Op):
    """Copies the model so far with a ratio applied; reports the ratio as a metric."""
    name, major = "test.shrink", 1
    consumes = {"model": "model", "data": "dataset"}
    produces = {"model": "model"}

    @dataclass
    class Params:
        model: str = p("")
        data: str = p("")
        ratio: float = p(0.5)

    def run(self, ctx):
        out = ctx.out("model")
        out.mkdir(parents=True, exist_ok=True)
        rows = len(open(ctx.params.data).read().splitlines())
        (out / "weights.json").write_text(json.dumps({"from": ctx.params.model, "ratio": ctx.params.ratio, "rows": rows}))
        ctx.progress(50, "half way")
        ctx.metric(kept=1 - ctx.params.ratio)


@register
class Upload(Op):
    name, major = "test.upload", 1

    @dataclass
    class Params:
        pass

    def run(self, ctx):
        (ctx.job.root / "uploaded").write_text("yes")
