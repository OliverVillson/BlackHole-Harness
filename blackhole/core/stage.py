"""One stage in its own process: `python -m blackhole.core.stage --job DIR --stage ID`.

The runner wrote DIR/stages/ID/params.json with the op's resolved params; this
builds the op's Context and runs it. An exception becomes an error event and
exit code 1 (REAPer's stage_main)."""

from __future__ import annotations

import argparse
import json
import sys
import traceback

from blackhole.core import ops
from blackhole.core.events import emit
from blackhole.core.job import Job
from blackhole.core.schema import USES


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--stage", required=True)
    a = ap.parse_args(argv)
    job = Job(a.job)
    d = job.stage_dir(a.stage)
    spec = json.loads((d / "params.json").read_text())
    name, major = USES.match(spec["uses"]).groups()
    cls = ops.get(name, int(major))
    ctx = ops.Context(job=job, stage=a.stage, params=ops.make_params(cls, spec["params"]),
                      hw=job.read("hardware.json") or {}, stage_dir=d)
    try:
        cls().run(ctx)
    except Exception as e:
        traceback.print_exc()
        emit(a.stage, "error", msg=f"{type(e).__name__}: {e}"[:400])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
