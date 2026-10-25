"""`blackhole` / `bh`: the TUI with no arguments, else a subcommand.

    blackhole                                         the TUI (reattaches to a running job)
    blackhole run SCHEMA MODEL [--set k=v ...] [--fg] start a job (detached unless --fg)
    blackhole detect MODEL [--schemas DIR] [--json]    what a model is (any HF id, fuzzy name or folder) and which schemas fit
    blackhole validate SCHEMA                          parse + check against the op catalog
    blackhole ops [--json]                             the op catalog
    blackhole inspect MODEL [QUERY]                    model structure and weights (with blackhole-dev installed)
    blackhole fn list|show|run                         named weight functions (with blackhole-dev installed)
    blackhole jobs | logs [JOB] | stop [JOB]           follow and control jobs
    blackhole describe "TEXT" [--model M]              frontier model drafts a schema (blackhole.agent)
    blackhole market search|pull|publish|submit        the registry (blackhole.market)
    blackhole hw [MODEL]                               hardware and plan (blackhole.hw)
    blackhole up|down                                  rent / release a GPU VM (blackhole.cloud)
"""

from __future__ import annotations

import argparse
import json
import sys

from blackhole.core import ops
from blackhole.core import schema as sch


def _sets(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        out[k] = v
    return out


COMMANDS = {"run", "validate", "detect", "ops", "inspect", "fn", "-h", "--help"}   # plus agent_cli.COMMANDS


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    from blackhole.cloud import cli as cloud_cli
    if argv and argv[0] in cloud_cli.COMMANDS:
        return cloud_cli.main(argv)
    from blackhole.agent import cli as agent_cli
    if not argv or argv[0] not in COMMANDS | set(agent_cli.COMMANDS):
        # anything else is the TUI (harness/blackhole_tui): `blackhole`, `blackhole flint --schema ...`
        try:
            from blackhole_tui.app import main as tui_main
        except ImportError:
            print("the TUI lives in harness/ (pip install -e harness); try `blackhole ops`", file=sys.stderr)
            return 2
        sys.argv = ["blackhole", *argv]
        tui_main()
        return 0
    ap = argparse.ArgumentParser(prog="blackhole", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("schema")
    r.add_argument("model")
    r.add_argument("--set", action="append", default=[], metavar="INPUT=VALUE")
    r.add_argument("--job")
    r.add_argument("--fg", action="store_true", help="run in this terminal instead of detached")
    v = sub.add_parser("validate")
    v.add_argument("schema")
    d = sub.add_parser("detect")
    d.add_argument("model")
    d.add_argument("--schemas", default="schemas", help="folder of schemas to check against")
    d.add_argument("--json", action="store_true")
    o = sub.add_parser("ops")
    o.add_argument("--json", action="store_true")
    agent_cli.add_commands(sub)
    try:                       # blackhole-dev (the dev training harness) adds inspect / fn when installed
        from blackhole_dev import cli as anatomy_cli
    except ImportError:
        anatomy_cli = None
    if anatomy_cli:
        anatomy_cli.add_commands(sub)
    a = ap.parse_args(argv)
    if a.cmd in agent_cli.COMMANDS:
        return agent_cli.dispatch(a)
    if anatomy_cli and a.cmd in anatomy_cli.COMMANDS:
        return anatomy_cli.dispatch(a)

    if a.cmd == "detect":
        from pathlib import Path
        from blackhole.hw.model import applicable, detect, resolve
        rid, others = resolve(a.model)
        if rid is None:
            print(f"✗ no model matches {a.model!r}")
            return 1
        try:
            info = detect(rid, rid if Path(rid).is_dir() else None)
        except RuntimeError as e:
            print(f"✗ {e}")
            return 1
        if a.json:
            print(json.dumps(info.to_dict(), indent=2))
            return 0
        print(f"{info.id}  {info.model_type}  {'/'.join(info.kinds)}  {info.params_b}B total, {info.active_b}B active"
              + (f", {info.experts} experts top-{info.top_k}" if info.experts else ""))
        print(f"  {info.layers} layers ({info.attention_layers} attention, {info.recurrent_layers} recurrent), "
              f"{info.checkpoint_gb} GB on disk, {info.quantized or info.dtype or '?'}, "
              f"KV {info.kv_bytes_per_token / 1024:.0f} KB/token at 16-bit")
        for n in info.notes:
            print(f"  · {n}")
        if others:
            print(f"  other matches: {', '.join(others)}")
        for f in sorted(Path(a.schemas).glob("*.toml")):
            try:
                why = applicable(info, sch.load(f))
            except sch.SchemaError:
                continue
            print(f"  {'✓' if not why else '✗'} {f.stem}" + (f": {why[0]}" if why else ""))
        return 0
    if a.cmd == "ops":
        rows = [ops.describe(c) for c in ops.catalog()]
        if a.json:
            print(json.dumps(rows, indent=2, default=str))
        else:
            for row in rows:
                print(f"{row['uses']:<24} {row['doc'].splitlines()[0] if row['doc'] else ''}")
        return 0
    if a.cmd == "validate":
        try:
            s = sch.load(a.schema)
        except sch.SchemaError as e:
            print(f"✗ {e}")
            return 1
        problems = ops.validate(s)
        for pr in problems:
            print(f"✗ {pr}")
        if not problems:
            print(f"✓ {s.ref}: {len(s.stages)} stages ({', '.join(x.id for x in s.stages)})")
        return 1 if problems else 0
    if a.cmd == "run":
        from blackhole.core import runner
        from blackhole.core.job import Job
        s = sch.load(a.schema)
        problems = ops.validate(s)
        if problems:
            print("\n".join(f"✗ {p}" for p in problems))
            return 1
        job = Job.create(a.schema, a.model, _sets(a.set), a.job)
        if a.fg:
            return runner.run(job)
        pid = runner.start_detached(job)
        print(f"started {job.root.name} (pid {pid}); `blackhole` or `blackhole logs` to follow")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
