"""`blackhole describe | connect | market | models`: the builder and marketplace from a shell.

blackhole.cli wires these in with two calls:

    from blackhole.agent import cli as agent_cli
    agent_cli.add_commands(sub)                  # after the other sub.add_parser calls
    if a.cmd in agent_cli.COMMANDS:
        return agent_cli.dispatch(a)
"""

from __future__ import annotations

import getpass
import json
import os
from pathlib import Path

COMMANDS = ("describe", "connect", "market", "models")


def add_commands(sub) -> None:
    from blackhole.agent.providers import PROVIDERS
    from blackhole.market import DEFAULT_INDEX
    from blackhole.market.models import DEVICES

    d = sub.add_parser("describe", help="a frontier model drafts a schema from your words")
    d.add_argument("text")
    d.add_argument("--base", help="base model to start from")
    d.add_argument("--device", default="8GB-VRAM", choices=list(DEVICES), help="the PC the result must run on")
    d.add_argument("--provider", choices=PROVIDERS)
    d.add_argument("--author", default=os.environ.get("USER", "me"))
    d.add_argument("--out", help="where to write the schema (default: <name>.toml)")
    d.add_argument("--registry", default=DEFAULT_INDEX)
    d.add_argument("--max-usd", type=float, help="hard API spend cap for this draft")

    c = sub.add_parser("connect", help="store an API key for a frontier provider")
    c.add_argument("name", choices=PROVIDERS)
    c.add_argument("--base-url")
    c.add_argument("--model")
    c.add_argument("--default", action="store_true")

    m = sub.add_parser("market", help="the schema marketplace")
    m.add_argument("action", choices=["search", "pull", "publish", "submit", "index"])
    m.add_argument("arg", nargs="?", default="", help="query | id[@version] | schema.toml | receipt.json | repo dir")
    m.add_argument("--registry", default=DEFAULT_INDEX)
    m.add_argument("--clone", help="local clone of the registry repo (publish, submit)")
    m.add_argument("--device", default="8GB-VRAM", choices=list(DEVICES))
    m.add_argument("--arch", choices=["moe", "dense", "hybrid"])
    m.add_argument("--allow-code", action="store_true", help="pull a schema that ships a lab script")
    m.add_argument("--json", action="store_true")

    mo = sub.add_parser("models", help="base models: runs here / fits your PC")
    mo.add_argument("--device", default="8GB-VRAM", choices=list(DEVICES))
    mo.add_argument("--context", type=int, default=16384)


def dispatch(a) -> int:
    from blackhole.agent.providers import ProviderError

    try:
        return {"describe": _describe, "connect": _connect, "market": _market, "models": _models}[a.cmd](a)
    except (ProviderError, KeyError, ValueError, PermissionError, OSError) as e:
        print(f"✗ {e}")
        return 1


def _describe(a) -> int:
    from blackhole.agent import draft
    from blackhole.hw import detect
    from blackhole.market import Registry

    try:
        reg = Registry.open(a.registry)
    except (OSError, ValueError) as e:
        print(f"(marketplace unreachable, drafting from scratch: {e})")
        reg = None

    def answer(questions):
        print()
        return [input(f"? {q}\n> ").strip() for q in questions]

    d = draft(a.text, base=a.base, hw=detect(), provider=a.provider or "", device=a.device, author=a.author,
              registry=reg, answer=answer, on_text=lambda t: print(t, end="", flush=True), budget_usd=a.max_usd)
    print()
    if not d.schema_toml:
        print("✗ no schema:\n  " + "\n  ".join(d.problems))
        return 1
    import tomllib
    name = tomllib.loads(d.schema_toml)["id"].split("/")[-1]
    out = Path(a.out or f"{name}.toml")
    out.write_text(d.schema_toml)
    est = d.estimate
    cost = f"${d.cost_usd:.2f}" if d.cost_usd is not None else "unknown cost"
    print(f"✓ {out}  ~{est.get('gpu_hours')} {est.get('gpu')} GPU-hours, "
          f"~{est.get('footprint_gb', '?')} GB on {a.device}, frontier data spend ${est.get('frontier_usd', 0)}")
    print(f"  drafted in {d.drafts} attempt(s), {d.usage.turns} turns, {cost}")
    print(f"  next: blackhole validate {out} && blackhole run {out} <model>")
    return 0


def _connect(a) -> int:
    from blackhole.agent import providers

    key = getpass.getpass(f"{a.name} API key (hidden; blank for none): ").strip()
    prov = providers.get(a.name, api_key=key, base_url=a.base_url or "", model=a.model or "")
    print(f"✓ {a.name}: {prov.check()}")
    path = providers.save_key(a.name, key, a.base_url or "", a.model or prov.model, default=a.default)
    print(f"  saved to {path} (0600){'  key ' + providers.mask(key) if key else ''}")
    return 0


def _market(a) -> int:
    from blackhole import market

    if a.action == "index":
        print(json.dumps(market.build_index(Path(a.arg or ".")), indent=2))
        return 0
    if a.action in ("publish", "submit"):
        if not a.clone:
            raise ValueError("--clone <registry checkout> is needed to publish or submit")
        if a.action == "publish":
            reg = market.Registry.open(a.registry)
            out = market.publish(Path(a.arg), Path(a.clone), reg)
            print(f"✓ committed {out['file']} on {out['branch']}\n  open the PR with:\n  {out['pr_command']}")
        else:
            out = market.submit(Path(a.arg), Path(a.clone), os.environ.get("USER", "anon"))
            print(f"✓ committed {out['file']} on {out['branch']}; push it and open a PR")
        return 0
    reg = market.Registry.open(a.registry)
    if a.action == "pull":
        print(f"✓ {reg.pull(a.arg, allow_code=a.allow_code)}")
        return 0
    rows = reg.search(a.arg, arch=a.arch or "", device=a.device)
    if a.json:
        print(json.dumps(rows, indent=2))
        return 0
    for r in rows:
        fits = {True: "✓ fits", False: "✗ too big", None: "  ?"}[r["fits_your_pc"]]
        exp = " [experimental]" if r["experimental"] else ""
        print(f"{r['badge']:<9} {r['id'] + '@' + str(r['version']):<40} {fits:<10} {r['title']}{exp}")
        for base, metrics in r["results"].items():
            print(f"{'':<10}  {base}: " + ", ".join(f"{k} {v}" for k, v in metrics.items()))
    return 0


def _models(a) -> int:
    from blackhole.hw import detect
    from blackhole.market.models import table

    hw = detect()
    print(f"VM: {hw.gpu_name or 'no GPU'} x{hw.gpu_count}, {hw.gpu_mem_gb} GB each, {hw.disk_free_gb} GB free"
          f"  |  your PC: {a.device}")
    for r in table(hw, a.device, a.context):
        print(f"{r['model']:<22} {r['arch']:<6} {r['params_b']:>6}B {r['experts']:>10}  "
              f"{'✓' if r['runs_here'] else '✗'} runs here  {'✓' if r['fits_target'] else '✗'} {r['recipe']}")
    return 0
