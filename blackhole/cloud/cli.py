"""`blackhole up | down | ssh | machines`, run from a laptop. blackhole.cli hands these words here:

    if argv and argv[0] in cloud_cli.COMMANDS:
        return cloud_cli.main(argv)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from blackhole import cloud
from blackhole.cloud import CloudError, Machine

COMMANDS = ("up", "down", "ssh", "machines")


def _tty_ask(question: str) -> str | None:
    try:
        return input(question)
    except EOFError:
        return None


def main(argv: list[str] | None = None, say=print, ask=None) -> int:
    # no one at a terminal (the app, a script): no questions, and several machines need --yes
    ask = ask or (_tty_ask if sys.stdin.isatty() else None)
    ap = argparse.ArgumentParser(prog="blackhole", description=cloud.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up", help="rent a GPU VM and install the harness on it (prices first, rents with --yes)")
    u.add_argument("--provider", choices=cloud.PROVIDERS,
                   help="default $BLACKHOLE_PROVIDER, else nebius; with --host, where that machine runs (for `down`)")
    u.add_argument("--gpu", default="b200", help=f"one of {', '.join(cloud.GPU_GB)} (what the provider has)")
    u.add_argument("--spot", action="store_true", help="preemptible: far cheaper, may be stopped mid-run")
    u.add_argument("--disk", type=int, default=500, help="boot disk GB: the model, its pruned copies, caches")
    u.add_argument("--label", default="", help="part of the machine's name")
    u.add_argument("--host", help="skip renting: install on this user@ip (RunPod, Vast, your own machine)")
    u.add_argument("--port", type=int, help="SSH port for --host")
    u.add_argument("--resume", metavar="NAME", help="start a stopped machine again")
    u.add_argument("--vllm", choices=["auto", "yes", "no"], default="auto")
    u.add_argument("--llamacpp", action="store_true", help="also build llama.cpp for the GGUF ops")
    u.add_argument("--from", dest="source", help="what install.sh installs: a git URL to clone instead of BlackHole-Harness")
    u.add_argument("--ref", help="branch or tag of BlackHole-Harness to install (default main)")
    u.add_argument("-y", "--yes", action="store_true", help="really rent it")
    d = sub.add_parser("down", help="delete (or --keep-disk: stop) a machine, or --all of them; --forget drops the record only")
    d.add_argument("name", nargs="?", help="a name from `blackhole machines`, or its user@host")
    d.add_argument("--all", action="store_true", help="every machine, after one confirmation")
    d.add_argument("--keep-disk", action="store_true", help="stop it: the disk and models stay (and are billed)")
    d.add_argument("--forget", action="store_true",
                   help="remove it from the list without touching the VM (it keeps running and billing)")
    d.add_argument("-y", "--yes", action="store_true", help="no question, and delete even when models are still on it")
    s = sub.add_parser("ssh", help="open the TUI on a machine")
    s.add_argument("name", nargs="?")
    s.add_argument("rest", nargs=argparse.REMAINDER, help="arguments for blackhole there")
    sub.add_parser("machines", help="what this laptop has rented and not returned")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "machines":
            ms = Machine.all()
            for m in ms:
                say(_line(m))
            if not ms:
                say("no machines rented")
            return 0
        if a.cmd == "up":
            args = ["--vllm", a.vllm] + (["--llamacpp", "yes"] if a.llamacpp else []) + \
                (["--from", a.source] if a.source else []) + (["--ref", a.ref] if a.ref else [])
            if a.host:
                cloud.bring_own(a.host, port=a.port, install_args=args, say=say, provider_name=a.provider or "")
            elif a.resume:
                cloud.resume(a.resume, say=say)
            else:
                cloud.up(a.provider or os.environ.get("BLACKHOLE_PROVIDER", "nebius"), a.gpu, yes=a.yes, spot=a.spot, disk_gb=a.disk, label=a.label,
                         install_args=args, say=say)
            return 0
        if a.cmd == "down":
            return _down(a, say, ask)
        name = a.name or _only(say)
        if name is None:
            return 2
        if a.cmd == "ssh":
            m = Machine.load(name)
            identity, _ = cloud.ssh_keys()
            return subprocess.call(cloud.ssh_cmd(m.host, identity, port=m.port, tty=True) + ["blackhole", *a.rest])
    except CloudError as e:
        say(f"✗ {e}")
        return 1
    return 2


def _line(m: Machine) -> str:
    hours = (time.time() - m.created) / 3600
    if not m.managed:
        return f"{m.name}  {m.provider or 'own'} machine  {m.host}  (set up with --host; blackhole can only forget it)"
    price = f", ~{m.usd_per_hour * hours:.0f} so far" if m.usd_per_hour and m.state == "running" else ""
    return (f"{m.name}  {m.provider} {m.gpu}  {m.state}  {m.host if m.address else '(no address)'}  "
            f"{hours:.1f} h{price}")


def _pick(ms: list[Machine], say, ask) -> list[Machine] | None:
    """Which machines a bare `down` means: the only one, or the user's pick from a numbered list."""
    if len(ms) == 1:
        return ms
    for i, m in enumerate(ms, 1):
        say(f"  {i}. {_line(m)}")
    got = (ask(f"Which? 1-{len(ms)}, 'all', or Enter for none: ") or "").strip().lower()
    if got == "all":
        return ms
    picked = [ms[int(x) - 1] for x in got.replace(",", " ").split() if x.isdigit() and 0 < int(x) <= len(ms)]
    return picked or None


def _down(a, say, ask) -> int:
    ms = Machine.all()
    if a.all:
        targets = ms
    elif a.name:
        m = Machine.find(a.name)
        if m is None:
            if "@" in a.name or a.name.replace(".", "").isdigit():
                say(f"blackhole has no record of {a.name}: it did not rent it, so it cannot delete it")
                for line in cloud.still_billing():
                    say(f"  {line}")
                return 0
            raise CloudError(f"no machine {a.name} (see `blackhole machines`)")
        targets = [m]
    else:
        targets = ms
    if not targets:
        say("nothing to delete: no machines recorded here (`blackhole machines`)")
        say("  a VM you started by hand in a console is not listed; delete it there:")
        for n, u in cloud.CONSOLES.items():
            say(f"    {n}: {u}")
        return 0
    if not a.name and not a.all and len(targets) > 1 and ask is None:
        for m in targets:
            say(f"  {_line(m)}")
        say("name one: `blackhole down NAME`, or `blackhole down --all`")
        return 2
    if not a.name and not a.all:
        targets = _pick(targets, say, ask)
        if not targets:
            say("nothing deleted")
            return 0

    if a.forget:
        for m in targets:
            cloud.forget(m.name, say=say)
        return 0

    verb = "stop" if a.keep_disk else "delete"
    rented = [m for m in targets if m.managed]
    own = [m for m in targets if not m.managed]
    yes = a.yes
    if not yes and ask is None and len(targets) > 1:
        say(f"add --yes to {verb} {len(targets)} machines without a question")
        return 2
    if not yes and ask is not None:
        for m in rented:
            say(f"{verb} {_line(m)}")
            if not a.keep_disk:
                for o in cloud.outputs(m):
                    say(f"    model still on it, would be lost: {o}")
        for m in own:
            say(f"forget {m.name} ({m.host}): not rented by blackhole, so it is only removed from the list")
        what = " and ".join(x for x in [
            f"{verb} {len(rented)} VM{'s' if len(rented) != 1 else ''}" if rented else "",
            f"forget {len(own)} host{'s' if len(own) != 1 else ''}" if own else ""] if x)
        answer = ask(f"{what[0].upper() + what[1:]}?{' This cannot be undone.' if rented and not a.keep_disk else ''} [y/N] ")
        if (answer or "").strip().lower() not in ("y", "yes"):
            say("nothing deleted")
            return 0
        yes = True   # the models on it were listed in the question, so no second refusal
    failed = 0
    for m in targets:
        try:
            cloud.down(m.name, keep_disk=a.keep_disk, yes=yes, say=say)
        except CloudError as e:
            failed += 1
            say(f"✗ {m.name}: {e}")
    return 1 if failed else 0


def _only(say) -> str | None:
    ms = Machine.all()
    if len(ms) == 1:
        return ms[0].name
    say("name a machine: " + (", ".join(m.name for m in ms) if ms else "none rented"))
    return None


if __name__ == "__main__":
    sys.exit(main())
