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


def main(argv: list[str] | None = None, say=print) -> int:
    ap = argparse.ArgumentParser(prog="blackhole", description=cloud.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up", help="rent a GPU VM and install the harness on it (prices first, rents with --yes)")
    u.add_argument("--provider", choices=cloud.PROVIDERS, default=os.environ.get("BLACKHOLE_PROVIDER", "nebius"))
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
    d = sub.add_parser("down", help="stop (--keep-disk) or delete a machine")
    d.add_argument("name", nargs="?")
    d.add_argument("--keep-disk", action="store_true", help="stop it: the disk and models stay (and are billed)")
    d.add_argument("-y", "--yes", action="store_true", help="delete even when models are still on it")
    s = sub.add_parser("ssh", help="open the TUI on a machine")
    s.add_argument("name", nargs="?")
    s.add_argument("rest", nargs=argparse.REMAINDER, help="arguments for blackhole there")
    sub.add_parser("machines", help="what this laptop has rented and not returned")
    a = ap.parse_args(argv)

    try:
        if a.cmd == "machines":
            ms = Machine.all()
            for m in ms:
                hours = (time.time() - m.created) / 3600
                price = f", ~{m.usd_per_hour * hours:.0f} so far" if m.usd_per_hour and m.state == "running" else ""
                say(f"{m.name}  {m.provider} {m.gpu}  {m.state}  {m.host if m.address else '(no address)'}  "
                    f"{hours:.1f} h{price}")
            if not ms:
                say("no machines rented")
            return 0
        if a.cmd == "up":
            args = ["--vllm", a.vllm] + (["--llamacpp", "yes"] if a.llamacpp else []) + \
                (["--from", a.source] if a.source else []) + (["--ref", a.ref] if a.ref else [])
            if a.host:
                cloud.bring_own(a.host, port=a.port, install_args=args, say=say)
            elif a.resume:
                cloud.resume(a.resume, say=say)
            else:
                cloud.up(a.provider, a.gpu, yes=a.yes, spot=a.spot, disk_gb=a.disk, label=a.label,
                         install_args=args, say=say)
            return 0
        name = a.name or _only(say)
        if name is None:
            return 2
        if a.cmd == "down":
            cloud.down(name, keep_disk=a.keep_disk, yes=a.yes, say=say)
            return 0
        if a.cmd == "ssh":
            m = Machine.load(name)
            identity, _ = cloud.ssh_keys()
            return subprocess.call(cloud.ssh_cmd(m.host, identity, tty=True) + ["blackhole", *a.rest])
    except CloudError as e:
        say(f"✗ {e}")
        return 1
    return 2


def _only(say) -> str | None:
    ms = Machine.all()
    if len(ms) == 1:
        return ms[0].name
    say("name a machine: " + (", ".join(m.name for m in ms) if ms else "none rented"))
    return None


if __name__ == "__main__":
    sys.exit(main())
