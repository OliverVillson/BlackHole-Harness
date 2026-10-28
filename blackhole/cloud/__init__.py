"""GPU VM bring-up and the installer. Owned by the "VM setup and installer" thread.

    blackhole up --provider nebius --gpu b200        # from a laptop: price, then rent, install, print `ssh -t ... blackhole`
    blackhole up --host ubuntu@203.0.113.7            # a machine you already have (RunPod, Vast, your own box)
    blackhole ssh [name]                              # open the TUI on it
    blackhole down [name] [--keep-disk]               # stop (disk kept) or delete (the compile cache is saved here first)
    blackhole down --all                              # every machine, after one confirmation
    blackhole down <name|user@host> --forget          # drop the record only; the VM itself is not touched
    blackhole machines                                # what this laptop has rented and not returned

Providers are small classes with offer/create/status/stop/start/delete (nebius.py, evroc.py,
lambdacloud.py). `up` prints the provider, GPU, region and price an hour first and rents nothing
without --yes. Every VM is then set up the same way: install.sh (shipped next to this file) runs
over SSH, so its output streams here, and then `ssh -t <host> blackhole` opens the TUI.

Every machine has a record in $BLACKHOLE_HOME/machines/<name>.json, written before the first
resource is asked for, so a machine whose `up` died half way is still found and returned
(the pattern from mugge.cloud, which drives evroc this way for `mugge run --host evroc`).
`up --host` writes one too (managed false), so a machine you set up yourself is listed and can be
forgotten; `down` never deletes those, it says where to delete them by hand.
"""

from __future__ import annotations

import json
import os
import secrets
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

INSTALLER = Path(__file__).with_name("install.sh")

# One GPU's memory, for the "fits" check against a schema's needs. Keys are what --gpu takes.
GPU_GB = {"b300": 288, "b200": 180, "h200": 141, "h100": 80, "rtx6000": 96, "l40s": 48, "a100": 80}


class CloudError(RuntimeError):
    pass


def home() -> Path:
    return Path(os.environ.get("BLACKHOLE_HOME") or Path.home() / ".blackhole")


def machines_dir() -> Path:
    return home() / "machines"


@dataclass
class Offer:
    """What `up` would rent, shown before anything is created."""

    provider: str
    gpu: str
    size: str            # the provider's name for it: a Nebius preset, an evroc size, a Lambda type
    region: str
    usd_per_hour: float | None   # the VM an hour, in `currency`; disk and address are listed in `extras`
    login: str           # the SSH user the image gives
    spot: bool = False
    currency: str = "$"
    extras: str = ""     # disk, address, anything billed besides the VM
    note: str = ""

    def line(self) -> str:
        price = f"{self.currency}{self.usd_per_hour:.2f}/h" if self.usd_per_hour is not None else "price unknown"
        kind = "spot, may be stopped" if self.spot else "on-demand"
        return f"{self.provider} · 1x {self.gpu.upper()} ({self.size}) · {self.region} · {price} {kind}" + \
            (f" + {self.extras}" if self.extras else "")


@dataclass
class Machine:
    """One VM and what was made for it. Saved while it exists, so `down` finds it again."""

    name: str
    provider: str
    gpu: str
    size: str
    region: str
    login: str
    created: float = field(default_factory=time.time)
    address: str | None = None
    id: str | None = None            # the provider's id for the VM
    resources: list[list[str]] = field(default_factory=list)   # provider-specific, in the order made
    state: str = "creating"          # creating | running | stopped
    usd_per_hour: float | None = None
    managed: bool = True             # False: set up with `up --host`; blackhole cannot delete it
    port: int | None = None          # SSH port, for `up --host user@ip --port N`

    @property
    def host(self) -> str:
        return f"{self.login}@{self.address}"

    def save(self, folder: Path | None = None) -> None:
        folder = folder or machines_dir()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{self.name}.json").write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    def forget(self, folder: Path | None = None) -> None:
        ((folder or machines_dir()) / f"{self.name}.json").unlink(missing_ok=True)

    @classmethod
    def load(cls, name: str, folder: Path | None = None) -> Machine:
        path = (folder or machines_dir()) / f"{name}.json"
        if not path.exists():
            raise CloudError(f"no machine {name} (see `blackhole machines`)")
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def matches(self, what: str) -> bool:
        """By name, user@address or bare address, so `down` takes what `machines` prints."""
        return what in (self.name, self.address) or bool(self.address and what == self.host)

    @classmethod
    def find(cls, what: str, folder: Path | None = None) -> Machine | None:
        return next((m for m in cls.all(folder) if m.matches(what)), None)

    @classmethod
    def all(cls, folder: Path | None = None) -> list[Machine]:
        folder = folder or machines_dir()
        return [cls.load(p.stem, folder) for p in sorted(folder.glob("*.json"))] if folder.exists() else []


class Provider:
    """A cloud that rents one-GPU Ubuntu VMs. Subclasses fill in the five calls."""

    name = ""
    can_stop = True      # False: the VM can only be deleted (Lambda), so `down --keep-disk` is refused

    def offer(self, gpu: str, spot: bool = False) -> Offer:
        raise NotImplementedError

    def create(self, machine: Machine, offer: Offer, public_key: str, disk_gb: int, say: Callable) -> None:
        """Rent it, fill machine.address and machine.id, and append to machine.resources (saving as you go)."""
        raise NotImplementedError

    def stop(self, machine: Machine) -> None:
        raise CloudError(f"{self.name} VMs cannot be stopped, only deleted")

    def start(self, machine: Machine) -> None:
        raise CloudError(f"{self.name} VMs cannot be stopped, only deleted")

    def delete(self, machine: Machine) -> None:
        raise NotImplementedError


def provider(name: str) -> Provider:
    if name == "nebius":
        from blackhole.cloud.nebius import Nebius
        return Nebius()
    if name == "evroc":
        from blackhole.cloud.evroc import EvrocProvider
        return EvrocProvider()
    if name == "lambda":
        from blackhole.cloud.lambdacloud import Lambda
        return Lambda()
    raise CloudError(f"unknown provider {name!r}: nebius, evroc or lambda (or --host user@ip for any other machine)")


PROVIDERS = ["nebius", "evroc", "lambda"]

# Where to delete a VM by hand: one blackhole did not create, or one it was told to forget.
CONSOLES = {
    "nebius": "https://console.nebius.com (Compute > Virtual machines)",
    "evroc": "https://cloud.evroc.com",
    "lambda": "https://cloud.lambda.ai/instances",
}


def still_billing(provider_name: str = "") -> list[str]:
    """What to tell someone whose VM blackhole will not (or cannot) delete."""
    if provider_name in CONSOLES:
        return [f"the VM itself is still running and billing: delete it in the {provider_name.capitalize()} console, "
                f"{CONSOLES[provider_name]}"]
    return ["if it is a cloud VM it is still running and billing: delete it in your provider's console",
            *(f"  {n}: {u}" for n, u in CONSOLES.items())]


# ---------------------------------------------------------------- SSH


def ssh_keys(private: str | None = None) -> tuple[Path, str]:
    path = Path(os.path.expanduser(private or os.environ.get("BLACKHOLE_SSH_KEY") or "~/.ssh/id_ed25519"))
    public = path.with_name(path.name + ".pub")
    if not public.exists():
        raise CloudError(f"no public key at {public}: run `ssh-keygen -t ed25519`, or set BLACKHOLE_SSH_KEY "
                         "to a private key whose .pub sits beside it")
    return path, public.read_text(encoding="utf-8").strip()


def ssh_cmd(host: str, identity: Path | None = None, port: int | None = None, tty: bool = False) -> list[str]:
    return ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30",
            *(["-t"] if tty else ["-o", "BatchMode=yes"]),
            *(["-p", str(port)] if port else []), *(["-i", str(identity)] if identity else []), host]


def wait(what: str, check: Callable[[], bool], timeout: float, poll: float, sleep: Callable = time.sleep) -> None:
    end = time.monotonic() + timeout
    while not check():
        if time.monotonic() > end:
            raise CloudError(f"timed out after {timeout:.0f} s waiting for {what}")
        sleep(poll)


# Ready once cloud-init is over (done or error: its apt run has ended either way, so install.sh's
# apt-get does not find dpkg locked) and nvidia-smi sees a GPU. Images with no cloud-init are ready.
READY = 'case "$(cloud-init status 2>/dev/null)" in *running*|*"not started"*) exit 1;; esac; nvidia-smi -L'


def wait_ready(host: str, identity: Path | None, say: Callable, *, port: int | None = None, poll: float = 10.0,
               sleep: Callable = time.sleep, run: Callable = subprocess.run) -> None:
    base = ssh_cmd(host, identity, port)[:-1] + ["-o", "ConnectTimeout=10", host]
    say(f"waiting for SSH on {host}")
    wait(f"SSH on {host}", lambda: run(base + ["true"], capture_output=True).returncode == 0, 900, poll, sleep)
    say("waiting for cloud-init and the NVIDIA driver")

    def sees() -> bool:
        done = run(base + [READY], capture_output=True)
        return done.returncode == 0 and b"GPU" in done.stdout

    wait(f"the GPU on {host}", sees, 900, poll, sleep)


def install(host: str, identity: Path | None, args: list[str], *, port: int | None = None,
            run: Callable = subprocess.run) -> None:
    """install.sh over SSH, its output streaming to this terminal. Rerunning it is safe: each step
    checks whether it is done."""
    script = INSTALLER.read_bytes()
    cmd = ssh_cmd(host, identity, port) + ["bash -s -- " + " ".join(shlex.quote(a) for a in ["--yes", *args])]
    if run(cmd, input=script).returncode != 0:
        raise CloudError(f"install.sh failed on {host}; rerun with `blackhole up --host {host}` "
                         "after a look at ~/.blackhole/install.log there")


# ---------------------------------------------------------------- the compile cache
#
# vLLM's first start on a fresh VM spends most of its time compiling (torch.compile, Inductor and
# Triton kernels, FlashInfer JIT): 413 s of a 480 s calibrate stage on a B200, against 50 s warm.
# install.sh's launcher points all of those caches at ~/.blackhole/cache/compile, so one tar holds
# them. `down` copies it to this laptop before a delete, and `up` puts it back after the install.
# The archive is named by GPU architecture and the vLLM and torch versions, since a cache built by
# other versions is never used; within one name vLLM's own hashes decide what is reused.

CACHE_KEY = ('B=${BLACKHOLE_HOME:-$HOME/.blackhole}; [ -x "$B/vllm/bin/python" ] || exit 3; '
             'cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d " ."); '
             '"$B/vllm/bin/python" -c "from importlib.metadata import version as v; '
             'print(\'sm${cc:-0}-vllm\' + v(\'vllm\') + \'-torch\' + v(\'torch\').split(\'+\')[0])"')
CACHE_SAVE = ('B=${BLACKHOLE_HOME:-$HOME/.blackhole}; [ -n "$(ls -A "$B/cache/compile" 2>/dev/null)" ] || exit 3; '
              'tar -C "$B/cache" -czf - compile')
CACHE_LOAD = 'B=${BLACKHOLE_HOME:-$HOME/.blackhole}; mkdir -p "$B/cache" && tar -C "$B/cache" -xzf -'


def cache_store() -> Path:
    return home() / "compile-cache"


def cache_key(host: str, identity: Path | None = None, run: Callable = subprocess.run) -> str | None:
    """None when the VM has no vLLM venv (nothing to cache) or the key cannot be read."""
    done = run(ssh_cmd(host, identity) + [CACHE_KEY], capture_output=True, text=True, timeout=120)
    key = (done.stdout or "").strip().splitlines()[-1:] if done.returncode == 0 else []
    return key[0] if key and key[0].startswith("sm") else None


def save_cache(host: str, identity: Path | None = None, say: Callable = print, run: Callable = subprocess.run,
               store: Path | None = None) -> Path | None:
    """The VM's compile cache, into store/<key>.tar.gz here. Best effort: a failure costs only a cold start."""
    key = cache_key(host, identity, run)
    if not key:
        return None
    store = store or cache_store()
    store.mkdir(parents=True, exist_ok=True)
    dest, part = store / f"{key}.tar.gz", store / f".{key}.tar.gz.part"
    with open(part, "wb") as out:
        done = run(ssh_cmd(host, identity) + [CACHE_SAVE], stdout=out, timeout=1800)
    if done.returncode != 0 or part.stat().st_size == 0:
        part.unlink(missing_ok=True)
        return None
    part.replace(dest)
    say(f"saved the vLLM compile cache ({dest.stat().st_size / 1e6:.0f} MB) to {dest}")
    return dest


def restore_cache(host: str, identity: Path | None = None, say: Callable = print, run: Callable = subprocess.run,
                  store: Path | None = None) -> bool:
    key = cache_key(host, identity, run)
    src = (store or cache_store()) / f"{key}.tar.gz" if key else None
    if not src or not src.exists():
        return False
    with open(src, "rb") as f:
        done = run(ssh_cmd(host, identity) + [CACHE_LOAD], stdin=f, timeout=1800)
    if done.returncode == 0:
        say(f"restored the vLLM compile cache {src.name}: the teacher starts warm")
    return done.returncode == 0


# ---------------------------------------------------------------- the flows


def up(provider_name: str, gpu: str, *, yes: bool = False, spot: bool = False, disk_gb: int = 500,
       label: str = "", install_args: list[str] | None = None, say: Callable = print,
       get_provider: Callable[[str], Provider] = provider, ready: Callable = wait_ready,
       do_install: Callable = install, restore: Callable = restore_cache,
       folder: Path | None = None) -> Machine | Offer:
    """Price first; with yes, rent, wait for the GPU, install and return the machine. On any failure
    before the install, what was rented is deleted again."""
    p = get_provider(provider_name)
    offer = p.offer(gpu.lower(), spot)
    say(offer.line())
    if offer.note:
        say(f"  {offer.note}")
    if not yes:
        say("nothing rented. Add --yes to rent it.")
        return offer
    identity, public_key = ssh_keys()
    name = f"bh-{(label or gpu)[:24]}-{secrets.token_hex(2)}".lower()
    m = Machine(name, p.name, offer.gpu, offer.size, offer.region, offer.login, usd_per_hour=offer.usd_per_hour)
    m.save(folder)
    try:
        p.create(m, offer, public_key, disk_gb, say)
        m.save(folder)
        ready(m.host, identity, say)
    except BaseException:
        say(f"{name} did not come up; deleting what was made")
        try:
            p.delete(m)
            m.forget(folder)
        except Exception as e:  # keep the record so `blackhole down` can retry
            say(f"could not delete {name} ({e}); run `blackhole down {name}`")
        raise
    m.state = "running"
    m.save(folder)
    say(f"{name} is up at {m.host}; installing")
    do_install(m.host, identity, install_args or [])
    try:
        restore(m.host, identity, say)
    except Exception as e:  # a cold start, not a failed `up`
        say(f"no compile cache restored ({e})")
    say(f"ready. Open the TUI with:  ssh -t {m.host} blackhole   (or `blackhole ssh {name}`)")
    return m


def own_name(host: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in host.lower()).strip("-")


def bring_own(host: str, *, port: int | None = None, install_args: list[str] | None = None, say: Callable = print,
              do_install: Callable = install, restore: Callable = restore_cache, provider_name: str = "",
              folder: Path | None = None) -> Machine:
    """Any Ubuntu machine with an NVIDIA driver: a RunPod or Vast box, a lab server, a desktop. It is
    recorded (managed false) so `machines` lists it and `down` can forget it, never delete it."""
    identity = None
    key = os.environ.get("BLACKHOLE_SSH_KEY")
    if key:
        identity = Path(os.path.expanduser(key))
    say(f"installing on {host}")
    do_install(host, identity, install_args or [], port=port)
    if not port:  # the cache helpers speak plain ssh to port 22
        try:
            restore(host, identity, say)
        except Exception as e:
            say(f"no compile cache restored ({e})")
    login, _, address = host.rpartition("@")
    m = Machine.find(host, folder) or Machine(own_name(host), provider_name, "", "", "", login or "root",
                                              address=address, managed=False)
    m.state, m.provider, m.port = "running", provider_name or m.provider, port
    m.save(folder)
    say(f"ready. Open the TUI with:  ssh -t {host} blackhole")
    return m


# Shown before a delete: what would be lost. Jobs keep their final model in jobs/<job>/out.
OUTPUTS = 'ls -d ~/.blackhole/jobs/*/out ~/.reaper/jobs/*/out 2>/dev/null; true'


def outputs(m: Machine, run: Callable = subprocess.run) -> list[str]:
    """Finished models still on the VM. Empty when it has no address or does not answer."""
    if not m.address:
        return []
    try:
        found = run(ssh_cmd(m.host, port=m.port) + [OUTPUTS], capture_output=True, text=True, timeout=30)
    except Exception:
        return []
    return [line for line in (found.stdout or "").splitlines() if line.strip()]


def forget(name: str, *, say: Callable = print, folder: Path | None = None) -> Machine:
    """Drop the record only. Nothing is asked of the provider, so the VM keeps running if it exists."""
    m = Machine.find(name, folder)
    if m is None:
        raise CloudError(f"no machine {name} (see `blackhole machines`)")
    m.forget(folder)
    say(f"forgot {m.name}" + (f" ({m.host})" if m.address else ""))
    if m.address or not m.managed:
        for line in still_billing(m.provider):
            say(f"  {line}")
    return m


def down(name: str, *, keep_disk: bool = False, yes: bool = False, say: Callable = print,
         get_provider: Callable[[str], Provider] = provider, run: Callable = subprocess.run,
         save: Callable = save_cache, folder: Path | None = None) -> None:
    m = Machine.find(name, folder) or Machine.load(name, folder)
    name = m.name
    if not m.managed:
        if keep_disk:
            raise CloudError(f"{name} was not rented by blackhole, so it cannot stop it; stop it where you rented it")
        say(f"{name} was set up with `blackhole up --host`, not rented by blackhole, so it cannot be deleted from here")
        forget(name, say=say, folder=folder)
        return
    p = get_provider(m.provider)
    if keep_disk:
        if not p.can_stop:
            raise CloudError(f"{p.name} cannot stop a VM, only delete it; copy the model off first, then "
                             f"`blackhole down {name}`")
        p.stop(m)
        m.state = "stopped"
        m.save(folder)
        say(f"{name} stopped; the disk and the models on it are kept (and billed). `blackhole up --resume {name}` "
            "starts it again")
        return
    if m.address and not yes:
        outs = outputs(m, run)
        if outs:
            say(f"{name} still has models that deleting would lose:")
            for o in outs:
                say(f"  {o}")
            say(f"copy them off first:  rsync -avP {m.host}:{outs[0]} ./   (or run the sync.bucket op), then "
                f"`blackhole down {name} --yes`; or `--keep-disk` to stop it instead")
            return
    if m.address and m.state == "running":
        try:  # the next fresh VM's teacher then starts in ~50 s, not ~7 min
            save(m.host, None, say)
        except Exception as e:
            say(f"compile cache not saved ({e}); the next VM starts cold")
    p.delete(m)
    m.forget(folder)
    say(f"{name} deleted")


def resume(name: str, *, say: Callable = print, get_provider: Callable[[str], Provider] = provider,
           ready: Callable = wait_ready, folder: Path | None = None) -> Machine:
    m = Machine.load(name, folder)
    p = get_provider(m.provider)
    p.start(m)       # a restarted VM can get a new address; start() refreshes it
    m.state = "running"
    m.save(folder)
    identity, _ = ssh_keys()
    ready(m.host, identity, say)
    say(f"{name} is back at {m.host}:  ssh -t {m.host} blackhole")
    return m
