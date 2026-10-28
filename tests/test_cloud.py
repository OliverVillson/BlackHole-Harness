"""blackhole.cloud without a cloud: fake CLIs and HTTP, nothing rented."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import pytest

from blackhole import cloud
from blackhole.cloud import CloudError, Machine, Offer, Provider
from blackhole.cloud.evroc import EvrocProvider
from blackhole.cloud.lambdacloud import Lambda
from blackhole.cloud.nebius import Nebius, _address, _cloud_init


class Done:
    def __init__(self, out="", code=0, err=""):
        self.stdout, self.returncode, self.stderr = out, code, err


# ---------------------------------------------------------------- up / down


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, fail_create=False):
        self.fail_create, self.calls = fail_create, []

    def offer(self, gpu, spot=False):
        return Offer("fake", gpu, "f-1", "here", 1.0, "me", spot)

    def create(self, machine, offer, public_key, disk_gb, say):
        self.calls.append("create")
        machine.resources.append(["vm", machine.name])
        if self.fail_create:
            raise CloudError("no capacity")
        machine.address, machine.id = "203.0.113.7", "vm-1"

    def stop(self, machine):
        self.calls.append("stop")

    def delete(self, machine):
        self.calls.append("delete")


@pytest.fixture
def keys(tmp_path, monkeypatch):
    key = tmp_path / "id_ed25519"
    key.write_text("private")
    key.with_name("id_ed25519.pub").write_text("ssh-ed25519 AAAAC3Nza laptop")
    monkeypatch.setenv("BLACKHOLE_SSH_KEY", str(key))
    monkeypatch.setenv("BLACKHOLE_HOME", str(tmp_path / "bh"))
    return key


def test_up_without_yes_rents_nothing(keys):
    p, said = FakeProvider(), []
    got = cloud.up("fake", "b200", say=said.append, get_provider=lambda n: p)
    assert isinstance(got, Offer) and p.calls == []
    assert any("--yes" in s for s in said)
    assert Machine.all() == []


def test_up_rents_waits_installs_and_records(keys):
    p, installed = FakeProvider(), []
    m = cloud.up("fake", "b200", yes=True, say=lambda s: None, get_provider=lambda n: p,
                 ready=lambda host, ident, say: None, do_install=lambda host, ident, args: installed.append((host, args)),
                 install_args=["--vllm", "yes"], restore=lambda host, ident, say: None)
    assert m.state == "running" and m.host == "me@203.0.113.7"
    assert installed == [("me@203.0.113.7", ["--vllm", "yes"])]
    assert [x.name for x in Machine.all()] == [m.name]


def test_up_failure_deletes_what_was_made(keys):
    p = FakeProvider(fail_create=True)
    with pytest.raises(CloudError):
        cloud.up("fake", "b200", yes=True, say=lambda s: None, get_provider=lambda n: p)
    assert p.calls == ["create", "delete"]
    assert Machine.all() == []


def _machine(tmp_path) -> Machine:
    m = Machine("bh-x", "fake", "b200", "f-1", "here", "me", address="203.0.113.7")
    m.save()
    return m


def test_down_refuses_when_models_are_still_there(keys, tmp_path):
    m, p, said = _machine(tmp_path), FakeProvider(), []
    cloud.down(m.name, say=said.append, get_provider=lambda n: p,
               run=lambda *a, **k: Done("/home/me/.blackhole/jobs/flint/out\n"))
    assert p.calls == [] and any("rsync" in s for s in said)
    cloud.down(m.name, yes=True, say=said.append, get_provider=lambda n: p)
    assert p.calls == ["delete"] and Machine.all() == []


def test_down_keep_disk_stops(keys, tmp_path):
    m, p = _machine(tmp_path), FakeProvider()
    cloud.down(m.name, keep_disk=True, say=lambda s: None, get_provider=lambda n: p)
    assert p.calls == ["stop"] and Machine.load(m.name).state == "stopped"


def test_install_streams_the_script_over_ssh(keys):
    seen = {}

    def run(cmd, input=None, **k):
        seen["cmd"], seen["input"] = cmd, input
        return Done()

    cloud.install("me@203.0.113.7", Path(str(keys)), ["--vllm", "no"], run=run)
    assert seen["cmd"][0] == "ssh" and seen["cmd"][-1] == "bash -s -- --yes --vllm no"
    assert seen["input"].startswith(b"#!/usr/bin/env bash")


def test_wait_ready_needs_cloud_init_done_and_a_gpu(keys):
    answers = iter([Done(code=255), Done(), Done(code=1), Done(b"GPU 0: NVIDIA B200 (UUID: x)")])
    cloud.wait_ready("me@h", None, lambda s: None, poll=0, sleep=lambda s: None, run=lambda *a, **k: next(answers))


# ---------------------------------------------------------------- Nebius


def test_nebius_offer_and_create_command():
    calls = []
    inst = {"metadata": {"id": "computeinstance-e00x"},
            "status": {"network_interfaces": [{"public_ip_address": {"address": "198.51.100.4/32"}}]}}

    def run(cmd, **k):
        calls.append(cmd)
        if cmd[1:4] == ["vpc", "subnet", "list"]:
            return Done(json.dumps({"items": [{"metadata": {"id": "vpcsubnet-1"}}]}))
        return Done(json.dumps(inst))

    n = Nebius(run=run, project="project-e00b200")
    o = n.offer("b200")
    assert o.size == "gpu-b200-sxm/1gpu-20vcpu-224gb" and o.usd_per_hour == 8.50 and "us-central1" in o.region
    assert n.offer("b200", spot=True).usd_per_hour == 0.99
    m = Machine("bh-b200-ab12", "nebius", "b200", o.size, o.region, o.login)
    n.create(m, o, "ssh-ed25519 AAAA laptop", 500, lambda s: None)
    create = calls[1]
    assert create[:3] == ["nebius", "compute", "instance"] and "--parent-id" in create
    assert create[create.index("--boot-disk-managed-disk-source-image-family-image-family") + 1] == "ubuntu24.04-cuda13.0"
    assert json.loads(create[create.index("--network-interfaces") + 1])[0]["subnet_id"] == "vpcsubnet-1"
    assert m.address == "198.51.100.4" and m.id == "computeinstance-e00x"
    with pytest.raises(CloudError, match="no one-GPU"):
        n.offer("a10")


def test_nebius_cloud_init_adds_the_login():
    ci = _cloud_init("ssh-ed25519 AAAA laptop")
    assert ci.startswith("#cloud-config") and "name: blackhole" in ci and "ssh-ed25519 AAAA laptop" in ci


def test_address_is_found_anywhere():
    assert _address({"a": [{"public_ip_address": {"address": "1.2.3.4/32"}}]}) == "1.2.3.4"
    assert _address({"a": 1}) is None


def test_nebius_quota_error_is_readable():
    n = Nebius(run=lambda cmd, **k: Done(code=1, err="ResourceExhausted: quota gpu-b200 exceeded"), project="p")
    with pytest.raises(CloudError, match="quota"):
        n.cli("compute", "instance", "list")


# ---------------------------------------------------------------- Lambda


class Http:
    """urlopen stand-in: answers by (method, path)."""

    def __init__(self, routes):
        self.routes, self.seen = routes, []

    def __call__(self, req, timeout=None):
        path = req.full_url.split("/api/v1", 1)[-1]
        self.seen.append((req.get_method(), path, json.loads(req.data) if req.data else None))
        body = self.routes[(req.get_method(), path)]
        return io.BytesIO(json.dumps({"data": body() if callable(body) else body}).encode())


def test_lambda_picks_cheapest_type_with_capacity_and_launches():
    types = {
        "gpu_1x_h100_pcie": {"instance_type": {"price_cents_per_hour": 249}, "regions_with_capacity_available": []},
        "gpu_1x_h100_sxm5": {"instance_type": {"price_cents_per_hour": 399},
                             "regions_with_capacity_available": [{"name": "us-east-1"}]},
        "gpu_8x_h100_sxm5": {"instance_type": {"price_cents_per_hour": 3192},
                             "regions_with_capacity_available": [{"name": "us-east-1"}]},
    }
    http = Http({("GET", "/instance-types"): types, ("GET", "/ssh-keys"): [],
                 ("POST", "/ssh-keys"): {}, ("POST", "/instance-operations/launch"): {"instance_ids": ["i-1"]},
                 ("GET", "/instances/i-1"): {"status": "active", "ip": "192.0.2.9"},
                 ("POST", "/instance-operations/terminate"): {}})
    lam = Lambda(opener=http, key="k", sleep=lambda s: None)
    o = lam.offer("h100")
    assert (o.size, o.region, o.usd_per_hour) == ("gpu_1x_h100_sxm5", "us-east-1", 3.99)
    m = Machine("bh-h100-1", "lambda", "h100", o.size, o.region, o.login)
    lam.create(m, o, "ssh-ed25519 AAAAkey laptop", 500, lambda s: None)
    assert m.host == "ubuntu@192.0.2.9" and m.id == "i-1"
    lam.delete(m)
    assert http.seen[-1] == ("POST", "/instance-operations/terminate", {"instance_ids": ["i-1"]})
    assert not lam.can_stop


def test_lambda_without_key_says_where_to_get_one(monkeypatch):
    monkeypatch.delenv("LAMBDA_API_KEY", raising=False)
    with pytest.raises(CloudError, match="LAMBDA_API_KEY"):
        Lambda(opener=Http({})).offer("h100")


# ---------------------------------------------------------------- evroc


def test_evroc_reads_the_cli_login_and_prices_l40s(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("currentProfile: default\nprofiles:\n  default:\n    project: proj-1\n    region: se-sto\n"
                   "    user:\n      refreshToken: rt\n")
    e = EvrocProvider(env={}, cli_path=cfg)
    assert (e.project, e.refresh) == ("proj-1", "rt")
    o = e.offer("l40s")
    assert o.size == "gn-l40s.s" and o.usd_per_hour == 1.50 and o.currency == "€"
    assert "no price" in e.offer("b200").note


def test_installer_ships_beside_the_module():
    assert cloud.INSTALLER.exists()
    assert subprocess.run(["bash", "-n", str(cloud.INSTALLER)]).returncode == 0


def test_installer_copies_match():
    # the root copy is what curl | bash fetches, the package copy is what `blackhole up` sends
    root = Path(__file__).resolve().parents[1] / "install.sh"
    assert root.read_bytes() == cloud.INSTALLER.read_bytes(), "cp install.sh blackhole/cloud/install.sh"


def test_installer_dry_run_installs_every_package(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    done = subprocess.run(["bash", str(cloud.INSTALLER), "--dry-run", "--no-apt", "--from", str(repo),
                           "--home", str(tmp_path / "bh"), "--torch-index", "https://example.invalid/whl"],
                          capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    out = done.stdout
    for part in ("reaper[pipeline]", "/harness", "/runtime", f"-e {repo}[gpu]", "flash-linear-attention"):
        assert part in out, part
    assert "nothing was changed" in out and not (tmp_path / "bh" / "venv").exists()


# ---------------------------------------------------------------- the compile cache


def local_ssh(vm_home):
    """subprocess.run stand-in that runs the remote command here, with the VM's home in vm_home."""
    def run(cmd, **kw):
        assert cmd[0] == "ssh"
        return subprocess.run(["bash", "-c", cmd[-1]], env={"PATH": "/usr/bin:/bin", "BLACKHOLE_HOME": str(vm_home)},
                              **kw)
    return run


def fake_vm(root: Path) -> Path:
    """A VM home with a vLLM venv whose python reports vllm 0.11.0 and torch 2.9.1."""
    import sys
    b = root / ".blackhole"
    site = b / "vllm" / "site"
    for name, ver in (("vllm", "0.11.0"), ("torch", "2.9.1+cu130")):
        d = site / f"{name}-{ver.split('+')[0]}.dist-info"
        d.mkdir(parents=True)
        (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {ver}\n")
    py = b / "vllm" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(f"#!/bin/sh\nPYTHONPATH={site} exec {sys.executable} \"$@\"\n")
    py.chmod(0o755)
    return b


def test_compile_cache_goes_home_and_back(tmp_path):
    vm1 = fake_vm(tmp_path / "vm1")
    kernel = vm1 / "cache" / "compile" / "vllm" / "torch_compile_cache" / "abc" / "kernel.so"
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(b"\x7fELF compiled")
    store, said = tmp_path / "laptop", []
    saved = cloud.save_cache("me@vm1", say=said.append, run=local_ssh(vm1), store=store)
    assert saved.name == "sm0-vllm0.11.0-torch2.9.1.tar.gz" and saved.stat().st_size > 0

    vm2 = fake_vm(tmp_path / "vm2")       # a fresh VM with the same versions
    assert cloud.restore_cache("me@vm2", say=said.append, run=local_ssh(vm2), store=store)
    assert (vm2 / "cache" / "compile" / "vllm" / "torch_compile_cache" / "abc" / "kernel.so").read_bytes() == b"\x7fELF compiled"


def test_compile_cache_skips_cleanly(tmp_path):
    vm = fake_vm(tmp_path / "vm")
    store = tmp_path / "laptop"
    assert cloud.save_cache("me@vm", say=lambda s: None, run=local_ssh(vm), store=store) is None   # nothing compiled yet
    assert not cloud.restore_cache("me@vm", say=lambda s: None, run=local_ssh(vm), store=store)    # no archive here
    bare = tmp_path / "novllm" / ".blackhole"
    bare.mkdir(parents=True)
    assert cloud.cache_key("me@x", run=local_ssh(bare)) is None


def test_down_saves_the_cache_before_deleting(keys, tmp_path):
    m, p, order = _machine(tmp_path), FakeProvider(), []
    m.state = "running"
    m.save()
    p.delete = lambda machine: order.append("delete")
    cloud.down(m.name, yes=True, say=lambda s: None, get_provider=lambda n: p,
               save=lambda host, ident, say: order.append(("save", host)))
    assert order == [("save", "me@203.0.113.7"), "delete"]
