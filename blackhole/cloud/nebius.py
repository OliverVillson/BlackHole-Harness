"""Nebius AI Cloud, through the `nebius` CLI (https://docs.nebius.com/cli): REAPer's B200 ran here.

    nebius profile create              # once: log in; the profile's parent-id is the project
    blackhole up --provider nebius --gpu b200 [--spot] --yes

A one-GPU preset, a managed boot disk from the CUDA image family, a public IP, and cloud-init that
adds the `blackhole` login with this laptop's key. A project lives in one region, and B200s only in
us-central1 and me-west1, so NEBIUS_PROJECT_ID (else the CLI profile's parent-id) has to be a
project in the GPU's region.

The managed boot disk is deleted with the VM: stop it (`blackhole down --keep-disk`) to keep the
models, delete it only after copying them off.

Prices and presets read 2026-10-23 from https://docs.nebius.com/compute/resources/pricing and
https://docs.nebius.com/compute/virtual-machines/types. Spot (preemptible) prices are dynamic from
2026-10-08; the figure is the published minimum.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Callable

from blackhole.cloud import CloudError, Machine, Offer, Provider

LOGIN = "blackhole"
# B200 and B300 need driver 580 (the 570 in ubuntu24.04-cuda12 is refused for their 1-GPU presets)
IMAGE = "ubuntu24.04-cuda13.0"
DISK_TYPE = "network_ssd"

#        platform          one-GPU preset        regions                                 $/GPU-h  spot from
PRESETS = {
    "b300": ("gpu-b300-sxm", "1gpu-24vcpu-346gb", ["uk-south1", "eu-west2", "us-north1"], 9.50, 0.99),
    "b200": ("gpu-b200-sxm", "1gpu-20vcpu-224gb", ["us-central1"], 8.50, 0.99),
    "b200-a": ("gpu-b200-sxm-a", "1gpu-20vcpu-224gb", ["me-west1"], 8.50, 0.99),
    "h200": ("gpu-h200-sxm", "1gpu-16vcpu-200gb", ["eu-north1", "eu-west1", "us-central1"], 5.40, 0.79),
    "h100": ("gpu-h100-sxm", "1gpu-16vcpu-200gb", ["eu-north1"], 4.50, 0.79),
    "l40s": ("gpu-l40s-a", "1gpu-16vcpu-64gb", ["eu-north1"], 1.35, 0.65),
    "rtx6000": ("gpu-rtx6000", "1gpu-24vcpu-218gb", ["us-central1"], None, None),
}


def _cloud_init(public_key: str) -> str:
    return "\n".join([
        "#cloud-config",
        "users:",
        f"  - name: {LOGIN}",
        "    sudo: ALL=(ALL) NOPASSWD:ALL",
        "    shell: /bin/bash",
        "    ssh_authorized_keys:",
        f"      - {public_key}",
        "package_update: false",   # install.sh does its own apt run, after cloud-init is over
        ""])


def _address(doc) -> str | None:
    """The public IPv4 anywhere under status.network_interfaces, without its /32."""
    if isinstance(doc, dict):
        pub = doc.get("public_ip_address")
        if isinstance(pub, dict) and pub.get("address"):
            return pub["address"].split("/")[0]
        for v in doc.values():
            if (found := _address(v)):
                return found
    elif isinstance(doc, list):
        for v in doc:
            if (found := _address(v)):
                return found
    return None


class Nebius(Provider):
    name = "nebius"

    def __init__(self, run: Callable = subprocess.run, project: str | None = None):
        self.run = run
        self.project = project or os.environ.get("NEBIUS_PROJECT_ID")

    def cli(self, *args: str) -> dict:
        if self.run is subprocess.run and not shutil.which("nebius"):
            raise CloudError("no `nebius` CLI: curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash, "
                             "then `nebius profile create`")
        done = self.run(["nebius", *args, "--format", "json"], capture_output=True, text=True)
        if done.returncode != 0:
            err = (done.stderr or done.stdout or "").strip()
            if "quota" in err.lower():
                raise CloudError(f"Nebius quota: {err[:300]} (ask for a GPU quota in the console, or use another region)")
            raise CloudError(f"nebius {' '.join(args[:3])} failed: {err[:500]}")
        return json.loads(done.stdout) if done.stdout.strip() else {}

    def parent(self) -> list[str]:
        return ["--parent-id", self.project] if self.project else []

    def offer(self, gpu: str, spot: bool = False) -> Offer:
        if gpu not in PRESETS:
            raise CloudError(f"Nebius has no one-GPU {gpu}: one of {', '.join(PRESETS)}")
        platform, preset, regions, price, spot_price = PRESETS[gpu]
        return Offer(self.name, gpu.split("-")[0], f"{platform}/{preset}", " or ".join(regions),
                     spot_price if spot else price, LOGIN, spot,
                     extras="network SSD boot disk",
                     note="spot: Nebius stops it when it needs the GPU back; the disk stays" if spot else
                     "project must be in that region (NEBIUS_PROJECT_ID)")

    def create(self, machine: Machine, offer: Offer, public_key: str, disk_gb: int, say: Callable) -> None:
        platform, preset = offer.size.split("/")
        subnets = self.cli("vpc", "subnet", "list", *self.parent()).get("items") or []
        if not subnets:
            raise CloudError("no subnet in this Nebius project: create the default network in the console first")
        subnet = subnets[0]["metadata"]["id"]
        nic = [{"name": "eth0", "subnet_id": subnet, "ip_address": {}, "public_ip_address": {}}]
        machine.resources.append(["instance", machine.name])
        args = ["compute", "instance", "create", *self.parent(), "--name", machine.name,
                "--resources-platform", platform, "--resources-preset", preset,
                "--boot-disk-attach-mode", "READ_WRITE",
                "--boot-disk-managed-disk-name", f"{machine.name}-boot",
                "--boot-disk-managed-disk-type", DISK_TYPE,
                "--boot-disk-managed-disk-size-gibibytes", str(disk_gb),
                "--boot-disk-managed-disk-source-image-family-image-family", IMAGE,
                "--cloud-init-user-data", _cloud_init(public_key),
                "--network-interfaces", json.dumps(nic),
                "--labels", f"blackhole={machine.name}"]
        if offer.spot:
            args += ["--preemptible-on-preemption", "stop"]
        say(f"renting {machine.name}: {platform} {preset}, {disk_gb} GB disk, image {IMAGE}")
        vm = self.cli(*args)          # waits for the operation; returns the instance
        machine.id = vm.get("metadata", {}).get("id") or self._by_name(machine.name).get("metadata", {}).get("id")
        machine.address = _address(vm) or _address(self.cli("compute", "instance", "get", "--id", machine.id))
        if not machine.address:
            raise CloudError(f"{machine.name} has no public address")

    def _by_name(self, name: str) -> dict:
        return self.cli("compute", "instance", "get-by-name", *self.parent(), "--name", name)

    def _id(self, machine: Machine) -> str:
        return machine.id or self._by_name(machine.name)["metadata"]["id"]

    def stop(self, machine: Machine) -> None:
        self.cli("compute", "instance", "stop", "--id", self._id(machine))

    def start(self, machine: Machine) -> None:
        self.cli("compute", "instance", "start", "--id", self._id(machine))
        machine.address = _address(self.cli("compute", "instance", "get", "--id", self._id(machine))) or machine.address

    def delete(self, machine: Machine) -> None:
        try:
            vid = self._id(machine)
        except CloudError as e:
            if "not found" in str(e).lower() or "NotFound" in str(e):
                return       # never made, or already gone
            raise
        self.cli("compute", "instance", "delete", "--id", vid)
