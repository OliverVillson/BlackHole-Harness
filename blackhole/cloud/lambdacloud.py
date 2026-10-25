"""Lambda Cloud, through its REST API (https://docs.lambda.ai/public-cloud/cloud-api/).

    export LAMBDA_API_KEY=...          # cloud.lambda.ai -> API keys
    blackhole up --provider lambda --gpu h100 --yes

Instance types, prices and the regions with capacity are read live from /instance-types, so
nothing here goes stale; the cheapest one-GPU type of the asked family with capacity wins. The
image is Lambda Stack (Ubuntu with the driver), login `ubuntu`. Lambda cannot stop a VM, only
terminate it, so `blackhole down --keep-disk` is refused: copy the model off first.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable

from blackhole.cloud import CloudError, Machine, Offer, Provider, wait

API = os.environ.get("LAMBDA_API_URL", "https://cloud.lambda.ai/api/v1")
LOGIN = "ubuntu"


class Lambda(Provider):
    name = "lambda"
    can_stop = False

    def __init__(self, opener: Callable = urllib.request.urlopen, key: str | None = None, sleep=None):
        self.open, self.key = opener, key or os.environ.get("LAMBDA_API_KEY")
        self.sleep = sleep

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        if not self.key:
            raise CloudError("no LAMBDA_API_KEY: make one at cloud.lambda.ai under API keys")
        req = urllib.request.Request(API + path, method=method, data=json.dumps(body).encode() if body else None,
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        try:
            with self.open(req, timeout=60) as r:
                return json.load(r).get("data", {})
        except urllib.error.HTTPError as e:
            raise CloudError(f"lambda {method} {path} failed ({e.code}): {e.read().decode(errors='replace')[:400]}") from e
        except (urllib.error.URLError, OSError) as e:
            raise CloudError(f"lambda {method} {path}: no answer ({e})") from e

    def _types(self, gpu: str) -> list[tuple[int, str, list[str]]]:
        """(cents an hour, type, regions with capacity) for one-GPU types of this family, cheapest first."""
        out = []
        for name, t in self.call("GET", "/instance-types").items():
            it = t.get("instance_type", {})
            if not name.startswith("gpu_1x_") or gpu not in name.lower():
                continue
            out.append((it.get("price_cents_per_hour", 0), name,
                        [r["name"] for r in t.get("regions_with_capacity_available", [])]))
        return sorted(out)

    def offer(self, gpu: str, spot: bool = False) -> Offer:
        if spot:
            raise CloudError("Lambda has no spot VMs")
        types = self._types(gpu)
        if not types:
            raise CloudError(f"Lambda has no one-GPU {gpu} type")
        with_cap = [t for t in types if t[2]]
        cents, name, regions = (with_cap or types)[0]
        note = "" if with_cap else "no region has capacity right now; try again later or another GPU"
        return Offer(self.name, gpu, name, regions[0] if regions else "-", cents / 100, LOGIN,
                     extras="no separate disk charge (the disk goes with the VM)", note=note)

    def _key_name(self, public_key: str) -> str:
        for k in self.call("GET", "/ssh-keys") or []:
            if k.get("public_key", "").split()[:2] == public_key.split()[:2]:
                return k["name"]
        name = "blackhole-" + public_key.split()[1][-8:].replace("/", "_").replace("+", "-")
        self.call("POST", "/ssh-keys", {"name": name, "public_key": public_key})
        return name

    def create(self, machine: Machine, offer: Offer, public_key: str, disk_gb: int, say: Callable) -> None:
        if offer.region == "-":
            raise CloudError(f"no Lambda region has a free {offer.size}")
        key = self._key_name(public_key)
        say(f"renting {machine.name}: {offer.size} in {offer.region}")
        machine.resources.append(["instance", "?"])
        ids = self.call("POST", "/instance-operations/launch", {
            "region_name": offer.region, "instance_type_name": offer.size, "ssh_key_names": [key],
            "name": machine.name}).get("instance_ids") or []
        if not ids:
            raise CloudError("Lambda accepted the launch but returned no instance id")
        machine.id = ids[0]
        machine.resources[-1] = ["instance", machine.id]

        def booted() -> bool:
            inst = self.call("GET", f"/instances/{machine.id}")
            if inst.get("status") in ("terminated", "unhealthy"):
                raise CloudError(f"{machine.name} is {inst['status']}")
            machine.address = inst.get("ip")
            return inst.get("status") == "active" and bool(machine.address)

        kw = {"sleep": self.sleep} if self.sleep else {}
        wait("the Lambda instance to boot", booted, 1200, 10, **kw)

    def delete(self, machine: Machine) -> None:
        if machine.id:
            self.call("POST", "/instance-operations/terminate", {"instance_ids": [machine.id]})
