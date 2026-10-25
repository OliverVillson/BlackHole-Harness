"""evroc (https://evroc.com), the Swedish cloud, through its REST API. Ported from mugge.cloud, which
rents and returns evroc machines for `mugge run --host evroc`; the calls and their order are the same.

    evroc login                         # once; or EVROC_REFRESH_TOKEN / EVROC_USERNAME + EVROC_PASSWORD
    export EVROC_PROJECT=...            # the id under the project's name at cloud.evroc.com
    blackhole up --provider evroc --gpu l40s --yes

A boot disk from Ubuntu 24.04, a static public IP and the VM, in a security group that lets SSH in
(made once, kept). evroc's image has no driver at first boot; its cloud-init installs one, which is
why `up` waits for nvidia-smi and for cloud-init to finish before running install.sh. Login is
`evroc-user`. Stopping sets spec.running = false with a PUT, which mugge never did: untested
against the live API, so check it once before relying on `down --keep-disk` here.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from blackhole.cloud import CloudError, Machine, Offer, Provider, wait

API_URL = "https://api.evroc.com"
TOKEN_URL = "https://authn.iam.evroc.com/realms/evroc-customer/protocol/openid-connect/token"
CLIENT_ID = "evroc-cli"
VERSION = "v1beta2"
REGION, ZONE = "se-sto", "a"
IMAGE = "ubuntu.24-04.1"
LOGIN = "evroc-user"
SECURITY_GROUP = "blackhole-ssh"
ANYWHERE = [("v4", "0.0.0.0/0"), ("v6", "::/0")]

SIZES = {"l40s": "gn-l40s.s", "b200": "gn-b200.s"}
# Euros an hour. L40S from an evroc invoice (2026-09-30, as in mugge); evroc publishes no table by
# size. Disk €0.000137/GB/h and a public IP €0.01/h from https://evroc.com/cloud-services/virtual-machines/
EUROS = {"gn-l40s.s": 1.50}
DISK_EUROS_PER_GB_HOUR, IP_EUROS_PER_HOUR = 0.000137, 0.01


class NotFound(CloudError):
    pass


def _cli_profile(path: Path) -> dict:
    """The evroc CLI's current profile from ~/.evroc/config.yaml: plain nested mappings, no YAML library."""
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#") or ":" not in raw:
            continue
        indent = len(raw) - len(raw.lstrip())
        key, _, value = raw.strip().partition(":")
        while stack[-1][0] >= indent:
            stack.pop()
        key, value = key.strip().strip("'\""), value.strip().strip("'\"")
        if value:
            stack[-1][1][key] = value
        else:
            stack[-1][1][key] = child = {}
            stack.append((indent, child))
    return root.get("profiles", {}).get(root.get("currentProfile", ""), {})


class EvrocProvider(Provider):
    name = "evroc"

    def __init__(self, opener: Callable = urllib.request.urlopen, env: dict | None = None,
                 cli_path: Path | None = None, sleep: Callable = time.sleep):
        env = os.environ if env is None else env
        self.open, self.sleep = opener, sleep
        self.username, self.password = env.get("EVROC_USERNAME"), env.get("EVROC_PASSWORD")
        self.refresh, self.project = env.get("EVROC_REFRESH_TOKEN"), env.get("EVROC_PROJECT")
        self.region, self.api_url, self.token_url = env.get("EVROC_REGION") or REGION, API_URL, TOKEN_URL
        cli_path = cli_path or Path.home() / ".evroc" / "config.yaml"
        if not (self.username and self.password) and not self.refresh and cli_path.exists():
            prof = _cli_profile(cli_path)
            self.refresh = prof.get("user", {}).get("refreshToken")
            self.project = self.project or prof.get("project")
            self.region = env.get("EVROC_REGION") or prof.get("region") or REGION
            self.api_url = (prof.get("apiURL") or API_URL).rstrip("/")
            if prof.get("issuerURL"):
                self.token_url = prof["issuerURL"].rstrip("/") + "/protocol/openid-connect/token"
        self.token, self.expires = None, 0.0

    # ---- API

    def _token(self) -> str:
        if self.token and time.time() < self.expires - 30:
            return self.token
        if not (self.username and self.password) and not self.refresh:
            raise CloudError("no evroc credentials: run `evroc login`, or set EVROC_REFRESH_TOKEN "
                             "(or EVROC_USERNAME and EVROC_PASSWORD)")
        form = ({"grant_type": "password", "username": self.username, "password": self.password}
                if self.username and self.password else {"grant_type": "refresh_token", "refresh_token": self.refresh})
        req = urllib.request.Request(self.token_url, data=urllib.parse.urlencode({**form, "client_id": CLIENT_ID}).encode(),
                                     headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with self.open(req, timeout=30) as r:
                data = json.load(r)
        except urllib.error.HTTPError as e:
            raise CloudError(f"evroc login failed ({e.code}); run `evroc login` again") from e
        self.refresh = data.get("refresh_token") or self.refresh
        self.token, self.expires = data["access_token"], time.time() + float(data.get("expires_in", 300))
        return self.token

    def ref(self, service: str, collection: str, name: str) -> str:
        return f"/{service}/projects/{self.project}/regions/{self.region}/{collection}/{name}"

    def path(self, service: str, collection: str, name: str = "") -> str:
        return f"/{service}/{VERSION}/projects/{self.project}/regions/{self.region}/{collection}" + (f"/{name}" if name else "")

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        if not self.project:
            raise CloudError("no evroc project: set EVROC_PROJECT")
        req = urllib.request.Request(self.api_url + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self._token()}",
                                              "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with self.open(req, timeout=60) as r:
                text = r.read()
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            if e.code == 404:
                raise NotFound(f"{method} {path}: not found") from e
            if e.code == 403 and "quota" in msg:
                raise CloudError("evroc's GPU quota is used up: stop or delete another GPU VM, or ask evroc for more") from e
            raise CloudError(f"evroc {method} {path} failed ({e.code}): {msg}") from e
        except (urllib.error.URLError, OSError) as e:
            raise CloudError(f"evroc {method} {path}: no answer ({e})") from e
        return json.loads(text) if text.strip() else {}

    def get(self, service, collection, name):
        return self.call("GET", self.path(service, collection, name))

    def exists(self, service, collection, name) -> bool:
        try:
            self.get(service, collection, name)
            return True
        except NotFound:
            return False

    # ---- Provider

    def offer(self, gpu: str, spot: bool = False) -> Offer:
        if spot:
            raise CloudError("evroc has no spot VMs")
        if gpu not in SIZES:
            raise CloudError(f"evroc has no one-GPU {gpu}: one of {', '.join(SIZES)}")
        size = SIZES[gpu]
        return Offer(self.name, gpu, size, self.region, EUROS.get(size), LOGIN, currency="€",
                     extras=f"disk €{DISK_EUROS_PER_GB_HOUR * 1000:.3f}/h per 1000 GB, IP €{IP_EUROS_PER_HOUR:.2f}/h",
                     note="" if size in EUROS else "evroc publishes no price for this size; check your contract")

    def create(self, machine: Machine, offer: Offer, public_key: str, disk_gb: int, say: Callable) -> None:
        name, labels = machine.name, {"blackhole": machine.name}
        meta = {"id": name, "userLabels": labels}
        if not self.exists("networking", "securityGroups", SECURITY_GROUP):
            say(f"making the security group {SECURITY_GROUP} (SSH in, anything out; kept)")
            self.call("POST", self.path("networking", "securityGroups"), {
                "apiVersion": f"networking/{VERSION}", "kind": "SecurityGroup", "metadata": {"id": SECURITY_GROUP},
                "spec": {"rules": [
                    *({"direction": "Ingress", "name": f"allow-ssh-{f}", "port": 22, "protocol": "TCP",
                       "remote": {"address": {"ipAddressOrCIDR": c}}} for f, c in ANYWHERE),
                    # both families out: a download that picks IPv6 with no way out hangs (mugge lost 40 GPU-minutes)
                    *({"direction": "Egress", "name": f"allow-all-egress-{f}", "protocol": "All",
                       "remote": {"address": {"ipAddressOrCIDR": c}}} for f, c in ANYWHERE)],
                    "vpcRef": self.ref("networking", "virtualPrivateClouds", f"default-{self.region}")}})
        say(f"renting {name}: {offer.size}, {disk_gb} GB disk, in {self.region}")
        # each resource is recorded before it is asked for, so a timed-out request is still cleaned up
        machine.resources.append(["networking", "publicIPs", name])
        self.call("POST", self.path("networking", "publicIPs"),
                  {"apiVersion": f"networking/{VERSION}", "kind": "PublicIP", "metadata": meta, "spec": {}})
        machine.resources.append(["compute", "disks", name])
        self.call("POST", self.path("compute", "disks"), {
            "apiVersion": f"compute/{VERSION}", "kind": "Disk", "metadata": meta,
            "spec": {"diskSize": {"amount": disk_gb, "unit": "GB"}, "placement": {"zone": ZONE},
                     "source": {"type": "image", "diskImageRef": f"/compute/global/diskImages/evroc/{IMAGE}"}}})
        wait("the disk", lambda: _ready(self.get("compute", "disks", name)), 600, 5, self.sleep)
        machine.resources.append(["compute", "virtualMachines", name])
        self.call("POST", self.path("compute", "virtualMachines"), {
            "apiVersion": f"compute/{VERSION}", "kind": "VirtualMachine", "metadata": meta,
            "spec": {
                "computeProfileRef": f"/compute/global/computeProfiles/{offer.size}",
                "disks": [{"bootFrom": True, "diskRef": self.ref("compute", "disks", name)}],
                "networking": {
                    "publicIPv4Address": {"static": {"publicIPRef": self.ref("networking", "publicIPs", name)}},
                    "securityGroupSettings": {"securityGroupMemberRefs": [self.ref("networking", "securityGroups", SECURITY_GROUP)]},
                    "subnetRef": self.ref("networking", "subnets", f"default-{self.region}-{ZONE}")},
                "osSettings": {"ssh": {"authorizedKeys": [{"value": public_key}]}},
                "placement": {"zone": ZONE},
                "running": True}})

        def vm_ready() -> bool:
            vm = self.get("compute", "virtualMachines", name)
            st = vm.get("status", {})
            if st.get("virtualMachineStatus") == "ProvisioningFailed":
                raise CloudError(f"{name}: provisioning failed")
            return _ready(vm)

        wait("the VM to start", vm_ready, 1200, 5, self.sleep)

        def address() -> bool:
            machine.address = self.get("networking", "publicIPs", name).get("status", {}).get("publicIPv4Address")
            return bool(machine.address)

        wait("its public address", address, 300, 5, self.sleep)
        machine.id = name

    def _running(self, machine: Machine, running: bool) -> None:
        vm = self.get("compute", "virtualMachines", machine.name)
        vm.setdefault("spec", {})["running"] = running
        self.call("PUT", self.path("compute", "virtualMachines", machine.name), vm)

    def stop(self, machine: Machine) -> None:
        self._running(machine, False)

    def start(self, machine: Machine) -> None:
        self._running(machine, True)   # the static IP keeps machine.address

    def delete(self, machine: Machine) -> None:
        """The VM first (it holds the disk and the address), then the rest, newest first."""
        for service, collection, name in reversed(machine.resources):
            try:
                self.call("DELETE", self.path(service, collection, name))
            except NotFound:
                continue
            if collection == "virtualMachines":
                wait(f"{name} to go", lambda: not self.exists(service, collection, name), 600, 5, self.sleep)


def _ready(resource: dict) -> bool:
    return any(c.get("type") == "Ready" and c.get("status") == "True"
               for c in resource.get("status", {}).get("conditions") or [])
