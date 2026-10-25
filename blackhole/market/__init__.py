"""The schema marketplace: a git-backed registry of schemas and run receipts.

Registry repo layout (one GitHub repo; every change is a reviewed pull request):

    schemas/<author>/<name>/<version>.toml
    receipts/<author>/<name>/<version>/<output_sha256 or job>.json
    index.json      # built by registry CI (build_index) from the two trees; clients read only this

- `Registry.open(url_or_dir)` reads index.json from a raw URL or a local clone.
- `search(query, arch, device)` gives the listing rows: id, version, title, tags, tested-on,
  badge, best receipt per base model, and "Fits your PC" for the target device class.
- `pull(ref)` resolves "author/name@^1.2", downloads the file, checks its sha256 against the
  index, and stores it under ~/.blackhole/schemas/.
- `publish(path, clone)` validates a schema, places it in a registry clone on a new branch
  and commits; `gh pr create` then opens the PR (the CLI asks first).
- `submit(receipt, clone)` does the same for a job's receipt.json.

Schemas are data, not code: a schema can only name ops installed on the VM, so pulling one
from a stranger cannot run anything the user did not install. The one exception is lab.*
ops (lab.script runs a Python file shipped with the schema): such schemas are listed as
experimental, never get past "community", and pull() refuses them unless the user passes
allow_code=True after reading the script. Spending is capped by data_policy.frontier_spend_usd,
which the TUI shows before a run.

Badges, per schema version and base model:
    draft      no passing receipt yet
    community  at least one receipt passed the schema's own eval.gates
    verified   a registry maintainer's runner reproduced a passing receipt
    proven     3+ passing receipts from 3+ submitters (or 2 + a verified one),
               metrics agreeing within 3% relative
Receipts only count when their schema_sha256 matches the published file.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from blackhole.core import ops
from blackhole.core import schema as sch
from blackhole.core.lint import lint

DEFAULT_INDEX = os.environ.get("BLACKHOLE_REGISTRY",
                               "https://raw.githubusercontent.com/blackhole-harness/registry/main/index.json")
BADGES = ("draft", "community", "verified", "proven")
TOLERANCE = 0.03


def home() -> Path:
    return Path(os.environ.get("BLACKHOLE_HOME", Path.home() / ".blackhole"))


# ---------------------------------------------------------------- versions

_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z.-]+))?$")


def vkey(v: str) -> tuple:
    m = _SEMVER.match(v)
    if not m:
        raise ValueError(f"not a semver version: {v}")
    pre = m[4] or ""
    return int(m[1]), int(m[2]), int(m[3]), pre == "", pre  # a release sorts above its pre-releases


def matches(version: str, constraint: str) -> bool:
    """'' / latest / *, '1.2.3', '1', '1.2', '^1.2', '~1.2.3', '>=1.0'. Pre-releases only match exactly."""
    a, b, c, release, _ = vkey(version)
    if constraint == version:
        return True
    if not release:
        return False
    if constraint in ("", "latest", "*"):
        return True
    if constraint.startswith(">="):
        return vkey(version) >= vkey(_pad(constraint[2:]))
    if constraint[0] in "^~":
        base = vkey(_pad(constraint[1:]))[:3]
        if (a, b, c) < base:
            return False
        if constraint[0] == "~" or base[0] == 0:
            return (a, b) == base[:2]
        return a == base[0]
    parts = constraint.split(".")
    return len(parts) < 3 and [str(a), str(b)][: len(parts)] == parts


def _pad(v: str) -> str:
    parts = v.split(".")
    return ".".join(parts + ["0"] * (3 - len(parts)))


# ---------------------------------------------------------------- receipts and badges

def _agree(receipts: list[dict]) -> bool:
    shared = set.intersection(*(set(r.get("metrics", {})) for r in receipts))
    for metric in shared:
        vals = [r["metrics"][metric] for r in receipts if isinstance(r["metrics"][metric], (int, float))]
        if len(vals) > 1 and max(vals) and (max(vals) - min(vals)) / abs(max(vals)) > TOLERANCE:
            return False
    return True


def badge(receipts: list[dict], experimental: bool = False) -> str:
    """Badge for one schema version on one base model."""
    passing = [r for r in receipts if r.get("gates_passed") and r.get("outcome", "done") == "done"]
    if not passing:
        return "draft"
    verified = any(r.get("verified_by") for r in passing)
    submitters = {r.get("submitted_by") or r.get("verified_by") or r.get("created_at") for r in passing}
    if experimental:
        return "community"
    if len(passing) >= 3 and (len(submitters) >= 3 or (verified and len(submitters) >= 2)) and _agree(passing):
        return "proven"
    return "verified" if verified else "community"


def badges_by_base(receipts: list[dict], experimental: bool = False) -> dict[str, str]:
    by: dict[str, list[dict]] = {}
    for r in receipts:
        by.setdefault(r.get("base_model", "?"), []).append(r)
    return {base: badge(rs, experimental) for base, rs in sorted(by.items())}


def experimental(raw: dict) -> bool:
    """True when the schema runs code shipped with it (lab.* ops)."""
    return any(str(st.get("kind", "")).startswith("lab.") for st in raw.get("stage") or [])


def best(badges: dict[str, str]) -> str:
    return max(badges.values(), key=BADGES.index, default="draft")


# ---------------------------------------------------------------- index

@dataclass
class Version:
    version: str
    sha256: str
    path: str
    parent: str = ""
    yanked: bool = False
    receipts: list[dict] = field(default_factory=list)
    experimental: bool = False

    @property
    def badges(self) -> dict[str, str]:
        return badges_by_base(self.receipts, self.experimental)

    @property
    def badge(self) -> str:
        return best(self.badges)

    def best_receipts(self) -> dict[str, dict]:
        """Newest passing receipt per base model: the metrics the listing shows."""
        out: dict[str, dict] = {}
        for r in sorted(self.receipts, key=lambda r: r.get("created_at", "")):
            if r.get("gates_passed"):
                out[r.get("base_model", "?")] = r
        return out


@dataclass
class Entry:
    id: str
    title: str = ""
    description: str = ""
    author: str = ""
    license: str = ""
    tags: list[str] = field(default_factory=list)
    base_compat: list[str] = field(default_factory=list)
    targets: dict = field(default_factory=dict)
    footprint: dict = field(default_factory=dict)
    versions: list[Version] = field(default_factory=list)

    def resolve(self, constraint: str = "") -> Version | None:
        ok = [v for v in self.versions if (not v.yanked or v.version == constraint) and matches(v.version, constraint)]
        return max(ok, key=lambda v: vkey(v.version)) if ok else None

    @property
    def latest(self) -> Version | None:
        return self.resolve()


def _fits(entry: Entry, version: Version | None, device: str) -> bool | None:
    """Measured footprint from receipts when there is one, else the schema's declared budget."""
    from blackhole.market.models import DEVICES

    budget = DEVICES.get(device)
    if budget is None:
        return None
    measured = [m for r in (version.best_receipts().values() if version else [])
                for k, m in r.get("metrics", {}).items() if k.endswith("footprint_gb")]
    if measured:
        return min(measured) <= budget
    declared = entry.footprint.get("max_gb")
    if declared:
        return declared <= budget
    dev = entry.footprint.get("device")
    return DEVICES.get(dev, 1e9) <= budget if dev else None


class Registry:
    def __init__(self, index: dict, base: str):
        self.base = base  # URL or directory that index paths are relative to
        self.entries: dict[str, Entry] = {}
        for e in index.get("schemas", []):
            versions = [Version(v["version"], v["sha256"], v["path"], v.get("parent", ""), v.get("yanked", False),
                                v.get("receipts", []), v.get("experimental", False)) for v in e.get("versions", [])]
            fields = {k: e[k] for k in Entry.__dataclass_fields__ if k in e and k != "versions"}
            self.entries[e["id"]] = Entry(**fields, versions=versions)

    @classmethod
    def open(cls, where: str = DEFAULT_INDEX) -> "Registry":
        if where.startswith(("http://", "https://")):
            with urllib.request.urlopen(where, timeout=20) as r:
                return cls(json.load(r), where.rsplit("/", 1)[0])
        p = Path(where).expanduser()
        if p.is_dir():
            p = p / "index.json"
        return cls(json.loads(p.read_text()), str(p.parent))

    def search(self, query: str = "", arch: str = "", device: str = "", tags: list[str] | None = None) -> list[dict]:
        words = query.lower().split()
        rows = []
        for e in self.entries.values():
            hay = " ".join([e.id, e.title, e.description, *e.tags, *e.base_compat]).lower()
            if not all(w in hay for w in words) or (tags and not set(tags) <= set(e.tags)):
                continue
            if arch and e.targets.get("architectures") and arch not in e.targets["architectures"]:
                continue
            v = e.latest
            rows.append({
                "id": e.id, "version": v.version if v else None, "title": e.title, "description": e.description,
                "tags": e.tags, "tested_on": sorted(set(e.base_compat) | set(v.badges if v else {})),
                "badge": v.badge if v else "draft", "badges": v.badges if v else {},
                "results": {base: r.get("metrics", {}) for base, r in (v.best_receipts() if v else {}).items()},
                "fits_your_pc": _fits(e, v, device) if device else None,
                "parent": v.parent if v else "",
                "experimental": v.experimental if v else False,
            })
        return sorted(rows, key=lambda r: (-BADGES.index(r["badge"]), r["id"]))

    def get(self, ref: str) -> tuple[Entry, Version]:
        sid, _, constraint = ref.partition("@")
        entry = self.entries.get(sid)
        if entry is None:
            raise KeyError(f"no schema {sid!r} in the registry")
        v = entry.resolve(constraint)
        if v is None:
            raise KeyError(f"no version of {sid} matches {constraint or 'latest'!r}")
        return entry, v

    def text(self, ref: str) -> str:
        """The schema file's text, checked against the index sha256."""
        _, v = self.get(ref)
        loc = f"{self.base}/{v.path}"
        if loc.startswith(("http://", "https://")):
            with urllib.request.urlopen(loc, timeout=20) as r:
                text = r.read().decode()
        else:
            text = Path(loc).read_text()
        got = sch.parse(tomllib.loads(text)).sha256
        if got != v.sha256:
            raise ValueError(f"{ref}: sha256 mismatch (index {v.sha256[:12]}, file {got[:12]}); not using it")
        return text

    def pull(self, ref: str, allow_code: bool = False) -> Path:
        entry, v = self.get(ref)
        if v.experimental and not allow_code:
            raise PermissionError(f"{entry.id}@{v.version} runs a script shipped with it (lab ops); "
                                  "read it first, then pull with allow_code=True")
        text = self.text(f"{entry.id}@{v.version}")
        dest = home() / "schemas" / entry.id / f"{v.version}.toml"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text)
        return dest


def _listing_key(v: str) -> tuple:
    return vkey(v)[3], vkey(v)  # any release beats any pre-release


def build_index(repo: Path) -> dict:
    """What registry CI runs on every PR: parse and validate every schema, attach receipts
    whose schema_sha256 matches, and fail on anything a client should never see."""
    repo = Path(repo)
    entries: dict[str, dict] = {}
    for path in sorted((repo / "schemas").glob("*/*/*.toml")):
        rel = path.relative_to(repo).as_posix()
        raw = tomllib.loads(path.read_text())
        s = sch.parse(raw)
        problems = ops.validate(s) + lint(raw, publish=True)
        if problems:
            raise ValueError(f"{rel}: " + "; ".join(problems))
        if rel != f"schemas/{s.id}/{s.version}.toml":
            raise ValueError(f"{rel}: must live at schemas/{s.id}/{s.version}.toml")
        receipts = []
        for rp in sorted((repo / "receipts" / s.id / s.version).glob("*.json")):
            r = json.loads(rp.read_text())
            if r.get("schema_sha256") == s.sha256 and r.get("schema") == s.ref:
                receipts.append(r)
        e = entries.setdefault(s.id, {"id": s.id, "versions": [], "_top": "0.0.0-0"})
        if _listing_key(s.version) > _listing_key(e["_top"]):  # listing metadata: newest release
            e.update(_top=s.version, title=raw.get("title", ""), description=raw.get("description", ""),
                     author=raw.get("author", ""), license=raw.get("license", ""), tags=raw.get("tags", []),
                     base_compat=raw.get("base_compat", []), targets=s.targets, footprint=s.footprint)
        exp = experimental(raw)
        e["versions"].append({"version": s.version, "sha256": s.sha256, "path": rel, "parent": raw.get("parent", ""),
                              "experimental": exp, "badges": badges_by_base(receipts, exp), "receipts": receipts})
    for e in entries.values():
        del e["_top"]
        e["versions"].sort(key=lambda v: vkey(v["version"]), reverse=True)
    return {"format": 1, "schemas": sorted(entries.values(), key=lambda e: e["id"])}


# ---------------------------------------------------------------- publish and submit

def _git(clone: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(clone), *args], check=True, capture_output=True, text=True).stdout


def publish(path: Path, clone: Path, index: Registry | None = None) -> dict:
    """Place a schema in a registry clone on a new branch and commit it. Returns
    {branch, file, pr_command}. Versions are immutable: a published id@version never changes."""
    raw = tomllib.loads(Path(path).read_text())
    s = sch.parse(raw)
    problems = ops.validate(s) + lint(raw, publish=True)
    if problems:
        raise ValueError("not publishable: " + "; ".join(problems))
    if index and s.id in index.entries:
        e = index.entries[s.id]
        if any(v.version == s.version for v in e.versions):
            raise ValueError(f"{s.ref} is already published; bump the version")
        if e.latest and vkey(s.version) < vkey(e.latest.version):
            raise ValueError(f"{s.version} is older than the published {e.latest.version}")
    dest = Path(clone) / "schemas" / s.id / f"{s.version}.toml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
    branch = f"publish/{s.id.replace('/', '-')}-{s.version}"
    _git(clone, "checkout", "-b", branch)
    _git(clone, "add", str(dest.relative_to(clone)))
    _git(clone, "commit", "-m", f"publish {s.ref}")
    title = f"Publish {s.ref}: {raw.get('title', '')}"
    return {"branch": branch, "file": str(dest),
            "pr_command": f"git -C {clone} push -u origin {branch} && gh pr create --repo <registry> "
                          f"--head {branch} --title {json.dumps(title)} --body 'sha256 {s.sha256}'"}


def submit(receipt_path: Path, clone: Path, submitted_by: str) -> dict:
    """Add a job's receipt.json to a registry clone on a new branch."""
    r = json.loads(Path(receipt_path).read_text())
    sid, _, version = r["schema"].partition("@")
    r["submitted_by"] = submitted_by
    name = re.sub(r"[^a-z0-9]+", "-", (r.get("output_sha256") or r["created_at"]).lower()).strip("-")[:16]
    dest = Path(clone) / "receipts" / sid / version / f"{name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(r, indent=2) + "\n")
    branch = f"receipt/{sid.replace('/', '-')}-{version}-{name[:8]}"
    _git(clone, "checkout", "-b", branch)
    _git(clone, "add", str(dest.relative_to(clone)))
    _git(clone, "commit", "-m", f"receipt for {r['schema']} on {r.get('base_model')}")
    return {"branch": branch, "file": str(dest)}
