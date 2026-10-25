"""Job directories and the shared stage cache.

    $BLACKHOLE_HOME (default ~/.blackhole)/
        models/<org>--<name>/      downloaded base models (the weights stage)
        cache/<key>/               one finished stage, shared by every job: out/, metrics.json, log
        jobs/<job>/
            schema.toml            frozen copy of the schema this job runs (.json if it was JSON)
            job.json               {schema, schema_sha256, base, inputs, created}
            events.log             progress events, one JSON object per line (the TUI tails this)
            pid                    the runner's pid while it runs
            hardware.json          what was detected when the run started
            plan.json              per-stage plan (memory, disk, hours, cost), refused early if it cannot run
            stages/<id> -> ../../../cache/<key>   finished stages are symlinks into the cache
            stages/<id>/params.json  the op's resolved params
            receipt.json           the marketplace receipt, written when the run ends

A stage's cache key hashes its op, the op's major version and its resolved
params, where an artifact from an earlier stage stands in as that stage's key.
So a rerun skips every stage whose inputs did not change, and a second job on
the same base model with a different prune ratio reuses its calibration data
and REAP scores (REAPer's `--prune` rerun, generalised to any schema).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

from blackhole.core import schema as sch


def home() -> Path:
    return Path(os.environ.get("BLACKHOLE_HOME") or Path.home() / ".blackhole")


def job_name(schema: sch.Schema, base: str, inputs: dict) -> str:
    model = re.sub(r"[^A-Za-z0-9._-]+", "-", base.rstrip("/").split("/")[-1])
    tag = re.sub(r"[^a-z0-9]+", "-", str(inputs.get("purpose") or "").lower()).strip("-")[:24]
    return f"{model}-{schema.id.split('/')[-1]}" + (f"-{tag}" if tag else "")


class Job:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        (self.root / "stages").mkdir(parents=True, exist_ok=True)
        self.meta = self.read("job.json") or {}
        self.schema = sch.load(next(self.root.glob("schema.*")))
        self.inputs = sch.input_values(self.schema, self.meta.get("inputs") or {})

    @classmethod
    def create(cls, schema_path: str | Path, base: str, inputs: dict, root: str | Path | None = None,
               check: bool = True) -> "Job":
        """A new job. With check, the base model is detected first (a few KB) and a schema
        that cannot run on it is refused here, before a directory or a download."""
        schema = sch.load(schema_path)
        values = sch.input_values(schema, inputs)   # fail before making a directory
        if check:
            from blackhole.hw.model import applicable, detect
            base_dir = base if Path(base).expanduser().is_dir() else None
            why = applicable(detect(base, base_dir), schema)
            if why:
                raise sch.SchemaError("; ".join(why))
        root = Path(root) if root else home() / "jobs" / job_name(schema, base, values)
        root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(schema_path, root / ("schema" + Path(schema_path).suffix))
        (root / "job.json").write_text(json.dumps({
            "schema": schema.ref, "schema_sha256": schema.sha256, "base": base, "inputs": inputs,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2) + "\n")
        return cls(root)

    @property
    def base(self) -> str:
        return self.meta["base"]

    @property
    def base_dir(self) -> Path:
        p = Path(self.base).expanduser()
        if p.is_dir():
            return p.resolve()
        return home() / "models" / self.base.replace("/", "--")

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def read(self, rel: str):
        p = self.path(rel)
        return json.loads(p.read_text()) if p.exists() else None

    def write(self, rel: str, obj) -> None:
        p = self.path(rel)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2) + "\n")
        tmp.replace(p)

    # ------------------------------------------------------------------ stage cache

    @staticmethod
    def cache_dir(key: str) -> Path:
        return home() / "cache" / key[:24]

    def stage_dir(self, sid: str) -> Path:
        return self.path("stages", sid)

    def is_cached(self, key: str) -> bool:
        return (self.cache_dir(key) / ".done").exists()

    def work_dir(self, sid: str, key: str) -> Path:
        """A fresh directory for a stage about to run, linked from stages/<id>."""
        d = self.cache_dir(key)
        if d.exists():
            shutil.rmtree(d)                 # a half-written or forced rerun of the same stage
        d.mkdir(parents=True, exist_ok=True)
        self.link(sid, key)
        return d

    def link(self, sid: str, key: str) -> None:
        link = self.stage_dir(sid)
        if link.is_symlink() or link.exists():
            link.unlink() if link.is_symlink() else shutil.rmtree(link)
        link.symlink_to(self.cache_dir(key), target_is_directory=True)

    def mark_done(self, key: str) -> None:
        (self.cache_dir(key) / ".done").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    def metrics(self, sid: str) -> dict:
        p = self.stage_dir(sid) / "metrics.json"
        return json.loads(p.read_text()) if p.exists() else {}
