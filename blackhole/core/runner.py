"""Run a job: every stage of its schema, in order, each in its own process.

    python -m blackhole.core.runner --job DIR [--from STAGE] [--only STAGE]

This is REAPer's reaper_pipeline/__main__.py with the stage list taken from the
schema instead of hard-coded:

- each stage runs as `python -m blackhole.core.stage --job DIR --stage ID`, its
  output relayed to stdout (the TUI tails DIR/events.log) and teed to the
  stage's log; SIGTERM stops the running stage so the GPU is freed
- a stage whose cache key is already done is skipped ("cached"), in this job or
  any other job on this machine (blackhole.core.job)
- after a stage, its gates are checked against its metrics.json. A failed
  `stop` gate ends the run cleanly: every later stage is skipped except
  `always: true` ones (uploads), and the run exits 0 with status "stopped",
  like BlackHole's Event Horizon gates
- at the end the schema's eval.gates are checked and receipt.json is written

Events are LobBot's protocol (blackhole.core.events). The last line is
{"stage": "pipeline", "status": "done"|"error", "model": <final artifact>, "outcome": ...}.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from blackhole import __version__
from blackhole.core import ops
from blackhole.core import schema as sch
from blackhole.core.events import emit, parse
from blackhole.core.job import Job

_child: subprocess.Popen | None = None
DETECT = sch.Stage("detect", "model.detect@1")
FETCH = sch.Stage("weights", "model.fetch@1")


def detect_hw() -> dict:
    try:
        from blackhole.hw import detect
        return detect().to_dict()
    except Exception as e:  # no GPU stack on this machine: plans say so
        return {"error": f"{type(e).__name__}: {e}", "gpu_count": 0, "gpu_mem_gb": 0}


def key_of(stage: sch.Stage, params: dict) -> str:
    blob = json.dumps({"uses": stage.uses, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class Wiring:
    """Resolves each stage's params as the run goes: real paths for the op, and a
    path-free twin for the cache key (artifacts stand in as producer key + name)."""

    def __init__(self, job: Job, hw: dict):
        self.job = job
        base = {"model": str(job.base_dir), "id": job.base}
        self.real = {"inputs": job.inputs, "base": base, "hw": hw}
        self.keyed = {"inputs": job.inputs, "base": {"model": f"base:{job.base}", "id": job.base}, "hw": {}}
        self.last_model: str | None = None   # "${<stage>.<artifact>}" of the model so far
        self.last_data: str | None = None    # ... and of the data so far
        if job.schema.raw.get("data"):
            self.real["schema"] = self.keyed["schema"] = {"data": job.schema.raw["data"]}
            self.last_data = "${schema.data}"

    def params(self, stage: sch.Stage, cls: type[ops.Op]) -> tuple[dict, dict]:
        raw = dict(stage.params)
        for pname, kind in cls.consumes.items():
            if pname not in raw and kind in ("model", "gguf"):
                raw[pname] = self.last_model or "${base.model}"
            elif pname not in raw and kind == "dataset" and pname == "data" and self.last_data:
                raw[pname] = self.last_data
        real, keyed = sch.resolve(raw, self.real), sch.resolve(raw, self.keyed)
        ops.make_params(cls, real)   # unknown or mistyped params fail here, before the process starts
        return real, keyed

    def produced(self, stage: sch.Stage, cls: type[ops.Op], key: str) -> None:
        d = self.job.stage_dir(stage.id)
        self.real[stage.id] = {a: str(d / "out" / a) for a in cls.produces}
        self.real[stage.id]["metrics"] = self.job.metrics(stage.id)
        self.keyed[stage.id] = {a: f"{key}/{a}" for a in cls.produces}
        self.keyed[stage.id]["metrics"] = self.real[stage.id]["metrics"]
        if cls.name == "model.detect":   # what the base is: ${base.kinds}, ${base.experts}, ...
            info = json.loads((d / "out" / "info").read_text())
            self.real["base"] = {**info, **self.real["base"]}
            self.keyed["base"] = {**info, **self.keyed["base"]}
        for art, kind in cls.produces.items():
            if kind in ("model", "gguf"):
                self.last_model = f"${{{stage.id}.{art}}}"
            elif kind == "dataset" and art == "data":
                self.last_data = f"${{{stage.id}.{art}}}"


def run_process(job: Job, stage: sch.Stage, cls: type[ops.Op], env: dict) -> tuple[int, list[str], dict | None]:
    global _child
    py = os.environ.get("BLACKHOLE_VLLM_PY") if "vllm" in cls.needs.tools else None
    cmd = [py or sys.executable, "-u", "-m", "blackhole.core.stage", "--job", str(job.root), "--stage", stage.id]
    tail: list[str] = []
    last = None
    with open(job.stage_dir(stage.id) / "log", "a") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        _child = subprocess.Popen(cmd, env={**os.environ, **env, "PYTHONUNBUFFERED": "1"},
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  errors="replace")
        assert _child.stdout
        for line in _child.stdout:
            log.write(line)
            log.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
            ev = parse(line)
            if ev is None:
                if line.strip():
                    tail = (tail + [line.rstrip()])[-30:]
            else:
                last = ev
        rc = _child.wait()
        _child = None
    return rc, tail, last


def error_message(rc: int, tail: list[str]) -> str:
    for line in reversed(tail):
        if line and not line.startswith(" ") and ("Error" in line or "Exception" in line):
            return line[:400]
    if rc < 0:
        return f"stage killed by signal {-rc}"
    return tail[-1][:400] if tail else f"stage exited with {rc}"


def stages_of(job: Job) -> list[sch.Stage]:
    return [DETECT] + ([] if Path(job.base).expanduser().is_dir() else [FETCH]) + job.schema.stages


def run(job: Job, start: str | None = None, only: str | None = None) -> int:
    hw = detect_hw()
    job.write("hardware.json", hw)
    wiring = Wiring(job, hw)
    stages = stages_of(job)
    ids = [s.id for s in stages]
    rerun = set(ids[ids.index(start):]) if start else ({only} if only else set())
    outcome, failed, receipts = "done", None, []
    for stage in stages:
        cls = ops.get(stage.op, stage.major)
        if outcome != "done" and not stage.always:
            emit(stage.id, "skipped", msg=f"after {failed}: {outcome}")
            continue
        if not sch.resolve(stage.when, wiring.real):
            emit(stage.id, "skipped", 100, "when: false")
            continue
        try:
            real, keyed = wiring.params(stage, cls)
        except (sch.SchemaError, ValueError, TypeError) as e:
            emit(stage.id, "error", msg=str(e)[:400])
            outcome, failed = "error", stage.id
            continue
        key = key_of(stage, keyed)
        forced = stage.id in rerun
        if only and stage.id != only:
            if not job.is_cached(key):
                if ids.index(stage.id) < ids.index(only):
                    emit(stage.id, "error", msg="not cached; --only needs every earlier stage done")
                    outcome, failed = "error", stage.id
                else:
                    emit(stage.id, "skipped", msg="--only")
                continue
        if job.is_cached(key) and not forced:
            job.link(stage.id, key)
            wiring.produced(stage, cls, key)
            emit(stage.id, "skipped", 100, "cached")
        else:
            d = job.work_dir(stage.id, key)
            (d / "params.json").write_text(json.dumps({"uses": stage.uses, "params": real}, indent=2) + "\n")
            emit(stage.id, "running", 0, "starting")
            env = {k: str(v) for k, v in ((job.read("plan.json") or {}).get(stage.id, {}).get("env") or {}).items()}
            rc, tail, last = run_process(job, stage, cls, env)
            if rc != 0:
                if not (last and last["status"] == "error"):
                    emit(stage.id, "error", msg=error_message(rc, tail))
                outcome, failed = "error", stage.id
                continue
            write_lineage(job, stage, cls, real)
            job.mark_done(key)
            wiring.produced(stage, cls, key)
            emit(stage.id, "done", 100, (last or {}).get("msg", ""), **{"metrics": job.metrics(stage.id)})
        if stage is DETECT:
            from blackhole.hw.model import ModelInfo, applicable
            info = {k: v for k, v in wiring.real["base"].items() if k in ModelInfo.__dataclass_fields__}
            why = applicable(ModelInfo(**info), job.schema)
            if why:
                emit("detect", "error", msg="; ".join(why)[:400])
                outcome, failed = "error", "detect"
        got = job.metrics(stage.id)
        for gate in stage.gates:
            gate = sch.Gate(gate.metric, gate.op, sch.resolve(gate.value, wiring.real), gate.on_fail)
            if gate.check(got.get(gate.metric)):
                continue
            emit(stage.id, "running", 100, f"gate failed: {gate.describe(got.get(gate.metric))}", gate="failed")
            if gate.on_fail == "stop" and outcome == "done":
                outcome, failed = "stopped", f"{stage.id} gate {gate.metric}"
    final = wiring.last_model and sch.resolve(wiring.last_model, wiring.real)
    receipt = write_receipt(job, hw, outcome, final)
    if outcome == "error":
        emit("pipeline", "error", msg=f"{failed} failed")
        return 1
    emit("pipeline", "done", 100, final if outcome == "done" else f"stopped: {failed}",
         model=final if outcome == "done" else None, outcome=outcome, gates_passed=receipt["gates_passed"])
    return 0


def write_lineage(job: Job, stage: sch.Stage, cls: type[ops.Op], params: dict) -> None:
    """Every model a stage makes carries blackhole-lineage.json: its base and every step
    since (mugge's mugge-lineage.json), so a file found on a laptop says how it was made."""
    src = Path(params.get("model") or "")
    prev = json.loads((src / "blackhole-lineage.json").read_text()) if (src / "blackhole-lineage.json").exists() \
        else {"base": job.base, "steps": []}
    for art, kind in cls.produces.items():
        out = job.stage_dir(stage.id) / "out" / art
        if kind == "model" and out.is_dir():
            step = {"schema": job.schema.ref, "stage": stage.id, "uses": stage.uses,
                    "params": {k: v for k, v in params.items() if k != "model"}}
            (out / "blackhole-lineage.json").write_text(
                json.dumps({**prev, "steps": prev["steps"] + [step]}, indent=2, default=str) + "\n")


def write_receipt(job: Job, hw: dict, outcome: str, final: str | None) -> dict:
    """The run record the marketplace reads: a schema is "proven" on a base model once
    enough receipts with gates_passed come back from independent machines."""
    metrics = {f"{s.id}.{k}": v for s in job.schema.stages for k, v in job.metrics(s.id).items()}
    passed = outcome == "done" and all(g.check(metrics.get(g.metric)) for g in job.schema.eval_gates)
    sha = None
    if final and Path(final).is_file():
        h = hashlib.sha256()
        with open(final, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        sha = h.hexdigest()
    elif final and Path(final).is_dir():
        sha = (Path(final) / "manifest.sha256").read_text().strip() if (Path(final) / "manifest.sha256").exists() else None
    receipt = {"schema": job.schema.ref, "schema_sha256": job.schema.sha256, "base_model": job.base,
               "hardware": {"gpu": hw.get("gpu_name"), "count": hw.get("gpu_count")}, "metrics": metrics,
               "outcome": outcome, "gates_passed": passed, "output_sha256": sha, "runner_version": __version__,
               "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    job.write("receipt.json", receipt)
    return receipt


def start_detached(job: Job, *args: str) -> int:
    """Start the runner for the TUI: own session, output appended to events.log, so it
    survives the TUI closing and the SSH connection dropping. Returns the pid."""
    with open(job.path("events.log"), "a") as out:
        proc = subprocess.Popen([sys.executable, "-u", "-m", "blackhole.core.runner", "--job", str(job.root), *args],
                                stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    job.path("pid").write_text(str(proc.pid))
    return proc.pid


def _stop(signum, _frame):
    if _child and _child.poll() is None:
        _child.terminate()
        try:
            _child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _child.kill()
    emit("pipeline", "error", msg="stopped")
    sys.exit(128 + signum)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True)
    ap.add_argument("--from", dest="start", help="rerun from this stage onward")
    ap.add_argument("--only", help="rerun this one stage (earlier ones must be cached)")
    a = ap.parse_args(argv)
    job = Job(a.job)
    job.path("pid").write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    return run(job, a.start, a.only)


if __name__ == "__main__":
    sys.exit(main())
