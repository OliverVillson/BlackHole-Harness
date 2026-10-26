"""`blackhole_pipeline --recipe X.bhr`: bundle checks, the job config a recipe writes, recipe-order
processes with job functions, and a run whose only new stage is a job function. No GPU, no torch;
the bundles are written here directly (the private blackhole_dev package is never imported)."""

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

from blackhole_pipeline import recipe as rc

pytest.importorskip("reaper_pipeline")


def raw_recipe(*stages: dict, inputs: dict | None = None) -> dict:
    r = {"api": "blackhole/v1", "id": "me/test", "version": "0.1.0", "base": "Qwen/Qwen3-30B-A3B",
         "stage": list(stages)}
    if inputs:
        r["inputs"] = inputs
    return r


def make_bundle(path: Path, raw: dict, **manifest) -> Path:
    files = {"recipe.py": b"# the source\n", "recipe.json": json.dumps({"raw": raw, "meta": {"target": "mac16"}}).encode()}
    rc.write_bundle(path, files, {"id": raw["id"], "version": raw["version"], "runner": rc.runner_of(raw), **manifest})
    return path


def test_bundle_is_checked_and_reproducible(tmp_path):
    raw = raw_recipe({"id": "reap", "kind": "pipeline.reap", "prune": 0.25})
    a = make_bundle(tmp_path / "a.bhr", raw)
    b = make_bundle(tmp_path / "b.bhr", raw)
    assert a.read_bytes() == b.read_bytes()
    man, files = rc.read_bundle(a)
    assert man["format"] == "bhr/1" and set(man["files"]) == {"recipe.py", "recipe.json"}
    with tarfile.open(a) as tf:
        members = {m.name: tf.extractfile(m).read() for m in tf.getmembers()}

    def repack(name: str, mutate) -> Path:
        m = dict(members)
        mutate(m)
        p = tmp_path / name
        with tarfile.open(p, "w:gz") as tf:
            for n, d in m.items():
                ti = tarfile.TarInfo(n)
                ti.size = len(d)
                tf.addfile(ti, io.BytesIO(d))
        return p

    with pytest.raises(rc.RecipeBundleError, match="does not match"):
        rc.read_bundle(repack("t1.bhr", lambda m: m.update({"recipe.py": b"import os\n"})))
    with pytest.raises(rc.RecipeBundleError, match="not pinned"):
        rc.read_bundle(repack("t2.bhr", lambda m: m.update({"functions/evil.py": b"x\n"})))
    with pytest.raises(rc.RecipeBundleError, match="unsafe"):
        rc.read_bundle(repack("t3.bhr", lambda m: m.update({"../x": b""})))


def test_job_config_and_process_order(tmp_path):
    raw = raw_recipe({"id": "reap", "kind": "pipeline.reap", "prune": "${inputs.prune}", "calib": "code"},
                     {"id": "distil", "kind": "job.fn", "fn": "distil", "rounds": 2},
                     {"id": "mlx", "kind": "pipeline.mlx", "mlx_hex": 1},
                     {"id": "bench", "kind": "pipeline.bench"},
                     inputs={"prune": {"type": "float", "default": 0.25}})
    assert rc.runner_of(raw) == "pipeline" and rc.problems(raw) == []
    b = make_bundle(tmp_path / "r.bhr", raw, inputs={"prune": 0.375})
    job = tmp_path / "job"
    cfg = rc.prepare(job, b, inputs={"prune": "0.5"})
    assert cfg["stage_list"] == ["detect", "plan", "weights", "calibrate", "score", "prune", "eval", "distil", "mlx", "bench"]
    assert cfg["steps"][0] == {"kind": "reap", "params": {"prune": 0.5, "calib": "code"}}
    assert cfg["steps"][1] == {"kind": "fn", "id": "distil", "fn": "distil", "params": {"rounds": 2}}
    assert cfg["model"] == "Qwen/Qwen3-30B-A3B" and cfg["target"] == "mac16"
    assert (job / "recipe" / "recipe.py").exists() and (job / "recipe" / "schema.json").exists()
    procs = rc.processes(cfg["stage_list"])
    assert [p[0] for p in procs] == ["detect", "plan", "weights", "calibrate", "reap", "eval", "distil", "mlx", "bench"]
    assert procs[6] == ("distil", ["distil"], "fn")
    from blackhole_pipeline.job import params
    assert params(cfg)["prune"] == 0.5 and params(cfg)["mlx_hex"] == 1
    # a rerun keeps the existing config (inputs given now are ignored), another recipe is refused
    assert rc.prepare(job, b, inputs={"prune": "0.1"}) == cfg
    other = make_bundle(tmp_path / "o.bhr", {**raw, "id": "me/other"})
    with pytest.raises(rc.RecipeBundleError, match="belongs to"):
        rc.prepare(job, other)
    bad = raw_recipe({"kind": "pipeline.reap"}, {"kind": "pipeline.reap"}, {"kind": "model.edit", "fn": "x"})
    assert len(rc.problems(bad)) == 2 and rc.runner_of(bad) == "mixed"
    with pytest.raises(rc.RecipeBundleError, match="mixes"):
        rc.prepare(tmp_path / "job2", make_bundle(tmp_path / "m.bhr", bad))


def test_a_job_function_stage_runs_in_recipe_order(tmp_path, monkeypatch):
    """detect..calibrate are already done (cached), so the run starts the job function: a module
    found through the blackhole.jobfn entry point (patched here) that emits events and marks done."""
    mods = tmp_path / "mods"
    mods.mkdir()
    (mods / "fake_jobfn.py").write_text(
        "import argparse, json, pathlib\n"
        "ap = argparse.ArgumentParser(); ap.add_argument('--job'); ap.add_argument('--stage'); a = ap.parse_args()\n"
        "cfg = json.loads((pathlib.Path(a.job) / 'config.json').read_text())\n"
        "step = next(s for s in cfg['steps'] if s.get('id') == a.stage)\n"
        "print(json.dumps({'stage': a.stage, 'status': 'running', 'pct': 50, 'msg': step['fn']}), flush=True)\n"
        "d = pathlib.Path(a.job) / 'reaper' / '.done'; d.mkdir(parents=True, exist_ok=True); (d / a.stage).write_text('{}')\n"
        "print(json.dumps({'stage': a.stage, 'status': 'done', 'pct': 100}), flush=True)\n")
    monkeypatch.setenv("PYTHONPATH", str(mods) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setattr(rc, "jobfn_module", lambda: "fake_jobfn")
    raw = raw_recipe({"id": "distil", "kind": "job.fn", "fn": "distil-short"})
    b = make_bundle(tmp_path / "r.bhr", raw)
    job = tmp_path / "job"
    rc.prepare(job, b)
    from blackhole_pipeline.job import BHJob
    rj = BHJob(job).reaper()
    for s in rc.ALWAYS_FIRST:
        rj.mark_done(s)
    from blackhole_pipeline import __main__ as pm
    lines = []
    monkeypatch.setattr(pm, "emit", lambda stage, status="running", pct=None, msg="", **kw:
                        lines.append({"stage": stage, "status": status, "msg": msg, **kw}))
    monkeypatch.setattr("builtins.print", lambda *a, **k: lines.append(a[0] if a else ""))
    assert pm.run(BHJob(job)) == 0
    assert rj.is_done("distil")
    evs = [json.loads(x) if isinstance(x, str) and x.startswith("{") else x for x in lines]
    assert any(isinstance(e, dict) and e.get("stage") == "distil" and e.get("msg") == "distil-short" for e in evs)
    assert any(isinstance(e, dict) and e.get("stage") == "detect" and e.get("status") == "skipped" for e in evs)
    assert evs[-1]["stage"] == "pipeline" and evs[-1]["status"] == "done"
    # --from accepts the recipe's own stage names and refuses others
    assert pm.run(BHJob(job), "distil") == 0
    assert pm.run(BHJob(job), "qheal") == 2


def test_main_reports_a_bad_bundle_as_an_event(tmp_path, capsys):
    from blackhole_pipeline import __main__ as pm
    (tmp_path / "bad.bhr").write_bytes(b"not a tar")
    assert pm.main(["--job", str(tmp_path / "job"), "--recipe", str(tmp_path / "bad.bhr")]) == 2
    ev = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert ev["stage"] == "pipeline" and ev["status"] == "error" and "recipe:" in ev["msg"]
    with pytest.raises(SystemExit):
        pm.main(["--job", str(tmp_path / "job"), "--model", "x"])
