"""Recipe bundles (.bhr) for the pipeline: `python -m blackhole_pipeline --job DIR --recipe X.bhr`.

A .bhr is a gzip tar written by the BlackHole-Dev app (the private blackhole_dev package):

    manifest.json     format "bhr/1", id, version, base_model, target, runner, stages, inputs,
                      files {path: sha256}, functions, blackhole_dev_version, needs
    recipe.py         the recipe as the developer wrote it
    recipe.json       {"raw": <the blackhole/v1 schema dict>, "meta": {...}}, compiled from recipe.py
    functions/*.py    the developer's own function files the recipe calls

This module reads only manifest.json and recipe.json, so the open pipeline runs a recipe
without blackhole_dev. Two kinds of recipe:

  pipeline  stages `pipeline.<stage>` (this pipeline's stages: reap, heal, gptq, reaq, mlx, qheal,
            bench, quant, package) and `job.fn` (a named job function, e.g. a forked distil, run as
            `python -m <module> --job DIR --stage ID` with <module> from the `blackhole.jobfn`
            entry point group), in recipe order. detect, plan, weights and calibrate always run
            first; pipeline.reap reports score and prune, then eval.
  core      catalog ops (blackhole.ops, incl. blackhole_dev's model.edit through the `blackhole.ops`
            entry point), run by blackhole.core.runner.

The job's config.json is written from the bundle when missing, so the TUI, `--from STAGE` and
reruns work as they do for TUI jobs.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tarfile
from importlib.metadata import entry_points
from pathlib import Path

from blackhole_pipeline import PROCESSES, STAGES

FORMAT = "bhr/1"
PIPELINE_STEPS = ("reap", "heal", "gptq", "reaq", "mlx", "qheal", "bench", "quant", "package")
ALWAYS_FIRST = ["detect", "plan", "weights", "calibrate"]
REF = re.compile(r"\$\{([A-Za-z_][\w.-]*)\}")


class RecipeBundleError(ValueError):
    pass


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_bundle(path: str | Path) -> tuple[dict, dict[str, bytes]]:
    """(manifest, {member: bytes}) after checking every file against the manifest's sha256."""
    try:
        with tarfile.open(path, "r:gz") as tf:
            files = {}
            for m in tf.getmembers():
                if not m.isfile():
                    continue
                name = m.name.removeprefix("./")
                if name.startswith("/") or ".." in Path(name).parts:
                    raise RecipeBundleError(f"{path}: unsafe member {m.name!r}")
                files[name] = tf.extractfile(m).read()
    except (tarfile.TarError, OSError) as e:
        raise RecipeBundleError(f"{path}: not a readable .bhr ({e})") from None
    if "manifest.json" not in files:
        raise RecipeBundleError(f"{path}: no manifest.json")
    man = json.loads(files["manifest.json"])
    if man.get("format") != FORMAT:
        raise RecipeBundleError(f"{path}: format {man.get('format')!r}, this pipeline reads {FORMAT}")
    listed = man.get("files") or {}
    for name, want in listed.items():
        if name not in files:
            raise RecipeBundleError(f"{path}: {name} is in the manifest but not in the bundle")
        got = sha256_bytes(files[name])
        if got != want:
            raise RecipeBundleError(f"{path}: {name} sha256 {got[:12]} does not match the manifest ({want[:12]})")
    extra = set(files) - set(listed) - {"manifest.json"}
    if extra:
        raise RecipeBundleError(f"{path}: files not pinned by the manifest: {sorted(extra)}")
    if "recipe.json" not in files:
        raise RecipeBundleError(f"{path}: no recipe.json")
    return man, files


def extract(path: str | Path, dest: Path) -> tuple[dict, dict]:
    """Check and unpack a bundle into dest (replacing it). Returns (manifest, recipe.json)."""
    man, files = read_bundle(path)
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    for name, data in files.items():
        p = tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    return man, json.loads(files["recipe.json"])


def write_bundle(path: str | Path, files: dict[str, bytes], manifest: dict) -> str:
    """Write a .bhr (manifest gets files' sha256s). Reproducible bytes: fixed mtimes and order.
    Returns the bundle's sha256. blackhole_dev builds bundles with this too."""
    manifest = {**manifest, "format": FORMAT, "files": {n: sha256_bytes(b) for n, b in sorted(files.items())}}
    members = {"manifest.json": (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(), **files}
    buf = io.BytesIO()
    import gzip
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as tf:
            for name in sorted(members):
                data = members[name]
                ti = tarfile.TarInfo(name)
                ti.size, ti.mtime, ti.mode = len(data), 0, 0o644
                tf.addfile(ti, io.BytesIO(data))
    Path(path).write_bytes(buf.getvalue())
    return sha256_bytes(buf.getvalue())


# --------------------------------------------------------------------------- recipe -> job


def _op(stage: dict) -> str:
    return str(stage.get("kind", "")).split("@")[0]


def runner_of(raw: dict) -> str:
    fams = set()
    for st in raw.get("stage") or []:
        op = _op(st)
        fams.add("pipeline" if op.startswith("pipeline.") or op == "job.fn" else "core")
    if not fams or fams == {"pipeline"}:
        return "pipeline"
    return "core" if fams == {"core"} else "mixed"


def stage_id(stage: dict) -> str:
    op = _op(stage)
    if op.startswith("pipeline."):
        return op.split(".", 1)[1]
    return stage.get("id") or op.split(".")[0]


def problems(raw: dict) -> list[str]:
    """What stops a pipeline recipe from running here (empty = runs)."""
    out = []
    seen = set()
    for i, st in enumerate(raw.get("stage") or []):
        op = _op(st)
        if op.startswith("pipeline."):
            name = op.split(".", 1)[1]
            if name not in PIPELINE_STEPS:
                out.append(f"stage {i + 1}: {op} is not a pipeline stage (one of {', '.join(PIPELINE_STEPS)})")
            if name in seen:
                out.append(f"stage {i + 1}: {op} appears twice; the pipeline runs each stage once")
            seen.add(name)
        elif op == "job.fn":
            if not st.get("fn"):
                out.append(f"stage {i + 1}: job.fn needs fn = \"<function name>\"")
            sid = stage_id(st)
            if sid in STAGES or sid in seen:
                out.append(f"stage {i + 1}: id {sid!r} is taken by a pipeline stage; give the step another id")
            seen.add(sid)
        else:
            out.append(f"stage {i + 1}: {op} is a catalog op; a pipeline recipe has only pipeline.* and job.fn stages")
    return out


def stage_list(raw: dict) -> list[str]:
    """Every stage name a pipeline recipe reports, in run order."""
    out = list(ALWAYS_FIRST)
    for st in raw.get("stage") or []:
        op = _op(st)
        if op == "pipeline.reap":
            out += ["score", "prune", "eval"]
        else:
            out.append(stage_id(st))
    return out


def processes(stages: list[str]) -> list[tuple[str, list[str], str]]:
    """The (process, stages, who) list run() walks, in the recipe's order. Stages that are no
    pipeline process are job functions: who = "fn"."""
    by_stage = {s: p for p in PROCESSES for s in p[1]}
    out, have = [], set()
    for s in stages:
        if s in by_stage:
            p = by_stage[s]
            if p[0] not in have:
                out.append(p)
                have.add(p[0])
        elif s not in have:
            out.append((s, [s], "fn"))
            have.add(s)
    return out


def _resolve(value, scope: dict):
    def lookup(ref: str):
        cur = scope
        for part in ref.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise RecipeBundleError(f"${{{ref}}} cannot be resolved in a pipeline recipe "
                                        "(inputs.<name>, base.model and base.id can)")
            cur = cur[part]
        return cur

    if isinstance(value, str):
        m = REF.fullmatch(value.strip())
        if m:
            return lookup(m.group(1))
        return REF.sub(lambda m: str(lookup(m.group(1))), value)
    if isinstance(value, dict):
        return {k: _resolve(v, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, scope) for v in value]
    return value


def input_values(raw: dict, given: dict) -> dict:
    from blackhole.core import schema as sch
    return sch.input_values(sch.parse(raw), given)


def job_config(rec: dict, manifest: dict, model: str | None = None, inputs: dict | None = None,
               bundle_sha: str = "") -> dict:
    """The TUI-style config.json for a pipeline recipe: model, schema, steps, stage_list."""
    raw, meta = rec["raw"], rec.get("meta") or {}
    probs = problems(raw)
    if probs:
        raise RecipeBundleError("; ".join(probs))
    model = model or manifest.get("base_model") or raw.get("base")
    if not model:
        raise RecipeBundleError("no base model: give --model, or set base= in the recipe")
    given = {**(manifest.get("inputs") or {}), **(inputs or {})}
    vals = input_values(raw, given)
    scope = {"inputs": vals, "base": {"model": model, "id": model}}
    steps = []
    for st in raw.get("stage") or []:
        params = {k: v for k, v in st.items() if k not in ("id", "kind", "needs", "gates", "when", "always")}
        params = _resolve(params, scope)
        op = _op(st)
        if op == "job.fn":
            steps.append({"kind": "fn", "id": stage_id(st), "fn": params.pop("fn"), "params": params})
        else:
            steps.append({"kind": op.split(".", 1)[1], "params": params})
    purpose = next((s["params"]["purpose"] for s in steps if "purpose" in s["params"]), None)
    if purpose is None:
        purpose = vals.get("purpose") or raw.get("purpose") or ""
    return {"model": model, "schema": raw["id"], "schema_version": raw.get("version"), "recipe": bundle_sha,
            "target": manifest.get("target") or meta.get("target") or "", "purpose": purpose,
            "steps": steps, "stage_list": stage_list(raw), "inputs": vals, "runner": "pipeline"}


def jobfn_module() -> str:
    """The module that runs job functions (the `blackhole.jobfn` entry point; blackhole_dev ships one)."""
    eps = list(entry_points(group="blackhole.jobfn"))
    if not eps:
        raise RecipeBundleError("this recipe calls a job function, and no package provides the blackhole.jobfn "
                                "entry point here (install blackhole-dev in this venv)")
    return eps[0].value.split(":")[0]


def prepare(job_root: Path, bundle: str | Path, model: str | None = None, inputs: dict | None = None) -> dict:
    """Check and unpack the bundle into <job>/recipe/ and write <job>/config.json if it is missing.
    Returns the job config (the existing one on a rerun)."""
    job_root.mkdir(parents=True, exist_ok=True)
    man, rec = extract(bundle, job_root / "recipe")
    cfg_p = job_root / "config.json"
    if cfg_p.exists():
        cfg = json.loads(cfg_p.read_text())
        if cfg.get("schema") != rec["raw"]["id"]:
            raise RecipeBundleError(f"{cfg_p} belongs to {cfg.get('schema')!r}, the bundle is {rec['raw']['id']!r}; "
                                    "use a new job directory")
        return cfg
    sha = sha256_bytes(Path(bundle).read_bytes())
    runner = man.get("runner") or runner_of(rec["raw"])
    if runner == "pipeline":
        cfg = job_config(rec, man, model, inputs, sha)
    elif runner == "core":
        model = model or man.get("base_model") or rec["raw"].get("base")
        if not model:
            raise RecipeBundleError("no base model: give --model, or set base= in the recipe")
        cfg = {"model": model, "schema": rec["raw"]["id"], "schema_version": rec["raw"].get("version"),
               "recipe": sha, "target": man.get("target") or "", "runner": "core",
               "inputs": {**(man.get("inputs") or {}), **(inputs or {})}}
    else:
        raise RecipeBundleError("this recipe mixes pipeline.* / job.fn stages with catalog ops, which no runner "
                                "takes yet; split it into two recipes")
    (job_root / "recipe" / "schema.json").write_text(json.dumps(rec["raw"], indent=2) + "\n")
    cfg_p.write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg


def run_core(job_root: Path, cfg: dict, start: str | None = None) -> int:
    """A catalog-op recipe: blackhole.core.runner in <job>/core (model.edit comes from blackhole_dev
    through the blackhole.ops entry point when it is installed)."""
    from blackhole.core import runner
    from blackhole.core.job import Job
    core = job_root / "core"
    if (core / "job.json").exists():
        job = Job(core)
    else:
        job = Job.create(job_root / "recipe" / "schema.json", cfg["model"], cfg.get("inputs") or {}, root=core)
        # functions = ["functions/x.py"] in the recipe are relative to the bundle; the op reads them from the job
        link = core / "functions"
        if (job_root / "recipe" / "functions").is_dir() and not link.exists():
            link.symlink_to(job_root / "recipe" / "functions")
        if not (core / "recipe.py").exists() and (job_root / "recipe" / "recipe.py").exists():
            (core / "recipe.py").symlink_to(job_root / "recipe" / "recipe.py")
    return runner.run(job, start)
