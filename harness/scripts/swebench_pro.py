"""Helpers for swebench-pro.sh (SWE-bench Pro V2 on one GPU VM with a local vLLM model).

Standalone on purpose: it runs in the Harbor venv, which has PyYAML but not the pipeline.

    python swebench_pro.py config SRC.yaml OUT.yaml --api-base URL [--max-tokens N]
    python swebench_pro.py ids TASKS_DIR {smoke|hard|all|N} OUT.txt
    python swebench_pro.py score JOB_DIR [--out results.json]

config: Scale's locked mini-swe-agent config with only the endpoint settings changed (local
        api_base, dummy key, no cost tracking, a per-reply token cap); prompts stay untouched.
ids:    the task list. smoke = 20 tasks spread over all 11 repos, hard = HARD-51, N = every
        (642/N)-th task, all = the 642 V2 tasks.
score:  {instance_id: true|false} plus a summary (resolved / total / errors).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def write_config(src: Path, out: Path, api_base: str, max_tokens: int) -> dict:
    import yaml

    cfg = yaml.safe_load(src.read_text())
    model = cfg.setdefault("model", {})
    kw = dict(model.get("model_kwargs") or {})
    kw.pop("reasoning_effort", None)
    kw.update(api_base=api_base, api_key="local", drop_params=True)
    if max_tokens:
        kw["max_tokens"] = max_tokens
    model["model_kwargs"] = kw
    model["cost_tracking"] = "ignore_errors"
    cfg.setdefault("agent", {})["cost_limit"] = 0
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg


def pick_ids(tasks: Path, which: str, hard_file: Path | None = None) -> list[str]:
    all_ids = sorted(p.name for p in tasks.iterdir() if (p / "task.toml").exists())
    if which == "all":
        return all_ids
    if which == "hard":
        hard_file = hard_file or tasks.parent / "hard51_ids.txt"
        return [x.strip() for x in hard_file.read_text().splitlines() if x.strip()]
    if which == "smoke":
        by_repo: dict[str, list[str]] = {}
        for i in all_ids:
            by_repo.setdefault(i.split("__")[0], []).append(i)
        picked, k = [], 0
        while len(picked) < min(20, len(all_ids)):
            for repo in sorted(by_repo):
                if k < len(by_repo[repo]) and len(picked) < 20:
                    picked.append(by_repo[repo][(k * 7) % len(by_repo[repo])])
            k += 1
        return sorted(set(picked))
    n = int(which)
    if n <= 0:
        raise SystemExit("task count must be positive")
    if n >= len(all_ids):
        return all_ids
    step = len(all_ids) / n
    return [all_ids[int(i * step)] for i in range(n)]


def score(job: Path) -> dict:
    results, errors = {}, []
    for rj in sorted(job.glob("*/result.json")):
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        name = r.get("task_name")
        if not name:
            continue
        iid = name.split("/")[-1]
        rewards = (r.get("verifier_result") or {}).get("rewards") or {}
        reward = rewards.get("reward", next(iter(rewards.values()), 0) if rewards else 0)
        results[iid] = bool(reward and float(reward) >= 1.0)
        if r.get("exception_info") and not results[iid]:
            errors.append(iid)
    total = len(results)
    resolved = sum(results.values())
    return {"job": str(job.resolve()), "resolved": resolved, "total": total,
            "resolved_rate": round(resolved / total, 4) if total else 0.0,
            "errors": len(errors), "error_ids": errors, "results": results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("config")
    c.add_argument("src", type=Path)
    c.add_argument("out", type=Path)
    c.add_argument("--api-base", required=True)
    c.add_argument("--max-tokens", type=int, default=16384)
    i = sub.add_parser("ids")
    i.add_argument("tasks", type=Path)
    i.add_argument("which")
    i.add_argument("out", type=Path)
    s = sub.add_parser("score")
    s.add_argument("job", type=Path)
    s.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "config":
        write_config(a.src, a.out, a.api_base, a.max_tokens)
        print(a.out)
    elif a.cmd == "ids":
        ids = pick_ids(a.tasks, a.which)
        a.out.write_text("".join(x + "\n" for x in ids))
        print(len(ids))
    else:
        s = score(a.job)
        if a.out:
            a.out.write_text(json.dumps(s["results"], indent=1, sort_keys=True) + "\n")
            a.out.with_name(a.out.stem + ".summary.json").write_text(json.dumps(
                {k: v for k, v in s.items() if k != "results"}, indent=1) + "\n")
        print(f"{s['resolved']}/{s['total']} resolved ({s['resolved_rate']:.1%}), {s['errors']} errored")
    return 0


if __name__ == "__main__":
    sys.exit(main())
