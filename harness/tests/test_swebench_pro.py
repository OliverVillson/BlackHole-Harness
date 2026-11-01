"""swebench_pro.py: locked config with only the endpoint changed, task lists, Harbor job scoring."""
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("swebench_pro", SCRIPTS / "swebench_pro.py")
sbp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sbp)


def test_config_keeps_prompts_and_points_at_local_endpoint(tmp_path):
    src = tmp_path / "mini.yaml"
    src.write_text(yaml.safe_dump({
        "agent": {"system_template": "SYS", "instance_template": "{{task}}", "step_limit": 0, "cost_limit": 3.0},
        "model": {"model_class": "litellm_textbased", "model_kwargs": {"drop_params": True, "reasoning_effort": "high"}},
    }))
    out = tmp_path / "out.yaml"
    sbp.write_config(src, out, "http://127.0.0.1:8000/v1", 16384)
    cfg = yaml.safe_load(out.read_text())
    assert cfg["agent"]["system_template"] == "SYS" and cfg["agent"]["instance_template"] == "{{task}}"
    assert cfg["agent"]["cost_limit"] == 0 and cfg["model"]["cost_tracking"] == "ignore_errors"
    kw = cfg["model"]["model_kwargs"]
    assert kw["api_base"] == "http://127.0.0.1:8000/v1" and kw["max_tokens"] == 16384
    assert "reasoning_effort" not in kw and cfg["model"]["model_class"] == "litellm_textbased"


def _tasks(root: Path, names):
    for n in names:
        (root / n).mkdir(parents=True)
        (root / n / "task.toml").write_text("")


def test_ids_smoke_spreads_over_repos_and_counts(tmp_path):
    tasks = tmp_path / "v2" / "tasks"
    names = [f"instance_{r}__{r}-{i:03d}" for r in ("a", "b", "c") for i in range(30)]
    _tasks(tasks, names)
    (tmp_path / "v2" / "hard51_ids.txt").write_text("instance_a__a-001\ninstance_b__b-002\n")
    smoke = sbp.pick_ids(tasks, "smoke")
    assert len(smoke) == 20 and {s.split("__")[0] for s in smoke} == {"instance_a", "instance_b", "instance_c"}
    assert sbp.pick_ids(tasks, "hard") == ["instance_a__a-001", "instance_b__b-002"]
    assert len(sbp.pick_ids(tasks, "all")) == 90
    ten = sbp.pick_ids(tasks, "10")
    assert len(ten) == len(set(ten)) == 10


def test_score_reads_harbor_trials(tmp_path):
    job = tmp_path / "job"
    for i, (name, reward, exc) in enumerate([("t1", 1.0, None), ("t2", 0.0, None), ("t3", None, {"exception_type": "X"})]):
        d = job / f"trial{i}"
        d.mkdir(parents=True)
        d.joinpath("result.json").write_text(json.dumps({
            "task_name": f"swebench-pro/{name}",
            "verifier_result": None if reward is None else {"rewards": {"reward": reward}},
            "exception_info": exc}))
    (job / "result.json").write_text("{}")
    s = sbp.score(job)
    assert s["results"] == {"t1": True, "t2": False, "t3": False}
    assert (s["resolved"], s["total"], s["errors"]) == (1, 3, 1)


def test_shell_script_parses():
    subprocess.run(["bash", "-n", str(SCRIPTS / "swebench-pro.sh")], check=True)
