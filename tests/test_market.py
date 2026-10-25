"""Registry index, version resolution, badges, pull integrity, publish and the model table."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from blackhole.core import schema as sch
from blackhole.hw import Hardware
from blackhole.market import Registry, badge, build_index, lint, matches, publish, submit
from blackhole.market import models as mm

ROOT = Path(__file__).resolve().parents[1]


def _schema(version: str, title: str = "Prune a MoE for one purpose") -> str:
    text = (ROOT / "schemas" / "reap-purpose.toml").read_text()
    return text.replace('version = "1.0.0"', f'version = "{version}"').replace(
        'title = "Prune a MoE for one purpose"', f'title = "{title}"')


def _receipt(text: str, base: str, by: str, agree: float = 0.83, passed: bool = True, verified: str = "") -> dict:
    import tomllib
    s = sch.parse(tomllib.loads(text))
    return {"schema": s.ref, "schema_sha256": s.sha256, "base_model": base, "hardware": {"gpu": "B200", "count": 1},
            "metrics": {"check.top1_agreement": agree}, "outcome": "done", "gates_passed": passed,
            "output_sha256": f"{by}{base}"[:16], "runner_version": "0.1.0", "created_at": f"2026-10-2{len(by)}T00:00:00Z",
            "submitted_by": by, "verified_by": verified}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "registry"
    d = r / "schemas" / "blackhole" / "reap-purpose"
    d.mkdir(parents=True)
    v1, v11, v2pre = _schema("1.0.0"), _schema("1.1.0", "Prune a MoE, v1.1"), _schema("2.0.0-rc.1")
    for v, t in (("1.0.0", v1), ("1.1.0", v11), ("2.0.0-rc.1", v2pre)):
        (d / f"{v}.toml").write_text(t)
    rd = r / "receipts" / "blackhole" / "reap-purpose" / "1.1.0"
    rd.mkdir(parents=True)
    rs = [_receipt(v11, "Qwen/Qwen3-30B-A3B-Instruct-2507", by, verified="maintainer" if by == "ann" else "")
          for by in ("ann", "bob", "cy")]
    rs.append(_receipt(v11, "RecursiveSI/flint", "dee"))
    rs.append(_receipt(v1, "RecursiveSI/flint", "eve"))  # sha of 1.0.0: must not count for 1.1.0
    for i, rec in enumerate(rs):
        (rd / f"{i}.json").write_text(json.dumps(rec))
    (r / "index.json").write_text(json.dumps(build_index(r)))
    return r


def test_index_and_badges(repo):
    reg = Registry.open(str(repo))
    e = reg.entries["blackhole/reap-purpose"]
    assert e.title == "Prune a MoE, v1.1"  # listing metadata from the newest release, not the rc
    assert [v.version for v in e.versions] == ["2.0.0-rc.1", "1.1.0", "1.0.0"]
    v = e.latest
    assert v.version == "1.1.0"
    assert v.badges == {"Qwen/Qwen3-30B-A3B-Instruct-2507": "proven", "RecursiveSI/flint": "community"}
    assert v.badge == "proven" and len(v.receipts) == 4
    row = reg.search("moe purpose", device="24GB-VRAM")[0]
    assert row["badge"] == "proven" and row["fits_your_pc"] is True
    assert row["results"]["RecursiveSI/flint"] == {"check.top1_agreement": 0.83}
    assert reg.search("sql") == []


def test_badge_rules():
    import tomllib  # noqa: F401
    t = _schema("1.0.0")
    base = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    assert badge([]) == "draft"
    assert badge([_receipt(t, base, "a", passed=False)]) == "draft"
    assert badge([_receipt(t, base, "a")]) == "community"
    assert badge([_receipt(t, base, "a", verified="m")]) == "verified"
    three = [_receipt(t, base, n) for n in ("a", "bb", "ccc")]
    assert badge(three) == "proven"
    assert badge(three, experimental=True) == "community"
    disagree = three[:2] + [_receipt(t, base, "ccc", agree=0.70)]
    assert badge(disagree) == "community"


def test_version_constraints():
    assert matches("1.4.2", "^1.2") and not matches("2.0.0", "^1.2") and not matches("1.1.9", "^1.2")
    assert matches("0.3.5", "^0.3") and not matches("0.4.0", "^0.3")
    assert matches("1.2.9", "~1.2.3") and not matches("1.3.0", "~1.2.3")
    assert matches("1.9.0", "1") and matches("1.2.7", "1.2") and not matches("1.3.0", "1.2")
    assert not matches("2.0.0-rc.1", "") and matches("2.0.0-rc.1", "2.0.0-rc.1")
    assert matches("3.0.0", ">=1.0")


def test_pull_checks_sha(repo, tmp_path, monkeypatch):
    monkeypatch.setenv("BLACKHOLE_HOME", str(tmp_path / "home"))
    reg = Registry.open(str(repo))
    p = reg.pull("blackhole/reap-purpose@^1.0")
    assert p.name == "1.1.0.toml" and p.read_text() == _schema("1.1.0", "Prune a MoE, v1.1")
    assert reg.get("blackhole/reap-purpose@2.0.0-rc.1")[1].version == "2.0.0-rc.1"
    (repo / "schemas" / "blackhole" / "reap-purpose" / "1.1.0.toml").write_text(_schema("1.1.0", "tampered"))
    with pytest.raises(ValueError, match="sha256 mismatch"):
        reg.pull("blackhole/reap-purpose@1.1.0")
    with pytest.raises(KeyError):
        reg.get("blackhole/reap-purpose@^3")


def test_build_index_rejects_bad_files(tmp_path):
    r = tmp_path / "r"
    d = r / "schemas" / "someone" / "x"
    d.mkdir(parents=True)
    (d / "1.0.0.toml").write_text(_schema("1.0.0"))  # id says blackhole/reap-purpose
    with pytest.raises(ValueError, match="must live at"):
        build_index(r)
    (d / "1.0.0.toml").write_text(_schema("1.0.0").replace('kind = "score.reap"', 'kind = "score.magic"'))
    with pytest.raises(ValueError, match="score.magic"):
        build_index(r)


def test_lint():
    import tomllib
    raw = tomllib.loads(_schema("1.0.0"))
    assert lint(raw, publish=True) == []
    raw["inputs"]["purpose"]["advanced"] = True
    raw["stage"].append({"kind": "lab.script", "file": "x.py"})
    raw["stage"].append({"kind": "data.dataset", "source": "hf:x", "license": "CC-BY-NC-4.0"})
    raw["data_policy"] = {"licenses_allowed": ["MIT"]}
    raw["outputs"] = ["gguf"]
    problems = " ".join(lint(raw, publish=True))
    for needle in ("cannot be advanced", "code_sha256", "CC-BY-NC-4.0", "package.gguf"):
        assert needle in problems


@pytest.mark.skipif(not shutil.which("git"), reason="needs git")
def test_publish_and_submit(repo, tmp_path):
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "-C", str(repo), "config", k, v], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "init"], check=True)
    reg = Registry.open(str(repo))
    mine = tmp_path / "mine.toml"
    mine.write_text(_schema("1.1.0"))
    with pytest.raises(ValueError, match="already published"):
        publish(mine, repo, reg)
    mine.write_text(_schema("1.2.0"))
    out = publish(mine, repo, reg)
    assert out["branch"] == "publish/blackhole-reap-purpose-1.2.0" and "gh pr create" in out["pr_command"]
    assert (repo / "schemas/blackhole/reap-purpose/1.2.0.toml").exists()
    rec = tmp_path / "receipt.json"
    rec.write_text(json.dumps(_receipt(_schema("1.2.0"), "Qwen/Qwen3-30B-A3B-Instruct-2507", "zed")))
    sub = submit(rec, repo, "zed")
    assert Path(sub["file"]).exists() and json.loads(Path(sub["file"]).read_text())["submitted_by"] == "zed"


def test_model_table_fits():
    b200 = Hardware(gpu_name="NVIDIA B200", gpu_count=1, gpu_mem_gb=180, disk_free_gb=2000)
    rows = {r["model"]: r for r in mm.table(b200, "8GB-VRAM")}
    assert rows["qwen3-30b-a3b"]["runs_here"] and rows["qwen3-30b-a3b"]["fits_target"]
    assert not rows["kimi-k2"]["runs_here"] and not rows["kimi-k2"]["fits_target"]
    assert rows["granite-4.0-h-tiny"]["recipe"].startswith("prune 0%")
    q = mm.find("qwen3-30b-a3b")
    assert 28 < q.expert_b < 30  # routed experts are ~95% of a 30B-A3B
    assert mm.estimate_gb(q, 0.5, "Q4_K_M")["total_gb"] < mm.estimate_gb(q, 0.0, "Q4_K_M")["total_gb"]


def test_any_model_via_detect(tmp_path):
    cfg = {"architectures": ["Qwen3MoeForCausalLM"], "model_type": "qwen3_moe", "hidden_size": 2048,
           "num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
           "intermediate_size": 6144, "moe_intermediate_size": 768, "num_experts": 128, "num_experts_per_tok": 8,
           "vocab_size": 151936, "max_position_embeddings": 32768}
    d = tmp_path / "my-moe"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    m = mm.lookup(str(d))
    assert m.arch == "moe" and m.experts == 128 and 25 < m.total_b < 35 and m.expert_b > 0.8 * m.total_b
    assert m.kv_bytes_per_token == 2 * 48 * 4 * 128 * 2
    assert mm.best_fit(m, "24GB-VRAM") is not None
    assert mm.lookup(str(tmp_path / "nothing-here"), detect=True) is None
