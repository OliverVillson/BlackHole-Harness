import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from blackhole.core import ops, runner  # noqa: E402
from blackhole.core import schema as sch  # noqa: E402
from blackhole.core.job import Job  # noqa: E402

SCHEMA = """
api = "blackhole/v1"
id = "test/shrink"
version = "1.0.0"

[inputs.ratio]
type = "float"
default = 0.5
min = 0.1
max = 0.9

[inputs.min_kept]
type = "float"
default = 0.2

[[stage]]
kind = "test.gen"
n = 4

[[stage]]
kind = "test.shrink"
ratio = "${inputs.ratio}"
  [[stage.gates]]
  metric = "kept"
  op = ">="
  value = "${inputs.min_kept}"

[[stage]]
id = "again"
kind = "test.shrink"
ratio = 0.0

[[stage]]
kind = "test.upload"
always = true

[eval]
  [[eval.gates]]
  metric = "test2.kept"
  op = ">="
  value = 0.3
"""


DENSE = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 4096, "num_hidden_layers": 32,
         "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336, "vocab_size": 128256,
         "max_position_embeddings": 131072, "torch_dtype": "bfloat16"}
QWEN3_MOE = {"model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"], "hidden_size": 2048,
             "num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
             "intermediate_size": 6144, "moe_intermediate_size": 768, "num_experts": 128, "num_experts_per_tok": 8,
             "vocab_size": 151936, "max_position_embeddings": 262144}
QWEN3_NEXT = {**QWEN3_MOE, "model_type": "qwen3_next", "num_experts": 512, "num_experts_per_tok": 10,
              "full_attention_interval": 4, "shared_expert_intermediate_size": 512}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("BLACKHOLE_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("BLACKHOLE_OPS", "tests.fake_ops")
    monkeypatch.setenv("PYTHONPATH", str(ROOT))
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.chdir(ROOT)
    ops._loaded = False
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "config.json").write_text(json.dumps(DENSE))
    (tmp_path / "s.toml").write_text(SCHEMA)
    return tmp_path


def events(capsys):
    return [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]


def test_examples_parse_and_validate():
    for f in (ROOT / "schemas").glob("*.toml"):
        assert ops.validate(sch.load(f)) == [], f.name


def test_stage_ids_and_refs():
    s = sch.parse({"api": "blackhole/v1", "id": "a/b", "version": "1.0.0", "stage": [
        {"kind": "data.dataset"}, {"kind": "data.decontam"}, {"kind": "prune.reap", "ratio": 0.5}]})
    assert [x.id for x in s.stages] == ["data", "data2", "prune"]
    with pytest.raises(sch.SchemaError):
        sch.parse({"api": "blackhole/v1", "id": "a/b", "version": "1.0.0",
                   "stage": [{"kind": "prune.reap", "saliency": "${score.saliency}"}]})


def test_run_cache_and_rerun(env, capsys):
    job = Job.create(env / "s.toml", str(env / "base"), {"ratio": 0.6}, env / "job")
    assert runner.run(job) == 0
    ev = events(capsys)
    assert ev[-1]["status"] == "done" and ev[-1]["outcome"] == "done"
    model = Path(ev[-1]["model"])
    w = json.loads((model / "weights.json").read_text())
    assert w["ratio"] == 0.0 and w["rows"] == 4 and w["from"].endswith("test2/out/model")   # chained
    lineage = json.loads((model / "blackhole-lineage.json").read_text())
    assert [s["stage"] for s in lineage["steps"]] == ["test2", "again"]
    receipt = job.read("receipt.json")
    assert receipt["gates_passed"] and receipt["schema"] == "test/shrink@1.0.0"
    assert (job.root / "uploaded").exists()

    assert runner.run(job) == 0                              # everything cached
    assert all(e["status"] == "skipped" for e in events(capsys) if e["stage"] in ("test", "test2", "again"))

    other = Job.create(env / "s.toml", str(env / "base"), {"ratio": 0.4}, env / "job2")
    assert runner.run(other) == 0                           # another job: gen reused, shrink redone
    st = {e["stage"]: e for e in events(capsys) if e["status"] in ("done", "skipped")}
    assert st["test"]["status"] == "skipped" and st["test2"]["status"] == "done"


def test_gate_stops_but_upload_runs(env, capsys):
    job = Job.create(env / "s.toml", str(env / "base"), {"ratio": 0.9, "min_kept": 0.5}, env / "job")
    assert runner.run(job) == 0
    ev = events(capsys)
    assert ev[-1]["outcome"] == "stopped" and ev[-1]["model"] is None
    assert any(e["stage"] == "again" and e["status"] == "skipped" for e in ev)
    assert (job.root / "uploaded").exists()
    assert job.read("receipt.json")["gates_passed"] is False


def test_bad_input_refused(env):
    with pytest.raises(sch.SchemaError):
        Job.create(env / "s.toml", str(env / "base"), {"ratio": 2.0}, env / "job")


def test_detect_kinds():
    from blackhole.hw.model import from_config
    d = from_config("meta-llama/Llama-3.1-8B-Instruct", DENSE)
    assert d.kinds == ["dense"] and 7.5 < d.params_b < 8.5 and d.experts == 0
    m = from_config("Qwen/Qwen3-30B-A3B-Instruct-2507", QWEN3_MOE)
    assert m.kinds == ["moe"] and m.experts == 128 and m.top_k == 8 and 28 < m.params_b < 33 and 2.5 < m.active_b < 4
    h = from_config("Qwen/Qwen3-Next", QWEN3_NEXT)
    assert h.kinds == ["hybrid", "moe"] and h.attention_layers == 12 and h.recurrent_layers == 36 and h.shared_experts == 1


def test_wrong_model_refused_early(env):
    schema = ROOT / "schemas" / "reap-purpose.toml"
    with pytest.raises(sch.SchemaError, match="targets moe"):
        Job.create(schema, str(env / "base"), {"purpose": "SQL"}, env / "job")


def test_detect_feeds_base_refs(env, capsys):
    job = Job.create(env / "s.toml", str(env / "base"), {}, env / "job")
    assert runner.run(job) == 0
    ev = events(capsys)
    assert any(e["stage"] == "detect" and e["status"] == "done" for e in ev)
    assert job.metrics("detect")["kinds"] == ["dense"]


def test_moefy_turns_dense_into_moe_for_later_stages():
    from blackhole.hw.model import applicable, from_config
    s = sch.load(ROOT / "schemas" / "lab" / "dense-moefy-reap.toml")
    assert ops.validate(s) == []
    assert applicable(from_config("meta-llama/Llama-3.1-8B-Instruct", DENSE), s) == []
    assert any("targets dense" in w for w in applicable(from_config("q", QWEN3_MOE), s))


def test_dense_default_fits_dense_not_moe():
    from blackhole.hw.model import applicable, from_config
    s = sch.load(ROOT / "schemas" / "dense-prune-int4.toml")
    assert ops.validate(s) == []
    assert applicable(from_config("Qwen/Qwen3-8B", DENSE), s) == []
    assert applicable(from_config("q", QWEN3_MOE), s)


def test_fit_quantises_harder_before_pruning(tmp_path):
    from blackhole.ops.plan import Fit

    def plan(max_gb):
        d = tmp_path / str(max_gb)
        d.mkdir(exist_ok=True)
        ctx = ops.Context(job=None, stage="fit", params=Fit.Params(params_b=8.2, max_gb=max_gb), hw={}, stage_dir=d)
        Fit().run(ctx)
        return json.loads((d / "metrics.json").read_text())
    assert plan(0)["bits"] == "Q4_K_M" and plan(0)["prune"] == 0        # no target: GPTQ 4-bit
    assert plan(4.5)["bits"] == "Q3_K_M" and plan(4.5)["prune"] == 0    # 3-bit before any pruning
    assert plan(3.7)["prune"] == 0.125 and plan(3.3)["prune"] == 0.25
    assert plan(2.0)["fits"] is False                                   # stop; never layer drop


def test_fit_moe_ladder_and_50_percent_cap(tmp_path):
    from blackhole.ops.plan import Fit
    ladder = [{"bits": "Q4_K_M", "prune": 0.25}, {"bits": "Q3_K_M", "prune": 0.25}, {"bits": "Q3_K_M", "prune": 0.375}]

    def plan(max_gb, lad=ladder):
        d = tmp_path / f"m{max_gb}"
        d.mkdir(exist_ok=True)
        ctx = ops.Context(job=None, stage="fit", stage_dir=d, hw={},
                          params=Fit.Params(params_b=30.5, prunable_b=29.0, max_gb=max_gb, ladder=lad))
        Fit().run(ctx)
        return json.loads((d / "metrics.json").read_text())
    assert plan(0)["bits"] == "Q4_K_M" and plan(0)["prune"] == 0.25
    assert plan(12)["bits"] == "Q3_K_M" and plan(12)["prune"] == 0.25    # 3 bits before pruning more
    assert plan(10)["prune"] == 0.375 and plan(5)["fits"] is False
    with pytest.raises(ValueError):
        plan(0, [{"bits": "Q4_K_M", "prune": 0.75}])


def test_reaq_floor_and_codebook_guards():
    import pytest
    from blackhole.ops.quant import ExpertCodebook, Reaq
    with pytest.raises(ValueError):
        Reaq.Params(min_bits="Q2_K")
    assert Reaq.Params(target="codebook").levels == [8, 16]
    with pytest.raises(ValueError):
        ExpertCodebook.Params(export="packed")
    with pytest.raises(ValueError):
        ExpertCodebook.Params(levels=32)
