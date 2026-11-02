"""A blackhole-hex folder check (check_folder.py), the converter's refusals, and the patch that lets
vLLM's Qwen3.8-Flash-Next n-gram embedding load under the blackhole-hex quantization method."""

import json
import os
import sys
import types

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("triton")

from blackhole_runtime.hex_ref import random_parts  # noqa: E402


def write_folder(d, E=6, H=64, I=32, hot=3, drop=None):
    from safetensors.torch import save_file
    t = {}
    for li in range(2):
        pre = f"model.language_model.layers.{li}.mlp.experts"
        for L, ids in ((16, list(range(hot))), (8, list(range(hot, E)))):
            if drop is not None and drop in ids:
                ids = [i for i in ids if i != drop]
            bits = L.bit_length() - 1
            t[f"{pre}.tier{L}.experts"] = torch.tensor(ids, dtype=torch.int32)
            for proj, (o, i) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
                for k, v in random_parts(len(ids), o, i, bits, seed=li * 10 + L).items():
                    t[f"{pre}.tier{L}.{proj}.{k}"] = v
    save_file(t, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps({"model_type": "qwen4_exp",
                                               "text_config": {"num_experts": E, "num_experts_per_tok": 2},
                                               "quantization_config": {"quant_method": "blackhole-hex"}}))


def test_check_folder_ok_and_missing_expert(tmp_path):
    from blackhole_runtime.check_folder import check
    write_folder(tmp_path)
    r = check(tmp_path, "cpu")
    assert r["ok"] and r["layers"] == 2 and r["checked"] == 2, r
    bad = tmp_path / "bad"
    bad.mkdir()
    write_folder(bad, drop=4)
    r = check(bad, "cpu")
    assert not r["ok"] and "expected each of 0..5 once" in r["problems"][0]


def test_convert_refuses_hex_folders_and_other_models(tmp_path):
    from blackhole_runtime.convert import convert
    write_folder(tmp_path)
    with pytest.raises(SystemExit, match="already a blackhole-hex folder"):
        convert(tmp_path, tmp_path / "out")
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mixtral"}))
    with pytest.raises(SystemExit, match="hexvllm"):
        convert(tmp_path, tmp_path / "out")


def test_qwen4_exp_ple_embedding_patch(monkeypatch):
    """vLLM's Qwen4ExpPLEEmbeddingMethod.from_quant_config raises for unknown quant configs; under
    blackhole-hex it must be asked as if the model were unquantized (other configs untouched)."""
    from blackhole_runtime import vllm_hex
    calls = []

    class Method:
        @staticmethod
        def from_quant_config(quant_config, prefix, embedding_dtype=None):
            calls.append(quant_config)
            if quant_config is not None and quant_config.get_name() != "fp8":
                raise NotImplementedError("unsupported")
            return "plain" if quant_config is None else "fp8"

    ng = types.ModuleType("vllm.models.qwen4_exp.common.ngram_embedding")
    ng.Qwen4ExpPLEEmbeddingMethod = Method
    for name in ("vllm", "vllm.models", "vllm.models.qwen4_exp", "vllm.models.qwen4_exp.common"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["vllm.models.qwen4_exp.common"].ngram_embedding = ng
    monkeypatch.setitem(sys.modules, ng.__name__, ng)

    class Q:
        def __init__(self, n):
            self.n = n

        def get_name(self):
            return self.n

    assert vllm_hex.plain_ple_embedding() and vllm_hex.plain_ple_embedding()   # idempotent
    assert Method.from_quant_config(Q(vllm_hex.NAME), "model.layers.1.ple") == "plain"
    assert Method.from_quant_config(Q("online"), "model.layers.1.ple") == "plain"   # --quantization fp8_per_block
    assert Method.from_quant_config(Q("fp8"), "x") == "fp8"
    with pytest.raises(NotImplementedError):
        Method.from_quant_config(Q("awq"), "x")
