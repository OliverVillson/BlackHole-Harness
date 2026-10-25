"""The schema builder and the provider layer, with scripted fake Claude and OpenAI clients (no network)."""

import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from blackhole.agent import Builder, check, draft, estimate, providers

ROOT = Path(__file__).resolve().parents[1]
GOOD = (ROOT / "schemas" / "reap-purpose.toml").read_text().replace('id = "blackhole/reap-purpose"',
                                                                     'id = "me/sql-helper"')
BAD = GOOD.replace('kind = "score.reap"', 'kind = "score.magic"')


# ---------------------------------------------------------------- fake Claude

class FakeStream:
    def __init__(self, msg):
        self.msg = msg
        self.text_stream = [b.text for b in msg.content if b.type == "text"]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return self.msg


class FakeClaude:
    """Plays back a script of assistant turns and records every request."""

    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.beta = NS(messages=NS(stream=self._stream))

    def _stream(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        return FakeStream(self.script.pop(0))


def text(t):
    return NS(type="text", text=t)


def tool(name, inp, i):
    return NS(type="tool_use", id=f"tu_{i}", name=name, input=inp)


def turn(*blocks, stop="tool_use"):
    usage = NS(input_tokens=1000, output_tokens=200, cache_creation_input_tokens=0, cache_read_input_tokens=500)
    return NS(content=list(blocks), stop_reason=stop, usage=usage)


def claude(script):
    fake = FakeClaude(script)
    return providers.ClaudeProvider(api_key="test", client=fake), fake


def test_builder_repairs_until_accepted():
    prov, fake = claude([
        turn(text("Checking what fits."), tool("list_models", {"device": "24GB-VRAM"}, 1)),
        turn(tool("propose_schema", {"toml": BAD}, 2)),
        turn(tool("propose_schema", {"toml": GOOD}, 3)),
        turn(text("A SQL helper pruned from Qwen3-30B."), stop="end_turn"),
    ])
    d = draft("a SQL helper for my 24 GB card", provider=prov, device="24GB-VRAM", author="me")
    assert d.schema_toml == GOOD and not d.problems and d.drafts == 2
    assert d.summary == "A SQL helper pruned from Qwen3-30B."
    assert d.estimate["gpu_hours"] > 0 and d.estimate["base"] == "qwen3-30b-a3b"
    assert d.usage.turns == 4 and d.cost_usd == pytest.approx((4000 * 4 + 800 * 20 + 2000 * 0.2) / 1e6)

    first = fake.requests[0]
    assert first["model"] == "claude-opus-5-5" and first["fallbacks"] == "default"
    assert first["betas"] == ["server-side-fallback-2026-07-01"]
    assert all(t["eager_input_streaming"] for t in first["tools"])
    assert "score.reap" in first["system"] and "lab.script" not in first["system"]
    # The rejected draft's problems went back as an error result, in one user message.
    rejected = fake.requests[2]["messages"][-1]
    assert rejected["role"] == "user" and rejected["content"][0]["is_error"]
    assert "score.magic" in rejected["content"][0]["content"]
    # Assistant turns are passed back as the exact content objects (thinking blocks survive).
    assert fake.requests[1]["messages"][1]["content"] is fake.requests[1]["messages"][1]["content"]


def test_questions_stop_the_loop_for_the_tui():
    prov, fake = claude([turn(tool("ask", {"questions": ["Which PC will it run on?", "GGUF or MLX?"]}, 1))])
    d = draft("make me a model", provider=prov)
    assert d.questions == ["Which PC will it run on?", "GGUF or MLX?"] and not d.schema_toml
    assert len(fake.requests) == 1


def test_answers_are_passed_back_on_the_next_draft():
    prov, fake = claude([turn(tool("propose_schema", {"toml": GOOD}, 1)), turn(text("ok"), stop="end_turn")])
    draft("make me a model", provider=prov, answers={"Which PC?": "RTX 4060 laptop"})
    assert "RTX 4060 laptop" in fake.requests[0]["messages"][0]["content"]


def test_cli_style_answer_callback_keeps_going():
    prov, fake = claude([
        turn(tool("ask", {"questions": ["Which PC?"]}, 1)),
        turn(tool("propose_schema", {"toml": GOOD}, 2)),
        turn(text("done"), stop="end_turn"),
    ])
    d = Builder(prov, answer=lambda qs: ["M-series 16GB"]).run("a model")
    assert d.schema_toml and json.loads(fake.requests[1]["messages"][-1]["content"][0]["content"]) == {
        "Which PC?": "M-series 16GB"}


def test_gives_up_after_max_repairs():
    script = [turn(tool("propose_schema", {"toml": BAD}, i)) for i in range(5)]
    prov, _ = claude(script)
    d = Builder(prov, max_repairs=3).run("x")
    assert not d.schema_toml and any("score.magic" in p for p in d.problems) and d.drafts == 4


def test_refusal_and_truncated_tool_input():
    prov, _ = claude([turn(stop="refusal")])
    with pytest.raises(providers.ProviderError, match="declined"):
        draft("x", provider=prov)
    # eager streaming can hand over a cut-off input; it must come back as an error, not run
    prov, fake = claude([turn(tool("propose_schema", {}, 1)), turn(text("gave up"), stop="end_turn")])
    d = draft("x", provider=prov)
    assert "missing required field 'toml'" in fake.requests[1]["messages"][-1]["content"][0]["content"]
    assert d.problems


def test_check_reports_model_actionable_problems():
    assert check(GOOD) == []
    assert any("score.magic" in p for p in check(BAD))
    assert check("api = ")[0].startswith("TOML does not parse")
    lab = GOOD.replace('kind = "package.hf"', 'kind = "lab.script"\nfile = "x.py"')
    assert any("not available to the builder" in p for p in check(lab))
    no_out = GOOD.replace('outputs = ["safetensors-fp8", "safetensors-bf16"]', 'outputs = ["gguf"]')
    assert any("package.gguf" in p for p in check(no_out))


def test_estimate_uses_input_defaults():
    import tomllib
    est = estimate(tomllib.loads(GOOD))
    assert est["footprint_gb"] < 25 and est["frontier_usd"] == 0
    assert [r["kind"] for r in est["stages"]][:3] == ["data.teacher_gen", "score.reap", "prune.reap"]


# ---------------------------------------------------------------- fake OpenAI-compatible

class Msg(NS):
    def model_dump(self, exclude_none=True):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name, "arguments": c.function.arguments}}
                               for c in self.tool_calls]
        return d


def oa_turn(content=None, calls=(), finish="stop"):
    tcs = [NS(id=f"call_{i}", function=NS(name=n, arguments=a)) for i, (n, a) in enumerate(calls)]
    return NS(choices=[NS(message=Msg(content=content, tool_calls=tcs or None), finish_reason=finish)],
              usage=NS(prompt_tokens=900, completion_tokens=100))


def test_openai_compatible_provider_runs_the_same_builder():
    script = [oa_turn(calls=[("propose_schema", json.dumps({"toml": BAD}))]),
              oa_turn(calls=[("propose_schema", "{not json")]),
              oa_turn(calls=[("propose_schema", json.dumps({"toml": GOOD}))]),
              oa_turn("Here is your schema.")]
    seen = []

    def create(**kw):
        seen.append(json.loads(json.dumps(kw["messages"])))
        return script.pop(0)

    client = NS(chat=NS(completions=NS(create=create)), base_url="http://localhost:8000/v1")
    prov = providers.OpenAICompatProvider("local", model="served", client=client)
    d = draft("x", provider=prov)
    assert d.schema_toml == GOOD and d.cost_usd == 0.0
    assert seen[1][-1]["role"] == "tool" and seen[1][-1]["content"].startswith("ERROR: REJECTED")
    assert "INVALID_JSON" in seen[2][-1]["content"]


# ---------------------------------------------------------------- keys

def test_keys_file_is_private_and_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("BLACKHOLE_HOME", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    p = providers.save_key("claude", api_key="sk-ant-stored-0000000000")
    providers.save_key("nebius", api_key="neb-key", model="openai/gpt-oss-120b")
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600
    data = providers.load_keys()
    assert data["default"] == "claude" and data["providers"]["nebius"]["model"] == "openai/gpt-oss-120b"
    assert providers.get().client.api_key == "sk-ant-stored-0000000000"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-1111111111111")
    assert providers.get("claude").client.api_key == "sk-ant-env-1111111111111"
    neb = providers.get("nebius")
    assert neb.model == "openai/gpt-oss-120b" and "nebius" in str(neb.client.base_url)
    assert providers.mask("sk-ant-env-1111111111111") == "sk-ant-…1111"


def test_spend_cap_shrinks_max_tokens_then_stops():
    prov, fake = claude([turn(tool("list_models", {}, i)) for i in range(40)])
    with pytest.raises(providers.ProviderError, match="spend cap"):
        draft("x", provider=prov, budget_usd=0.10)
    asked = [r["max_tokens"] for r in fake.requests]
    assert asked and all(1024 <= m <= 0.10 / 20e-6 for m in asked)  # every reply could max out and still fit
    # worst case of every turn stayed under the cap
    assert len(fake.requests) * (1000 * 4 + 200 * 20 + 500 * 0.2) / 1e6 <= 0.10
    with pytest.raises(providers.ProviderError, match="prices"):
        providers.Budget(0.1, None)
