"""Frontier-model providers: Claude (Anthropic SDK) and anything OpenAI-compatible.

Every provider does one thing: run a tool-use loop to completion.

    provider.run(system, prompt, tools, on_text=print) -> RunResult

Callers never see provider message formats. Each provider keeps its own history,
so Claude's thinking blocks go back untouched and OpenAI tool_call ids line up,
without a lowest-common-denominator message type in between.

Keys, first match wins: BLACKHOLE_<NAME>_API_KEY, then ANTHROPIC_API_KEY /
OPENAI_API_KEY, then ~/.blackhole/keys.toml (0600, written by the TUI's /connect
screen or `blackhole connect`). With no key the Anthropic SDK still picks up an
`ant auth login` profile.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


def home() -> Path:
    return Path(os.environ.get("BLACKHOLE_HOME", Path.home() / ".blackhole"))


class ProviderError(Exception):
    """Something the TUI shows the user as-is: bad key, rate limit, refusal, no network."""


class ToolError(Exception):
    """Raised by a tool function; goes back to the model as an error result so it can fix its call."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[[dict], str]


@dataclass
class Usage:
    input_tokens: int = 0       # uncached input, including cache writes
    cache_read_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0

    def cost_usd(self, prices: tuple[float, float, float] | None) -> float | None:
        """prices = (input, output, cache read) per million tokens; None when unknown."""
        if not prices:
            return None
        pin, pout, pcache = prices
        return round((self.input_tokens * pin + self.output_tokens * pout + self.cache_read_tokens * pcache) / 1e6, 4)


@dataclass
class RunResult:
    text: str
    usage: Usage = field(default_factory=Usage)
    stopped_by_tool: bool = False


class StopLoop(Exception):
    """Raised by a tool to end the loop at once (the user's question needs the TUI, or we are done)."""

    def __init__(self, result: str):
        super().__init__(result)
        self.result = result


class Provider(Protocol):
    name: str
    model: str
    prices: tuple[float, float, float] | None

    def run(self, system: str, prompt: str, tools: list[Tool], *, max_turns: int = 30,
            on_text: Callable[[str], None] | None = None, budget_usd: float | None = None) -> RunResult: ...

    def check(self) -> str: ...


class Budget:
    """Hard spend cap for one run. Before each turn it checks the worst case of that turn
    (the whole history re-read uncached plus a full max_tokens reply) against what is left,
    and lowers max_tokens or stops, so the cap holds even if every turn maxes out."""

    def __init__(self, usd: float | None, prices: tuple[float, float, float] | None):
        if usd is not None and not prices:
            raise ProviderError("a spend cap needs this provider's prices; pass prices=(in, out, cache) per million")
        self.usd, self.prices = usd, prices

    def max_tokens(self, usage: "Usage", context_tokens: int, want: int) -> int:
        if self.usd is None:
            return want
        pin, pout, _ = self.prices
        left = self.usd - (usage.cost_usd(self.prices) or 0.0) - context_tokens * pin / 1e6
        allowed = int(left / pout * 1e6)
        if allowed < 1024:
            raise ProviderError(f"stopped at the ${self.usd:.2f} spend cap "
                                f"(spent ${usage.cost_usd(self.prices):.4f}); the draft is incomplete")
        return min(want, allowed)


def _check_args(schema: dict, args: Any) -> str | None:
    """Minimal JSON-schema check for tool inputs (required keys, top-level types).

    With eager input streaming the API no longer validates tool input, so a cut-off
    stream can hand us a partial object; this catches it before the tool runs."""
    if not isinstance(args, dict):
        return "INVALID_JSON: tool input must be a JSON object; send the call again"
    for key in schema.get("required", []):
        if key not in args:
            return f"missing required field {key!r}; send the call again"
    types = {"string": str, "integer": int, "number": (int, float), "array": list, "object": dict, "boolean": bool}
    for key, val in args.items():
        want = schema.get("properties", {}).get(key, {}).get("type")
        if want and not isinstance(val, types[want]):
            return f"field {key!r} must be {want}; send the call again"
    return None


def call_tool(tool: Tool | None, args: Any) -> tuple[str, bool]:
    """Validate the model's arguments and run the tool: (content, is_error). StopLoop passes through."""
    if tool is None:
        return "unknown tool", True
    problem = _check_args(tool.input_schema, args)
    if problem:
        return problem, True
    try:
        return tool.fn(args), False
    except ToolError as e:
        return str(e), True


# ---------------------------------------------------------------- Claude

# Claude API prices per million tokens (input, output, cache read), 2026-10.
CLAUDE_PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-fable-5-1": (10.00, 50.00, 0.25),
}


class ClaudeProvider:
    name = "claude"

    def __init__(self, api_key: str | None = None, model: str = "claude-opus-5-5", effort: str = "high",
                 max_tokens: int = 64000, client=None):
        import anthropic

        self._a = anthropic
        self.client = client or (anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic())
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.prices = CLAUDE_PRICES.get(model)

    def check(self) -> str:
        """Cheap key check for /connect: one models.retrieve call, no tokens spent."""
        a = self._a
        try:
            return self.client.models.retrieve(self.model).display_name
        except a.AuthenticationError as e:
            raise ProviderError("Anthropic rejected the API key") from e
        except a.NotFoundError as e:
            raise ProviderError(f"{self.model} is not available to this key") from e
        except a.APIConnectionError as e:
            raise ProviderError(f"cannot reach the Anthropic API: {e}") from e

    def run(self, system, prompt, tools, *, max_turns=30, on_text=None, budget_usd=None) -> RunResult:
        a = self._a
        budget = Budget(budget_usd, self.prices)
        context = (len(system) + len(prompt) + len(json.dumps([t.input_schema for t in tools]))) // 3  # ~3 chars/token, high side
        by_name = {t.name: t for t in tools}
        # Tool inputs (a whole schema) stream as they are written; call_tool validates them.
        tool_defs = [{"name": t.name, "description": t.description, "input_schema": t.input_schema,
                      "eager_input_streaming": True} for t in tools]
        messages: list[dict] = [{"role": "user", "content": prompt}]
        usage = Usage()
        for _ in range(max_turns):
            try:
                max_tokens = budget.max_tokens(usage, context, self.max_tokens)
                with self.client.beta.messages.stream(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    tools=tool_defs,
                    messages=messages,
                    output_config={"effort": self.effort},
                    cache_control={"type": "ephemeral"},  # system + tools + history are re-sent each turn
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",  # a declined request re-runs on Anthropic's recommended fallback model
                ) as stream:
                    for chunk in stream.text_stream:
                        if on_text:
                            on_text(chunk)
                    msg = stream.get_final_message()
            except a.AuthenticationError as e:
                raise ProviderError("Anthropic rejected the API key; run /connect claude") from e
            except a.RateLimitError as e:
                raise ProviderError("Anthropic rate limit; wait a minute and try again") from e
            except a.APIStatusError as e:
                raise ProviderError(f"Anthropic API error {e.status_code}: {e.message}") from e
            except a.APIConnectionError as e:
                raise ProviderError(f"cannot reach the Anthropic API: {e}") from e
            except (a.CredentialsError, TypeError) as e:  # no key, no auth token, no `ant auth login` profile
                if isinstance(e, TypeError) and "authentication" not in str(e):
                    raise
                raise ProviderError("no Anthropic credentials; run `blackhole connect claude` "
                                    "or set ANTHROPIC_API_KEY") from e

            u = msg.usage
            context = u.input_tokens + (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0) \
                + u.output_tokens + 2000  # next turn re-reads all of this plus the tool results
            usage.turns += 1
            usage.input_tokens += u.input_tokens + (u.cache_creation_input_tokens or 0)
            usage.cache_read_tokens += u.cache_read_input_tokens or 0
            usage.output_tokens += u.output_tokens
            if msg.stop_reason == "refusal":
                raise ProviderError("Claude declined this request, and so did the fallback model")
            # The whole content goes back (thinking blocks included) so the next turn stays valid.
            messages.append({"role": "assistant", "content": msg.content})
            text = "".join(b.text for b in msg.content if b.type == "text")
            if msg.stop_reason == "max_tokens":
                raise ProviderError("the model ran out of output tokens; raise max_tokens")
            if msg.stop_reason != "tool_use":
                return RunResult(text, usage)
            results = []
            for block in msg.content:
                if block.type == "tool_use":
                    try:
                        content, is_error = call_tool(by_name.get(block.name), block.input)
                    except StopLoop as stop:
                        return RunResult(stop.result, usage, stopped_by_tool=True)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": content,
                                    "is_error": is_error})
            messages.append({"role": "user", "content": results})  # every result in one message
        raise ProviderError(f"no result after {max_turns} turns")


# ---------------------------------------------------------------- OpenAI-compatible

# name: (base_url, default model, prices). Chat Completions is the common denominator:
# OpenAI, Nebius Token Factory, OpenRouter and a vLLM on the VM all speak it.
OPENAI_PRESETS: dict[str, tuple[str, str, tuple[float, float, float] | None]] = {
    "openai": ("https://api.openai.com/v1", "gpt-5", (1.25, 10.00, 0.125)),  # list price at launch; check
    "nebius": ("https://api.studio.nebius.com/v1", "openai/gpt-oss-120b", None),  # mugge's default designer
    "openrouter": ("https://openrouter.ai/api/v1", "openai/gpt-5", None),
    "local": ("http://localhost:8000/v1", "", (0.0, 0.0, 0.0)),  # `vllm serve` on the VM: no API spend
}


class OpenAICompatProvider:
    def __init__(self, name: str = "openai", api_key: str | None = None, base_url: str | None = None,
                 model: str | None = None, max_tokens: int = 32000, client=None,
                 prices: tuple[float, float, float] | None = None):
        import openai

        url, default_model, preset_prices = OPENAI_PRESETS.get(name, (None, "", None))
        self._o = openai
        self.name, self.max_tokens = name, max_tokens
        same_model = name == "local" or (model or default_model) == default_model
        self.prices = prices if prices is not None else (preset_prices if same_model else None)
        self.model = model or default_model
        self.client = client or openai.OpenAI(api_key=api_key or "none", base_url=base_url or url)

    def check(self) -> str:
        o = self._o
        try:
            ids = [m.id for m in self.client.models.list()]
        except o.AuthenticationError as e:
            raise ProviderError(f"{self.name} rejected the API key") from e
        except o.APIConnectionError as e:
            raise ProviderError(f"cannot reach {self.client.base_url}: {e}") from e
        if not self.model and ids:
            self.model = ids[0]  # a local server: use whatever it serves
        return self.model

    def run(self, system, prompt, tools, *, max_turns=30, on_text=None, budget_usd=None) -> RunResult:
        o = self._o
        budget = Budget(budget_usd, self.prices)
        context = (len(system) + len(prompt) + len(json.dumps([t.input_schema for t in tools]))) // 3
        if not self.model:
            self.check()
        by_name = {t.name: t for t in tools}
        tool_defs = [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                       "parameters": t.input_schema}} for t in tools]
        messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        usage = Usage()
        for _ in range(max_turns):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=tool_defs,
                    max_completion_tokens=budget.max_tokens(usage, context, self.max_tokens))
            except o.AuthenticationError as e:
                raise ProviderError(f"{self.name} rejected the API key; run /connect {self.name}") from e
            except o.RateLimitError as e:
                raise ProviderError(f"{self.name} rate limit; wait and try again") from e
            except o.APIStatusError as e:
                raise ProviderError(f"{self.name} API error {e.status_code}: {e.message}") from e
            except o.APIConnectionError as e:
                raise ProviderError(f"cannot reach {self.client.base_url}: {e}") from e
            usage.turns += 1
            if resp.usage:
                usage.input_tokens += resp.usage.prompt_tokens
                usage.output_tokens += resp.usage.completion_tokens
                context = resp.usage.prompt_tokens + resp.usage.completion_tokens + 2000
            choice = resp.choices[0]
            msg = choice.message
            if msg.content and on_text:
                on_text(msg.content)
            messages.append(msg.model_dump(exclude_none=True))
            if not msg.tool_calls:
                if choice.finish_reason == "length":
                    raise ProviderError("the model ran out of output tokens")
                return RunResult(msg.content or "", usage)
            for call in msg.tool_calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = None
                try:
                    content, is_error = call_tool(by_name.get(call.function.name), args)
                except StopLoop as stop:
                    return RunResult(stop.result, usage, stopped_by_tool=True)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": ("ERROR: " if is_error else "") + content})
        raise ProviderError(f"no result after {max_turns} turns")


# ---------------------------------------------------------------- keys

ENV_KEYS = {"claude": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
PROVIDERS = ("claude", *OPENAI_PRESETS)


def keys_path() -> Path:
    return home() / "keys.toml"


def load_keys() -> dict:
    p = keys_path()
    return tomllib.loads(p.read_text()) if p.exists() else {}


def _toml_str(s: str) -> str:
    return json.dumps(s)  # a JSON string is a valid TOML basic string


def save_key(name: str, api_key: str = "", base_url: str = "", model: str = "", default: bool = False) -> Path:
    """Store one provider in keys.toml. The file is created 0600 before a key is written to it."""
    data = load_keys()
    entry = {k: v for k, v in (("api_key", api_key), ("base_url", base_url), ("model", model)) if v}
    data.setdefault("providers", {})[name] = entry
    if default or "default" not in data:
        data["default"] = name
    lines = [f"default = {_toml_str(data['default'])}", ""]
    for pname, e in data["providers"].items():
        lines.append(f"[providers.{pname}]")
        lines += [f"{k} = {_toml_str(v)}" for k, v in e.items()]
        lines.append("")
    p = keys_path()
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines))
    os.chmod(p, 0o600)
    return p


def get(name: str | None = None, **overrides) -> Provider:
    """The named provider (else keys.toml's default, else whichever env key is set)."""
    data = load_keys()
    if not name:
        name = data.get("default") or ("claude" if os.environ.get("ANTHROPIC_API_KEY") or
                                       not os.environ.get("OPENAI_API_KEY") else "openai")
    if name not in PROVIDERS:
        raise ProviderError(f"unknown provider {name!r}; one of {', '.join(PROVIDERS)}")
    entry = {**data.get("providers", {}).get(name, {}), **{k: v for k, v in overrides.items() if v}}
    key = (os.environ.get(f"BLACKHOLE_{name.upper()}_API_KEY") or os.environ.get(ENV_KEYS.get(name, ""))
           or entry.get("api_key"))
    if name == "claude":
        return ClaudeProvider(api_key=key, **{k: entry[k] for k in ("model", "effort") if k in entry})
    return OpenAICompatProvider(name, api_key=key, base_url=entry.get("base_url"), model=entry.get("model"))


def mask(key: str) -> str:
    return f"{key[:7]}…{key[-4:]}" if len(key) > 14 else "…"
