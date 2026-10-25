"""package.*: the thing the user downloads."""

from dataclasses import dataclass

from blackhole.core.ops import Op, p, register


@register
class Hf(Op):
    """Model card, schema, receipt and eval next to the safetensors; manifest.sha256."""
    name, major = "package.hf", 1
    consumes = {"model": "model"}
    produces = {"model": "model"}
    ported_from = "REAPer stages/package.py"

    @dataclass
    class Params:
        model: str = p("")
        name: str = p("", "default <base>-<schema>")


@register
class Gguf(Op):
    """The GGUF plus an Ollama Modelfile with the right chat and tool-call template, and
    a LM Studio / llama.cpp run line."""
    name, major = "package.gguf", 1
    consumes = {"model": "gguf"}
    produces = {"gguf": "gguf", "modelfile": "file"}
    ported_from = "LobBot stages/package.py"

    @dataclass
    class Params:
        model: str = p("")
        name: str = p("")
        context: int = p(16384)
