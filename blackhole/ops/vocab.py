"""vocab.*: shrinking the embedding and head to the tokens a purpose uses."""

from dataclasses import dataclass

from blackhole.core.ops import Op, p, register


@register
class Trim(Op):
    """Keep the tokens seen in the data (plus specials and byte fallbacks), remap the
    tokenizer, slice embedding and head. Checks llama-tokenize == HF after."""
    name, major = "vocab.trim", 1
    consumes = {"model": "model", "data": "dataset"}
    produces = {"model": "model", "kept": "file"}
    ported_from = "BH event_horizon.py trim step"

    @dataclass
    class Params:
        model: str = p("")
        data: str = p("")
        keep: int = p(96000, "target vocabulary size")
