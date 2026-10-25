"""blackhole.anatomy: the open format layer of the BlackHole model API.

    st        safetensors headers, lazy reads, byte-range writes, bf16 / fp8 conversion (no torch)
    hex       HEX (Header-Encoded eXperts): the packed layout the Metal and Triton kernels read,
              the per-expert table quantizer (nearest or GPTQ onto the table), pack / unpack,
              the torch reference dequant. The one definition of the layout.
    formats   format names, bits per weight, and a round-to-nearest error probe

The developer API that edits models through these (parts, queries, weight edits, named
functions, the `model.edit` op) is the separate blackhole-dev package.
"""

from blackhole.anatomy import formats, hex, st

__all__ = ["st", "hex", "formats"]
