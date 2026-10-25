"""sync.*: getting results off a rented VM before it is deleted."""

from dataclasses import dataclass

from blackhole.core.ops import Op, p, register


@register
class Bucket(Op):
    """Upload the job's outputs, receipt and logs to an S3-compatible bucket (or `hf
    upload`), each artifact under its cache key, so a fresh VM restores whatever matches
    instead of recomputing it. Use with `always: true`; `shutdown` powers the VM off after."""
    name, major = "sync.bucket", 1
    ported_from = "BH event_horizon.py upload + restore + --shutdown"

    @dataclass
    class Params:
        target: str = p("", "s3://bucket/prefix | hf:<user>/<repo>; empty = $BLACKHOLE_BUCKET")
        include: list = p(["out", "receipt.json", "events.log"])
        shutdown: bool = p(False)
