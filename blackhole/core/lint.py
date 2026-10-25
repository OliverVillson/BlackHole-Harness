"""Schema rules beyond structure and the op catalog: things that parse and would run, but
would mislead a user or the marketplace. ops.validate() appends these, so `blackhole
validate`, the schema builder's repair loop and registry CI all apply the same rules."""

from __future__ import annotations



def lint(raw: dict, publish: bool = False) -> list[str]:
    """Rules on top of schema.parse and the op catalog. publish=True adds what the marketplace
    and the schema builder also require (metadata, an output, an eval stage)."""
    out = []
    inputs = raw.get("inputs") or {}
    for name, spec in inputs.items():
        if spec.get("required") and spec.get("advanced"):
            out.append(f"inputs.{name}: required inputs cannot be advanced (simple mode would never ask)")
    policy = raw.get("data_policy") or {}
    allowed = policy.get("licenses_allowed")
    kinds = []
    for st in raw.get("stage") or []:
        kind = st.get("kind", "").split("@")[0]
        kinds.append(kind)
        lic = st.get("license")
        if kind == "data.dataset" and allowed and isinstance(lic, str) and lic not in allowed:
            out.append(f"stage {kind}: licence {lic!r} is not in data_policy.licenses_allowed")
        teacher = str(st.get("teacher", "")) + str(st.get("model", ""))
        if kind.startswith("lab.") and not st.get("code_sha256"):
            out.append(f"stage {kind}: lab ops must pin code_sha256 so reviewers know which script runs")
        if "frontier:" in teacher and not policy.get("frontier_spend_usd"):
            out.append(f"stage {kind}: a frontier teacher needs data_policy.frontier_spend_usd > 0")
    need = {"gguf": "package.gguf", "mlx": "quant.mlx", "safetensors-bf16": "package.hf",
            "safetensors-fp8": "package.hf", "safetensors-w4a16": "package.hf"}
    for o in raw.get("outputs") or []:
        if o in need and need[o] not in kinds:
            out.append(f"outputs: {o!r} needs a {need[o]} stage")
    if publish:
        if not raw.get("outputs"):
            out.append("outputs: list at least one output format (top level, before the first [table])")
        if not any(k.startswith("eval.") for k in kinds):
            out.append("no eval stage: the marketplace could never show results or award a badge")
        for key in ("title", "description", "license"):
            if not raw.get(key):
                out.append(f"{key}: required to publish")
    return out
