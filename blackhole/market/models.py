"""Model marketplace: base models, footprint estimates, and the two "fits" columns.

- "Runs here": can this VM run a pipeline on the model (bf16 weights on disk, the
  teacher in GPU memory)? REAPer's "Fits here" column.
- "Fits your PC": after a prune ratio and quant, will the result run on the user's
  own machine (a device class from blackhole.hw)? best_fit() finds the gentlest
  recipe that does, which the schema builder uses to pick a base and a ratio.

Numbers are rounded public spec-sheet values. eval.footprint measures the real
thing at the end of a run, and receipts carry it into the marketplace.
"""

from __future__ import annotations

from dataclasses import dataclass

from blackhole.hw import DEVICES as DEVICE_SPECS, Hardware

GB = 1e9


@dataclass(frozen=True)
class BaseModel:
    key: str
    hf_id: str
    arch: str  # moe | dense | hybrid
    total_b: float
    active_b: float
    experts: int = 0
    top_k: int = 0
    layers: int = 0
    # KV cache bytes per token at 16-bit, all layers. Hybrids (DeltaNet/Mamba) are small.
    kv_bytes_per_token: int = 0
    license: str = ""
    detected_expert_b: float = 0.0  # from blackhole.hw.model.detect; overrides the estimate below

    @property
    def expert_b(self) -> float:
        """Billions of params in routed experts, from total/active/top-k (ignores shared experts)."""
        if self.detected_expert_b:
            return self.detected_expert_b
        if self.arch == "dense" or not self.experts:
            return 0.0
        frac_active = self.top_k / self.experts
        return max(0.0, (self.total_b - self.active_b) / (1 - frac_active))

    @property
    def bf16_gb(self) -> float:
        return self.total_b * 2


# Shortcut list: REAPer's marketplace plus the BlackHole bases, so the table needs no network.
# Any other model goes through lookup() -> blackhole.hw.model.detect.
CATALOG: list[BaseModel] = [
    BaseModel("qwen3-30b-a3b", "Qwen/Qwen3-30B-A3B-Instruct-2507", "moe", 30.5, 3.3, 128, 8, 48, 98_304, "Apache-2.0"),
    BaseModel("qwen3-coder-30b-a3b", "Qwen/Qwen3-Coder-30B-A3B-Instruct", "moe", 30.5, 3.3, 128, 8, 48, 98_304, "Apache-2.0"),
    BaseModel("qwen3-next-80b-a3b", "Qwen/Qwen3-Next-80B-A3B-Instruct", "hybrid", 80, 3, 512, 10, 48, 24_576, "Apache-2.0"),
    BaseModel("gpt-oss-20b", "openai/gpt-oss-20b", "moe", 21, 3.6, 32, 4, 24, 49_152, "Apache-2.0"),
    BaseModel("gpt-oss-120b", "openai/gpt-oss-120b", "moe", 117, 5.1, 128, 4, 36, 73_728, "Apache-2.0"),
    BaseModel("glm-4.5-air", "zai-org/GLM-4.5-Air", "moe", 106, 12, 128, 8, 46, 188_416, "MIT"),
    BaseModel("glm-4.6", "zai-org/GLM-4.6", "moe", 355, 32, 160, 8, 92, 376_832, "MIT"),
    BaseModel("minimax-m2", "MiniMaxAI/MiniMax-M2", "moe", 230, 10, 256, 8, 62, 253_952, "MIT"),
    BaseModel("deepseek-v3.1", "deepseek-ai/DeepSeek-V3.1", "moe", 671, 37, 256, 8, 61, 70_272, "MIT"),
    BaseModel("kimi-k2", "moonshotai/Kimi-K2-Instruct", "moe", 1000, 32, 384, 8, 61, 70_272, "Modified-MIT"),
    BaseModel("flint", "RecursiveSI/flint", "moe", 39.8, 3.5, 256, 8, 48, 49_152, "see model card"),
    BaseModel("granite-4.0-h-tiny", "ibm-granite/granite-4.0-h-tiny", "hybrid", 7, 1, 64, 6, 40, 8_192, "Apache-2.0"),
    BaseModel("qwen3.6-35b-a3b", "Qwen/Qwen3.6-35B-A3B", "hybrid", 35, 3, 256, 8, 40, 20_480, "Apache-2.0"),
    BaseModel("qwen3.8-27b", "Qwen/Qwen3.8-27B", "hybrid", 26.9, 26.9, 0, 0, 64, 32_768, "Apache-2.0"),
]

# Share of a device's memory a model may use (weights + KV + buffers), leaving room for the OS/apps.
_USABLE = {"cuda": 0.9, "metal": 0.65, "cpu": 0.75}
DEVICES: dict[str, float] = {name: round(mem * _USABLE[kind], 1) for name, (mem, _bw, kind) in DEVICE_SPECS.items()}

BITS = {  # effective bits per weight, including scales
    "bf16": 16, "fp8": 8.25, "Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7, "Q4_K_M": 4.85,
    "IQ4_XS": 4.3, "Q3_K_M": 3.9, "IQ3_XXS": 3.1, "w4a16": 4.25,
}
KV_BITS = {"f16": 16, "q8_0": 8.5, "fp8": 8, "q4_0": 4.5}


def from_info(info) -> BaseModel:
    """A detected model (blackhole.hw.model.ModelInfo) as a catalog row."""
    kinds = set(info.kinds)
    arch = "hybrid" if "hybrid" in kinds else "moe" if "moe" in kinds else "dense"
    return BaseModel(info.id.rsplit("/", 1)[-1].lower(), info.id, arch, info.params_b, info.active_b or info.params_b,
                     info.experts, info.top_k, info.layers, info.kv_bytes_per_token,
                     detected_expert_b=info.expert_params_b)


def lookup(name: str, detect: bool = True) -> BaseModel | None:
    """Catalog shortcut first; otherwise resolve the name and detect it from its config
    (a few KB from a local folder or the Hub), so any model works, not just listed ones."""
    m = find(name)
    if m or not detect:
        return m
    from blackhole.hw.model import detect as detect_model, resolve
    try:
        rid, _ = resolve(name, known=[c.hf_id for c in CATALOG])
        return from_info(detect_model(rid)) if rid else None
    except Exception:  # offline, gated, not a model: the caller reports "unknown model"
        return None


def find(key: str) -> BaseModel | None:
    key = key.lower()
    for m in CATALOG:
        if key in (m.key, m.hf_id.lower()):
            return m
    matches = [m for m in CATALOG if key in m.key or key in m.hf_id.lower()]
    return matches[0] if len(matches) == 1 else None


def estimate_gb(m: BaseModel, ratio: float = 0, quant: str = "bf16", experts_quant: str | None = None,
                context: int = 16_384, kv: str = "q8_0") -> dict:
    """Estimated footprint after pruning and quant: weights + KV at `context` + ~5% buffers."""
    expert_b = m.expert_b * (1 - ratio)
    other_b = m.total_b - m.expert_b
    w_gb = (other_b * BITS[quant] + expert_b * BITS[experts_quant or quant]) / 8
    kv_gb = m.kv_bytes_per_token * context * (KV_BITS[kv] / 16) / GB
    total = (w_gb + kv_gb) * 1.05
    return {"params_b": round(other_b + expert_b, 1), "weights_gb": round(w_gb, 2),
            "kv_gb": round(kv_gb, 2), "total_gb": round(total, 2)}


# Rough quality cost of each step, so best_fit prefers "prune less, quantise less".
_PRUNE_COST = {0.0: 0, 0.25: 1, 0.5: 2, 0.625: 2.8, 0.75: 4.5}
_QUANT_COST = {"Q8_0": 0, "Q6_K": 0.6, "Q5_K_M": 1.0, "Q4_K_M": 1.5, "IQ4_XS": 2.0, "Q3_K_M": 3.2}


def best_fit(m: BaseModel, device: str, context: int = 16_384) -> dict | None:
    """Cheapest-in-quality (prune %, GGUF quant) that fits the device, or None."""
    budget = DEVICES[device]
    prunes = [0.0] if m.arch == "dense" or not m.experts else list(_PRUNE_COST)
    options = sorted(((_PRUNE_COST[p] + _QUANT_COST[q], p, q) for p in prunes for q in _QUANT_COST))
    for _, p, q in options:
        est = estimate_gb(m, p, q, context=context)
        if est["total_gb"] <= budget:
            return {"ratio": p, "quant": q, **est}
    return None


def runs_here(m: BaseModel, hw: Hardware) -> tuple[bool, str]:
    """REAPer's rule of thumb: bf16 weights + pruned copy + data on disk, bf16 teacher in GPU memory.
    blackhole.hw.plan_schema gives the real answer per schema; this is the quick table column."""
    need_disk = m.bf16_gb * 2.2 + 20
    vram = hw.gpu_mem_gb * hw.gpu_count
    if hw.disk_free_gb < need_disk:
        return False, f"needs ~{need_disk:.0f} GB disk, {hw.disk_free_gb:.0f} free"
    if vram < m.bf16_gb * 1.1:
        return False, f"needs ~{m.bf16_gb * 1.1:.0f} GB GPU memory for the teacher, {vram:.0f} here"
    return True, "ok"


def table(hw: Hardware, device: str, context: int = 16_384) -> list[dict]:
    rows = []
    for m in CATALOG:
        ok, why = runs_here(m, hw)
        fit = best_fit(m, device, context)
        rows.append({
            "model": m.key, "arch": m.arch, "params_b": m.total_b, "active_b": m.active_b,
            "experts": f"{m.experts}/top{m.top_k}" if m.experts else "-",
            "bf16_gb": round(m.bf16_gb), "runs_here": ok, "runs_here_why": why,
            "fits_target": fit is not None,
            "recipe": f"prune {fit['ratio']:.0%} + {fit['quant']} = {fit['total_gb']} GB" if fit else "too big",
        })
    return rows
