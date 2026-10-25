"""A BlackHole MLX HEX export -> a "blackhole-hex" folder that vLLM serves with the HEX kernel.

    python -m blackhole_runtime.convert <mlx folder> <out folder>

The routed experts keep their packed HEX parts as they are (no requantization, the same
numbers the Mac runs): the MLX tiers (experts_hot first, then experts; the router rows are
already in that order) become `<mlp>.experts.tier<L>` with their expert ids. Everything else
(attention, linear attention, router, shared expert, embeddings, LM head) is dequantized to
bf16, so a Qwen3.6 build is a few GB bigger than on the Mac. An output-trimmed LM head is
expanded back to the full vocabulary and its kept token ids go into the quantization config,
where vllm_hex.py masks every other token. Needs the pipeline package (blackhole_pipeline) and
an architecture vLLM supports (Qwen3.5 / 3.6 MoE; Flint has no vLLM model).
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")
ARCH = {"qwen3_5_moe": "Qwen3_5MoeForCausalLM"}
_PART = re.compile(r"^(.*\.mlp)\.(experts_hot|experts)\.(gate_proj|up_proj|down_proj)\.(idx|block_scale|expert_scale|levels)$")


def hex_tensors(parts: dict, tiers: dict) -> dict:
    """MLX-tier HEX parts (PyTorch names) -> blackhole-hex tier tensors, grouped by table size."""
    by = {}
    for k, t in parts.items():
        m = _PART.match(k)
        if not m:
            raise ValueError(f"unexpected HEX tensor {k}")
        by.setdefault((m.group(1), m.group(2)), {}).setdefault(m.group(3), {})[m.group(4)] = t
    out, groups = {}, {}
    for (mlp, tier), projs in by.items():
        layer = int(re.search(r"layers\.(\d+)\.", mlp).group(1))
        n_hot = tiers.get(layer, 0)
        n = projs["down_proj"]["levels"].shape[0]
        first = 0 if tier == "experts_hot" else n_hot
        L = projs["down_proj"]["levels"].shape[-1]
        groups.setdefault((mlp, L), []).append((first, n, projs))
    for (mlp, L), gs in groups.items():
        gs.sort(key=lambda g: g[0])
        pre = f"{mlp}.experts.tier{L}"
        out[f"{pre}.experts"] = torch.cat([torch.arange(f, f + n, dtype=torch.int32) for f, n, _ in gs])
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for p in ("idx", "block_scale", "expert_scale", "levels"):
                out[f"{pre}.{proj}.{p}"] = torch.cat([g[2][proj][p] for g in gs]).contiguous()
    return out


def convert(src: Path, dst: Path) -> dict:
    from safetensors.torch import save_file

    from blackhole_pipeline.serve import _mlx_state
    cfg = json.loads((src / "config.json").read_text())
    arch = ARCH.get(cfg.get("model_type"))
    if arch is None:
        raise SystemExit(f"{src}: model type {cfg.get('model_type')!r} has no vLLM model here (supported: {sorted(ARCH)})")
    if not cfg.get("hex"):
        raise SystemExit(f"{src} has no HEX experts (config 'hex' is empty): nothing for the HEX kernel")
    tiers = {int(k): v for k, v in (cfg.get("expert_tiers") or {}).items()}
    parts: dict = {}
    state = _mlx_state(src, torch.bfloat16, hex_parts=parts)
    leftover = [k for k in state if re.search(r"\.mlp\.experts\.(gate_up_proj|down_proj)$|\.mlp\.experts\.\d+\.", k)]
    if leftover:
        raise SystemExit(f"{src}: some MoE layers are not HEX ({leftover[0]}, ...); vLLM's HEX method needs all of them")
    head_ids = state.pop("lm_head_ids", None)
    state.update(hex_tensors(parts, tiers))

    dst.mkdir(parents=True, exist_ok=True)
    index, shard, size, n = {}, {}, 0, 0

    def flush():
        nonlocal shard, size, n
        if shard:
            n += 1
            name = f"model-{n:05d}.safetensors"
            save_file(shard, str(dst / name), metadata={"format": "pt"})
            index.update({k: name for k in shard})
        shard, size = {}, 0

    total = 0
    for k in sorted(state):
        t = state[k].contiguous()
        shard[k] = t
        size += t.numel() * t.element_size()
        total += t.numel() * t.element_size()
        if size > 4 * 1024 ** 3:
            flush()
    flush()
    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": index}, indent=2) + "\n")
    for f in src.iterdir():
        if f.is_dir() or f.name.endswith(WEIGHT_SUFFIXES) or f.suffix == ".py" \
                or f.name in ("model.safetensors.index.json", "config.json", "README.md"):
            continue
        shutil.copy2(f, dst / f.name)
    for a in ("quantization", "model_file", "expert_tiers", "head_vocab", "hex"):
        cfg.pop(a, None)
    cfg["architectures"] = [arch]
    tl: dict = {}   # experts per table size, over all layers
    for k, t in state.items():
        m = re.search(r"tier(\d+)\.experts$", k)
        if m:
            tl[m.group(1)] = tl.get(m.group(1), 0) + int(t.numel())
    qc = {"quant_method": "blackhole-hex", "version": 1, "block": 32, "source": "mlx", "tiers": tl}
    if head_ids is not None:
        qc["head_ids"] = [int(i) for i in head_ids.tolist()]
    cfg["quantization_config"] = qc
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return {"out": str(dst), "gb": round(total / 1e9, 2), "tiers": tl, "head_ids": None if head_ids is None else len(head_ids)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    a = ap.parse_args(argv)
    print(json.dumps(convert(a.src, a.dst), indent=2))


if __name__ == "__main__":
    main()
