"""Check a "blackhole-hex" folder before serving it: every MoE layer's HEX tiers load, cover every
expert once, have the model's shapes, and the HEX kernel on them equals the reference MoE
computed from the dequantized tiers, for a few random tokens and routings per layer.

    python -m blackhole_runtime.check_folder <hex folder> [--device cuda] [--layers 4]

Works on any blackhole-hex folder (the pipeline's hexvllm stage, BlackHole-Dev's hex-pack, or
convert.py from an MLX export) and any MoE architecture: tensors are found by name
(<...>.experts.tier<L>.<proj>.<part>), shapes come from the tiers and config.json. Prints one
JSON line at the end; exit code 1 on any problem. On a CPU set TRITON_INTERPRET=1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

_TIER = re.compile(r"^(?P<pre>.+)\.tier(?P<L>\d+)\.(?:experts|(?P<proj>gate_proj|up_proj|down_proj)\.(?P<part>idx|"
                   r"block_scale|expert_scale|levels))$")


def tiers_by_layer(folder: Path) -> dict[str, dict[int, dict]]:
    """{experts prefix: {levels: {"experts": ids, proj: {part: tensor}}}} of every HEX tier."""
    from safetensors import safe_open
    out: dict = {}
    for f in sorted(folder.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for n in h.keys():
                m = _TIER.match(n)
                if not m:
                    continue
                t = out.setdefault(m["pre"], {}).setdefault(int(m["L"]), {})
                if m["proj"]:
                    t.setdefault(m["proj"], {})[m["part"]] = h.get_tensor(n)
                else:
                    t["experts"] = h.get_tensor(n)
    return out


def check(folder: Path, device: str, n_layers: int = 0, tokens: int = 5) -> dict:
    from blackhole_runtime.hex_moe import HexTier, hex_moe
    from blackhole_runtime.hex_ref import moe_reference

    cfg = json.loads((folder / "config.json").read_text())
    q = cfg.get("quantization_config") or {}
    problems = []
    if q.get("quant_method") != "blackhole-hex":
        problems.append(f"config.json quantization_config.quant_method is {q.get('quant_method')!r}, not 'blackhole-hex'")
    tc = cfg.get("text_config") or cfg
    E = next((tc[k] for k in ("num_experts", "num_local_experts", "n_routed_experts") if isinstance(tc.get(k), int)), None)
    k = int(tc.get("num_experts_per_tok") or 2)
    layers = tiers_by_layer(folder)
    if not layers:
        problems.append("no HEX tiers (<...>.experts.tier<L>.*) in the safetensors files")
    dev = torch.device(device)
    dt = torch.float32 if dev.type == "cpu" else torch.bfloat16
    worst, checked = 0.0, 0
    for li, (pre, tiers) in enumerate(sorted(layers.items(), key=lambda kv: [int(x) if x.isdigit() else x
                                                                             for x in re.split(r"(\d+)", kv[0])])):
        seen = torch.cat([t["experts"].long() for t in tiers.values()])
        n_exp = E or int(seen.max()) + 1
        if sorted(seen.tolist()) != list(range(n_exp)):
            problems.append(f"{pre}: tiers hold experts {sorted(seen.tolist())[:6]}..., expected each of 0..{n_exp - 1} once")
            continue
        if n_layers and checked >= n_layers:
            continue
        ref_tiers, run_tiers = [], []
        for L, t in sorted(tiers.items(), reverse=True):
            bits = L.bit_length() - 1
            parts = {p: {x: t[p][x] for x in ("idx", "block_scale", "expert_scale", "levels")}
                     for p in ("gate_proj", "up_proj", "down_proj")}
            ref_tiers.append({"bits": bits, "experts": t["experts"].long(), **parts})
            run_tiers.append(HexTier(bits, t["experts"].to(dev), *[{x: v.to(dev) for x, v in parts[p].items()}
                                                                   for p in ("gate_proj", "up_proj", "down_proj")])
                             .prepare(n_exp))
        H = ref_tiers[0]["gate_proj"]["idx"].shape[-1] * 32 // ref_tiers[0]["bits"]
        g = torch.Generator().manual_seed(li)
        topk = torch.stack([torch.randperm(n_exp, generator=g)[:min(k, n_exp)] for _ in range(tokens)])
        w = torch.softmax(torch.randn(tokens, topk.shape[1], generator=g), -1)
        x = torch.randn(tokens, H, generator=g).to(dt)
        want = moe_reference(x.float(), w, topk, ref_tiers)
        got = hex_moe(x.to(dev), w.to(dev), topk.to(dev), run_tiers).float().cpu()
        worst = max(worst, float((got - want).abs().max() / (want.abs().max() + 1e-12)))
        checked += 1
    tol = 1e-4 if dt == torch.float32 else 2e-2
    if worst >= tol:
        problems.append(f"HEX kernel differs from the reference MoE by {worst:.1e} (tolerance {tol:.0e})")
    return {"layers": len(layers), "checked": checked, "kernel_rel": worst, "problems": problems, "ok": not problems}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layers", type=int, default=0, help="kernel-check only the first N layers (0 = all)")
    ap.add_argument("--tokens", type=int, default=5)
    a = ap.parse_args(argv)
    res = check(a.folder, a.device, a.layers, a.tokens)
    for p in res["problems"]:
        print(f"problem: {p}", file=sys.stderr)
    print(f"{res['layers']} HEX layers, {res['checked']} kernel-checked, max rel diff {res['kernel_rel']:.1e} "
          f"({'OK' if res['ok'] else 'PROBLEMS'})", file=sys.stderr)
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
