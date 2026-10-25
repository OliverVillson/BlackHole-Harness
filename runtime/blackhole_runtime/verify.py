"""Convert an MLX HEX export for vLLM and prove the result computes what the Mac computes.

    python -m blackhole_runtime.verify <MLX HEX export> <out folder> [--device cuda]

1. convert.py writes the blackhole-hex folder (skipped with --no-convert if it exists).
2. Every layer's HEX tiers, dequantized, equal the MLX folder's own dequantized experts.
3. The Triton kernel on the converted tiers, for a few random tokens and routings per layer,
   equals the reference MoE computed from the MLX experts (fp32 on the CPU, bf16 on a GPU).
Prints one JSON line at the end; exit code 1 on any mismatch. On the CPU set TRITON_INTERPRET=1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-convert", action="store_true")
    ap.add_argument("--tokens", type=int, default=5)
    a = ap.parse_args(argv)

    from safetensors import safe_open

    from blackhole_pipeline.serve import _mlx_state
    from blackhole_runtime.convert import convert
    from blackhole_runtime.hex_moe import HexTier, hex_moe
    from blackhole_runtime.hex_ref import hex_dequant

    res = {} if a.no_convert else convert(a.src, a.dst)
    dev = torch.device(a.device)
    dt = torch.float32 if dev.type == "cpu" else torch.bfloat16
    cfg = json.loads((a.src / "config.json").read_text())
    tc = cfg.get("text_config") or cfg
    E, k = tc["num_experts"], tc["num_experts_per_tok"]
    out = {}
    for f in sorted(a.dst.glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            out.update({n: h.get_tensor(n) for n in h.keys() if ".tier" in n})
    ref = _mlx_state(a.src, dt)   # bf16 on a GPU machine: a full Qwen3.6 in fp32 would not fit in RAM
    layers = sorted({int(n.split(".")[2]) for n in out})
    worst_w, worst_y, ok = 0.0, 0.0, bool(layers)
    for li in layers:
        pre = f"model.layers.{li}.mlp.experts"
        gu, dn = ref.pop(f"{pre}.gate_up_proj").float(), ref.pop(f"{pre}.down_proj").float()   # [E, 2I, H], [E, H, I]
        I = dn.shape[-1]
        tiers = []
        for L in (16, 8):
            t = f"{pre}.tier{L}"
            if f"{t}.experts" not in out:
                continue
            bits = L.bit_length() - 1
            ids = out[f"{t}.experts"].long()
            parts = {p: {q: out[f"{t}.{p}.{q}"] for q in ("idx", "block_scale", "expert_scale", "levels")}
                     for p in ("gate_proj", "up_proj", "down_proj")}
            W = {p: hex_dequant(*parts[p].values(), bits) for p in parts}
            for got_w, want_w in ((W["gate_proj"], gu[ids, :I]), (W["up_proj"], gu[ids, I:]), (W["down_proj"], dn[ids])):
                worst_w = max(worst_w, float((got_w - want_w).abs().max() / (want_w.abs().max() + 1e-12)))
            tiers.append(HexTier(bits, ids.to(dev), *[{q: v.to(dev) for q, v in parts[p].items()}
                                                      for p in ("gate_proj", "up_proj", "down_proj")]).prepare(E))
        g = torch.Generator().manual_seed(li)
        T = a.tokens
        topk = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])
        w = torch.softmax(torch.randn(T, k, generator=g), -1)
        x = torch.randn(T, gu.shape[-1], generator=g).to(dt).float()
        want = torch.zeros_like(x)
        for s in range(k):
            h = torch.einsum("th,toh->to", x, gu[topk[:, s]])
            act = (torch.nn.functional.silu(h[:, :I]) * h[:, I:]).to(dt).float()
            want += torch.einsum("ti,thi->th", act, dn[topk[:, s]]) * w[:, s, None]
        got = hex_moe(x.to(dev, dt), w.to(dev), topk.to(dev), tiers).float().cpu()
        worst_y = max(worst_y, float((got - want).abs().max() / (want.abs().max() + 1e-12)))
    tol_y = 1e-4 if dt == torch.float32 else 2e-2
    ok = ok and worst_w < (1e-5 if dt == torch.float32 else 1e-2) and worst_y < tol_y
    print(f"{len(layers)} HEX layers · weights max rel diff {worst_w:.1e} · kernel vs MLX experts {worst_y:.1e} "
          f"({'OK' if ok else 'MISMATCH'})", file=sys.stderr)
    print(json.dumps({**res, "layers": len(layers), "weight_rel": worst_w, "kernel_rel": worst_y, "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
