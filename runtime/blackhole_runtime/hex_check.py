"""Check and time the HEX Triton MoE kernel on an NVIDIA GPU (the counterpart of mlx/hex_check.py).

    python -m blackhole_runtime.hex_check                    # Qwen3.6-35B-A3B REAP 31.25% shape
    python -m blackhole_runtime.hex_check --experts 128 --hidden 2048 --inter 768 --topk 8
    python -m blackhole_runtime.hex_check --sweep            # also try a few tile sizes

1. Numbers: the kernel against the torch reference (dequantize, then matmul in fp32) for one
   token, a few tokens and a prompt, with 4-bit and 3-bit tiers, in bf16. Passes below 2e-2
   relative error (bf16 activations; the reference keeps fp32 weights).
2. Speed, per MoE layer (gate + up + down for every routed slot), next to vLLM's own fused MoE
   on the same shapes: bf16 experts and int4 W4A16 (group 128, what GPTQ int4 exports use).
   Decode batch sizes 1 / 8 / 32 tokens and a 1024- and 4096-token prompt.
Memory: a few hundred MB of random experts, no model download; safe next to a running job.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from blackhole_runtime.hex_moe import HexTier, hex_moe
from blackhole_runtime.hex_ref import moe_reference, random_parts


def build(E, H, I, hot, dev, seed=0):
    perm = torch.randperm(E, generator=torch.Generator().manual_seed(seed))
    tiers, ref = [], []
    for i, (ids, bits) in enumerate(((perm[:hot], 4), (perm[hot:], 3))):
        if len(ids) == 0:
            continue
        parts = [random_parts(len(ids), o, n, bits, dev, seed + 10 * i + j)
                 for j, (o, n) in enumerate(((I, H), (I, H), (H, I)))]
        ref.append({"bits": bits, "experts": ids.to(dev), "gate_proj": dict(parts[0]),
                    "up_proj": dict(parts[1]), "down_proj": dict(parts[2])})
        tiers.append(HexTier(bits, ids.to(dev), *[dict(p) for p in parts]).prepare(E))
    return tiers, ref


def routing(T, E, k, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.argsort(torch.rand(T, E, generator=g), -1)[:, :k].to(dev).to(torch.int32)
    w = torch.softmax(torch.randn(T, k, generator=g), -1).to(dev)
    return ids, w


def timeit(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    try:   # time with a CUDA graph when the call allows it (as vLLM decodes), else eagerly
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            fn()
        run = g.replay
    except Exception:  # noqa: BLE001
        run = fn
    run()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def vllm_baselines(E, H, I, dev):
    """{name: fn(x, w, ids)} for vLLM's fused MoE with bf16 and int4 W4A16 experts, if vLLM is here."""
    try:
        from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    except Exception as e:  # noqa: BLE001
        print(f"  (vLLM fused MoE not importable here: {type(e).__name__}: {e}; HEX timings only)")
        return {}
    w1 = torch.randn(E, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(E, H, I, device=dev, dtype=torch.bfloat16) * 0.02
    out = {"vLLM bf16": lambda x, w, ids: fused_experts(x, w1, w2, w, ids)}
    g = 128
    q1 = torch.randint(0, 256, (E, 2 * I, H // 2), device=dev, dtype=torch.uint8)
    q2 = torch.randint(0, 256, (E, H, I // 2), device=dev, dtype=torch.uint8)
    s1 = torch.rand(E, 2 * I, H // g, device=dev, dtype=torch.bfloat16) * 0.01
    s2 = torch.rand(E, H, I // g, device=dev, dtype=torch.bfloat16) * 0.01
    qc = int4_w4a16_moe_quant_config(w1_scale=s1, w2_scale=s2, block_shape=[0, g])
    out["vLLM int4 W4A16"] = lambda x, w, ids: fused_experts(x, q1, q2, w, ids, quant_config=qc)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experts", type=int, default=176)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=512)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--hot", type=float, default=0.10, help="share of experts in the 4-bit tier")
    ap.add_argument("--sweep", action="store_true", help="also time a few tile sizes per batch size")
    a = ap.parse_args(argv)
    assert torch.cuda.is_available(), "needs an NVIDIA (or ROCm) GPU; the CPU tests are in runtime/tests"
    dev = "cuda"
    E, H, I, k = a.experts, a.hidden, a.inter, a.topk
    hot = int(round(E * a.hot))
    print(f"{torch.cuda.get_device_name()} · {E} experts ({hot} 4-bit, {E - hot} 3-bit), hidden {H}, "
          f"expert width {I}, top-{k}")
    tiers, ref = build(E, H, I, hot, dev)
    ok = True
    for T in (1, 7, 300):
        ids, w = routing(T, E, k, dev, seed=T)
        x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
        got = hex_moe(x, w, ids, tiers).float()
        r = moe_reference(x, w, ids, ref)
        err = float((got - r).abs().max() / r.abs().max())
        print(f"  {T:4d} tokens: max rel error vs reference {err:.2e}")
        ok &= err < 2e-2

    base = vllm_baselines(E, H, I, dev)
    gb = 3 * H * I * (3.25 * (E - hot) + 4.25 * hot) / E / 8 / 1e9   # bytes per routed expert, HEX
    print("  per MoE layer, microseconds (lower is better):")
    print(f"  {'tokens':>7} {'HEX':>9} " + " ".join(f"{n:>16}" for n in base))
    for T in (1, 8, 32, 1024, 4096):
        ids, w = routing(T, E, k, dev, seed=T)
        x = (torch.randn(T, H, device=dev) * 0.5).to(torch.bfloat16)
        t_hex = timeit(lambda: hex_moe(x, w, ids, tiers), iters=20 if T > 512 else 100)
        row = f"  {T:7d} {t_hex:9.1f} "
        for fn in base.values():
            try:
                row += f"{timeit(lambda: fn(x, w, ids), iters=20 if T > 512 else 100):16.1f} "
            except Exception as e:  # noqa: BLE001
                row += f"{'err ' + type(e).__name__:>16} "
        if T == 1:
            row += f"   HEX weight traffic {k * gb / (t_hex * 1e-6):.0f} GB/s"
        print(row)
        if a.sweep:
            best = []
            for tiles in ("16,32,64", "16,64,64", "16,64,128", "16,128,64", "32,64,64", "64,64,64", "64,128,64",
                          "128,128,64", "64,64,128"):
                for warps in (4, 8):
                    os.environ["BLACKHOLE_HEX_TILES"] = f"{tiles},{warps},3"
                    try:
                        best.append((timeit(lambda: hex_moe(x, w, ids, tiers), iters=20), f"{tiles} w{warps}"))
                    except Exception:  # noqa: BLE001
                        pass
            os.environ.pop("BLACKHOLE_HEX_TILES", None)
            best.sort()
            print("          best tiles BM,BN,BK: " + ", ".join(f"{n} {t:.1f}" for t, n in best[:3]))
    print("HEX Triton kernel OK" if ok else "HEX Triton kernel MISMATCH: send this output")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
