"""Torch reference for HEX experts (Header-Encoded eXperts, BlackHole-Harness).

The layout is owned by blackhole.anatomy.hex; this module uses it when the root package is
installed and otherwise carries an identical copy so the pipeline (and its CI job) works alone.
Per tier, stacked over the tier's n experts, for one projection [out, in] (in a multiple of 32):
  idx           uint32 [n, out, in / 32 * bits]   `bits` (3 or 4) per weight, one little-endian
                                                  bit stream per block of 32 inputs = `bits` words
  block_scale   uint8  [n, out, in / 32]          fp8 e4m3 scale per block
  expert_scale  float32 [n]                       one scale per expert matrix
  levels        float32 [n, 2**bits]              the expert's own sorted level table
  weight[e, o, i] = levels[e, idx] * fp8(block_scale[e, o, i // 32]) * expert_scale[e]
"""

from __future__ import annotations

import torch

try:  # the source of truth, when the root blackhole package is installed
    from blackhole.anatomy.hex import HEX_BLOCK, HEX_SKIP, hex_dequant  # noqa: F401
except ImportError:
    HEX_BLOCK = 32
    HEX_SKIP = 0xFFFFFFFF

    def hex_dequant(idx, block_scale, expert_scale, levels, bits: int):
        """Weights [..., out, in] (float32) from stacked HEX parts with matching leading dims."""
        *lead, out, nw = idx.shape
        nblk = nw // bits
        w = idx.reshape(*lead, out, nblk, bits).to(torch.int64) & 0xFFFFFFFF
        vals = []
        for j in range(HEX_BLOCK):
            off = j * bits
            wi, sh = off // 32, off % 32
            v = w[..., wi] >> sh
            if sh + bits > 32:
                v = v | (w[..., wi + 1] << (32 - sh))
            vals.append(v & ((1 << bits) - 1))
        u = torch.stack(vals, -1).reshape(*lead, out, nblk * HEX_BLOCK)
        L = levels.shape[-1]
        lv = levels.reshape(-1, L).float()
        v = torch.gather(lv, 1, u.reshape(lv.shape[0], -1)).reshape(u.shape)
        c = block_scale.to(torch.int64)
        ex = ((c >> 3) & 15).float()
        mt = (c & 7).float()
        s = torch.where(ex == 0, mt / 8.0 * 0.015625, (1.0 + mt / 8.0) * torch.pow(2.0, ex - 7.0))
        s = torch.where((c & 128) > 0, -s, s) * expert_scale.reshape(*lead, 1, 1).float()
        return v * s.repeat_interleave(HEX_BLOCK, dim=-1)


def pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """[..., in] codes in [0, 2**bits) -> [..., in / 32 * bits] uint32 (the HEX bit stream)."""
    *lead, n = codes.shape
    x = codes.to(torch.int64).reshape(*lead, n // HEX_BLOCK, HEX_BLOCK)
    words = torch.zeros(*lead, n // HEX_BLOCK, bits, dtype=torch.int64, device=codes.device)
    for j in range(HEX_BLOCK):
        off = j * bits
        wi, sh = off // 32, off % 32
        words[..., wi] |= (x[..., j] << sh) & 0xFFFFFFFF
        if sh + bits > 32:
            words[..., wi + 1] |= x[..., j] >> (32 - sh)
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return words.reshape(*lead, -1).view(torch.uint32)


def random_parts(n: int, out: int, inp: int, bits: int, device="cpu", seed: int = 0) -> dict:
    """Random but realistic HEX parts for one projection of n experts (tests and benchmarks)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    codes = torch.randint(0, 2 ** bits, (n, out, inp), generator=g)
    return {
        "idx": pack(codes, bits).to(device),
        "block_scale": torch.randint(40, 70, (n, out, inp // HEX_BLOCK), generator=g, dtype=torch.int32)
        .to(torch.uint8).to(device),                                      # fp8 e4m3 ~ 0.1 .. 1
        "expert_scale": (torch.rand(n, generator=g) * 0.02 + 0.01).to(device),
        "levels": torch.sort(torch.rand(n, 2 ** bits, generator=g) * 2 - 1, dim=-1).values.to(device),
    }


def moe_reference(x, topk_weights, topk_ids, tiers: list[dict]) -> torch.Tensor:
    """Routed-expert output [T, H] of a SwiGLU MoE whose experts are split over HEX tiers.
    Each tier: {"bits", "experts": int ids [n], "gate_proj"/"up_proj"/"down_proj": parts}."""
    T, k = topk_ids.shape
    out = torch.zeros(T, x.shape[1], dtype=torch.float32, device=x.device)
    xf = x.float()
    for t in tiers:
        W = {p: hex_dequant(*(t[p][q] for q in ("idx", "block_scale", "expert_scale", "levels")), t["bits"])
             for p in ("gate_proj", "up_proj", "down_proj")}
        for local, e in enumerate(t["experts"].tolist()):
            tok, slot = (topk_ids == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            a = xf[tok]
            h = torch.nn.functional.silu(a @ W["gate_proj"][local].T) * (a @ W["up_proj"][local].T)
            h = h.to(x.dtype).float()   # the kernel hands the activation over in the model's dtype
            out.index_add_(0, tok, (h @ W["down_proj"][local].T) * topk_weights[tok, slot, None].float())
    return out
