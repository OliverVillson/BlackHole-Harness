"""Triton fused MoE over HEX experts (Header-Encoded eXperts, BlackHole-Harness).

The NVIDIA counterpart of flint_mlx.py's Metal kernels: a grouped GEMM over the experts the
router chose, dequantizing HEX weights tile by tile in registers (no bf16 copy of the experts):
3- or 4-bit indices from the packed bit stream, the expert's own 8/16-level table, the fp8
e4m3 block scale and the expert scale (layout: hex_ref.py / blackhole.anatomy.hex).

One MoE layer = two launches per tier:
  glu   act[slot] = silu(x[tok] @ gate[e].T) * (x[tok] @ up[e].T)     both GEMMs in one program
  down  y[slot]   = (act[slot] @ down[e].T) * topk_weight[slot]
then y is summed over the k slots of each token. Slots are grouped by expert and padded to
BLOCK_M (the moe_align step, here in plain torch with no host sync, so CUDA graphs can
capture it); each program owns BLOCK_M slots of one expert and BLOCK_N output rows. A tier
is a set of experts sharing one index width (REAQ: the hot experts 16-level / 4-bit, the rest
8-level / 3-bit); slots routed to another tier are simply not in this tier's blocks.

Runs on CUDA (and ROCm) GPUs; on a CPU with TRITON_INTERPRET=1 for the tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from blackhole_runtime.hex_ref import HEX_BLOCK

PARTS = ("idx", "block_scale", "expert_scale", "levels")


@triton.jit
def _fp8_e4m3(c):
    """Decode e4m3 codes (int32 tensor) to float32."""
    ex = (c >> 3) & 15
    mt = (c & 7).to(tl.float32)
    s = tl.where(ex == 0, mt * 0.001953125, (1.0 + mt * 0.125) * tl.exp2((ex - 7).to(tl.float32)))
    return tl.where((c & 128) != 0, -s, s)


@triton.jit
def _hex_tile(idx_ptr, bs_ptr, lev_ptr, e, offs_n, n_mask, nb0, s_ie, s_in, s_be, s_bn,
              BITS: tl.constexpr, NB: tl.constexpr, BLOCK_N: tl.constexpr):
    """W[e, offs_n, nb0*32 : (nb0+NB)*32].T as float32 [NB*32, BLOCK_N], block scales applied
    (the expert scale is applied once to the accumulator)."""
    L: tl.constexpr = 1 << BITS
    nb = nb0 + tl.arange(0, NB)
    m = n_mask[None, :]
    wp = idx_ptr + e * s_ie + offs_n[None, :] * s_in + nb[:, None] * BITS          # [NB, BN] word 0
    w0 = tl.load(wp, mask=m, other=0).to(tl.int64) & 0xFFFFFFFF
    w1 = tl.load(wp + 1, mask=m, other=0).to(tl.int64) & 0xFFFFFFFF
    w2 = tl.load(wp + 2, mask=m, other=0).to(tl.int64) & 0xFFFFFFFF
    j = tl.arange(0, 32)
    off = j * BITS
    lo = w0 | (w1 << 32)                                    # stream bits 0..63
    if BITS == 4:
        w3 = tl.load(wp + 3, mask=m, other=0).to(tl.int64) & 0xFFFFFFFF
        hi = w2 | (w3 << 32)                                # bits 64..127
        use_lo = off < 64
        sh_hi = tl.maximum(off - 64, 0)
    else:
        hi = w1 | (w2 << 32)                                # bits 32..95
        use_lo = off + BITS <= 64
        sh_hi = tl.maximum(off - 32, 0)
    sh_lo = tl.minimum(off, 63)
    code = tl.where(use_lo[None, :, None], lo[:, None, :] >> sh_lo[None, :, None],
                    hi[:, None, :] >> sh_hi[None, :, None]) & (L - 1)               # [NB, 32, BN]
    val = tl.load(lev_ptr + e * L + code)
    c = tl.load(bs_ptr + e * s_be + offs_n[None, :] * s_bn + nb[:, None], mask=m, other=0).to(tl.int32)
    w = val * _fp8_e4m3(c)[:, None, :]
    return tl.reshape(w, (NB * 32, BLOCK_N))


@triton.jit
def _hex_moe_kernel(a_ptr, out_ptr,
                    g_idx, g_bs, g_es, g_lev, u_idx, u_bs, u_es, u_lev,
                    sorted_ptr, expert_ptr, topk_w_ptr,
                    num_slots, N, K, s_am, s_om, s_ie, s_in, s_be, s_bn,
                    A_DIV: tl.constexpr, GLU: tl.constexpr, MUL_W: tl.constexpr, BITS: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n
    e = tl.load(expert_ptr + pid_m).to(tl.int64)
    if e < 0:
        return
    slot = tl.load(sorted_ptr + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    m_mask = slot < num_slots
    tok = slot // A_DIV
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_k = tl.arange(0, BLOCK_K)
    NB: tl.constexpr = BLOCK_K // 32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + tok[:, None] * s_am + (k0 + offs_k)[None, :], mask=m_mask[:, None], other=0.0)
        b = _hex_tile(g_idx, g_bs, g_lev, e, offs_n, n_mask, k0 // 32, s_ie, s_in, s_be, s_bn, BITS, NB, BLOCK_N)
        acc += tl.dot(a, b.to(a.dtype))
        if GLU:
            b2 = _hex_tile(u_idx, u_bs, u_lev, e, offs_n, n_mask, k0 // 32, s_ie, s_in, s_be, s_bn,
                           BITS, NB, BLOCK_N)
            acc2 += tl.dot(a, b2.to(a.dtype))
    acc = acc * tl.load(g_es + e)
    if GLU:
        acc = acc / (1.0 + tl.exp(-acc)) * (acc2 * tl.load(u_es + e))
    if MUL_W:
        acc = acc * tl.load(topk_w_ptr + slot, mask=m_mask, other=0.0).to(tl.float32)[:, None]
    tl.store(out_ptr + slot[:, None] * s_om + offs_n[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=m_mask[:, None] & n_mask[None, :])


# --------------------------------------------------------------------------- host side


@dataclass
class HexTier:
    """One tier of a MoE layer: experts sharing one index width, stacked in the HEX layout."""
    bits: int
    experts: torch.Tensor              # int global expert ids [n]
    gate_proj: dict                    # {"idx", "block_scale", "expert_scale", "levels"}
    up_proj: dict
    down_proj: dict
    lookup: torch.Tensor | None = None  # int32 [num_experts]: local index in this tier or -1

    def prepare(self, num_experts: int) -> "HexTier":
        """Kernel-ready views (uint32 words as int32) and the global -> local expert map."""
        for p in (self.gate_proj, self.up_proj, self.down_proj):
            if p["idx"].dtype == torch.uint32:
                p["idx"] = p["idx"].view(torch.int32)
            p["expert_scale"] = p["expert_scale"].float().reshape(-1).contiguous()
            p["levels"] = p["levels"].float().contiguous()
            for k in ("idx", "block_scale"):
                p[k] = p[k].contiguous()
        dev = self.gate_proj["idx"].device
        lk = torch.full((num_experts,), -1, dtype=torch.int32, device=dev)
        lk[self.experts.to(dev).long()] = torch.arange(len(self.experts), dtype=torch.int32, device=dev)
        self.lookup = lk
        return self


def align(local: torch.Tensor, n: int, block_m: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Group slots by (local) expert, each expert's run padded to block_m, without a host sync.
    local: int [S] (-1 = not this tier). Returns sorted slot ids int32 [B * block_m] (S = padding)
    and the expert of every block int32 [B] (-1 = unused), B a static upper bound."""
    S = local.numel()
    dev = local.device
    e = torch.where(local < 0, n, local).long()
    counts = torch.zeros(n + 1, dtype=torch.long, device=dev).scatter_add_(0, e, torch.ones_like(e))[:n]
    nblk = (counts + block_m - 1) // block_m
    max_blocks = -(-S // block_m) + min(n, S)
    blk_end = torch.cumsum(nblk, 0)
    start = (blk_end - nblk) * block_m
    order = torch.argsort(e, stable=True)
    se = e[order]
    sec = se.clamp(max=n - 1)
    first = torch.cumsum(counts, 0) - counts
    pos = start[sec] + torch.arange(S, device=dev) - first[sec]
    pos = torch.where(se < n, pos, max_blocks * block_m)
    sorted_ids = torch.full((max_blocks * block_m + 1,), S, dtype=torch.int32, device=dev)
    sorted_ids.scatter_(0, pos, order.to(torch.int32))
    b = torch.arange(max_blocks, device=dev)
    expert_ids = torch.searchsorted(blk_end, b, right=True)
    expert_ids = torch.where(b < blk_end[-1], expert_ids, -1).to(torch.int32)
    return sorted_ids[:-1], expert_ids


def default_config(slots: int, n_out: int, n_in: int, glu: bool) -> dict:
    """Tile sizes: small M tiles for decoding, bigger for prompts. BLOCK_K divides n_in."""
    env = os.environ.get("BLACKHOLE_HEX_TILES")   # "BM,BN,BK[,warps,stages]" overrides (benchmarking)
    if env:
        v = [int(t) for t in env.split(",")]
        cfg = dict(zip(("BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages"), v))
        cfg.setdefault("num_warps", 4)
        cfg.setdefault("num_stages", 2)
        return cfg
    bm = 16 if slots <= 128 else 64
    bk = 64 if n_in % 64 == 0 else 32
    bn = 64 if (glu or bm == 16) else 128
    return {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": 4, "num_stages": 2}


def _launch(a, out, w, w2, sorted_ids, expert_ids, topk_w, slots, a_div, bits, cfg):
    n_out, nw = w["idx"].shape[1], w["idx"].shape[2]
    n_in = nw // bits * HEX_BLOCK
    assert a.shape[1] == n_in, f"input width {a.shape[1]} != HEX width {n_in}"
    assert n_in % cfg["BLOCK_K"] == 0 and cfg["BLOCK_K"] % 32 == 0
    glu = w2 is not None
    u = w2 or w
    grid = (expert_ids.numel() * triton.cdiv(n_out, cfg["BLOCK_N"]),)
    _hex_moe_kernel[grid](
        a, out, *(w[p] for p in PARTS), *(u[p] for p in PARTS),
        sorted_ids, expert_ids, topk_w if topk_w is not None else a,
        slots, n_out, n_in, a.stride(0), out.stride(0),
        w["idx"].stride(0), w["idx"].stride(1), w["block_scale"].stride(0), w["block_scale"].stride(1),
        A_DIV=a_div, GLU=glu, MUL_W=topk_w is not None, BITS=bits,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])


def hex_moe(x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, tiers: list[HexTier],
            config: dict | None = None) -> torch.Tensor:
    """Routed-expert output [T, H] of a SwiGLU MoE layer whose experts are HEX tiers (prepared)."""
    T, k = topk_ids.shape
    S = T * k
    H = x.shape[1]
    I = tiers[0].gate_proj["idx"].shape[1]
    x = x.contiguous()
    flat = topk_ids.reshape(-1)
    w = topk_weights.reshape(-1).contiguous()
    act = torch.empty(S, I, dtype=x.dtype, device=x.device)
    y = torch.zeros(S, H, dtype=x.dtype, device=x.device)
    c1 = config or default_config(S, I, H, True)
    c2 = config or default_config(S, H, I, False)
    plans = []
    for t in tiers:
        local = t.lookup[flat]
        s1 = align(local, len(t.experts), c1["BLOCK_M"])
        s2 = s1 if c2["BLOCK_M"] == c1["BLOCK_M"] else align(local, len(t.experts), c2["BLOCK_M"])
        plans.append((t, s1, s2))
    for t, (sid, eid), _ in plans:
        _launch(x, act, t.gate_proj, t.up_proj, sid, eid, None, S, k, t.bits, c1)
    for t, _, (sid, eid) in plans:
        _launch(act, y, t.down_proj, None, sid, eid, w, S, 1, t.bits, c2)
    return y.view(T, k, H).sum(1)
