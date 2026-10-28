"""Triton fused MoE over HEX experts (Header-Encoded eXperts, BlackHole-Harness).

The NVIDIA counterpart of flint_mlx.py's Metal kernels (layout: hex_ref.py /
blackhole.anatomy.hex): 3- or 4-bit indices from the packed bit stream, the expert's own
8/16-level table, the fp8 e4m3 block scale and the expert scale.

Two ways through a MoE layer, picked by the number of (token, slot) pairs S:

fused (decoding and short prompts): a grouped GEMM over the chosen experts that decodes HEX
  tiles in registers, two launches per tier:
    glu   act[slot] = silu(x[tok] @ gate[e].T) * (x[tok] @ up[e].T)     both GEMMs in one program
    down  y[slot]   = (act[slot] @ down[e].T) * topk_weight[slot]
  then y is summed over each token's k slots. The level lookup is a select tree over the
  table held in registers (no memory gather per weight). Slots are grouped per expert and
  padded to BLOCK_M: for a handful of slots each gets its own block with no preparation at
  all; otherwise vLLM's moe_align CUDA op when vLLM is installed, else a torch version (no
  host sync either way, so CUDA graphs capture it). A tier is a set of experts sharing one
  index width (REAQ: hot experts 16-level / 4-bit, the rest 8-level / 3-bit); blocks of
  another tier's experts exit at once.

dequant (long prompts, S >= dequant_min): every expert is decoded once into a bf16 scratch
  buffer and vLLM's bf16 fused MoE runs on it (compute-bound work belongs on plain tensor
  cores; the decode cost is paid once per expert instead of once per token tile).

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
def _pick(b, hi, lo):
    return tl.where(b, hi, lo)


@triton.jit
def _lookup(lev_ptr, code, BITS: tl.constexpr):
    """levels[code] by a select tree over the table in registers (8 or 16 scalars)."""
    b0 = (code & 1) != 0
    b1 = (code & 2) != 0
    b2 = (code & 4) != 0
    v0 = _pick(b1, _pick(b0, tl.load(lev_ptr + 3), tl.load(lev_ptr + 2)),
               _pick(b0, tl.load(lev_ptr + 1), tl.load(lev_ptr + 0)))
    v1 = _pick(b1, _pick(b0, tl.load(lev_ptr + 7), tl.load(lev_ptr + 6)),
               _pick(b0, tl.load(lev_ptr + 5), tl.load(lev_ptr + 4)))
    v = _pick(b2, v1, v0)
    if BITS == 4:
        v2 = _pick(b1, _pick(b0, tl.load(lev_ptr + 11), tl.load(lev_ptr + 10)),
                   _pick(b0, tl.load(lev_ptr + 9), tl.load(lev_ptr + 8)))
        v3 = _pick(b1, _pick(b0, tl.load(lev_ptr + 15), tl.load(lev_ptr + 14)),
                   _pick(b0, tl.load(lev_ptr + 13), tl.load(lev_ptr + 12)))
        v = _pick((code & 8) != 0, _pick(b2, v3, v2), v)
    return v


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
    code = (tl.where(use_lo[None, :, None], lo[:, None, :] >> sh_lo[None, :, None],
                     hi[:, None, :] >> sh_hi[None, :, None]) & (L - 1)).to(tl.int32)  # [NB, 32, BN]
    val = _lookup(lev_ptr + e * L, code, BITS)
    c = tl.load(bs_ptr + e * s_be + offs_n[None, :] * s_bn + nb[:, None], mask=m, other=0).to(tl.int32)
    w = val * _fp8_e4m3(c)[:, None, :]
    return tl.reshape(w, (NB * 32, BLOCK_N))


@triton.jit
def _hex_moe_kernel(a_ptr, out_ptr,
                    g_idx, g_bs, g_es, g_lev, u_idx, u_bs, u_es, u_lev,
                    sorted_ptr, expert_ptr, lookup_ptr, npp_ptr, topk_w_ptr,
                    num_slots, N, K, s_am, s_om, s_ie, s_in, s_be, s_bn,
                    A_DIV: tl.constexpr, GLU: tl.constexpr, MUL_W: tl.constexpr, BITS: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n
    if pid_m * BLOCK_M >= tl.load(npp_ptr):
        return
    eg = tl.load(expert_ptr + pid_m)                      # global expert id of this block (-1: none)
    if eg < 0:
        return
    e = tl.load(lookup_ptr + eg).to(tl.int64)             # its index in this tier (-1: another tier)
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


@triton.jit
def _hex_dequant_kernel(idx_ptr, bs_ptr, es_ptr, lev_ptr, ids_ptr, out_ptr,
                        N, row_off, s_ie, s_in, s_be, s_bn, s_oe, s_on,
                        BITS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """out[ids[e], row_off + n, k] = W[e, n, k] (one tier's experts into the full bf16 stack)."""
    e = tl.program_id(0).to(tl.int64)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    k0 = tl.program_id(2) * BLOCK_K
    n_mask = offs_n < N
    w = _hex_tile(idx_ptr, bs_ptr, lev_ptr, e, offs_n, n_mask, k0 // 32, s_ie, s_in, s_be, s_bn,
                  BITS, BLOCK_K // 32, BLOCK_N) * tl.load(es_ptr + e)                  # [BK, BN]
    g = tl.load(ids_ptr + e).to(tl.int64)
    offs_k = k0 + tl.arange(0, BLOCK_K)
    tl.store(out_ptr + g * s_oe + (row_off + offs_n)[None, :] * s_on + offs_k[:, None],
             w.to(out_ptr.dtype.element_ty), mask=n_mask[None, :])


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
        self.experts = self.experts.to(dev).to(torch.int32)
        return self


def align(ids: torch.Tensor, n: int, block_m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group slots by expert, each expert's run padded to block_m, without a host sync.
    ids: int [S] global expert ids. Returns sorted slot ids int32 [B * block_m] (S = padding),
    the expert of every block int32 [B] (-1 = unused) and the padded length int32 [1]."""
    S = ids.numel()
    dev = ids.device
    e = ids.long()
    counts = torch.zeros(n, dtype=torch.long, device=dev).scatter_add_(0, e, torch.ones_like(e))
    nblk = (counts + block_m - 1) // block_m
    max_blocks = -(-S // block_m) + min(n, S)
    blk_end = torch.cumsum(nblk, 0)
    start = (blk_end - nblk) * block_m
    order = torch.argsort(e, stable=True)
    se = e[order]
    first = torch.cumsum(counts, 0) - counts
    pos = start[se] + torch.arange(S, device=dev) - first[se]
    sorted_ids = torch.full((max_blocks * block_m,), S, dtype=torch.int32, device=dev)
    sorted_ids.scatter_(0, pos, order.to(torch.int32))
    b = torch.arange(max_blocks, device=dev)
    expert_ids = torch.searchsorted(blk_end, b, right=True)
    expert_ids = torch.where(b < blk_end[-1], expert_ids, -1).to(torch.int32)
    return sorted_ids, expert_ids, (blk_end[-1:] * block_m).to(torch.int32)


_DIRECT: dict = {}


def direct(ids: torch.Tensor, block_m: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One block per slot (few slots: no sorting at all). Same outputs as align()."""
    S = ids.numel()
    key = (S, block_m, ids.device)
    if key not in _DIRECT:
        s = torch.full((S, block_m), S, dtype=torch.int32, device=ids.device)
        s[:, 0] = torch.arange(S, dtype=torch.int32, device=ids.device)
        _DIRECT[key] = (s.reshape(-1), torch.full((1,), S * block_m, dtype=torch.int32, device=ids.device))
    sid, npp = _DIRECT[key]
    return sid, ids.to(torch.int32), npp


def _vllm_align():
    try:
        from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
        return moe_align_block_size
    except Exception:  # noqa: BLE001
        return None


def plan(ids: torch.Tensor, n: int, block_m: int, mode: str = "auto"):
    """(sorted slot ids, block experts, padded length) for the fused kernels."""
    if mode == "direct" or (mode == "auto" and ids.numel() <= DIRECT_MAX):
        return direct(ids, block_m)
    if mode in ("auto", "vllm") and ids.is_cuda:
        fn = _vllm_align()
        if fn is not None:
            sid, eid, npp = fn(ids.reshape(1, -1).to(torch.int32), block_m, n)
            return sid, eid, npp
    return align(ids, n, block_m)


DIRECT_MAX = int(os.environ.get("BLACKHOLE_HEX_DIRECT_MAX", "32"))      # slots: 4 tokens at top-8
DEQUANT_MIN = int(os.environ.get("BLACKHOLE_HEX_DEQUANT_MIN", "1024"))  # slots: 128 tokens at top-8


def default_config(slots: int, n_out: int, n_in: int, glu: bool) -> dict:
    """Tile sizes (from the B300 sweep: decode wants narrow N tiles for more programs, prompts
    wider M tiles). BLOCK_K divides n_in."""
    env = os.environ.get("BLACKHOLE_HEX_TILES")   # "BM,BN,BK[,warps,stages]" overrides (benchmarking)
    if env:
        v = [int(t) for t in env.split(",")]
        cfg = dict(zip(("BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages"), v))
        cfg.setdefault("num_warps", 4)
        cfg.setdefault("num_stages", 2)
        return cfg
    bk = 64 if n_in % 64 == 0 else 32
    if slots <= DIRECT_MAX:
        bm, bn = 16, 32
    elif slots <= 512:
        bm, bn = 16, 64
    elif slots <= 4096:
        bm, bn = 32, 64
    else:
        bm, bn = 128, 128
    return {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": 4, "num_stages": 2}


def _launch(a, out, w, w2, sid, eid, npp, lookup, topk_w, slots, a_div, bits, cfg):
    n_out, nw = w["idx"].shape[1], w["idx"].shape[2]
    n_in = nw // bits * HEX_BLOCK
    assert a.shape[1] == n_in, f"input width {a.shape[1]} != HEX width {n_in}"
    assert n_in % cfg["BLOCK_K"] == 0 and cfg["BLOCK_K"] % 32 == 0
    glu = w2 is not None
    u = w2 or w
    grid = (eid.numel() * triton.cdiv(n_out, cfg["BLOCK_N"]),)
    _hex_moe_kernel[grid](
        a, out, *(w[p] for p in PARTS), *(u[p] for p in PARTS),
        sid, eid, lookup, npp, topk_w if topk_w is not None else a,
        slots, n_out, n_in, a.stride(0), out.stride(0),
        w["idx"].stride(0), w["idx"].stride(1), w["block_scale"].stride(0), w["block_scale"].stride(1),
        A_DIV=a_div, GLU=glu, MUL_W=topk_w is not None, BITS=bits,
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])


def hex_moe_fused(x, topk_weights, topk_ids, tiers: list[HexTier], config: dict | None = None,
                  align_mode: str = "auto") -> torch.Tensor:
    """Routed-expert output [T, H] through the fused HEX kernels."""
    T, k = topk_ids.shape
    S = T * k
    H = x.shape[1]
    I = tiers[0].gate_proj["idx"].shape[1]
    E = tiers[0].lookup.numel()
    x = x.contiguous()
    flat = topk_ids.reshape(-1)
    w = topk_weights.reshape(-1).contiguous()
    act = torch.empty(S, I, dtype=x.dtype, device=x.device)
    y = torch.zeros(S, H, dtype=x.dtype, device=x.device)
    c1 = config or default_config(S, I, H, True)
    c2 = config or default_config(S, H, I, False)
    p1 = plan(flat, E, c1["BLOCK_M"], align_mode)
    p2 = p1 if c2["BLOCK_M"] == c1["BLOCK_M"] else plan(flat, E, c2["BLOCK_M"], align_mode)
    for t in tiers:
        _launch(x, act, t.gate_proj, t.up_proj, *p1, t.lookup, None, S, k, t.bits, c1)
    for t in tiers:
        _launch(act, y, t.down_proj, None, *p2, t.lookup, w, S, 1, t.bits, c2)
    return y.view(T, k, H).sum(1)


def dequant_all(tiers: list[HexTier], dtype=torch.bfloat16) -> tuple[torch.Tensor, torch.Tensor]:
    """All experts as bf16 stacks: w13 [E, 2I, H] (gate rows, then up) and w2 [E, H, I]."""
    E = tiers[0].lookup.numel()
    I, Hw = tiers[0].gate_proj["idx"].shape[1:]
    H = Hw // tiers[0].bits * HEX_BLOCK
    dev = tiers[0].gate_proj["idx"].device
    w13 = torch.empty(E, 2 * I, H, dtype=dtype, device=dev)
    w2 = torch.empty(E, H, I, dtype=dtype, device=dev)
    BN, BK = 32, 64
    for t in tiers:
        n = len(t.experts)
        for p, out, row_off in ((t.gate_proj, w13, 0), (t.up_proj, w13, I), (t.down_proj, w2, 0)):
            N, K = p["idx"].shape[1], p["idx"].shape[2] // t.bits * HEX_BLOCK
            bk = BK if K % BK == 0 else 32
            _hex_dequant_kernel[(n, triton.cdiv(N, BN), K // bk)](
                *(p[q] for q in PARTS), t.experts, out, N, row_off,
                p["idx"].stride(0), p["idx"].stride(1), p["block_scale"].stride(0), p["block_scale"].stride(1),
                out.stride(0), out.stride(1), BITS=t.bits, BLOCK_N=BN, BLOCK_K=bk)
    return w13, w2


def _vllm_fused_experts():
    try:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
        return fused_experts
    except Exception:  # noqa: BLE001
        return None


def hex_moe_dequant(x, topk_weights, topk_ids, tiers: list[HexTier]) -> torch.Tensor:
    """Routed-expert output [T, H]: decode every expert to bf16 once, then a bf16 fused MoE
    (vLLM's when installed, else this module's kernels can't take bf16, so torch per expert)."""
    w13, w2 = dequant_all(tiers, x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.float32)
    fe = _vllm_fused_experts() if x.is_cuda else None
    if fe is not None:
        return fe(x.contiguous(), w13, w2, topk_weights.float(), topk_ids.to(torch.int32))
    T, k = topk_ids.shape
    I = w2.shape[-1]
    out = torch.zeros(T, x.shape[1], dtype=torch.float32, device=x.device)
    for s in range(k):
        h = torch.einsum("th,toh->to", x.to(w13.dtype), w13[topk_ids[:, s].long()]).float()
        a = (torch.nn.functional.silu(h[:, :I]) * h[:, I:]).to(x.dtype).to(w2.dtype)
        out += torch.einsum("ti,thi->th", a, w2[topk_ids[:, s].long()]).float() * topk_weights[:, s, None].float()
    return out.to(x.dtype)


def hex_moe(x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, tiers: list[HexTier],
            config: dict | None = None, path: str = "auto") -> torch.Tensor:
    """Routed-expert output [T, H] of a SwiGLU MoE layer whose experts are HEX tiers (prepared).
    path: "auto" (fused below DEQUANT_MIN slots, else dequant), "fused" or "dequant"."""
    capturing = x.is_cuda and torch.cuda.is_current_stream_capturing()   # no bf16 scratch inside CUDA graphs
    if path == "dequant" or (path == "auto" and config is None and not capturing
                             and topk_ids.numel() >= DEQUANT_MIN):
        return hex_moe_dequant(x, topk_weights, topk_ids, tiers)
    return hex_moe_fused(x, topk_weights, topk_ids, tiers, config)
