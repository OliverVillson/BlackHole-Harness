"""HEX (Header-Encoded eXperts) and REAQ, rebuilt on the anatomy API in numpy.

HEX stores an expert matrix W [out, in] as
  header   one fp32 expert scale  es = max|W| / 448, and the expert's own sorted table of
           L = 8 or 16 levels in [-1, 1], learned by weighted k-means on the block-normalised
           weights (weights = each input channel's activation energy when known)
  blocks   one fp8 (e4m3) scale per `block` inputs: fp8(max|w_block| / es)
  indices  log2(L) bits per weight into the table
  weight[o, i] = levels[idx[o, i]] * fp8(block_scale[o, i // block]) * es

That is 3.25 bpw at 8 levels and 4.25 at 16 with 32-weight blocks (3.5 / 4.5 at 16-weight
blocks). Rounding is nearest-level, or GPTQ onto the table when an input Hessian is given:
each column's rounding error is pushed onto the columns still to come through H^-1, so the
layer output error is minimised, not the weight error (the recipe that measured KL 0.054 at
3.25 bpw on Qwen3-30B REAP 25% against 0.083 for GPTQ int3).

REAQ (Router-weighted Expert Activation Quantization) chooses L per expert from the REAP score
(Model.reaq marks the experts; quantize() reads the marks), so the hot experts of a layer get
the 16-level tables and the average lands where the budget says, with no expert removed.

The packed layout (Hex.packed / unpack_expert / hex_dequant) is the one the kernels read
(flint_mlx.py's Metal kernel, the Triton kernel in harness/blackhole_pipeline/kernels/), per
expert matrix [out, in] with in padded to a multiple of HEX_BLOCK = 32:
  idx           uint32 [out, in / 32 * bits]   indices, `bits` (3 or 4) per weight, one little-endian
                                               bit stream per block of 32 inputs = `bits` words
  block_scale   uint8  [out, in / 32]          fp8 e4m3 scale per block
  expert_scale  float32 [1]                    the matrix scale
  levels        float32 [2**bits]              the expert's sorted level table in [-1, 1]
  weight[o, i] = levels[idx[o, i]] * fp8(block_scale[o, i // 32]) * expert_scale
This module is the source of truth for that layout; HEX_SKIP marks an index word that belongs
to no expert of a tier (a slot served by another tier).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from blackhole.anatomy import st

FP8_MAX = 448.0
HEX_BLOCK = 32
HEX_SKIP = 0xFFFFFFFF


def bpw(levels: int, block: int) -> float:
    return math.log2(levels) + 8 / block


# --------------------------------------------------------------------------- the header


def _fp8(x: np.ndarray) -> np.ndarray:
    """Round positive scales to e4m3 and back (what the kernel will multiply by)."""
    return st.decode(st.encode(x, "F8_E4M3"), "F8_E4M3")


def block_scales(W: np.ndarray, block: int) -> tuple[float, np.ndarray, np.ndarray]:
    """(expert scale, fp8-rounded dequant block scales [out, nblocks], block-normalised W)."""
    out, n = W.shape
    pad = -n % block
    x = np.pad(W.astype(np.float32), ((0, 0), (0, pad))).reshape(out, -1, block)
    amax = np.abs(x).max(-1)
    es = max(float(amax.max()) / FP8_MAX, 1e-30)
    blk = _fp8(np.minimum(amax / es, FP8_MAX)) * es
    blk = np.where(blk > 0, blk, 1.0).astype(np.float32)
    xn = np.clip(x / blk[..., None], -1, 1).reshape(out, -1)[:, :n]
    return es, blk, xn


def fit_table(xn: np.ndarray, L: int, weights: np.ndarray | None = None, iters: int = 20,
              sample: int = 1 << 17, seed: int = 0) -> np.ndarray:
    """Weighted 1-D k-means (Lloyd) on block-normalised values; returns L sorted levels."""
    x = xn.ravel().astype(np.float32)
    w = None if weights is None else weights.ravel().astype(np.float32)
    if x.size > sample:
        idx = np.random.default_rng(seed).integers(0, x.size, sample)
        x = x[idx]
        w = w[idx] if w is not None else None
    if w is None:
        w = np.ones_like(x)
    lev = np.quantile(x, np.linspace(0, 1, L + 2)[1:-1]).astype(np.float32)
    for _ in range(iters):
        a = np.searchsorted((lev[1:] + lev[:-1]) / 2, x)
        num = np.bincount(a, weights=x * w, minlength=L)
        den = np.bincount(a, weights=w, minlength=L)
        lev = np.sort(np.where(den > 0, num / np.maximum(den, 1e-30), lev)).astype(np.float32)
    return lev


def nearest(xn: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Index of the nearest level for every block-normalised value."""
    return np.searchsorted((levels[1:] + levels[:-1]) / 2, np.clip(xn, -1, 1)).astype(np.uint8)


@dataclass
class Hex:
    """One expert matrix in HEX form."""
    levels: np.ndarray       # [L] float32, sorted
    expert_scale: float
    block_scale: np.ndarray  # [out, nblocks] fp8 codes as uint8
    idx: np.ndarray          # [out, in] uint8 indices (unpacked)
    block: int
    shape: tuple

    @property
    def L(self) -> int:
        return len(self.levels)

    @property
    def bpw(self) -> float:
        return bpw(self.L, self.block)

    def scales(self) -> np.ndarray:
        """Per-element dequant scales [out, in]."""
        out, n = self.shape
        s = st.decode(self.block_scale, "F8_E4M3") * np.float32(self.expert_scale)
        return np.repeat(s, self.block, axis=1)[:, :n]

    def dequant(self) -> np.ndarray:
        return (self.levels[self.idx] * self.scales()).astype(np.float32)

    @property
    def bits(self) -> int:
        return int(math.log2(self.L))

    def packed(self) -> dict:
        """The saved parts (see the module docstring): idx uint32 bit stream, block_scale uint8,
        expert_scale float32 [1], levels float32 [L]. Inputs are padded to a multiple of the block."""
        return {"idx": pack_bits(self.idx, self.bits, self.block), "block_scale": self.block_scale,
                "expert_scale": np.array([self.expert_scale], np.float32), "levels": self.levels}


def pack_bits(idx: np.ndarray, bits: int, block: int = HEX_BLOCK) -> np.ndarray:
    """[out, in] small ints -> uint32 [out, ceil(in / block) * block * bits / 32]: per block of
    `block` inputs one little-endian bit stream, value i at bits i*bits .. i*bits+bits-1."""
    out, n = idx.shape
    pad = -n % block
    x = np.pad(idx.astype(np.uint64), ((0, 0), (0, pad))).reshape(out, -1, block)
    nwords = block * bits // 32
    words = np.zeros((out, x.shape[1], nwords), np.uint64)
    for i in range(block):
        off = i * bits
        wi, sh = off // 32, off % 32
        words[:, :, wi] |= (x[:, :, i] << sh) & 0xFFFFFFFF
        if sh + bits > 32:
            words[:, :, wi + 1] |= x[:, :, i] >> (32 - sh)
    return words.reshape(out, -1).astype(np.uint32)


def unpack_bits(words: np.ndarray, bits: int, n: int, block: int = HEX_BLOCK) -> np.ndarray:
    """Inverse of pack_bits: uint32 [out, nblocks * block * bits / 32] -> uint8 [out, n]."""
    out = words.shape[0]
    nwords = block * bits // 32
    w = words.reshape(out, -1, nwords).astype(np.uint64)
    vals = []
    for i in range(block):
        off = i * bits
        wi, sh = off // 32, off % 32
        v = w[:, :, wi] >> sh
        if sh + bits > 32:
            v |= w[:, :, wi + 1] << (32 - sh)
        vals.append(v & ((1 << bits) - 1))
    return np.stack(vals, -1).reshape(out, -1)[:, :n].astype(np.uint8)


def unpack_expert(parts: dict, shape: tuple, bits: int | None = None, block: int = HEX_BLOCK) -> Hex:
    """A Hex back from saved parts (one expert: idx [out, words], block_scale [out, nblk],
    expert_scale [1] or scalar, levels [L])."""
    out, n = shape
    levels = np.asarray(parts["levels"], np.float32).reshape(-1)
    bits = bits or int(math.log2(len(levels)))
    idx = unpack_bits(np.asarray(parts["idx"], np.uint32), bits, n, block)
    return Hex(levels, float(np.asarray(parts["expert_scale"]).ravel()[0]),
               np.asarray(parts["block_scale"], np.uint8), idx, block, (out, n))


def hex_dequant(idx, block_scale, expert_scale, levels, bits: int):
    """Torch reference with the kernels' signature: weights [..., out, in] (float32) from stacked
    HEX parts with matching leading dims (idx [..., out, in/32*bits] uint32, block_scale [..., out,
    in/32] uint8, expert_scale [...], levels [..., 2**bits])."""
    import torch
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
    v = torch.gather(levels.reshape(-1, L).float(), 1, u.reshape(len(levels.reshape(-1, L)), -1)).reshape(u.shape)
    c = block_scale.to(torch.int64)
    ex = ((c >> 3) & 15).float()
    mt = (c & 7).float()
    s = torch.where(ex == 0, mt / 8.0 * 0.015625, (1.0 + mt / 8.0) * torch.pow(2.0, ex - 7.0))
    s = torch.where((c & 128) > 0, -s, s) * expert_scale.reshape(*lead, 1, 1).float()
    return v * s.repeat_interleave(HEX_BLOCK, dim=-1)


def quantize(W: np.ndarray, L: int = 8, block: int = HEX_BLOCK, energy: np.ndarray | None = None,
             H: np.ndarray | None = None, damp: float = 0.01) -> Hex:
    """HEX-quantize one [out, in] matrix. energy [in] (mean x^2 per input channel) weights the
    table fit; H [in, in] (sum x x^T) switches rounding from nearest to GPTQ."""
    W = np.asarray(W, np.float32)
    if W.ndim != 2:
        raise ValueError(f"HEX quantizes 2-D matrices, got shape {W.shape}")
    es, blk, xn = block_scales(W, block)
    out, n = W.shape
    scales = np.repeat(blk, block, axis=1)[:, :n]
    wts = None
    if energy is not None:
        wts = (np.asarray(energy, np.float32)[None, :] * scales ** 2)   # error in weight units
    levels = fit_table(xn, L, wts)
    if H is None:
        idx = nearest(xn, levels)
    else:
        Q = gptq_table_auto(W, np.asarray(H, np.float64), scales, levels, damp)
        idx = nearest(Q / scales, levels)
    codes = st.encode(np.minimum(blk / es, FP8_MAX), "F8_E4M3")
    return Hex(levels, es, codes, idx, block, (out, n))


# --------------------------------------------------------------------------- GPTQ onto a table


def _prep(W: np.ndarray, H: np.ndarray, damp: float):
    W = W.astype(np.float32).copy()
    H = H.astype(np.float64).copy()
    n = W.shape[1]
    if not np.any(np.diag(H)):
        H = np.eye(n)
    dead = (np.diag(H) == 0) | (np.abs(W).max(0) == 0)
    live = (~dead).astype(np.float64)
    H = H * live[:, None] * live[None, :] + np.diag(dead.astype(np.float64))
    W[:, dead] = 0
    base = float((np.diag(H) * live).sum() / max(live.sum(), 1)) or 1.0
    d = damp
    for _ in range(6):
        try:
            Lc = np.linalg.cholesky(H + d * base * np.eye(n))
            Hinv = np.linalg.inv(Lc.T) @ np.linalg.inv(Lc)
            U = np.linalg.cholesky(Hinv).T          # upper
            if np.isfinite(U).all():
                return W, U.astype(np.float32), dead
        except np.linalg.LinAlgError:
            pass
        d *= 3
    raise RuntimeError("GPTQ: Hessian not positive definite even with heavy damping")


def gptq_table(W: np.ndarray, H: np.ndarray, scales: np.ndarray, levels: np.ndarray, damp: float = 0.01,
               block: int = 128) -> np.ndarray:
    """GPTQ column sweep rounding W [out, in] onto `levels` * `scales` (per element).
    Returns the dequantized matrix."""
    W, Hinv, dead = _prep(W, H, damp)
    n = W.shape[1]
    Q = np.zeros_like(W)
    mids = (levels[1:] + levels[:-1]) / 2
    for b0 in range(0, n, block):
        b1 = min(b0 + block, n)
        Wb = W[:, b0:b1].copy()
        Eb = np.zeros_like(Wb)
        Hb = Hinv[b0:b1, b0:b1]
        for j in range(b1 - b0):
            col = b0 + j
            s = scales[:, col]
            q = levels[np.searchsorted(mids, np.clip(Wb[:, j] / s, -1, 1))] * s
            Q[:, col] = q
            e = (Wb[:, j] - q) / Hb[j, j]
            Wb[:, j:] -= e[:, None] * Hb[j, j:][None, :]
            Eb[:, j] = e
        W[:, b1:] -= Eb @ Hinv[b0:b1, b1:]
    Q[:, dead] = 0
    return Q


def gptq_table_auto(W, H, scales, levels, damp: float = 0.01) -> np.ndarray:
    """The numpy sweep for small matrices (tests, laptops); torch on the GPU, or on the CPU for
    wide matrices, when torch is installed (the batched version lives in reaper/gpu_perf/gptq.py)."""
    try:
        import torch
    except ImportError:
        return gptq_table(W, H, scales, levels, damp)
    if W.shape[1] < 1024 and not torch.cuda.is_available():
        return gptq_table(W, H, scales, levels, damp)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return gptq_table_torch(torch.as_tensor(W, device=dev), torch.as_tensor(H, device=dev),
                            torch.as_tensor(scales, device=dev), torch.as_tensor(levels, device=dev), damp).cpu().numpy()


def gptq_table_torch(W, H, scales, levels, damp: float = 0.01, block: int = 128):
    """gptq_table in torch (one problem; W [out, in], H [in, in], scales [out, in], levels [L])."""
    import torch
    W = W.float().clone()
    H = H.double().clone()
    n = W.shape[1]
    eye = torch.eye(n, dtype=H.dtype, device=H.device)
    if not H.diagonal().any():
        H = eye.clone()
    dead = (H.diagonal() == 0) | (W.abs().amax(0) == 0)
    live = (~dead).to(H.dtype)
    H = H * live[:, None] * live[None, :] + eye * dead.to(H.dtype)[:, None]
    W[:, dead] = 0
    base = float((H.diagonal() * live).sum() / live.sum().clamp(min=1)) or 1.0
    d = damp
    for _ in range(6):
        Lc, info = torch.linalg.cholesky_ex(H + d * base * eye)
        U, info2 = torch.linalg.cholesky_ex(torch.cholesky_inverse(Lc), upper=True)
        if info == 0 and info2 == 0 and torch.isfinite(U).all():
            break
        d *= 3
    else:
        raise RuntimeError("GPTQ: Hessian not positive definite even with heavy damping")
    Hinv = U.float()
    S = scales.float()
    lv = levels.float()
    mids = (lv[1:] + lv[:-1]) / 2
    Q = torch.zeros_like(W)
    for b0 in range(0, n, block):
        b1 = min(b0 + block, n)
        Wb = W[:, b0:b1].clone()
        Eb = torch.zeros_like(Wb)
        Hb = Hinv[b0:b1, b0:b1]
        for j in range(b1 - b0):
            col = b0 + j
            s = S[:, col]
            q = lv[torch.searchsorted(mids, (Wb[:, j] / s).clamp(-1, 1).contiguous())] * s
            Q[:, col] = q
            e = (Wb[:, j] - q) / Hb[j, j]
            Wb[:, j:] -= e[:, None] * Hb[j, j:][None, :]
            Eb[:, j] = e
        W[:, b1:] -= Eb @ Hinv[b0:b1, b1:]
    Q[:, dead] = 0
    return Q


# --------------------------------------------------------------------------- stacked-matrix helper


def expert_matrices(expert) -> dict[str, list]:
    """gate_up ([2I, H]: gate rows then up rows) and down ([H, I]) of an anatomy Expert, as
    lists of the views that make them, when its tensors have the Linear [out, in] layout.
    Transposed layouts (gpt-oss [in, out]) are returned per tensor under their own role."""
    roles = expert.roles
    out: dict[str, list] = {}
    if "gate" in roles and "up" in roles and roles["gate"].shape == roles["up"].shape:
        out["gate_up"] = [roles["gate"], roles["up"]]
    elif "gate_up" in roles:
        out["gate_up"] = [roles["gate_up"]]
    if "down" in roles:
        out["down"] = [roles["down"]]
    for r, v in roles.items():
        if r not in ("gate", "up", "gate_up", "down") and len(v.shape) == 2:
            out[r] = [v]
    return out
