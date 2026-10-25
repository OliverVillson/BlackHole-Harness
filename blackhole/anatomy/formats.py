"""Storage formats a part can be given, their bits per weight, and a quick error probe.

Names:
  bf16 f16 f32 fp8                       plain
  intN[_gG]   N in 2..8, symmetric, group G (default 128): GPTQ's grids
  mlxN[_gG]   MLX affine, fp16 scale + bias per group (default 64)
  mxfp4 nvfp4
  q2_k .. q8_0                           llama.cpp types
  hex3 hex4[/bB]  HEX (Header-Encoded eXperts): a learned 8- or 16-level value table in each
              expert's header, 3/4-bit indices, an fp8 scale per block of B weights (default 32).
              hex3 = 3.25 bpw, hex4 = 4.25 bpw; hex4/b16 = 4.5 bpw.

measure() rounds to nearest on the format's grid (a learned per-tensor table for hex) and
returns the relative error. It is a probe for comparing parts and formats; the real
quantizers in the pipeline (GPTQ, HEX with GPTQ rounding) do better than round-to-nearest.
"""

from __future__ import annotations

import math
import re

import numpy as np

FIXED = {"bf16": 16.0, "f16": 16.0, "f32": 32.0, "fp8": 8.0 + 32 / 128 ** 2, "mxfp4": 4.25, "nvfp4": 4.5,
         "q2_k": 2.625, "q3_k": 3.4375, "q4_k": 4.5, "q5_k": 5.5, "q6_k": 6.5625, "q8_0": 8.5}
_INT = re.compile(r"^int([2-8])(?:_g(\d+))?$")
_MLX = re.compile(r"^mlx([2-8])(?:_g(\d+))?$")
_HEX = re.compile(r"^hex([34])(?:/b(\d+))?$")


def bpw(fmt: str) -> float:
    """Bits per weight of a format name; ValueError for an unknown one."""
    f = fmt.lower()
    if f in FIXED:
        return FIXED[f]
    if m := _INT.match(f):
        return int(m[1]) + 16 / int(m[2] or 128)
    if m := _MLX.match(f):
        return int(m[1]) + 32 / int(m[2] or 64)
    if m := _HEX.match(f):
        return int(m[1]) + 8 / int(m[2] or 32)
    raise ValueError(f"unknown format {fmt!r}: bf16 f16 fp8 intN[_gG] mlxN[_gG] mxfp4 nvfp4 q2_k..q8_0 hex3 hex4[/bB]")


def check(fmt: str) -> str:
    bpw(fmt)
    return fmt.lower()


def _groups(w: np.ndarray, g: int) -> tuple[np.ndarray, int]:
    w2 = w.reshape(-1, w.shape[-1]) if w.ndim > 1 else w.reshape(1, -1)
    pad = -w2.shape[1] % g
    w2 = np.pad(w2, ((0, 0), (0, pad)))
    return w2.reshape(w2.shape[0], -1, g), pad


def _ungroup(q: np.ndarray, pad: int, shape) -> np.ndarray:
    q = q.reshape(q.shape[0], -1)
    return (q[:, :q.shape[1] - pad] if pad else q).reshape(shape)


def _kmeans(x: np.ndarray, L: int, iters: int = 12, sample: int = 1 << 16) -> np.ndarray:
    rng = np.random.default_rng(0)
    s = x if x.size <= sample else x[rng.integers(0, x.size, sample)]
    lev = np.quantile(s, np.linspace(0, 1, L + 2)[1:-1])
    for _ in range(iters):
        a = np.searchsorted((lev[1:] + lev[:-1]) / 2, s)
        num = np.bincount(a, weights=s, minlength=L)
        den = np.bincount(a, minlength=L)
        lev = np.sort(np.where(den > 0, num / np.maximum(den, 1), lev))
    return lev


def fake_quant(w: np.ndarray, fmt: str) -> np.ndarray:
    """Round-to-nearest dequantized copy of w in `fmt` (float32)."""
    f = fmt.lower()
    w = np.asarray(w, np.float32)
    if f in ("bf16", "f32", "f16"):
        from blackhole.anatomy import st
        return st.decode(st.encode(w, {"bf16": "BF16", "f16": "F16", "f32": "F32"}[f]), {"bf16": "BF16", "f16": "F16", "f32": "F32"}[f])
    if f == "fp8":
        from blackhole.anatomy import st
        s = max(float(np.abs(w).max()) / 448.0, 1e-30)
        return st.decode(st.encode(w / s, "F8_E4M3"), "F8_E4M3") * s
    if m := (_INT.match(f) or _MLX.match(f)):
        bits, g = int(m[1]), int(m[2] or (128 if f.startswith("int") else 64))
        x, pad = _groups(w, g)
        if f.startswith("int"):
            q = 2 ** (bits - 1) - 1
            s = np.maximum(np.abs(x).max(-1, keepdims=True) / q, 1e-30)
            y = np.clip(np.rint(x / s), -q - 1, q) * s
        else:
            lo, hi = x.min(-1, keepdims=True), x.max(-1, keepdims=True)
            s = np.maximum((hi - lo) / (2 ** bits - 1), 1e-30)
            y = np.rint((x - lo) / s) * s + lo
        return _ungroup(y, pad, w.shape)
    if m := _HEX.match(f):
        L, block = 2 ** int(m[1]), int(m[2] or 32)
        x, pad = _groups(w, block)
        s = np.maximum(np.abs(x).max(-1, keepdims=True), 1e-30)
        xn = x / s
        lev = _kmeans(xn.ravel(), L)
        idx = np.searchsorted((lev[1:] + lev[:-1]) / 2, xn)
        return _ungroup(lev[idx] * s, pad, w.shape)
    if f in FIXED:   # mxfp4 / nvfp4 / gguf k-quants: approximate with the int grid of the same bit width
        bits = {"mxfp4": 4, "nvfp4": 4, "q2_k": 2, "q3_k": 3, "q4_k": 4, "q5_k": 5, "q6_k": 6, "q8_0": 8}[f]
        return fake_quant(w, f"int{bits}_g{16 if f == 'nvfp4' else 32}")
    raise ValueError(f"unknown format {fmt!r}")


def measure(w: np.ndarray, fmt: str) -> dict:
    """Relative squared error and SNR (dB) of round-to-nearest `fmt` on w."""
    w = np.asarray(w, np.float32)
    q = fake_quant(w, fmt)
    err = float(np.sum((w.astype(np.float64) - q) ** 2))
    ref = float(np.sum(w.astype(np.float64) ** 2)) or 1e-30
    rel = err / ref
    return {"format": fmt, "bpw": bpw(fmt), "rel_mse": rel, "snr_db": -10 * math.log10(max(rel, 1e-30))}
