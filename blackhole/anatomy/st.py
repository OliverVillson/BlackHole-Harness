"""safetensors without torch: headers, lazy reads, byte-range writes, dtype conversion.

A safetensors file is an 8-byte little-endian header length, a JSON header
{name: {dtype, shape, data_offsets}}, then raw bytes. Everything the anatomy API does to a
checkpoint is either a byte range of a source file (keep, rename, drop, take rows of a
row-major [E, ...] tensor) or a small computed tensor (a merged expert, an edited router row),
so numpy is enough and fp8 / mxfp4 blocks survive bit-exact whenever they are only moved.

bf16 and fp8 have no numpy dtype: they are held as uint16 / uint8 ("raw") and converted to
float32 by decode() when a value is read, and back by encode() when one is written.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# safetensors dtype -> (numpy storage dtype, float-convertible)
DTYPES: dict[str, tuple[type, bool]] = {
    "F64": (np.float64, True), "F32": (np.float32, True), "F16": (np.float16, True), "BF16": (np.uint16, True),
    "F8_E4M3": (np.uint8, True), "F8_E5M2": (np.uint8, True),
    "I64": (np.int64, False), "I32": (np.int32, False), "I16": (np.int16, False), "I8": (np.int8, False),
    "U64": (np.uint64, False), "U32": (np.uint32, False), "U16": (np.uint16, False), "U8": (np.uint8, False),
    "BOOL": (np.bool_, False),
}
BITS = {k: np.dtype(v[0]).itemsize * 8 for k, v in DTYPES.items()}
COPY_CHUNK = 1 << 28


def _fp8_table(exp_bits: int, man_bits: int, bias: int, fn: bool) -> np.ndarray:
    out = np.zeros(256, np.float32)
    for b in range(256):
        s = -1.0 if b >> 7 else 1.0
        e = (b >> man_bits) & ((1 << exp_bits) - 1)
        m = b & ((1 << man_bits) - 1)
        top = (1 << exp_bits) - 1
        if fn and e == top and m == (1 << man_bits) - 1:
            out[b] = np.nan                       # e4m3fn: only S.1111.111 is NaN, no infinities
        elif not fn and e == top:
            out[b] = s * np.inf if m == 0 else np.nan
        elif e == 0:
            out[b] = s * m / (1 << man_bits) * 2.0 ** (1 - bias)
        else:
            out[b] = s * (1 + m / (1 << man_bits)) * 2.0 ** (e - bias)
    return out


FP8 = {"F8_E4M3": _fp8_table(4, 3, 7, True), "F8_E5M2": _fp8_table(5, 2, 15, False)}


def decode(raw: np.ndarray, dtype: str) -> np.ndarray:
    """Storage array -> float32 (a copy)."""
    if dtype == "BF16":
        return (raw.astype(np.uint32) << 16).view(np.float32)
    if dtype in FP8:
        return FP8[dtype][raw]
    if not DTYPES[dtype][1]:
        return raw.astype(np.float32)
    return raw.astype(np.float32)


def encode(x: np.ndarray, dtype: str) -> np.ndarray:
    """float array -> storage array of `dtype` (round to nearest even for bf16, nearest for fp8)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if dtype == "BF16":
        u = x.view(np.uint32)
        r = ((u >> 16) & 1) + 0x7FFF
        out = ((u + r) >> 16).astype(np.uint16)
        return np.where(np.isnan(x), np.uint16(0x7FC0), out)
    if dtype in FP8:
        table = FP8[dtype]
        finite = np.where(np.isfinite(table))[0]
        order = finite[np.argsort(table[finite], kind="stable")]
        vals = table[order]
        lim = float(vals[-1])
        xc = np.clip(x, -lim, lim)
        i = np.clip(np.searchsorted(vals, xc), 1, len(vals) - 1)
        pick = np.where(np.abs(vals[i - 1] - xc) <= np.abs(vals[i] - xc), i - 1, i)
        return order[pick].astype(np.uint8)
    np_t = DTYPES[dtype][0]
    if not DTYPES[dtype][1]:
        return np.rint(x).astype(np_t)
    return x.astype(np_t)


# --------------------------------------------------------------------------- files


class Blob:
    """Read access to one safetensors file, local or on the Hub (HTTP range requests)."""

    def __init__(self, where: str, headers: dict | None = None):
        self.where = where
        self.remote = where.startswith(("http://", "https://"))
        self.http_headers = headers or {}
        self._header: tuple[dict, int] | None = None

    def __repr__(self) -> str:
        return f"Blob({self.where})"

    def read(self, off: int, n: int) -> bytes:
        if self.remote:
            return http_range(self.where, off, n, self.http_headers)
        with open(self.where, "rb") as f:
            f.seek(off)
            return f.read(n)

    def header(self) -> tuple[dict, int]:
        """({name: entry}, byte offset where tensor data starts)."""
        if self._header is None:
            n = struct.unpack("<Q", self.read(0, 8))[0]
            self._header = json.loads(self.read(8, n)), 8 + n
        return self._header

    def array(self, off: int, dtype: str, shape: tuple) -> np.ndarray:
        """The tensor at absolute byte offset `off`: a read-only memmap for local files (so
        reading one weight of a 2 GB tensor reads one page), bytes for remote ones."""
        np_t = DTYPES[dtype][0]
        n = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if n == 0:
            return np.zeros(shape, np_t)
        if not self.remote:
            return np.memmap(self.where, dtype=np_t, mode="r", offset=off, shape=tuple(shape) or (1,)).reshape(shape)
        return np.frombuffer(self.read(off, n * np.dtype(np_t).itemsize), dtype=np_t).reshape(shape)

    def copy_to(self, fd: int, off: int, n: int) -> None:
        """Append n bytes from `off` to the open file fd (in-kernel copy for local files)."""
        if self.remote:
            while n > 0:
                k = min(n, COPY_CHUNK)
                os.write(fd, self.read(off, k))
                off, n = off + k, n - k
            return
        src = os.open(self.where, os.O_RDONLY)
        try:
            while n > 0:
                try:
                    k = os.copy_file_range(src, fd, min(n, 1 << 30), offset_src=off)
                except (AttributeError, OSError):
                    k = 0
                if k == 0:                       # no copy_file_range here: plain pread/write
                    k = os.write(fd, os.pread(src, min(n, 64 << 20), off))
                off, n = off + k, n - k
        finally:
            os.close(src)


def http_range(url: str, off: int, n: int, headers: dict) -> bytes:
    import httpx
    r = httpx.get(url, headers={**headers, "Range": f"bytes={off}-{off + n - 1}"}, follow_redirects=True,
                  timeout=120)
    r.raise_for_status()
    return r.content


@dataclass
class Entry:
    """One tensor in a checkpoint."""
    name: str
    blob: Blob
    dtype: str
    shape: tuple
    offset: int          # absolute byte offset in the file

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * BITS[self.dtype] // 8 if self.shape else BITS[self.dtype] // 8


def entries(blob: Blob) -> list[Entry]:
    hdr, base = blob.header()
    return [Entry(k, blob, v["dtype"], tuple(v["shape"]), base + v["data_offsets"][0])
            for k, v in hdr.items() if k != "__metadata__"]


def checkpoint_files(root: Path) -> list[str]:
    idx = root / "model.safetensors.index.json"
    if idx.exists():
        return sorted(set(json.loads(idx.read_text())["weight_map"].values()))
    files = sorted(p.name for p in root.glob("*.safetensors") if not p.name.startswith("blackhole-"))
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint in {root}")
    return files


def write_file(path: Path, tensors: list[tuple[str, str, tuple, object]], metadata: dict | None = None) -> int:
    """Write a safetensors file. tensors: (name, dtype, shape, source) where source has
    nbytes and write(fd). Returns the data size."""
    hdr, off = {}, 0
    if metadata:
        hdr["__metadata__"] = metadata
    for name, dt, shape, src in tensors:
        n = src.nbytes
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
    blob = json.dumps(hdr, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, struct.pack("<Q", len(blob)) + blob)
        for _, _, _, src in tensors:
            src.write(fd)
    finally:
        os.close(fd)
    return off
