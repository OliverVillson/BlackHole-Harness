"""blackhole.anatomy, the open format layer: safetensors dtypes, the HEX layout and quantizer,
the format table. (The developer API that edits models through these is the separate
blackhole-dev package and has its own tests.)"""

from __future__ import annotations

import numpy as np
import pytest

from blackhole.anatomy import formats, hex as hx, st


def test_dtypes_roundtrip():
    x = np.array([0.0, 1.0, -1.5, 3.14159, 0.0625, 448.0, -0.0078125], np.float32)
    b = st.decode(st.encode(x, "BF16"), "BF16")
    assert np.allclose(b, x, rtol=1e-2)
    f = st.decode(st.encode(x, "F8_E4M3"), "F8_E4M3")
    assert np.allclose(f, x, rtol=0.13) and f[0] == 0 and f[-1] == -0.0078125
    assert st.encode(np.array([np.nan], np.float32), "BF16")[0] == 0x7FC0


def test_safetensors_write_and_read(tmp_path):
    class Src:
        def __init__(self, a):
            self.a, self.nbytes = a, a.nbytes

        def write(self, fd):
            import os
            os.write(fd, self.a.tobytes())

    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    b = st.encode(np.array([1.0, -2.0], np.float32), "BF16")
    p = tmp_path / "model.safetensors"
    st.write_file(p, [("a", "F32", a.shape, Src(a)), ("b", "BF16", b.shape, Src(b))], {"format": "pt"})
    es = {e.name: e for e in st.entries(st.Blob(str(p)))}
    assert set(es) == {"a", "b"} and es["a"].dtype == "F32" and es["b"].nbytes == 4
    got = es["a"].blob.array(es["a"].offset, "F32", es["a"].shape)
    assert np.array_equal(got, a)
    assert np.array_equal(st.decode(es["b"].blob.array(es["b"].offset, "BF16", (2,)), "BF16"), [1.0, -2.0])
    assert st.checkpoint_files(tmp_path) == ["model.safetensors"]


def test_hex_gptq_beats_nearest():
    rng = np.random.default_rng(3)
    W = rng.standard_normal((32, 128)).astype(np.float32)
    X = rng.standard_normal((400, 128)) * (1 + 3 * (np.arange(128) < 8))   # a few hot input channels
    Hm = X.T @ X
    plain = hx.quantize(W, 8, 32)
    g = hx.quantize(W, 8, 32, energy=np.diag(Hm) / 400, H=Hm)
    err = lambda h: float((((W - h.dequant()) @ X.T) ** 2).sum())  # noqa: E731
    assert err(g) < err(plain)
    assert plain.bpw == 3.25 and np.array_equal(hx.unpack_expert(g.packed(), W.shape).idx, g.idx)
    for bits in (3, 4):
        idx = rng.integers(0, 2 ** bits, (5, 70)).astype(np.uint8)
        words = hx.pack_bits(idx, bits)
        assert words.dtype == np.uint32 and words.shape == (5, 3 * bits)
        assert np.array_equal(hx.unpack_bits(words, bits, 70), idx)


def test_formats_table_and_probe():
    assert formats.bpw("int4") == 4.125 and formats.bpw("mlx4") == 4.5
    assert formats.bpw("hex4") == 4.25 and formats.bpw("hex3") == 3.25
    with pytest.raises(ValueError):
        formats.check("int9")
    w = np.random.default_rng(0).standard_normal((64, 128)).astype(np.float32)
    m = formats.measure(w, "int4")
    assert 0 < m["rel_mse"] < 0.05 and m["bpw"] == formats.bpw("int4") and m["snr_db"] > 13


def test_hex_torch_reference_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(5)
    hs = [hx.quantize(rng.standard_normal((8, 96)).astype(np.float32), L, 32) for L in (16, 16)]
    parts = [h.packed() for h in hs]
    stack = lambda k: torch.as_tensor(np.stack([p[k] for p in parts]))  # noqa: E731
    got = hx.hex_dequant(stack("idx"), stack("block_scale"), stack("expert_scale").reshape(2), stack("levels"), 4)
    want = np.stack([h.dequant() for h in hs])
    assert torch.allclose(got, torch.as_tensor(want), atol=1e-7)
