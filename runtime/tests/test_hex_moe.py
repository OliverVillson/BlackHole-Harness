"""HEX Triton MoE against the torch reference, on the CPU through the Triton interpreter."""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")

import pytest  # noqa: E402
import torch  # noqa: E402

pytest.importorskip("triton")

from blackhole_runtime.hex_moe import HexTier, align, hex_moe  # noqa: E402
from blackhole_runtime.hex_ref import HEX_BLOCK, hex_dequant, moe_reference, pack, random_parts  # noqa: E402


def make_tiers(E, H, I, split, bits=(4, 3), seed=0):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(E, generator=g)
    groups = [perm[:split], perm[split:]] if split else [perm]
    tiers = []
    for i, (ids, b) in enumerate(zip(groups, bits)):
        n = len(ids)
        tiers.append(HexTier(b, ids, random_parts(n, I, H, b, seed=seed + 10 * i + 1),
                             random_parts(n, I, H, b, seed=seed + 10 * i + 2),
                             random_parts(n, H, I, b, seed=seed + 10 * i + 3)))
    return tiers


def routing(T, E, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)])
    w = torch.softmax(torch.randn(T, k, generator=g), -1)
    return ids, w


def test_pack_roundtrip_matches_bitstream():
    for bits in (3, 4):
        codes = torch.randint(0, 2 ** bits, (2, 5, 64))
        words = pack(codes, bits)
        assert words.shape == (2, 5, 64 // HEX_BLOCK * bits)
        lv = torch.arange(2 ** bits, dtype=torch.float32).expand(2, -1).contiguous()
        bs = torch.full((2, 5, 2), 0x38, dtype=torch.uint8)            # fp8 1.0
        W = hex_dequant(words, bs, torch.ones(2), lv, bits)
        assert torch.equal(W, codes.float())


def test_align_groups_and_pads():
    local = torch.tensor([2, -1, 0, 2, 2, 1, -1, 0])
    sid, eid = align(local, 3, 4)
    S = local.numel()
    seen = []
    for b, e in enumerate(eid.tolist()):
        rows = sid[b * 4:(b + 1) * 4].tolist()
        if e < 0:
            assert all(r == S for r in rows)
            continue
        for r in rows:
            if r < S:
                assert local[r] == e
                seen.append(r)
    assert sorted(seen) == [i for i in range(S) if local[i] >= 0]


@pytest.mark.parametrize("T,split", [(1, 0), (3, 5), (40, 4)])
def test_hex_moe_matches_reference(T, split):
    E, H, I, k = 12, 128, 64, 4
    tiers = make_tiers(E, H, I, split, bits=(4, 3) if split else (3,), seed=T)
    ref_tiers = [{"bits": t.bits, "experts": t.experts, "gate_proj": dict(t.gate_proj),
                  "up_proj": dict(t.up_proj), "down_proj": dict(t.down_proj)} for t in tiers]
    for t in tiers:
        t.prepare(E)
    ids, w = routing(T, E, k, seed=T)
    x = torch.randn(T, H) * 0.5
    ref = moe_reference(x, w, ids, ref_tiers)
    for cfg in (None, {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 1}):
        got = hex_moe(x, w, ids, tiers, cfg)
        err = (got - ref).abs().max() / ref.abs().max()
        assert err < 1e-4, err


def test_mlx_tiers_become_hex_tiers_in_router_order():
    """convert.hex_tensors: MLX experts_hot (ids 0..n_hot-1) and experts (the rest) become tier
    tensors with those ids, and the kernel on them equals the reference on the MLX parts."""
    from blackhole_runtime.convert import hex_tensors
    E, H, I, n_hot, k = 10, 64, 32, 3, 3
    parts, mlx = {}, {}
    for tier, n, bits, s in (("experts_hot", n_hot, 4, 1), ("experts", E - n_hot, 3, 5)):
        for j, (proj, o, i) in enumerate((("gate_proj", I, H), ("up_proj", I, H), ("down_proj", H, I))):
            p = random_parts(n, o, i, bits, seed=s + j)
            mlx.setdefault(tier, {"bits": bits})[proj] = p
            for q, t in p.items():
                parts[f"model.layers.1.mlp.{tier}.{proj}.{q}"] = t
    out = hex_tensors(parts, {1: n_hot})
    assert out["model.layers.1.mlp.experts.tier16.experts"].tolist() == [0, 1, 2]
    assert out["model.layers.1.mlp.experts.tier8.experts"].tolist() == list(range(3, 10))
    ref_tiers = [{"bits": 4, "experts": torch.arange(0, 3), **{p: mlx["experts_hot"][p] for p in
                                                                 ("gate_proj", "up_proj", "down_proj")}},
                 {"bits": 3, "experts": torch.arange(3, 10), **{p: mlx["experts"][p] for p in
                                                                  ("gate_proj", "up_proj", "down_proj")}}]
    tiers = []
    for L, bits in ((16, 4), (8, 3)):
        pre = f"model.layers.1.mlp.experts.tier{L}"
        g = lambda proj: {q: out[f"{pre}.{proj}.{q}"].clone() for q in ("idx", "block_scale", "expert_scale", "levels")}  # noqa: E731
        tiers.append(HexTier(bits, out[f"{pre}.experts"], g("gate_proj"), g("up_proj"), g("down_proj")).prepare(E))
    ids, w = routing(5, E, k, seed=3)
    x = torch.randn(5, H)
    ref = moe_reference(x, w, ids, ref_tiers)
    got = hex_moe(x, w, ids, tiers)
    assert ((got - ref).abs().max() / ref.abs().max()) < 1e-4
