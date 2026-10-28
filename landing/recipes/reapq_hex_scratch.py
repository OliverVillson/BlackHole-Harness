# REAP/Q-HEX from scratch: REAP pruning, then REAQ picks 16 or 8 levels per expert and HEX
# quantizes every expert matrix (learned level table, fp8 block scales, GPTQ rounding onto the
# table). Only the anatomy API and numpy; torch just runs the model once per stage to capture
# the hidden states entering each MoE block. Qwen-style experts (gate/up/down, SiLU).
import json
from pathlib import Path

import numpy as np

from blackhole.anatomy import st
from blackhole.anatomy.hex import Hex
from blackhole_dev import function
from blackhole_dev.recipe import Recipe, gate


def moe_inputs(m, calib, max_seq, tokens, device):
    """{layer: hidden states [N, hidden] entering its MoE block}, first `tokens` per layer."""
    import torch
    p = Path(calib)
    texts = [json.loads(r)["text"] for r in p.read_text().splitlines() if r.strip()] if p.suffix == ".jsonl" \
        else [t for t in p.read_text().split("\n\n") if t.strip()]
    live, xs = m.live(device=device), {}

    def grab(i):
        def hook(_mod, args):
            if sum(map(len, xs.setdefault(i, []))) < tokens:
                xs[i].append(args[0].reshape(-1, args[0].shape[-1]).float().cpu().numpy())
        return hook
    hooks = [live.module_of(L.moe).register_forward_pre_hook(grab(L.index)) for L in m.layers if L.moe]
    with torch.no_grad():
        for t in texts:
            live.module(input_ids=live.tokenizer(t, return_tensors="pt").input_ids[:, :max_seq].to(live.device))
    for h in hooks:
        h.remove()
    return {i: np.concatenate(x)[:tokens] for i, x in xs.items()}


def route(m, moe, X):
    """Top-k expert ids [N, k] and their router weights [N, k]."""
    z = moe.router.logits(X)
    if m.tc.get("scoring_func") == "sigmoid":
        p = 1 / (1 + np.exp(-z))
    else:
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
    top = np.argsort(-p, 1)[:, :m.top_k]
    w = np.take_along_axis(p, top, 1)
    return top, w / w.sum(1, keepdims=True) if m.tc.get("norm_topk_prob") is not False else w


def gate_up(e):
    r = e.roles
    return [r["gate"], r["up"]] if "gate" in r else [r["gate_up"]]


def expert(e, X):
    """(down input act(gate) * up, expert output) for tokens X."""
    g, u = np.split(X @ np.concatenate([v.numpy() for v in gate_up(e)]).T, 2, axis=1)
    A = g / (1 + np.exp(-g)) * u
    return A, A @ e.roles["down"].numpy().T


def reap(m, moe, X):
    """REAP saliency per expert: mean over its routed tokens of router weight x |expert output|."""
    top, w = route(m, moe, X)
    s = np.zeros(len(moe.experts))
    for j, e in enumerate(moe.experts):
        t, k = np.nonzero(top == j)
        if len(t):
            s[j] = (w[t, k] * np.linalg.norm(expert(e, X[t])[1], axis=1)).mean()
    return s


def hex_quant(W, L, X=None, damp=0.01, iters=20):
    """HEX one [out, in] matrix: fp8 e4m3 scale per 32 inputs under one fp32 expert scale, an
    L-level table fit by weighted k-means, GPTQ rounding onto the table when inputs X are given."""
    out, n = W.shape
    amax = np.abs(W.reshape(out, -1, 32)).max(-1)
    es = max(float(amax.max()) / 448, 1e-30)
    codes = st.encode(np.minimum(amax / es, 448), "F8_E4M3")
    S = np.repeat(st.decode(codes, "F8_E4M3") * es, 32, 1)
    S = np.where(S > 0, S, 1).astype(np.float32)
    xn = np.clip(W / S, -1, 1)
    wt = S ** 2 * ((X ** 2).mean(0) if X is not None and len(X) else 1)       # error in output units
    pick = np.random.default_rng(0).integers(0, xn.size, min(xn.size, 1 << 17))
    x, wt = xn.ravel()[pick], np.broadcast_to(wt, xn.shape).ravel()[pick]
    lev = np.quantile(x, np.linspace(0, 1, L + 2)[1:-1])
    for _ in range(iters):
        a = np.searchsorted((lev[1:] + lev[:-1]) / 2, x)
        num, den = np.bincount(a, x * wt, L), np.bincount(a, wt, L)
        lev = np.sort(np.where(den > 0, num / np.maximum(den, 1e-30), lev))
    lev = lev.astype(np.float32)
    near = lambda v: np.searchsorted((lev[1:] + lev[:-1]) / 2, np.clip(v, -1, 1)).astype(np.uint8)  # noqa: E731
    if X is not None and len(X):
        H = X.astype(np.float64).T @ X
        d = np.diag(H)
        H[np.diag_indices(n)] += damp * d.mean() + (d == 0)
        U = np.linalg.cholesky(np.linalg.inv(H)).T                  # upper factor of H^-1
        W = W.astype(np.float64)
        for c in range(n):                                          # round a column, push its error right
            q = lev[near(W[:, c] / S[:, c])] * S[:, c]
            W[:, c:] -= np.outer((W[:, c] - q) / U[c, c], U[c, c:])
    return Hex(lev, es, codes, near(W / S), 32, (out, n))


@function("reapq-reap", per="model")
def reapq_reap(m, calib: str = "calib.jsonl", ratio: float = 0.25, max_seq: int = 2048, tokens: int = 8192,
               device: str = "cpu") -> dict:
    """REAP: drop the `ratio` lowest-saliency experts of every MoE layer (never below top-k)."""
    X, dropped = moe_inputs(m, calib, max_seq, tokens, device), 0
    for L in [L for L in m.layers if L.moe]:
        s, n = reap(m, L.moe, X[L.index]), len(L.moe.experts)
        m.set_scores(L.index, s.tolist())
        keep = np.sort(np.argsort(-s)[:max(m.top_k, n - round(n * ratio))])
        m.keep_experts(L.index, keep.tolist(), note="reap")
        dropped += n - len(keep)
    return {"experts_dropped": dropped}


@function("reapq-hex", per="model")
def reapq_hex(m, calib: str = "calib.jsonl", avg_bits: float = 3.75, gptq: bool = True, max_seq: int = 2048,
              tokens: int = 8192, device: str = "cpu") -> dict:
    """REAQ + HEX: in each MoE layer the hottest experts by REAP saliency get 16 levels (4.25 bpw),
    the rest 8 (3.25), so experts average `avg_bits`; never 2-bit. Writes the dequantized weights
    back and keeps the HEX headers for the packed layout."""
    if not 3.25 <= avg_bits <= 4.25:
        raise ValueError("avg_bits must be in [3.25, 4.25]: HEX tiers are 8 and 16 levels, never 2-bit")
    X, err, ref, bits, size = moe_inputs(m, calib, max_seq, tokens, device), 0.0, 0.0, 0.0, 0
    for L in [L for L in m.layers if L.moe]:
        x, ex = X[L.index], L.moe.experts
        top, _ = route(m, L.moe, x)
        hot = set(np.argsort(-reap(m, L.moe, x))[:round(len(ex) * (avg_bits - 3.25))].tolist())
        for j, e in enumerate(ex):
            xj = x[(top == j).any(1)]
            for role, views, inp in (("gate_up", gate_up(e), xj), ("down", [e.roles["down"]], expert(e, xj)[0])):
                W = np.concatenate([v.numpy() for v in views])
                h = hex_quant(W, 16 if j in hot else 8, inp if gptq else None)
                Q = h.dequant()
                err, ref = err + float(((W - Q) ** 2).sum()), ref + float((W ** 2).sum())
                bits, size = bits + h.bpw * W.size, size + W.size
                m.hex[(L.index, e.attrs["id"], role)] = h
                for v, q in zip(views, np.split(Q, len(views))):
                    v.set(q)
            e.quant("hex4" if j in hot else "hex3")
    return {"bpw": round(bits / max(size, 1), 4), "rel_mse": err / max(ref, 1e-30)}


recipe = Recipe(
    "blackhole/reapq-hex-scratch", "0.1.0",
    title="REAP/Q-HEX from scratch",
    description="REAP pruning, REAQ per-expert widths and HEX quantization written in one file; packed HEX out.",
    author="BlackHole-Harness", license="Apache-2.0",
    tags=["moe", "reap", "reaq", "hex", "blackhole-original"],
    targets={"architectures": ["moe", "hybrid"]},
)
prune = recipe.input("prune", "float", default=0.25, min=0.0, max=0.5, help="Share of experts REAP removes per layer")
avg_bits = recipe.input("avg_bits", "float", default=3.75, min=3.25, max=4.25, help="Average expert bits (hex3/hex4 mix)")
calib = recipe.input("calib", "path", default="calib.jsonl", help="Calibration texts (.jsonl with text, or .txt)")

recipe.fn("reapq-reap", "reap", calib=calib, ratio=prune)
recipe.fn("reapq-hex", "hex", calib=calib, avg_bits=avg_bits, pack="hex",
          gates=[gate("reapq-hex.rel_mse", "<=", 0.2)])
