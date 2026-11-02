"""Stage hexvllm: the REAP-pruned model with HEX experts as a folder vLLM serves on NVIDIA GPUs.

    vllm serve <out>/<name>-HEX-vLLM          (needs the BlackHole runtime: pip install -e runtime)

Every routed expert gets HEX (a BlackHole-Harness original): its own learned table of 2**bits
levels per matrix, fp8 block scales over 32 inputs, GPTQ rounding onto the table with the
expert's own Hessian (the tokens the router sends it). REAQ (also a BlackHole-Harness
original) picks the table size per expert: the `hex_hot` share of each layer's experts with the
highest REAP scores gets 2**hex_hot_bits levels, the rest 2**hex_bits. Sequential over chunks
of MoE layers like the gptq and mlx stages, so later layers see the earlier ones quantized.

Nothing else changes: attention, linear attention, routers, shared experts, embeddings (also
very large ones such as Qwen3.8-Flash-Next's n-gram table), vision tower and MTP layers are
copied from the pruned checkpoint byte for byte, in its own tensor names and layout. So this
works for any MoE checkpoint whose experts REAP found (fused [E, ...] tensors or per-expert
ones), as long as vLLM has a model class for it; the "blackhole-hex" quantization method
(runtime/blackhole_runtime/vllm_hex.py) runs the experts through the HEX Triton kernel.

Experts are written as tiers (layout: blackhole/anatomy/hex.py):
  <mlp>.experts.tier<L>.experts                         int32 [n] expert ids (after pruning)
  <mlp>.experts.tier<L>.{gate,up,down}_proj.{idx,block_scale,expert_scale,levels}
The quantized model is compared with the teacher (KL / top-1) on the held-out answers.
Writes work/hexvllm.json and out/<name>-HEX-vLLM/.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

from blackhole_pipeline.common import Pruned, emit, gib, stage_main, write_checkpoint

STAGE = "hexvllm"
HEX_BLOCK = 32
_FUSED = re.compile(r"^(?P<pre>.*layers\.(?P<L>\d+)\.(?:mlp|block_sparse_moe|feed_forward|moe))\.experts\."
                    r"(?:(?P<j>\d+)\.(?:gate_proj|up_proj|down_proj|w1|w2|w3)\.weight|gate_up_proj|down_proj"
                    r"|gate_up_proj_bias|down_proj_bias|(?P<j2>\d+)\.(?:gate_proj|up_proj|down_proj)\.bias)$")


def hex_dir(rj) -> Path:
    return rj.out_dir.with_name(rj.out_dir.name + "-HEX-vLLM")


def expert_tensors(src: Path, num_layers: int) -> tuple[dict[int, str], set[str]]:
    """({decoder layer: checkpoint prefix of its MoE block}, every routed-expert tensor name) of
    the checkpoint in src. Multi-token-prediction heads (mtp.*, layers past num_hidden_layers)
    are left alone: REAP did not score them and vLLM only builds them for speculative decoding."""
    from reaper_pipeline.ckpt import shard_map
    prefixes, names = {}, set()
    for keys in shard_map(src).values():
        for k in keys:
            if k.startswith(("mtp.", "model.mtp.")) or ".mtp." in k:
                continue
            m = _FUSED.match(k)
            if not m or int(m["L"]) >= num_layers:
                continue
            prefixes.setdefault(int(m["L"]), m["pre"])
            names.add(k)
    return prefixes, names


def tier_order(sal: list[float] | None, E: int, hot_share: float) -> tuple[list[int], int]:
    """Expert ids hottest first (REAP score) and how many go on the larger table."""
    if not sal or hot_share <= 0:
        return list(range(E)), 0
    if len(sal) != E:
        raise RuntimeError(f"{len(sal)} REAP scores for {E} experts")
    order = sorted(range(E), key=lambda j: -sal[j])
    return order, int(round(E * hot_share))


def hex_parts(W, Hm, levels: int):
    """HEX parts of the stacked expert matrices W [n, out, in] with Hessians Hm [n, in, in]:
    ({idx, block_scale, expert_scale, levels}, the dequantized weights)."""
    import torch
    from gptq import gptq_table
    from headers import header_parts

    from blackhole_pipeline.pack import (block_codes, codebook_dequant, expert_scale, mlx_pack, pack_nibbles,
                                         table_indices)
    S, Lv = header_parts(W, "expert", levels, HEX_BLOCK, Hm.diagonal(dim1=-2, dim2=-1), False)
    Q = gptq_table(W, Hm, S, Lv)
    es = expert_scale(W)
    idx = table_indices(Q, S, Lv)
    codes = block_codes(S, es, HEX_BLOCK)
    back = codebook_dequant(pack_nibbles(idx), W.shape[-1], codes, es, Lv, HEX_BLOCK)
    n_, o_, i_ = idx.shape
    bits = levels.bit_length() - 1
    words = mlx_pack(idx.long().reshape(n_, o_, i_ // HEX_BLOCK, HEX_BLOCK), bits).reshape(n_, o_, -1)
    parts = {"idx": words,
             "block_scale": codes.view(torch.uint8), "expert_scale": es.float().reshape(n_),
             "levels": Lv.float().reshape(n_, levels)}
    return parts, back


def hessian_layers_per_chunk(m, p: dict) -> int:
    """MoE layers per sequential chunk: the job's layer_chunk, fewer when the per-expert Hessians
    of that many layers would not fit in what the loaded model leaves free on its GPU."""
    import torch
    size = max(1, int(p["layer_chunk"]))
    if not torch.cuda.is_available():
        return size
    blk = m.blocks[0]
    v = m.views[blk.layer][0]
    H, I = v["gate"].shape[1], v["down"].shape[1]
    per_layer = blk.num_experts * (H * H + I * I) * 4
    dev = m.views[blk.layer][0]["gate"].device
    if dev.type != "cuda":
        return size
    free, _ = torch.cuda.mem_get_info(dev)
    fit = int(0.6 * free // max(per_layer, 1))
    return max(1, min(size, fit))


def run(bh, rj) -> None:
    import torch
    from reaper_pipeline.load import free
    from safetensors.torch import save_file

    p = bh.params
    bits, hot_bits, hot_share = int(p["hex_bits"]), int(p["hex_hot_bits"]), float(p["hex_hot"])
    for b in {bits, hot_bits} if hot_share > 0 else {bits}:
        if b not in (3, 4):
            raise ValueError(f"HEX tables have 8 or 16 levels (bits 3 or 4), not {b} bits")
    src = bh.healed_dir(rj) if bh.healed_dir(rj).exists() else rj.out_dir
    if not (src / "config.json").exists():
        raise RuntimeError(f"no pruned model in {src}: run REAP (reap-gpu) first")
    cfg = json.loads((src / "config.json").read_text())
    tc = cfg.get("text_config") or cfg
    emit(STAGE, pct=1, msg=f"loading {src.name}")
    m = Pruned(rj, p["gptq_calib"], src=src, reserve_gb=p["hex_reserve_gb"])
    prefixes, names = expert_tensors(src, int(tc.get("num_hidden_layers") or 10 ** 6))
    missing = sorted(b.layer for b in m.blocks if b.layer not in prefixes)
    if missing:
        raise RuntimeError(f"MoE layers {missing[:8]} have no routed-expert tensors in {src.name} that this "
                           "export recognises (expected <...>.layers.N.mlp.experts.{gate_up_proj,down_proj} or "
                           "<...>.experts.<j>.{gate,up,down}_proj.weight)")
    sal = {}
    if hot_share > 0:
        s = rj.read("work/saliency.json")
        if not s:
            raise RuntimeError("REAQ tiers need the REAP scores (work/saliency.json): run REAP first, or hex_hot=0")
        sal = {L["layer"]: [L["saliency"][j] for j in L["kept"]] for L in s["layers"]}

    work = rj.path("work", "hexvllm")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    nbytes, tiers, t0 = 0, {}, time.time()
    nL, done = len(m.blocks), 0
    chunk = hessian_layers_per_chunk(m, p)
    for c0 in range(0, nL, chunk):
        blocks = m.blocks[c0:c0 + chunk]
        emit(STAGE, pct=2 + 76 * done / nL, msg=f"Hessians for layers {blocks[0].layer}–{blocks[-1].layer}")
        hs = m.hessians(blocks)
        for blk in blocks:
            Hin, Hdn = hs.pop(blk.layer)
            E = blk.num_experts
            order, n_hot = tier_order(sal.get(blk.layer), E, hot_share)
            pre = prefixes[blk.layer] + ".experts"
            out = {}
            for js, nb in ((order[:n_hot], hot_bits), (order[n_hot:], bits)):
                if not js:
                    continue
                L = 2 ** nb
                gu, dn = m.stacked(blk.layer, js, Hin.device)
                I = gu.shape[1] // 2
                jj = torch.tensor(js, device=Hin.device)
                pg, bgu = hex_parts(gu, Hin[jj], L)
                pd, bdn = hex_parts(dn, Hdn[jj], L)
                m.put(blk.layer, js, bgu, bdn)
                t = f"{pre}.tier{L}"
                out[f"{t}.experts"] = torch.tensor(js, dtype=torch.int32)
                for proj, parts, sl in (("gate_proj", pg, slice(0, I)), ("up_proj", pg, slice(I, 2 * I)),
                                        ("down_proj", pd, slice(None))):
                    for k, v in parts.items():
                        out[f"{t}.{proj}.{k}"] = (v[:, sl] if v.dim() == 3 else v).cpu().clone().contiguous()
                tiers[str(L)] = tiers.get(str(L), 0) + len(js)
                del gu, dn, pg, pd, bgu, bdn
            nbytes += sum(v.numel() * v.element_size() for v in out.values())
            save_file(out, str(work / f"layer_{blk.layer:03d}.safetensors"))
            del Hin, Hdn, out
            done += 1
            emit(STAGE, pct=2 + 76 * done / nL, msg=f"layer {done}/{nL} · {E} experts ({n_hot} on {2 ** hot_bits} "
                                                     f"levels) · {time.time() - t0:.0f}s")
        del hs
        free()
    quant_s = time.time() - t0
    emit(STAGE, pct=80, msg="HEX model vs teacher on held-out answers")
    r = m.compare(lambda i, n: emit(STAGE, pct=80 + 12 * i / n, msg=f"held-out {i}/{n} vs teacher"))
    params, n_calib = m.params, m.n_calib
    del m
    free()

    def new_tensors():
        from safetensors.torch import load_file
        for f in sorted(work.glob("layer_*.safetensors")):
            yield f"hex-{f.stem}.safetensors", load_file(str(f))

    dst = hex_dir(rj)
    qc = {"quant_method": "blackhole-hex", "version": 1, "block": HEX_BLOCK, "source": "hexvllm", "tiers": tiers,
          "rank": "REAP saliency" if hot_share > 0 else None}
    total = write_checkpoint(src, dst, names, new_tensors, {"quantization_config": qc},
                             lambda i, n, what: emit(STAGE, pct=93 + 6 * i / n, msg=f"writing {dst.name}: {what}"))
    if isinstance(cfg.get("text_config"), dict):   # transformers' VLM configs keep a copy on the text config
        c = json.loads((dst / "config.json").read_text())
        c["text_config"].pop("quantization_config", None)
        (dst / "config.json").write_text(json.dumps(c, indent=2) + "\n")
    shutil.rmtree(work, ignore_errors=True)
    bh.link(dst)
    avg = bits * (1 - hot_share) + hot_bits * hot_share if hot_share > 0 else bits
    res = {"format": f"HEX experts for vLLM: {2 ** bits}-level per-expert tables"
                     + (f", top {hot_share:.0%} by REAP {2 ** hot_bits}-level" if hot_share > 0 else "")
                     + f", block {HEX_BLOCK}, fp8 block scales",
           "bpw": round(avg + 8 / HEX_BLOCK, 3), "expert_params": params, "expert_gb": round(gib(nbytes), 2),
           "size_gb": round(gib(total), 2), "out": str(dst), "tiers": tiers, "calib_seqs": n_calib,
           "quant_s": round(quant_s, 1), "healed": src != rj.out_dir}
    if r:
        res.update(top1=round(r["top1_agreement"], 4), kl=round(r["kl"], 4), ppl_teacher=round(r["ppl_teacher"], 3),
                   ppl_student=round(r["ppl_pruned"], 3))
    rj.write("work/hexvllm.json", res)
    rj.mark_done(STAGE, res)
    msg = f"HEX for vLLM {res['size_gb']:.1f} GB (experts {res['expert_gb']:.1f} GB)"
    if r:
        msg += f" · KL {res['kl']:.3f} · top-1 {res['top1']:.0%}"
    emit(STAGE, "done", 100, msg, **{k: res[k] for k in ("kl", "top1", "size_gb", "expert_gb", "out") if k in res})


if __name__ == "__main__":
    stage_main(run, STAGE)
