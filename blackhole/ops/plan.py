"""plan.*: decisions a schema makes from what was detected, before any GPU work."""

from dataclasses import dataclass

from blackhole.core.ops import Op, p, register

BITS = {"Q8_0": 8.5, "Q6_K": 6.56, "Q5_K_M": 5.69, "Q4_K_M": 4.85, "Q3_K_M": 3.91}   # bits per weight


@register
class Fit(Op):
    """Pick the gentlest recipe that meets a file-size target by walking a ladder, gentlest
    first: quantise harder before pruning. The default ladder is the dense frontier on
    Qwen3-8B (B200, 2026-10-23, GPTQ):
        Q4_K_M            ~6.1 GB  KL 0.028
        Q3_K_M            ~5.3 GB  KL 0.15   (beats every pruned recipe of similar size)
        Q3_K_M + 12.5%    ~5.0 GB  KL 0.25
        Q3_K_M + 25%
    Reports metrics later stages read: bits, prune (0 when none), need_prune, predicted_gb,
    fits. If the last rung misses the target, fits is false and the schema's gate stops the
    run with the reason; layer dropping is never on the ladder (KL 0.99 for 4 layers).

    MoE models use the same shape with experts as the prunable part (moe-prune-int4):
        REAP 25% + Q4_K_M  ~10.4 GB experts  KL 0.034   (Qwen3-30B-A3B, GPTQ)
        REAP 25% + Q3_K_M
        REAP 37.5% + Q3_K_M  6.9 GB          KL 0.121   (50% + int4 at the same size: 0.171)
    Pruning never goes past 50%: past that every MoE and dense run fell apart.

    Planned (not on the ladder yet): once quant.expert_codebook has a real kernel and a
    packed format, "REAP 25% + REAQ codebooks" replaces the Q3_K_M tiers for MoE. Lab
    numbers (Qwen3-30B-A3B, B200, 2026-10-23, simulated quant): 16/8-level REAQ tables with
    GPTQ rounding KL 0.033 at 9.5 GB vs Q4_K_M rung 0.034 at 10.4 GB; near 9 GB the best goes
    from 0.074 (REAP 37.5% + GPTQ int4) to 0.041. Not added now because nothing can serve it.

    It estimates the GGUF file from ${base.params_b} and the prunable share of the weights, so
    it runs on the CPU in a second. The frontier's sizes were safetensors with 16-bit
    embeddings, so GGUF files are smaller (Qwen3-8B Q4_K_M ~5.0 GB); eval.footprint
    measures the real file at the end."""
    name, major = "plan.fit", 1
    produces = {"plan": "file"}
    ported_from = "new; LobBot bits.py size estimate; market/models.py best_fit"

    @dataclass
    class Params:
        params_b: float = p(0.0, "base parameters, e.g. ${base.params_b}")
        ffn_share: float = p(0.65, "share of weights pruning can remove: FFNs in dense LLMs (~0.6-0.7)")
        prunable_b: float = p(0.0, "billions of prunable weights instead of ffn_share, e.g. ${base.expert_params_b} for MoE")
        max_gb: float = p(0.0, "file target; 0 = no target (first rung)")
        ladder: list = p([{"bits": "Q4_K_M", "prune": 0.0}, {"bits": "Q3_K_M", "prune": 0.0},
                          {"bits": "Q3_K_M", "prune": 0.125}, {"bits": "Q3_K_M", "prune": 0.25}],
                         "rungs tried in order, gentlest first")

    def run(self, ctx):
        import json
        P = ctx.params
        if any(r["prune"] > 0.5 for r in P.ladder):
            raise ValueError("a ladder rung prunes past 50%; every measured run fell apart there")
        share = P.prunable_b / P.params_b if P.prunable_b and P.params_b else P.ffn_share

        def size(r: dict) -> float:
            return round(P.params_b * (1 - share * r["prune"]) * BITS[r["bits"]] / 8, 2)

        pick = next((r for r in P.ladder if not P.max_gb or size(r) <= P.max_gb), P.ladder[-1])
        plan = {"bits": pick["bits"], "prune": pick["prune"], "need_prune": pick["prune"] > 0,
                "predicted_gb": size(pick), "fits": not P.max_gb or size(pick) <= P.max_gb,
                "rung": P.ladder.index(pick)}
        ctx.out("plan").write_text(json.dumps(plan, indent=2) + "\n")
        ctx.metric(**plan)
        ctx.progress(100, f"{pick['bits']}" + (f" + prune {round(pick['prune'] * 100)}%" if pick["prune"] else "")
                     + f": ~{size(pick)} GB")
