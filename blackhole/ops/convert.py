"""convert.*: changing what kind of model it is, so ops for another kind apply.

convert.moefy turns a dense model into a mixture of experts, which lets the MoE toolkit
(score.reap, prune.reap, quant.reaq) work on dense models too."""

from dataclasses import dataclass

from blackhole.core.ops import Needs, Op, p, register


@register
class Moefy(Op):
    """LAB ONLY: measured negative. On Qwen3-8B (B200, 2026-10-23) 64 experts at top-32 cost
    KL 0.53 (0.39 with an oracle router): SiLU FFNs are not sparse enough to split cleanly,
    and REAP on the new routers pruned about as badly as random (KL 0.67 vs 0.71). Dense
    models use score.neurons + prune.neurons; REAP stays for real MoEs.

    MoEfication: split every FFN's intermediate neurons into `experts` groups and train a
    small router per layer to pick `top_k` of them per token, so the dense model becomes a
    MoE with the same weights. Then REAP can score and drop whole experts, and inference
    only computes the chosen ones.

      split   cluster   k-means on each neuron's up/gate weight vectors (parameter clustering)
              coact     balanced graph partition of neuron co-activation on the data (the
                        MoEfication paper's best split; needs activations, so slower)
              random    equal random groups (a baseline)
      router  mlp       a linear (or 2-layer) router trained to predict which groups hold the
                        largest activations, on `router_tokens` tokens of the data so far
              centroid  no training: score each expert by the token's similarity to its
                        neurons' mean weight vector
      shared  fraction of neurons kept always on as a shared expert (the ones active on most
              tokens), which the LLaMA-MoE line of work found steadies quality

    Output is a standard MoE checkpoint so every later op and runtime loads it:
    Mixtral layout for Llama-style attention, Qwen3-MoE layout for Qwen3 (qk-norm),
    Qwen2-MoE layout when attention has biases or a shared expert is kept. Metrics:
    recon_kl (dense vs moefied on held-out text before any pruning), router_recall (share
    of the true top activation mass the router picks), active_b."""
    name, major = "convert.moefy", 1
    consumes = {"model": "model", "data": "dataset"}
    produces = {"model": "model", "router": "file", "report": "file"}
    needs = Needs(gpu=True, arch=("dense",))
    yields = ("moe",)
    ported_from = "new (MoEfication, Zhang et al. 2022; LLaMA-MoE); REAPer moe.py for the output layout"

    @dataclass
    class Params:
        model: str = p("")
        data: str = p("", "text for co-activation statistics and router training (the data so far)")
        experts: int = p(16, "experts per FFN layer")
        top_k: int = p(4, "experts per token; active FFN share = top_k / experts")
        split: str = p("cluster", "cluster | coact | random")
        router: str = p("mlp", "mlp | centroid")
        router_tokens: int = p(2_000_000, "tokens of activations the routers train on")
        router_hidden: int = p(0, "0 = linear router; >0 = 2-layer MLP of this width")
        shared: float = p(0.0, "share of neurons kept always on as a shared expert, e.g. 0.125")
        layers: str = p("all", "all | skip_first_last | a list like 2-29")
        target_format: str = p("auto", "auto | mixtral | qwen3_moe | qwen2_moe")
