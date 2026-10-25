"""vLLM support for HEX models: the "blackhole-hex" quantization method.

Installed with the runtime package, vLLM finds it as a general plugin (entry point
vllm.general_plugins -> register), in every vLLM process, so a HEX folder serves as is:

    vllm serve <hex folder> --max-model-len 32768

A HEX folder (blackhole.anatomy hexops `hex-pack`, or convert.py from an MLX HEX export) stores
each MoE layer's routed experts as tiers (`<mlp>.experts.tier16` / `.tier8`: experts sharing
one table size), each with the int32 ids of its experts and per projection the stacked HEX parts
(see hex_ref.py). Everything else is plain bf16. config.json carries
  quantization_config = {"quant_method": "blackhole-hex", "version": 1, "block": 32, ...,
                         optional "head_ids": [...]}   # output-trimmed LM head: other tokens masked
The experts run through the HEX Triton kernel (hex_moe.py). Single GPU (tensor and expert
parallel size 1): the target is one mid-tier NVIDIA card.

Written against vLLM 0.32's modular MoE interface (FusedMoEMethodBase.apply with topk ids).
"""

from __future__ import annotations

import re

import torch

NAME = "blackhole-hex"
HEAD_FILL = -1.0e4   # logit of a token an output-trimmed LM head cannot write
_TIER = re.compile(r"^(?:.*\.)?tier(\d+)\.(?:(experts)|(gate_proj|up_proj|down_proj)\.(idx|block_scale|expert_scale|levels))$")


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def register() -> None:
    """vLLM general-plugin entry point; safe to call more than once."""
    from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase, RoutedExperts
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, UnquantizedEmbeddingMethod

    from blackhole_runtime.hex_moe import HexTier, hex_moe

    class HexHeadMethod(UnquantizedEmbeddingMethod):
        """bf16 LM head whose logits outside the kept output tokens are HEAD_FILL."""

        def __init__(self, ids):
            super().__init__()
            self.ids = ids

        def process_weights_after_loading(self, layer) -> None:
            super().process_weights_after_loading(layer)
            mask = torch.full((layer.weight.shape[0],), HEAD_FILL, dtype=layer.weight.dtype,
                              device=layer.weight.device)
            mask[torch.as_tensor(self.ids, dtype=torch.long, device=mask.device)] = 0
            layer.register_buffer("hex_head_mask", mask, persistent=False)

        def apply(self, layer, x, bias=None):
            return super().apply(layer, x, bias) + layer.hex_head_mask

    class HexMoEMethod(FusedMoEMethodBase):
        def __init__(self, moe):
            super().__init__(moe)

        def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition,
                           params_dtype, **extra_weight_attrs):
            if getattr(self.moe, "tp_size", 1) != 1 or getattr(self.moe, "ep_size", 1) != 1:
                raise ValueError("blackhole-hex runs on one GPU: use --tensor-parallel-size 1 and no expert parallel")
            layer.hex_shape = (num_experts, hidden_size, intermediate_size_per_partition)
            layer.hex_tiers = []
            # The tiers' tensor shapes are only known from the checkpoint, so they are registered as
            # they load: this layer's loader replaces RoutedExperts.load_weights (per-expert names).
            layer.load_weights = lambda weights: self._load(layer, weights)

        def _load(self, layer, weights):
            dev = _device()
            for name, t in weights:
                m = _TIER.match(name)
                if not m:
                    raise ValueError(f"blackhole-hex: unexpected expert tensor {layer.layer_name}.{name} "
                                     "(expected <experts>.tier<L>.<proj>.<part> and .tier<L>.experts)")
                L = m.group(1)
                key = f"hex_t{L}_experts" if m.group(2) else f"hex_t{L}_{m.group(3)}_{m.group(4)}"
                layer.register_parameter(key, torch.nn.Parameter(t.to(dev), requires_grad=False))
                yield key

        def process_weights_after_loading(self, layer) -> None:
            E, H, I = layer.hex_shape
            levels = sorted({int(n.split("_")[1][1:]) for n, _ in layer.named_parameters(recurse=False)
                             if n.startswith("hex_t")}, reverse=True)
            if not levels:
                raise ValueError(f"blackhole-hex: no HEX experts were loaded for {layer.layer_name}")
            tiers, seen = [], torch.zeros(E, dtype=torch.bool)
            for L in levels:
                g = lambda proj: {p: getattr(layer, f"hex_t{L}_{proj}_{p}").data  # noqa: E731
                                  for p in ("idx", "block_scale", "expert_scale", "levels")}
                ids = getattr(layer, f"hex_t{L}_experts").data
                tier = HexTier(L.bit_length() - 1, ids, g("gate_proj"), g("up_proj"), g("down_proj")).prepare(E)
                if tier.gate_proj["idx"].shape[1] != I or tier.down_proj["idx"].shape[1] != H:
                    raise ValueError(f"blackhole-hex: {layer.layer_name} tier{L} shapes do not match the model")
                seen[ids.long().cpu()] = True
                tiers.append(tier)
            if not bool(seen.all()):
                missing = (~seen).nonzero().flatten().tolist()
                raise ValueError(f"blackhole-hex: {layer.layer_name} has no weights for experts {missing[:8]}")
            layer.hex_tiers = tiers

        def get_fused_moe_quant_config(self, layer):
            return None

        def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
            return hex_moe(x, topk_weights, topk_ids.long(), layer.hex_tiers)

    @register_quantization_config(NAME)
    class HexConfig(QuantizationConfig):
        def __init__(self, cfg: dict):
            super().__init__()
            self.cfg = cfg
            self.head_ids = cfg.get("head_ids")

        def get_name(self):
            return NAME

        def get_supported_act_dtypes(self):
            return [torch.bfloat16, torch.float16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 80

        @staticmethod
        def get_config_filenames():
            return []

        @classmethod
        def from_config(cls, config: dict):
            return cls(config)

        def get_quant_method(self, layer, prefix: str):
            if isinstance(layer, RoutedExperts):
                return HexMoEMethod(layer.moe_config)
            if isinstance(layer, ParallelLMHead) and self.head_ids:
                return HexHeadMethod(self.head_ids)
            if isinstance(layer, LinearBase):
                return UnquantizedLinearMethod()
            return None
