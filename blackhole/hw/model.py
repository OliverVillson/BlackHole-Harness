"""Bring any model: resolve a name to a checkpoint and detect what it is.

    from blackhole.hw.model import resolve, detect, applicable
    rid, others = resolve("qwen3 30b a3b")        # fuzzy name -> Hugging Face id (or a local dir)
    info = detect(rid)                             # config.json + safetensors index + tokenizer_config.json, never the weights
    info.kinds                                     # {"moe"} | {"dense"} | {"hybrid", "moe"} ...
    applicable(info, schema)                       # [] when the schema can run on it, else the reasons

Detection reads a few KB, from a local folder or the Hub, so the TUI can answer "what is
this and what can I do with it" before anything downloads. It is the first stage of every
run (model.detect), its result is model.json in the job and ${base.*} in schemas
(${base.experts}, ${base.kinds}, ${base.params_b}, ...), and the marketplace's model list is
only a shortcut of pre-detected entries on top of it.

Ported from REAPer's reaper_hw/model.py (sizes, experts, top-k, MoE layers, MLA) and
reaper_tui/backend.py (resolve_model, rank), with hybrid, vision, tokenizer and
remote-code detection added.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

GB = 1e9
EXPERT_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
TOP_K_KEYS = ("num_experts_per_tok", "moe_k", "moe_topk", "top_k", "num_selected_experts")
SHARED_KEYS = ("n_shared_experts", "num_shared_experts", "shared_expert_intermediate_size")
# Layer kinds that are not plain attention: linear attention, DeltaNet, Mamba/SSM.
RECURRENT = re.compile(r"linear|delta|mamba|ssm|recurrent|rwkv", re.I)
OFFICIAL = ("qwen", "deepseek-ai", "zai-org", "thudm", "openai", "mistralai", "meta-llama", "moonshotai",
            "minimaxai", "baidu", "tencent", "allenai", "google", "microsoft", "ibm-granite", "inclusionai",
            "xiaomimimo", "stepfun-ai", "nvidia", "lgai-exaone", "cohere", "arcee-ai", "meituan-longcat")
REUPLOAD = re.compile(r"gguf|awq|gptq|mlx|fp8|int4|int8|bnb|4bit|8bit|exl2|nvfp4|quant|reap|pruned|abliterated", re.I)


@dataclass
class ModelInfo:
    id: str
    model_type: str = "?"
    architecture: str = ""           # e.g. Qwen3MoeForCausalLM
    kinds: list[str] = field(default_factory=list)   # moe, dense, hybrid, vision (sorted)
    params_b: float = 0.0
    expert_params_b: float = 0.0     # routed expert weights (what REAP prunes)
    active_b: float = 0.0            # parameters per token
    checkpoint_gb: float = 0.0
    dtype: str = ""                  # torch_dtype of the checkpoint
    quantized: str | None = None     # fp8, mxfp4, compressed-tensors, ... when not 16-bit
    layers: int = 0
    hidden: int = 0
    vocab: int = 0
    experts: int = 0
    top_k: int = 0
    shared_experts: int = 0
    moe_layers: int = 0
    attention_layers: int = 0        # full/sliding attention layers (KV cache grows with these)
    recurrent_layers: int = 0        # linear attention / DeltaNet / Mamba layers (fixed-size state)
    kv_heads: int = 0
    head_dim: int = 0
    context: int = 0                 # max_position_embeddings
    chat_template: bool = False
    remote_code: bool = False        # ships its own modeling code (auto_map) transformers may not know
    notes: list[str] = field(default_factory=list)

    @property
    def kv_bytes_per_token(self) -> int:
        """16-bit KV cache bytes per token over all attention layers (recurrent state excluded)."""
        return 2 * self.attention_layers * self.kv_heads * self.head_dim * 2

    def to_dict(self) -> dict:
        return asdict(self) | {"kv_bytes_per_token": self.kv_bytes_per_token, "bf16_gb": round(self.params_b * 2, 1)}


def _text(cfg: dict) -> dict:
    return cfg.get("text_config") or cfg.get("language_config") or cfg


def _first_int(c: dict, keys) -> int:
    return next((c[k] for k in keys if isinstance(c.get(k), int) and c[k] > 0), 0)


def _moe_layers(c: dict, experts: int) -> int:
    if not experts:
        return 0
    L = c.get("num_hidden_layers", 0)
    if isinstance(c.get("mlp_only_layers"), list):
        L -= len(c["mlp_only_layers"])
    if c.get("decoder_sparse_step", 1) > 1:
        return L // c["decoder_sparse_step"]
    if c.get("interleave_moe_layer_step", 1) > 1:
        return L // c["interleave_moe_layer_step"]
    return L - (c.get("first_k_dense_replace") or 0)


def _layer_split(c: dict) -> tuple[int, int]:
    """(attention layers, recurrent layers) from the config's per-layer types."""
    L = c.get("num_hidden_layers", 0)
    types = c.get("layer_types") or c.get("layers_block_type") or c.get("hybrid_override_pattern")
    if isinstance(types, str):            # Nemotron-H style "M-M-*-M": M mamba, * attention
        types = ["mamba" if ch == "M" else "attention" for ch in types if ch in "M*"]
    if isinstance(types, list) and types:
        rec = sum(1 for t in types if RECURRENT.search(str(t)))
        return len(types) - rec, rec
    if c.get("full_attention_interval"):  # Qwen3-Next: one full attention layer every N
        full = L // c["full_attention_interval"]
        return full, L - full
    if any(k in c for k in ("mamba_d_state", "ssm_state_size", "mamba_n_heads")):
        return 0, L
    return L, 0


def estimate_params(c: dict, experts: int, moe_layers: int) -> tuple[float, float, float]:
    """(total, routed expert, active) parameters in billions from the config alone.
    Within ~5-10%; the safetensors size replaces it when the checkpoint is 16-bit."""
    H, L, V = c.get("hidden_size", 0), c.get("num_hidden_layers", 0), c.get("vocab_size", 32000)
    if not H or not L:
        return 0.0, 0.0, 0.0
    heads = c.get("num_attention_heads") or max(H // 128, 1)
    hd = c.get("head_dim") or H // heads
    kv = c.get("num_key_value_heads", heads)
    if c.get("kv_lora_rank"):  # DeepSeek MLA
        nope, rope, v = c.get("qk_nope_head_dim", 128), c.get("qk_rope_head_dim", 64), c.get("v_head_dim", 128)
        q = c.get("q_lora_rank") or H
        attn = H * q + q * heads * (nope + rope) + H * (c["kv_lora_rank"] + rope) \
            + c["kv_lora_rank"] * heads * (nope + v) + heads * v * H
    else:
        attn = H * hd * (heads * 2 + kv * 2)
    I_moe = c.get("moe_intermediate_size") or c.get("intermediate_size", 4 * H)
    I_dense = c.get("intermediate_size", 4 * H)
    shared = (c.get("n_shared_experts") or c.get("num_shared_experts") or 0) * 3 * H * I_moe
    if c.get("shared_expert_intermediate_size"):
        shared = 3 * H * c["shared_expert_intermediate_size"]
    k = _first_int(c, TOP_K_KEYS)
    expert = moe_layers * experts * 3 * H * I_moe
    dense_ffn = (L - moe_layers) * 3 * H * I_dense
    emb = V * H * (1 if c.get("tie_word_embeddings") else 2)
    total = L * attn + dense_ffn + expert + moe_layers * (experts * H + shared) + emb
    active = total - expert + moe_layers * k * 3 * H * I_moe
    return total / 1e9, expert / 1e9, active / 1e9


def from_config(model_id: str, cfg: dict, checkpoint_bytes: int = 0, tok_cfg: dict | None = None) -> ModelInfo:
    c = _text(cfg)
    experts = _first_int(c, EXPERT_KEYS)
    experts = experts if experts > 1 else 0
    moe_layers = _moe_layers(c, experts)
    attn_layers, rec_layers = _layer_split(c)
    q = cfg.get("quantization_config") or c.get("quantization_config")
    quant = str(q.get("quant_method") or q.get("fmt") or "quantized").lower() if q else None
    total, expert, active = estimate_params(c, experts, moe_layers)
    info = ModelInfo(model_id, c.get("model_type", cfg.get("model_type", "?")),
                     (cfg.get("architectures") or [""])[0])
    if checkpoint_bytes and not quant and total:   # 16-bit checkpoint: the file size is exact
        scale = checkpoint_bytes / 2 / 1e9 / total
        total, expert, active = total * scale, expert * scale, active * scale
    elif checkpoint_bytes and not total:
        total = checkpoint_bytes / 2 / 1e9
        info.notes.append("sizes from the checkpoint only; config has no hidden_size/num_hidden_layers")
    kinds = {"moe"} if experts else {"dense"}
    if rec_layers:
        kinds.add("hybrid")
    if cfg.get("vision_config") or cfg.get("vision_tower") or "vision" in info.architecture.lower():
        kinds.add("vision")
    heads = c.get("num_attention_heads") or 0
    info.kinds = sorted(kinds)
    info.params_b, info.expert_params_b, info.active_b = round(total, 2), round(expert, 2), round(active or total, 2)
    info.checkpoint_gb = round((checkpoint_bytes or total * 2e9) / GB, 1)
    info.dtype = str(c.get("torch_dtype") or cfg.get("torch_dtype") or c.get("dtype") or "")
    info.quantized = quant
    info.layers, info.hidden, info.vocab = c.get("num_hidden_layers", 0), c.get("hidden_size", 0), c.get("vocab_size", 0)
    info.experts, info.top_k, info.moe_layers = experts, _first_int(c, TOP_K_KEYS) if experts else 0, moe_layers
    info.shared_experts = c.get("n_shared_experts") or c.get("num_shared_experts") or (
        1 if c.get("shared_expert_intermediate_size") else 0)
    info.attention_layers, info.recurrent_layers = attn_layers, rec_layers
    info.kv_heads = c.get("num_key_value_heads") or heads
    info.head_dim = c.get("head_dim") or (info.hidden // heads if heads else 0)
    info.context = c.get("max_position_embeddings", 0)
    info.chat_template = bool((tok_cfg or {}).get("chat_template"))
    info.remote_code = bool(cfg.get("auto_map"))
    if c.get("num_nextn_predict_layers") or c.get("mtp_num_hidden_layers"):
        info.notes.append("has multi-token-prediction layers (dropped by pruning, usable for speculative decoding)")
    if not info.chat_template and tok_cfg is not None:
        info.notes.append("no chat template: a base model; schemas that train on chat data add one")
    return info


def _reader(model_id: str, local_dir: str | None):
    root = Path(local_dir or model_id).expanduser()

    def fetch(name: str) -> dict | None:
        if (root / name).is_file():
            return json.loads((root / name).read_text())
        if root.is_dir():
            return None
        try:
            from huggingface_hub import hf_hub_download
            return json.loads(Path(hf_hub_download(model_id, name)).read_text())
        except Exception:
            return None
    return root, fetch


def detect(model_id: str, local_dir: str | None = None) -> ModelInfo:
    """What a model is, from config.json, the safetensors index and tokenizer_config.json
    in a local folder or on the Hub (a few KB). Raises RuntimeError when no config can be
    read: offline, gated (needs `hf auth login`), a GGUF-only repo, or not a model."""
    root, fetch = _reader(model_id, local_dir)
    cfg = fetch("config.json")
    if cfg is None:
        raise RuntimeError(f"cannot read config.json for {model_id}: offline, gated, GGUF-only, or not a model repo")
    idx = fetch("model.safetensors.index.json") or {}
    size = int((idx.get("metadata") or {}).get("total_size") or 0)
    if not size and root.is_dir():
        size = sum(p.stat().st_size for p in root.glob("*.safetensors"))
    return from_config(model_id, cfg, size, fetch("tokenizer_config.json"))


def applicable(info: ModelInfo, schema) -> list[str]:
    """Why a schema cannot run on this model ([] when it can): its targets, and every op's
    architecture needs. The TUI greys out schemas with reasons; the runner refuses early."""
    why = []
    t = schema.targets
    if t.get("architectures") and not set(t["architectures"]) & set(info.kinds):
        why.append(f"{schema.ref} targets {'/'.join(t['architectures'])}; {info.id} is {'/'.join(info.kinds)}")
    if t.get("min_params_b") and info.params_b and info.params_b < t["min_params_b"]:
        why.append(f"needs at least {t['min_params_b']}B parameters; {info.id} has {info.params_b}B")
    if t.get("max_params_b") and info.params_b > t["max_params_b"]:
        why.append(f"needs at most {t['max_params_b']}B parameters; {info.id} has {info.params_b}B")
    if t.get("families") and info.model_type not in t["families"]:
        why.append(f"made for {', '.join(t['families'])}; {info.id} is {info.model_type}")
    from blackhole.core import ops
    kinds = set(info.kinds)          # what the model is at each stage: convert.moefy turns dense into moe
    for st in schema.stages:
        try:
            cls = ops.get(st.op, st.major)
        except KeyError:
            continue
        if cls.needs.arch and not set(cls.needs.arch) & kinds:
            why.append(f"stage {st.id} ({st.uses}) needs {'/'.join(cls.needs.arch)}; the model is {'/'.join(sorted(kinds))} there")
        if cls.yields:
            kinds = (kinds - {"dense", "moe"}) | set(cls.yields)
    return why


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "", s.lower())


def rank(query: str, ids: list[tuple[str, int]]) -> list[str]:
    """Repo ids (with download counts) best match first: the whole query in the name,
    the maker's own repo, not a quantized or pruned re-upload, the instruct version."""
    nq = _norm(query.split("/")[-1])

    def score(item) -> tuple:
        rid, downloads = item
        org, name = rid.split("/", 1) if "/" in rid else ("", rid)
        n = _norm(name)
        return (nq in n, org.lower() in OFFICIAL, not REUPLOAD.search(rid), "instruct" in n or "chat" in n,
                -abs(len(n) - len(nq)), downloads or 0)

    return [rid for rid, _ in sorted(ids, key=score, reverse=True)]


def resolve(query: str, api=None, known: list[str] = ()) -> tuple[str | None, list[str]]:
    """A fuzzy name to a Hugging Face id (or a local dir), plus up to five other candidates.
    `known` (the marketplace's ids) is tried before searching the Hub."""
    q = query.strip()
    if Path(q).expanduser().is_dir():
        return str(Path(q).expanduser().resolve()), []
    nq = _norm(q.split("/")[-1])
    local = [(k, 0) for k in known if k.lower() == q.lower() or nq in _norm(k.split("/")[-1])]
    if local:
        ranked = rank(q, local)
        return ranked[0], ranked[1:6]
    if api is None:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
        except ImportError:
            return (q if "/" in q else None), []
    if "/" in q:
        try:
            return api.model_info(q).id, []
        except Exception:
            q = q.split("/", 1)[1]
    parts = q.replace(" ", "-").split("-")   # drop trailing parts ("-A3B", "-27B") until the Hub finds something
    terms = [q, "-".join(parts)] + ["-".join(parts[:n]) for n in range(len(parts) - 1, 0, -1)]
    found: list = []
    for term in dict.fromkeys(terms):
        try:
            found = list(api.list_models(search=term, sort="downloads", limit=50))
        except Exception:
            return None, []
        if found:
            break
    if not found:
        return None, []
    ranked = rank(q, [(m.id, getattr(m, "downloads", 0)) for m in found])
    return ranked[0], ranked[1:6]
