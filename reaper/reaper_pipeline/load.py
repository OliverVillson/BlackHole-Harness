"""Load a model and tokenizer the way the run profile says."""

from __future__ import annotations

import json
from pathlib import Path

from reaper_pipeline.hardware import torch_dtype


def checkpoint_config(path: str | Path) -> dict:
    p = Path(path) / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_tokenizer(path, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    from reaper_pipeline.models import register_all
    register_all()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(path, prof: dict, trust_remote_code: bool = False):
    """bf16/fp16 spread over the GPUs (spilling to RAM if the plan says so), or CPU."""
    import transformers
    from transformers import AutoModelForCausalLM

    from reaper_pipeline.models import register_all
    register_all()  # Flint and other architectures transformers does not ship yet

    kw: dict = {"trust_remote_code": trust_remote_code}
    dkey = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    kw[dkey] = torch_dtype(prof.get("dtype", "bfloat16"))
    if prof.get("device") == "cuda":
        kw["device_map"] = "auto"
        r = prof.get("reap") or {}
        mm = r.get("max_memory")
        if r.get("offload_folder"):
            kw["offload_folder"] = r["offload_folder"]
        if mm:
            kw["max_memory"] = {(int(k) if str(k).isdigit() else k): v for k, v in mm.items()}
            dm = lookup_tables_on_cpu(path, kw["max_memory"], trust_remote_code)
            if dm is not None:
                kw["device_map"] = dm
    q = checkpoint_config(path).get("quantization_config") or {}
    if str(q.get("quant_method", "")).lower() == "mxfp4":
        # gpt-oss: REAP hooks need the experts as plain tensors, so unpack MXFP4 to bf16.
        # The pruned checkpoint is written from the original MXFP4 files, not from this copy.
        from transformers import Mxfp4Config
        kw["quantization_config"] = Mxfp4Config(dequantize=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **kw)
    except Exception as e:  # noqa: BLE001
        if not isinstance(kw.get("device_map"), dict):
            raise
        from reaper_pipeline.events import log
        log(f"load: the CPU-side lookup-table placement failed ({type(e).__name__}: {e}); loading with 'auto'")
        free()
        kw["device_map"] = "auto"
        model = AutoModelForCausalLM.from_pretrained(path, **kw)
    return model.eval()


def _gib(v) -> float:
    if isinstance(v, (int, float)):
        return v / 1024 ** 3
    v = str(v).upper().replace(" ", "")
    for unit, f in (("GIB", 1.0), ("GB", 1e9 / 1024 ** 3), ("MIB", 1 / 1024), ("MB", 1e6 / 1024 ** 3)):
        if v.endswith(unit):
            return float(v[:-len(unit)]) * f
    return float(v) / 1024 ** 3


def lookup_tables_on_cpu(path, max_memory: dict, trust_remote_code: bool = False) -> dict | None:
    """A device map that keeps a model's lookup-only tables (transformers' `_no_placement_params`,
    e.g. Qwen3.8-Flash-Next's ~100 GB n-gram embedding) in CPU RAM, where the model's own code
    looks them up and moves only the looked-up rows to the GPU, and places everything else with
    accelerate as "auto" would, with the GPU room those tables would have taken. None when the
    model has no such tables (then transformers' "auto" is used as before) or on any problem."""
    try:
        import torch
        from accelerate import infer_auto_device_map, init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote_code)
        with init_empty_weights():
            meta = AutoModelForCausalLM.from_config(cfg, trust_remote_code=trust_remote_code)
        keep = getattr(meta, "_no_placement_params", None)
        if not keep:
            return None
        params = dict(meta.named_parameters())
        skip = sorted(n for n in params if any(n == k or n.endswith("." + k) for k in keep))
        if not skip:
            return None
        skip_mods = {n.rsplit(".", 1)[0] for n in skip}
        skip_gib = sum(params[n].numel() * params[n].element_size() for n in skip) / 1024 ** 3
        gpus = [k for k in max_memory if isinstance(k, int)]
        if not gpus:
            return None
        mm = {k: (f"{int(_gib(v) + skip_gib / len(gpus))}GiB" if isinstance(k, int) else v) for k, v in max_memory.items()}
        dm = infer_auto_device_map(meta, max_memory=mm, no_split_module_classes=meta._no_split_modules or [],
                                   dtype=torch.bfloat16)

        def carve(dm: dict, target: str) -> dict:
            # replace the entry that holds `target` by its children, recursively, minus the target
            owner = next((k for k in dm if k == "" or target == k or target.startswith(k + ".")), None)
            if owner is None:
                return dm
            dev = dm.pop(owner)
            if owner == target:
                return dm
            mod = meta.get_submodule(owner) if owner else meta
            for name, _ in mod.named_children():
                full = f"{owner}.{name}" if owner else name
                dm[full] = dev
            for name, p in mod.named_parameters(recurse=False):
                dm[f"{owner}.{name}" if owner else name] = dev
            for name, b in mod.named_buffers(recurse=False):
                dm[f"{owner}.{name}" if owner else name] = dev
            return carve(dm, target)

        for t in sorted(skip_mods):
            dm = carve(dict(dm), t)
        from reaper_pipeline.events import log
        log(f"load: {skip_gib:.0f} GiB of lookup tables ({', '.join(sorted(skip_mods))[:120]}) stay in CPU RAM")
        return dm
    except Exception as e:  # noqa: BLE001  (never block a load on this; "auto" still works)
        try:
            from reaper_pipeline.events import log
            log(f"load: lookup tables stay where 'auto' puts them ({type(e).__name__}: {e})")
        except Exception:  # noqa: BLE001
            pass
        return None


def input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def free() -> None:
    """Release memory after the caller dropped its references."""
    import gc

    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
