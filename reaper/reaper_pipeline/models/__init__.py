"""PyTorch versions of MoE architectures transformers does not ship yet."""


def register_all() -> None:
    """Register them with the transformers Auto classes (ours win over transformers' own: the stages are written against them)."""
    from reaper_pipeline.models import flint
    flint.register()
