"""BlackHole-Harness: shrink large open-weight models into ones that run on a personal
computer, by running shareable schemas (recipes of pruning, distillation, healing and
quantisation ops) on a GPU VM, from a TUI over SSH.

    blackhole/core     schema format, op interface, job runner, stage cache, receipts
    blackhole/ops      the built-in op catalog (ported from REAPer, LobBot and BlackHole)
    blackhole/hw       hardware detection and run planning (REAPer's reaper_hw)
    blackhole/models   PyTorch ports of architectures transformers lacks (Flint)
    blackhole/agent    frontier models (Claude, GPT) that turn a description into a schema
    blackhole/market   the schema registry: search, pull, publish, receipts, proven badges
    blackhole/tui      the Textual TUI (`blackhole`, `bh`)
    blackhole/cloud    GPU VM bring-up on a provider, and the installer
"""

__version__ = "0.1.0"
