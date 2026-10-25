"""The built-in op catalog. One module per namespace; `blackhole ops` prints them all
with their params (blackhole.core.ops.describe).

Most ops are ports of code that already ran on real models; `ported_from` on each
op names the source so the port keeps its behaviour and its tests:

    REAPer   OliverVillson/LobBot @ claude/project-thread-7e6em9, reaper/reaper_pipeline
    LobBot   OliverVillson/LobBot @ main, stages/ (heal, quantize, bits, eval, package)
    BH       OliverVillson/llmap @ claude/centipede-train-pipeline, scripts/event_horizon.py, scripts/bh/
"""
