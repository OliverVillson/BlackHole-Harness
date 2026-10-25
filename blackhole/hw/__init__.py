"""Hardware detection and run planning: REAPer's reaper_hw moved here unchanged
(hardware.py, model.py, plan.py and their tests), plus two additions.

    from blackhole.hw import detect, plan
    hw = detect()                 # GPUs (name, mem_gb, cc), driver, CUDA, torch arches, CPUs, RAM, disk
    hw.to_dict()                  # what the runner writes to hardware.json and exposes as ${hw.*}

Additions for the Harness:
- model.py: bring any model. resolve() turns a fuzzy name into a Hub id or folder,
  detect() reads a few KB and says what it is (moe/dense/hybrid/vision, experts, top-k,
  sizes, KV per token, chat template), applicable() says which schemas can run on it.
  It runs as the first stage of every job (model.detect).
- plan_schema(hw, schema, base) asks every stage's op for its StagePlan (memory, disk,
  hours, frontier spend) and sums them into the run plan the TUI shows before start,
  with a GPU-hour cost from blackhole.cloud's price table when the VM is rented.
- Device classes for the *target* machine, the one the user will run the result on:
  "8GB-VRAM", "12GB-VRAM", "16GB-VRAM", "M-series-16GB", "M-series-32GB", "CPU-32GB".
  Each has memory and bandwidth, so eval.footprint and the marketplace's "Fits your
  PC" column can say whether the output runs there and at about what tok/s
  (LobBot's laptop_bandwidth_gb_s estimate).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

DEVICES = {
    #                memory GB, bandwidth GB/s, kind
    "8GB-VRAM": (8, 272, "cuda"),          # RTX 4060 class
    "12GB-VRAM": (12, 504, "cuda"),        # RTX 4070 class
    "16GB-VRAM": (16, 448, "cuda"),        # RTX 4060 Ti 16G / 5060 Ti class
    "24GB-VRAM": (24, 1008, "cuda"),       # RTX 4090 class
    "M-series-16GB": (16, 120, "metal"),   # M4/M5 Air; unified memory, ~10 GB usable for weights
    "M-series-32GB": (32, 273, "metal"),
    "CPU-32GB": (32, 60, "cpu"),
}


@dataclass
class Hardware:
    gpu_name: str = ""
    gpu_count: int = 0
    gpu_mem_gb: float = 0.0
    cc: str = ""
    driver: str = ""
    cuda: str = ""
    cpus: int = 0
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def detect() -> Hardware:
    """Port target: reaper_hw.hardware.detect (nvidia-smi, then torch). Until the port
    lands this reads nvidia-smi only, like REAPer's TUI fallback."""
    import os
    import shutil
    import subprocess
    hw = Hardware(cpus=os.cpu_count() or 0, disk_free_gb=round(shutil.disk_usage(os.path.expanduser("~")).free / 1e9, 1))
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        rows = [r.split(", ") for r in out.strip().splitlines() if r.strip()]
        if rows:
            hw.gpu_name, hw.gpu_count = rows[0][0], len(rows)
            hw.gpu_mem_gb, hw.cc, hw.driver = round(float(rows[0][1]) / 1024, 1), rows[0][2], rows[0][3]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        pass
    return hw
