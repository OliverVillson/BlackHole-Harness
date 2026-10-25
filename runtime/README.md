# BlackHole runtime

Fast kernels and serving for models made with BlackHole-Harness. Kept apart from the pipeline
(`harness/`): the pipeline never imports this package, and this package only needs the pipeline
to convert an MLX export.

First piece: **HEX experts on NVIDIA**. HEX (Header-Encoded eXperts, a BlackHole-Harness format)
stores each MoE expert with its own learned 8- or 16-level value table in its header, fp8 block
scales per 32 weights and 3- or 4-bit indices. Until now only the Mac could run it (Metal kernel
in `harness/blackhole_pipeline/mlx/flint_mlx.py`). This package adds:

| file | what |
| --- | --- |
| `blackhole_runtime/hex_moe.py` | Triton fused MoE: grouped GEMM over the routed experts, HEX weights decoded tile by tile in registers, gate and up in one pass, no bf16 copy of the experts |
| `blackhole_runtime/hex_ref.py` | torch reference (the layout's source of truth is `blackhole.anatomy.hex`) |
| `blackhole_runtime/hex_check.py` | GPU check against the reference + micro-benchmark against vLLM's bf16 and int4 fused MoE |
| `blackhole_runtime/vllm_hex.py` | the `blackhole-hex` quantization method, registered in vLLM as a plugin |
| `blackhole_runtime/convert.py` | MLX HEX export → `blackhole-hex` folder for vLLM |
| `blackhole_runtime/verify.py` | converts, then proves every layer's experts and the kernel's output match the MLX export |

## Test on a CPU

```bash
pip install torch triton -e 'runtime[dev]'
TRITON_INTERPRET=1 pytest -q runtime/tests
```

## Check and benchmark on an NVIDIA GPU

Use a separate clone and venv, so a pipeline job running on the same machine is not touched:

```bash
git clone -b claude/project-thread-8ja0p5 https://github.com/OliverVillson/BlackHole-Harness ~/bh-runtime
python3 -m venv ~/runtime-venv && source ~/runtime-venv/bin/activate
pip install -U pip && pip install vllm ninja && pip install -e ~/bh-runtime/runtime
blackhole-hex-check            # Qwen3.6-35B-A3B REAP 31.25% MoE shape: 176 experts, 2048 x 512, top-8
blackhole-hex-check --sweep    # also tries tile sizes; send the "best tiles" lines
```

It uses a few hundred MB of GPU memory and random weights, no download. Timings are only
meaningful while nothing else is using the GPU.

## Serve a HEX model with vLLM

```bash
pip install -e ~/bh-runtime/harness                               # the converter reads MLX exports with it
python -m blackhole_runtime.verify <MLX HEX export> ~/models/<name>-HEX-vllm  # converts, then checks every layer
vllm serve ~/models/<name>-HEX-vllm --max-model-len 32768
```

One GPU (no tensor or expert parallelism). If CUDA-graph capture fails, add `--enforce-eager`
and report it. Non-expert weights are bf16 in the converted folder for now.
