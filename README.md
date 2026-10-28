# BlackHole-Harness

Shrink a large open-weight model into one that runs on your own PC. You SSH into a GPU VM,
run `blackhole`, pick or describe a **schema** (a shareable recipe of pruning, distillation,
healing and quantisation steps), point it at any base model (a Hugging Face id or a folder,
MoE or dense, auto-detected), and get a GGUF, MLX or safetensors model out, with a receipt of
every step and how much quality it cost.

Site: [blackholeharness.dev](https://blackholeharness.dev) (source in `landing/`).

## What makes it different

Beyond REAP expert pruning and GPTQ, the harness ships techniques that originate in this
project:

- **REAQ** (Router-weighted Expert Activation Quantization): REAP's per-expert router-weighted
  score sets each expert's bit width or table size at a fixed average size, without removing
  experts.
- **HEX** (Header-Encoded eXperts): per-expert learned 8/16-level value tables in the expert
  header, 3/4-bit indices, FP8 block scales, GPTQ rounding. On Qwen3-30B (REAP 25%) it reaches
  KL 0.054 at 3.25 bpw against 0.083 for GPTQ int3, with a Metal kernel for Apple silicon.
- **Verified shortest-trace distillation**: the teacher solves contest problems several
  times, the tests run, and the student trains on the shortest passing trace.
- **qheal**: healing after quantization, training attention, routers and shared experts
  around frozen low-bit experts.
- **MoEfy-anneal** (lab): turn a dense model into a MoE with every expert on, then distil
  while the active experts step down.

## The three BlackHole products

All three are open source under Apache-2.0.

1. **BlackHole-Harness** (this repo): the training pipeline harness. Preset recipes
   (schemas) that run on any open model, the TUI, the VM tooling, and the open format layer
   `blackhole.anatomy` (safetensors IO, the HEX layout and quantizer, format table).
2. **[BlackHole-Dev](https://github.com/OliverVillson/BlackHole-Dev)**: the dev training harness. The developer API that opens any model
   as objects (layers, heads, routers, experts with REAP scores, neurons, single weights),
   edits them, and packages manipulations as named functions schemas call through `model.edit`.
   Installs on top of this package.
3. **The BlackHole run harness**: serving BlackHole-made models with their own kernels (HEX on
   Metal today, Triton for NVIDIA in progress) and room for non-conventional model layouts.

## Layout

| Path | What |
|---|---|
| `blackhole/` | the `blackhole-harness` package: schema format, op catalog, runner, stage cache, receipts (`core`), hardware and model detection (`hw`), the open format layer: safetensors IO, HEX layout and quantizer, format table (`anatomy`), schema builder with Claude or GPT (`agent`), marketplace (`market`), GPU VM bring-up (`cloud`) |
| `schemas/` | example schemas; `schemas/lab/` holds experimental ones |
| `tests/` | core tests, no GPU or network needed |
| `install.sh` | the one-line VM installer and updater (served at blackholeharness.dev/install.sh; `blackhole/cloud/install.sh` is the copy `blackhole up` sends, kept identical by a test) |
| `harness/` | the pipeline that runs today: `blackhole_pipeline` (every stage, MLX export with the HEX Metal kernel, LiveCodeBench, `blackhole-serve`) and the Textual TUI `blackhole_tui` |
| `reaper/` | REAPer (`reaper_pipeline`, `reaper_hw`, `reaper_tui`): REAP calibration, scoring, pruning, eval, the Flint port; `reaper/gpu_perf/` holds GPTQ, REAQ/HEX tables and the B200 benchmarks |
| `runtime/` | the BlackHole runtime, kept apart from the pipeline: the HEX Triton MoE kernel for NVIDIA GPUs and the `blackhole-hex` vLLM plugin (see `runtime/README.md`) |
| `landing/` | the static landing page (Vercel, no build) |

`blackhole/` is where everything is heading: its ops are typed stubs today, and the working
code in `harness/` and `reaper/` moves in behind them op by op.

## Quick start

```bash
# laptop: browse, draft and validate schemas, rent a VM (no GPU needed)
pip install -e '.[dev,frontier]' && pytest
blackhole ops                                     # the op catalog
blackhole validate schemas/lab/reaq.toml
blackhole describe "a Python coder that fits a 16 GB Mac"
blackhole up --provider nebius --gpu b200         # prices first, rents nothing without --yes

# GPU VM: the real pipeline and TUI
curl -fsSL https://blackholeharness.dev/install.sh | bash
blackhole                                         # the TUI; reattaches to a running job
```

The installer is one command on any Ubuntu GPU box and the same command updates it: it clones
or updates `~/BlackHole-Harness` (`--ref BRANCH` for another branch), picks the CUDA torch the
driver can run, installs the CLI, REAPer, the pipeline, the TUI and the runtime editable into
`~/.blackhole/venv`, puts vLLM with REAPer and the `blackhole-hex` plugin in `~/.blackhole/vllm`,
and checks it all on the GPU. Venvs set up by hand earlier (`~/bh-venv`, `~/vllm-venv`) are
reused in place. `blackhole up` runs the same script over SSH.

With `harness/` installed, `blackhole` with no subcommand (or with a model name) opens the TUI;
`up`, `down`, `ssh`, `machines`, `run`, `validate`, `detect`, `ops`, `describe`, `market` and
`models` are the CLI.

## Tests

```bash
pip install -e '.[dev,frontier]' && pytest                    # core
pip install -e 'reaper[pipeline]' -e harness pytest
pytest reaper/tests harness/tests                             # pipeline on tiny models, CPU
TRITON_INTERPRET=1 pytest runtime/tests                        # HEX Triton kernel, CPU (pip install triton -e runtime)
```

## Licence

Apache-2.0, see [LICENSE](LICENSE). Models you make keep their base model's licence.
