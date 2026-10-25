#!/usr/bin/env bash
# BlackHole-Harness installer for a fresh Ubuntu GPU VM.
#
#   curl -fsSL https://raw.githubusercontent.com/OliverVillson/BlackHole-Harness/main/install.sh | bash
#   curl -fsSL .../install.sh | bash -s -- --vllm yes --from ./checkout
#
# What it does, each step skipped when already done:
#   1. apt: python3-venv python3-dev git tmux build-essential (only the missing ones)
#   2. reads the GPU with nvidia-smi and picks a torch build that has kernels for it
#      (torch >= 2.7: cu130 on driver 580+, cu128 on R570; Blackwell refuses older drivers)
#   3. venv in $BLACKHOLE_HOME/venv (Ubuntu's system Python is externally managed)
#   4. torch, then blackhole-harness[gpu]
#   5. vLLM in its own venv ($BLACKHOLE_HOME/vllm, found through BLACKHOLE_VLLM_PY), since
#      vLLM pins its own torch; llama.cpp for GGUF ops when asked
#   6. checks torch really runs on the GPU, and puts `blackhole` and `bh` on PATH
#
# Lessons carried over from REAPer on the Nebius B200: apt's python3-torch has no sm_100
# kernels, Blackwell needs a cu128 torch >= 2.7 and driver 570+, and FlashInfer's JIT
# kernels need nvcc, so vLLM runs with the prebuilt FLASH_ATTN backend.
set -euo pipefail

main() {
  BH_HOME="${BLACKHOLE_HOME:-$HOME/.blackhole}"
  BH_FROM="${BLACKHOLE_FROM:-git+https://github.com/OliverVillson/BlackHole-Harness.git}"
  BH_EXTRAS="gpu"
  VLLM="auto"         # auto | yes | no
  LLAMACPP="no"       # yes | no
  LLAMACPP_REF="${LLAMACPP_REF:-}"   # a llama.cpp tag; empty = the newest bNNNN release
  TORCH_INDEX=""      # override with --torch-index
  DRY=0

  APT=1
  while [ $# -gt 0 ]; do
    case "$1" in
      --from) BH_FROM="$2"; shift ;;
      --home) BH_HOME="$2"; shift ;;
      --extras) BH_EXTRAS="$2"; shift ;;
      --vllm) VLLM="$2"; shift ;;
      --llamacpp) LLAMACPP="$2"; shift ;;
      --torch-index) TORCH_INDEX="$2"; shift ;;
      --no-apt) APT=0 ;;
      --dry-run) DRY=1 ;;
      -y|--yes) ;;  # nothing asks yet; accepted so cloud-init lines stay stable
      -h|--help) usage; exit 0 ;;
      *) die "unknown option $1 (see --help)" ;;
    esac
    shift
  done
  VENV="$BH_HOME/venv"
  mkdir -p "$BH_HOME"
  LOG="$BH_HOME/install.log"
  : >"$LOG"

  banner
  check_os
  if [ "$APT" = 1 ]; then apt_deps; fi
  detect_gpu
  pick_torch
  make_venv
  install_torch
  install_harness
  install_vllm
  install_llamacpp
  verify
  link_bin
  done_msg
}

usage() {
  cat <<'EOF'
install.sh [options]
  --from SPEC         what to pip install: a git URL, a local checkout or a PyPI name
                      (default git+https://github.com/OliverVillson/BlackHole-Harness.git)
  --home DIR          where the venvs, models and jobs live (default $BLACKHOLE_HOME or ~/.blackhole)
  --extras LIST       pip extras of blackhole-harness (default gpu; e.g. gpu,frontier)
  --vllm auto|yes|no  vLLM in its own venv for a fast teacher (auto: yes when a GPU has >= 40 GB)
  --llamacpp yes|no   build llama.cpp for the GGUF ops (default no; schemas that need it ask)
  --torch-index URL   force a torch wheel index, e.g. https://download.pytorch.org/whl/cu126
  --no-apt            skip apt (no sudo); python3-venv must already be there
  --dry-run           print what would run, change nothing
  -y, --yes           no questions
EOF
}

# ---------- output, in the harness colours: orange, red, white on black
if [ -t 1 ]; then
  O=$'\033[38;5;208m'; R=$'\033[38;5;196m'; W=$'\033[1;97m'; D=$'\033[2m'; N=$'\033[0m'
else
  O=""; R=""; W=""; D=""; N=""
fi
say()  { printf '%s◉%s %s\n' "$O" "$N" "$*"; }
ok()   { printf '%s✓%s %s\n' "$W" "$N" "$*"; }
warn() { printf '%s!%s %s\n' "$O" "$N" "$*" >&2; }
die()  { printf '%s✗ %s%s\n' "$R" "$*" "$N" >&2; [ -n "${LOG:-}" ] && printf '%s  log: %s%s\n' "$D" "$LOG" "$N" >&2; exit 1; }

# run a command, or print it with --dry-run; output goes to the log
run() {
  if [ "$DRY" = 1 ]; then printf '%s  $ %s%s\n' "$D" "$*" "$N"; return 0; fi
  printf '\n$ %s\n' "$*" >>"$LOG"
  "$@" >>"$LOG" 2>&1 || die "failed: $*"
}

banner() {
  printf '%s' "$O"
  cat <<'EOF'
            .  ·  ˙  ·  .
        ·   ░░▒▒▓▓▓▓▒▒░░   ·
      ˙  ░▒▓██▀▀▀▀▀▀▀▀██▓▒░  ˙
     · ▒▓█▀              ▀█▓▒ ·
       ▓█     B L A C K    █▓
     · ▒▓█▄     H O L E  ▄█▓▒ ·
      ˙  ░▒▓██▄▄▄▄▄▄▄▄██▓▒░  ˙
        ·   ░░▒▒▓▓▓▓▒▒░░   ·
EOF
  printf '%s  %sharness installer%s\n\n' "$N" "$W" "$N"
}

check_os() {
  [ "$(uname -s)" = Linux ] || die "run this on the Linux GPU VM, not $(uname -s)"
  [ "$(uname -m)" = x86_64 ] || warn "$(uname -m): only x86_64 is tested (torch and vLLM wheels may be missing)"
  if [ -r /etc/os-release ]; then
    . /etc/os-release
    case "${ID:-}:${VERSION_ID:-}" in
      ubuntu:22.04|ubuntu:24.04) ok "$PRETTY_NAME" ;;
      *) warn "${PRETTY_NAME:-unknown OS}: tested on Ubuntu 22.04 and 24.04" ;;
    esac
  fi
  [ "$(id -u)" = 0 ] && warn "running as root: the venv goes in $BH_HOME (better: a normal user with sudo)"
  return 0
}

SUDO=""
apt_deps() {
  command -v apt-get >/dev/null || { warn "no apt-get: install python3-venv, python3-dev, git, tmux yourself"; return; }
  [ "$(id -u)" = 0 ] || SUDO="sudo"
  local want=(python3-venv python3-dev python3-pip git tmux build-essential curl rsync) missing=()
  [ "$LLAMACPP" = yes ] && want+=(cmake libcurl4-openssl-dev)
  for p in "${want[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  # Ubuntu ships python3.X-venv per version; python3-venv pulls the right one.
  if [ ${#missing[@]} -eq 0 ]; then ok "system packages already there"; return; fi
  say "apt install ${missing[*]}"
  run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq
  run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
  ok "system packages"
}

# ---------- hardware: same rules as reaper_hw/plan.py _software()
GPU_NAME=""; GPU_MEM=0; GPU_CC=""; DRIVER=""; DRV_CUDA=""; N_GPUS=0
detect_gpu() {
  if ! command -v nvidia-smi >/dev/null; then
    warn "no nvidia-smi: no NVIDIA driver on this machine. Installing the CPU build (demo mode only)."
    warn "on Nebius pick the ubuntu24.04-cuda13.0 image; elsewhere install driver 570+ first"
    return
  fi
  local q
  q=$(nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits 2>/dev/null) \
    || q=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits 2>/dev/null | sed 's/,\([^,]*\)$/,,\1/') \
    || die "nvidia-smi is installed but cannot talk to the driver (reboot after a driver install?)"
  N_GPUS=$(printf '%s\n' "$q" | grep -c .)
  IFS=',' read -r GPU_NAME GPU_MEM GPU_CC DRIVER <<<"$(printf '%s\n' "$q" | head -1)"
  GPU_NAME=$(echo "$GPU_NAME" | xargs); GPU_MEM=$(( $(echo "$GPU_MEM" | xargs | cut -d. -f1) / 1024 ))
  GPU_CC=$(echo "$GPU_CC" | xargs); DRIVER=$(echo "$DRIVER" | xargs)
  [ -n "$GPU_CC" ] || GPU_CC=$(cc_from_name "$GPU_NAME")
  DRV_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
  ok "${N_GPUS}x $GPU_NAME, ${GPU_MEM} GB, sm_${GPU_CC/./} · driver $DRIVER (CUDA $DRV_CUDA)"
}

cc_from_name() {
  case "$(echo "$1" | tr a-z A-Z)" in
    *B300*) echo 10.3 ;; *B200*|*B100*) echo 10.0 ;; *"RTX PRO 6000"*|*"RTX 50"*) echo 12.0 ;;
    *H200*|*H100*|*H800*|*H20*) echo 9.0 ;; *L40*|*L4*|*"RTX 40"*|*"ADA"*) echo 8.9 ;;
    *A100*|*A800*|*A30*) echo 8.0 ;; *A10*|*A40*|*A6000*|*"RTX 30"*) echo 8.6 ;; *T4*) echo 7.5 ;;
    *) echo "" ;;
  esac
}

ver_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

# Newest CUDA build the driver can run: cu130 on driver 580+ (Nebius ubuntu24.04-cuda13.0),
# else cu128 (R570). New torch releases drop old CUDA builds, so a cu128 pin ages badly.
cuda_index() {
  if ver_ge "${DRV_CUDA:-0}" 13.0; then echo "https://download.pytorch.org/whl/cu130"
  else echo "https://download.pytorch.org/whl/cu128"; fi
}

pick_torch() {
  if [ -n "$TORCH_INDEX" ]; then say "torch index forced: $TORCH_INDEX"; return; fi
  if [ -z "$GPU_NAME" ]; then TORCH_INDEX="https://download.pytorch.org/whl/cpu"; return; fi
  local major=${GPU_CC%%.*}
  if [ "${major:-0}" -ge 10 ]; then
    # Blackwell: cu128 or newer everywhere, torch >= 2.7, driver R570+
    [ -n "$DRV_CUDA" ] && ! ver_ge "$DRV_CUDA" 12.8 && \
      die "driver $DRIVER supports CUDA $DRV_CUDA; $GPU_NAME needs driver 570+ (CUDA 12.8). On Nebius use image ubuntu24.04-cuda13.0"
    TORCH_INDEX=$(cuda_index)
  elif [ -n "$DRV_CUDA" ] && ver_ge "$DRV_CUDA" 12.8; then
    TORCH_INDEX=$(cuda_index)   # Ada, Hopper, Ampere on a new driver
  elif [ -n "$DRV_CUDA" ] && ver_ge "$DRV_CUDA" 12.6; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu126"
  else
    die "driver $DRIVER (CUDA ${DRV_CUDA:-?}) is too old for torch 2.7; install driver 570 or newer"
  fi
  say "torch from ${TORCH_INDEX##*/} for sm_${GPU_CC/./}"
}

make_venv() {
  if [ -x "$VENV/bin/python" ]; then ok "venv $VENV"; return; fi
  command -v python3 >/dev/null || die "no python3"
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "python $(python3 -V) is too old, need 3.11+ (Ubuntu 24.04 has 3.12)"
  say "venv $VENV"
  run python3 -m venv "$VENV"
  run "$VENV/bin/python" -m pip install -q --upgrade pip wheel setuptools
}

PIP() { run "$VENV/bin/python" -m pip install -q "$@"; }

install_torch() {
  if [ "$DRY" = 0 ] && "$VENV/bin/python" - "$GPU_CC" <<'EOF' 2>/dev/null; then
import sys, torch
cc = sys.argv[1]
v = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
ok = v >= (2, 7)
if cc:  # a cubin for sm_XY runs on any sm_XZ with Z >= Y
    major, minor = (int(x) for x in cc.split("."))
    have = [a[3:] for a in torch.cuda.get_arch_list() if a.startswith("sm_")]
    ok = ok and any(int(a[:-1]) == major and int(a[-1]) <= minor for a in have if a.isdigit())
sys.exit(0 if ok else 1)
EOF
    ok "torch already fits this GPU"; return
  fi
  say "pip install torch>=2.7 (a few GB, takes a minute or two)"
  PIP 'torch>=2.7' --index-url "$TORCH_INDEX"
  ok "torch"
}

install_harness() {
  local spec="$BH_FROM"
  if [ -d "$spec" ]; then spec="$(cd "$spec" && pwd)"; fi
  # extras go in brackets; for a git URL that means "name[extras] @ git+..."
  if [ -n "$BH_EXTRAS" ]; then
    case "$spec" in
      git+*) spec="blackhole-harness[$BH_EXTRAS] @ $spec" ;;
      /*) spec="${spec}[$BH_EXTRAS]" ;;
      *\[*) ;;
      *) spec="${spec}[$BH_EXTRAS]" ;;
    esac
  fi
  say "pip install $spec"
  # keep the torch we just picked: pip must not swap it for PyPI's build
  PIP "$spec" --extra-index-url "$TORCH_INDEX"
  ok "harness"
}

install_vllm() {
  case "$VLLM" in
    no) return ;;
    auto) [ -n "$GPU_NAME" ] && [ "$GPU_MEM" -ge 40 ] || { say "skipping vLLM (--vllm yes to force)"; return; } ;;
    yes) ;;
    *) die "--vllm takes auto, yes or no" ;;
  esac
  # vLLM pins an exact torch; in the main venv it would replace the one picked above.
  VLLM_VENV="$BH_HOME/vllm"
  if [ -x "$VLLM_VENV/bin/python" ] && "$VLLM_VENV/bin/python" -c 'import vllm' 2>/dev/null; then
    ok "vLLM already there"; return
  fi
  say "vLLM in its own venv $VLLM_VENV (a few GB)"
  [ -x "$VLLM_VENV/bin/python" ] || run python3 -m venv "$VLLM_VENV"
  run "$VLLM_VENV/bin/python" -m pip install -q --upgrade pip
  run "$VLLM_VENV/bin/python" -m pip install -q vllm --extra-index-url "$TORCH_INDEX"
  ok "vLLM"
}

install_llamacpp() {
  [ "$LLAMACPP" = yes ] || return 0
  local src="$BH_HOME/llama.cpp" ref="$LLAMACPP_REF"
  if [ -x "$src/build/bin/llama-quantize" ]; then ok "llama.cpp already built"; return; fi
  if [ -z "$ref" ]; then
    ref=$(git ls-remote --tags --refs https://github.com/ggml-org/llama.cpp 'b*' | sed 's|.*refs/tags/||' | sort -V | tail -1)
  fi
  say "llama.cpp $ref (CPU build: convert and quantize need no GPU)"
  [ -d "$src/.git" ] || run git clone -q --depth 1 --branch "$ref" https://github.com/ggml-org/llama.cpp "$src"
  run cmake -S "$src" -B "$src/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  run cmake --build "$src/build" -j "$(nproc)" --target llama-quantize llama-cli llama-server
  run "$VENV/bin/python" -m pip install -q -r "$src/requirements/requirements-convert_hf_to_gguf.txt" --extra-index-url "$TORCH_INDEX"
  ok "llama.cpp in $src"
}

verify() {
  [ "$DRY" = 1 ] && return
  say "checking torch on the GPU"
  local out
  out=$("$VENV/bin/python" - <<'EOF' 2>&1
import torch
print(f"torch {torch.__version__} cuda {torch.version.cuda} arches {' '.join(torch.cuda.get_arch_list()) or '-'}")
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    print(f"matmul ok on {torch.cuda.get_device_name(0)}")
else:
    print("no CUDA device visible")
EOF
) || die "torch does not run here: $out"
  printf '%s  %s%s\n' "$D" "$out" "$N" | sed "2,\$s/^/  /"
  if [ -n "$GPU_NAME" ] && ! grep -q "matmul ok" <<<"$out"; then
    die "torch cannot use the GPU. Try: $VENV/bin/pip install --force-reinstall 'torch>=2.7' --index-url $TORCH_INDEX"
  fi
  if [ -n "$GPU_NAME" ]; then ok "GPU works from torch"; else ok "torch works (CPU only: demo mode)"; fi
}

link_bin() {
  [ "$DRY" = 1 ] && return
  local bin="$HOME/.local/bin"
  mkdir -p "$bin"
  # one launcher: the harness's own command, or REAPer's while the harness package is being built
  cat >"$bin/blackhole" <<EOF
#!/usr/bin/env bash
export BLACKHOLE_HOME="$BH_HOME"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# every compile cache in one folder, which blackhole down/up carries between VMs (cold vLLM start:
# ~7 min on a B200, warm: ~50 s)
export VLLM_CACHE_ROOT="$BH_HOME/cache/compile/vllm" TORCHINDUCTOR_CACHE_DIR="$BH_HOME/cache/compile/inductor"
export TRITON_CACHE_DIR="$BH_HOME/cache/compile/triton" FLASHINFER_WORKSPACE_BASE="$BH_HOME/cache/compile"
$( [ -x "$BH_HOME/vllm/bin/python" ] && printf 'export BLACKHOLE_VLLM_PY=%s REAPER_VLLM_PY=%s\n' "$BH_HOME/vllm/bin/python" "$BH_HOME/vllm/bin/python" )
$( [ -d "$BH_HOME/llama.cpp/build/bin" ] && printf 'export BLACKHOLE_LLAMACPP=%s\n' "$BH_HOME/llama.cpp" )
$( [ "${GPU_CC%%.*}" -ge 10 ] 2>/dev/null && printf 'export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0\n' )
for cmd in blackhole bh reaper; do
  [ -x "$VENV/bin/\$cmd" ] && exec "$VENV/bin/\$cmd" "\$@"
done
echo "no harness command in $VENV/bin; rerun the installer" >&2; exit 1
EOF
  chmod +x "$bin/blackhole"
  ln -sf "$bin/blackhole" "$bin/bh"
  case ":$PATH:" in *":$bin:"*) ;; *)
    grep -qs 'BlackHole-Harness' "$HOME/.bashrc" || printf '\n# BlackHole-Harness\nexport PATH="$HOME/.local/bin:$PATH"\n' >>"$HOME/.bashrc"
  esac
  ok "blackhole and bh in $bin"
}

done_msg() {
  printf '\n%s  installed.%s  From your laptop:\n\n' "$W" "$N"
  printf '    %sssh -t %s@<vm-ip> blackhole%s\n\n' "$O" "$(id -un)" "$N"
  printf '  or here: %sblackhole%s (open a new shell first if it is not found)\n' "$O" "$N"
  [ "$DRY" = 1 ] && printf '%s  (dry run: nothing was changed)%s\n' "$D" "$N"
  return 0
}

# everything above is definitions: a half-downloaded `curl | bash` runs nothing
main "$@"
