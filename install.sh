#!/usr/bin/env bash
# BlackHole-Harness installer for a GPU VM (Ubuntu 22.04/24.04 with an NVIDIA driver).
#
#   curl -fsSL https://blackholeharness.dev/install.sh | bash
#   curl -fsSL https://blackholeharness.dev/install.sh | bash -s -- --ref some-branch
#
# One command, safe to rerun: every step checks first and only does what is missing, and a rerun
# updates the checkout and the packages. What it does:
#   1. apt: python3-venv python3-dev git tmux build-essential curl rsync (only the missing ones)
#   2. reads the GPU with nvidia-smi and picks a torch build with kernels for it
#      (torch >= 2.7: cu130 on driver 580+, cu128 on R570; Blackwell refuses older drivers)
#   3. clones or updates the BlackHole-Harness checkout (~/BlackHole-Harness, branch main)
#   4. one venv ($BLACKHOLE_HOME/venv) with torch, the whole harness, editable from the checkout:
#      blackhole (CLI), reaper[pipeline], harness (pipeline + TUI), runtime (Triton HEX kernels),
#      flash-linear-attention (Qwen3.5/3.6 training), mlx[cpu] (checking MLX exports)
#   5. vLLM in its own venv ($BLACKHOLE_HOME/vllm: vLLM pins its own torch) with REAPer and the
#      blackhole-hex vLLM plugin next to it
#   6. checks torch on the GPU and every package, puts `blackhole` and `bh` on PATH
#
# Machines set up by hand before this installer (~/bh-venv, ~/vllm-venv) are reused in place, and
# fresh installs get those two names as links, so older notes and the BlackHole-Dev app find them.
#
# Lessons carried over from REAPer on the Nebius B200: apt's python3-torch has no sm_100
# kernels, Blackwell needs a cu128 torch >= 2.7 and driver 570+, and FlashInfer's JIT
# kernels need nvcc, so vLLM runs with the prebuilt FLASH_ATTN backend.
set -euo pipefail

REPO_URL="https://github.com/OliverVillson/BlackHole-Harness.git"

main() {
  BH_HOME="${BLACKHOLE_HOME:-$HOME/.blackhole}"
  SRC="${BLACKHOLE_SRC:-$HOME/BlackHole-Harness}"
  FROM="${BLACKHOLE_FROM:-}"   # a local checkout to install as it is, or a git URL to clone
  REF="${BLACKHOLE_REF:-main}"
  BH_EXTRAS="gpu"
  VLLM="auto"         # auto | yes | no
  LLAMACPP="no"       # yes | no
  LLAMACPP_REF="${LLAMACPP_REF:-}"   # a llama.cpp tag; empty = the newest bNNNN release
  TORCH_INDEX=""      # override with --torch-index
  EXTRAS_OPT=1        # flash-linear-attention, mlx[cpu]: best effort
  DEV="auto"          # blackhole_dev (BlackHole-Dev's package, for recipes run from the app): auto | yes | no
  DEV_SRC="${BLACKHOLE_DEV_SRC:-$HOME/Blackhole-Dev}"
  DRY=0

  APT=1
  while [ $# -gt 0 ]; do
    case "$1" in
      --from) FROM="$2"; shift ;;
      --ref) REF="$2"; shift ;;
      --src) SRC="$2"; shift ;;
      --home) BH_HOME="$2"; shift ;;
      --extras) BH_EXTRAS="$2"; shift ;;
      --vllm) VLLM="$2"; shift ;;
      --llamacpp) LLAMACPP="$2"; shift ;;
      --torch-index) TORCH_INDEX="$2"; shift ;;
      --minimal) EXTRAS_OPT=0 ;;
      --dev) DEV="$2"; shift ;;
      --dev-src) DEV_SRC="$2"; shift ;;
      --no-apt) APT=0 ;;
      --dry-run) DRY=1 ;;
      -y|--yes) ;;  # nothing asks; accepted so cloud-init lines stay stable
      -h|--help) usage; exit 0 ;;
      *) die "unknown option $1 (see --help)" ;;
    esac
    shift
  done
  mkdir -p "$BH_HOME"
  LOG="$BH_HOME/install.log"
  : >"$LOG"

  banner
  check_os
  if [ "$APT" = 1 ]; then apt_deps; fi
  detect_gpu
  pick_torch
  checkout
  pick_venvs
  make_venv
  install_torch
  install_harness
  install_optional
  install_dev
  install_vllm
  install_llamacpp
  verify
  link_bin
  done_msg
}

usage() {
  cat <<'EOF'
install.sh [options]
  --ref REF           branch or tag of BlackHole-Harness to install (default main)
  --src DIR           where the checkout lives (default ~/BlackHole-Harness)
  --from PATH|URL     install this local checkout as it is (no git), or clone this git URL
  --home DIR          where the venvs, models and jobs live (default $BLACKHOLE_HOME or ~/.blackhole)
  --extras LIST       pip extras of blackhole-harness (default gpu; e.g. gpu,frontier)
  --vllm auto|yes|no  vLLM in its own venv for a fast teacher (auto: yes when a GPU has >= 40 GB)
  --llamacpp yes|no   build llama.cpp for the GGUF ops (default no; schemas that need it ask)
  --torch-index URL   force a torch wheel index, e.g. https://download.pytorch.org/whl/cu126
  --minimal           skip flash-linear-attention and mlx[cpu]
  --dev auto|yes|no   blackhole_dev, which recipes run from the BlackHole-Dev app need (auto: install
                      it when ~/Blackhole-Dev is there or the repo can be cloned; yes: fail otherwise)
  --dev-src DIR       where that checkout is (default ~/Blackhole-Dev; a copy without .git is fine)
  --no-apt            skip apt (no sudo); python3-venv, git and tmux must already be there
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
die()  {
  printf '%s✗ %s%s\n' "$R" "$*" "$N" >&2
  if [ -n "${LOG:-}" ] && [ -s "$LOG" ]; then
    printf '%s  last lines of %s:%s\n' "$D" "$LOG" "$N" >&2
    tail -n 15 "$LOG" | sed 's/^/    /' >&2
  fi
  exit 1
}

# run a command, or print it with --dry-run; output goes to the log
run() {
  if [ "$DRY" = 1 ]; then printf '%s  $ %s%s\n' "$D" "$*" "$N"; return 0; fi
  printf '\n$ %s\n' "$*" >>"$LOG"
  "$@" >>"$LOG" 2>&1 || die "failed: $*"
}

# the same, but a failure is only a warning (optional extras)
try() {
  if [ "$DRY" = 1 ]; then printf '%s  $ %s%s\n' "$D" "$*" "$N"; return 0; fi
  printf '\n$ %s\n' "$*" >>"$LOG"
  "$@" >>"$LOG" 2>&1
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
  [ "$(id -u)" = 0 ] && warn "running as root: everything goes in $BH_HOME (better: a normal user with sudo)"
  return 0
}

SUDO=""
apt_deps() {
  command -v apt-get >/dev/null || { warn "no apt-get: install python3-venv, python3-dev, git, tmux yourself"; return; }
  [ "$(id -u)" = 0 ] || SUDO="sudo"
  local v want missing=()
  v=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)
  # Ubuntu ships python3.X-venv per version, and Triton builds a small CUDA helper with gcc and
  # Python.h on first use
  want=(python3-venv python3-dev python3-pip git tmux build-essential curl rsync ca-certificates)
  [ -n "$v" ] && want+=("python$v-venv" "python$v-dev")
  [ "$LLAMACPP" = yes ] && want+=(cmake libcurl4-openssl-dev)
  for p in "${want[@]}"; do
    dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p")
  done
  if [ ${#missing[@]} -eq 0 ]; then ok "system packages already there"; return; fi
  say "apt install ${missing[*]}"
  run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get update -qq
  run $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${missing[@]}"
  ok "system packages"
}

# ---------- hardware: same rules as reaper_hw/plan.py _software()
GLOBAL=""
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

# ---------- the checkout: everything installs editable from it, so `git pull` + rerun updates
checkout() {
  local url="$REPO_URL"
  if [ -n "$FROM" ] && [ -d "$FROM" ]; then
    SRC="$(cd "$FROM" && pwd)"
    [ -f "$SRC/pyproject.toml" ] && [ -d "$SRC/harness" ] || die "$SRC is not a BlackHole-Harness checkout"
    ok "installing the checkout $SRC as it is"
    return
  fi
  [ -n "$FROM" ] && url="${FROM#git+}"
  command -v git >/dev/null || die "no git (drop --no-apt, or apt install git)"
  if [ ! -e "$SRC" ]; then
    say "git clone $url ($REF) into $SRC"
    run git clone -q --branch "$REF" "$url" "$SRC"
    ok "checkout $SRC at $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo "$REF")"
    return
  fi
  [ -d "$SRC/.git" ] || die "$SRC exists and is not a git checkout; move it away or pass --src DIR"
  local origin
  origin=$(git -C "$SRC" remote get-url origin 2>/dev/null || true)
  if [ "$(norm_url "$origin")" != "$(norm_url "$url")" ]; then
    # an older checkout of another repo (or a fork) in the same place: keep it, next to the new one
    local old="$SRC.old-$(date +%Y%m%d-%H%M%S)"
    warn "$SRC is a checkout of ${origin:-no remote}, not $url; moving it to $old"
    run mv "$SRC" "$old"
    run git clone -q --branch "$REF" "$url" "$SRC"
    ok "checkout $SRC at $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo "$REF")"
    return
  fi
  say "updating $SRC to $REF"
  run git -C "$SRC" fetch -q origin "$REF"
  if [ -n "$(git -C "$SRC" status --porcelain --untracked-files=no 2>/dev/null)" ]; then
    local stamp="blackhole install $(date +%Y-%m-%dT%H:%M:%S)"
    warn "local edits in $SRC kept in git stash (\"$stamp\"); git -C $SRC stash pop brings them back"
    run git -C "$SRC" stash push -q -m "$stamp"
  fi
  if git -C "$SRC" show-ref -q --verify "refs/remotes/origin/$REF"; then
    run git -C "$SRC" checkout -q -B "$REF" "origin/$REF"   # a branch: follow origin exactly
  else
    run git -C "$SRC" checkout -q FETCH_HEAD                 # a tag or a commit
  fi
  ok "checkout $SRC at $REF ($(git -C "$SRC" rev-parse --short HEAD))"
}

norm_url() { local u="${1%/}"; u="${u%.git}"; u="${u#git+}"; u="${u#https://}"; u="${u#http://}"; u="${u#git@}"; echo "${u/://}" | tr A-Z a-z; }

# ---------- venvs: the main one and vLLM's; earlier hand-made ones are reused where they are
pick_venvs() {
  VENV="$BH_HOME/venv"; VLLM_VENV="$BH_HOME/vllm"
  if [ ! -e "$VENV" ] && [ -x "$HOME/bh-venv/bin/python" ]; then VENV="$HOME/bh-venv"; say "reusing $VENV"; fi
  if [ ! -e "$VLLM_VENV" ] && [ -x "$HOME/vllm-venv/bin/python" ]; then VLLM_VENV="$HOME/vllm-venv"; say "reusing $VLLM_VENV"; fi
  return 0
}

make_venv() {
  if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then ok "venv $VENV"; return; fi
  command -v python3 >/dev/null || die "no python3"
  python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || die "python $(python3 -V) is too old, need 3.11+ (Ubuntu 24.04 has 3.12)"
  say "venv $VENV"
  [ "$DRY" = 1 ] || rm -rf "$VENV"
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
  local root="$SRC" extras=""
  [ -n "$BH_EXTRAS" ] && extras="[$BH_EXTRAS]"
  say "pip install the harness: blackhole, reaper, pipeline + TUI, runtime (editable from $SRC)"
  # torch stays the build picked above: its index is searched too, so pip never swaps it for PyPI's.
  # The root package goes last: its `blackhole` (up, validate, describe, market) still opens the TUI.
  PIP -e "$root/reaper[pipeline]" -e "$root/harness" -e "$root/runtime" --extra-index-url "$TORCH_INDEX"
  PIP -e "$root$extras" --extra-index-url "$TORCH_INDEX"
  ok "harness"
}

install_optional() {
  [ "$EXTRAS_OPT" = 1 ] || return 0
  # Qwen3.5 / 3.6 linear attention (Gated DeltaNet) kernels: without them training runs a slow
  # PyTorch reference
  if try "$VENV/bin/python" -m pip install -q flash-linear-attention --extra-index-url "$TORCH_INDEX"; then
    ok "flash-linear-attention"
  else
    warn "flash-linear-attention did not install: Qwen3.5/3.6 training will be slow"
  fi
  # mlx[cpu] lets the pipeline load its own MLX exports to check them before they ship
  if try "$VENV/bin/python" -m pip install -q "mlx[cpu]" mlx-lm --extra-index-url "$TORCH_INDEX"; then
    ok "mlx[cpu] for checking MLX exports"
  else
    warn "mlx[cpu] did not install: MLX exports are written but not test-loaded here"
  fi
}

# blackhole_dev: from a checkout or copy in ~/Blackhole-Dev (an rsync from the laptop works), else a
# clone when this machine can read the repo. Editable, so a fresh rsync takes effect without a rerun.
DEV_URL="https://github.com/OliverVillson/Blackhole-Dev.git"
install_dev() {
  [ "$DEV" = no ] && return 0
  if [ ! -f "$DEV_SRC/pyproject.toml" ] && [ ! -e "$DEV_SRC" ]; then
    if [ "$DRY" = 1 ] || GIT_TERMINAL_PROMPT=0 git clone -q "$DEV_URL" "$DEV_SRC" >>"$LOG" 2>&1; then
      [ "$DRY" = 1 ] && printf '%s  $ git clone %s %s%s\n' "$D" "$DEV_URL" "$DEV_SRC" "$N"
    else
      rm -rf "$DEV_SRC"
      local msg="blackhole_dev not installed: this machine cannot clone $DEV_URL. Recipes from the app need it; from the laptop: rsync -a --exclude .git --exclude app ~/src/Blackhole-Dev/ $(id -un)@$(vm_ip):Blackhole-Dev/ then run this installer again"
      [ "$DEV" = yes ] && die "$msg"
      warn "$msg"
      return 0
    fi
  elif [ -d "$DEV_SRC/.git" ]; then
    GIT_TERMINAL_PROMPT=0 try git -C "$DEV_SRC" pull -q --ff-only || warn "could not update $DEV_SRC (git pull); installing it as it is"
  fi
  [ "$DRY" = 1 ] || [ -f "$DEV_SRC/pyproject.toml" ] || die "$DEV_SRC is not a Blackhole-Dev checkout (no pyproject.toml)"
  say "pip install blackhole_dev (editable from $DEV_SRC)"
  PIP -e "$DEV_SRC" --extra-index-url "$TORCH_INDEX"
  ok "blackhole_dev $( [ "$DRY" = 1 ] || "$VENV/bin/python" -c 'from importlib.metadata import version as v; print(v("blackhole-dev"))' 2>/dev/null)"
}

install_vllm() {
  case "$VLLM" in
    no) return 0 ;;
    auto) if [ -z "$GPU_NAME" ] || [ "$GPU_MEM" -lt 40 ]; then say "skipping vLLM (--vllm yes to force)"; return 0; fi ;;
    yes) ;;
    *) die "--vllm takes auto, yes or no" ;;
  esac
  # vLLM pins an exact torch; in the main venv it would replace the one picked above. REAPer goes
  # in too (calibrate runs under vLLM's Python), and the runtime registers the blackhole-hex
  # quantization so `vllm serve` loads HEX models.
  if [ -x "$VLLM_VENV/bin/python" ] && "$VLLM_VENV/bin/python" -c 'import vllm' 2>/dev/null; then
    ok "vLLM already there ($VLLM_VENV)"
  else
    say "vLLM in its own venv $VLLM_VENV (a few GB)"
    [ "$DRY" = 1 ] || rm -rf "$VLLM_VENV"
    run python3 -m venv "$VLLM_VENV"
    run "$VLLM_VENV/bin/python" -m pip install -q --upgrade pip
    run "$VLLM_VENV/bin/python" -m pip install -q vllm ninja
    ok "vLLM"
  fi
  run "$VLLM_VENV/bin/python" -m pip install -q ninja -e "$SRC/reaper[pipeline]" -e "$SRC/runtime"
  ok "REAPer and the blackhole-hex plugin next to vLLM"
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
  out=$("$VENV/bin/python" -c 'import blackhole, blackhole_pipeline, blackhole_tui, reaper_pipeline, blackhole_runtime; import transformers; print("transformers", transformers.__version__)' 2>&1) \
    || die "the harness does not import: $out"
  ok "blackhole, pipeline, TUI, REAPer and runtime import ($out)"
  if [ -x "$VLLM_VENV/bin/python" ] && [ "$VLLM" != no ]; then
    out=$("$VLLM_VENV/bin/python" - <<'EOF' 2>&1
import vllm
from vllm import ModelRegistry as R
import reaper_pipeline, blackhole_runtime
qwen = "yes" if "Qwen3_5MoeForConditionalGeneration" in R.get_supported_archs() else "NO"
print(f"vllm {vllm.__version__} · Qwen3.5/3.6 MoE {qwen}")
EOF
) && ok "$out" || warn "vLLM does not import ($out): the pipeline falls back to slower transformers generation"
  fi
}

link_bin() {
  [ "$DRY" = 1 ] && return
  local bin="$HOME/.local/bin" vllm_py=""
  mkdir -p "$bin"
  [ -x "$VLLM_VENV/bin/python" ] && vllm_py="$VLLM_VENV/bin/python"
  # the old names, so notes that say `source ~/bh-venv/bin/activate` keep working, and the
  # pipeline's own fallback (~/vllm-venv) finds vLLM
  [ -e "$HOME/bh-venv" ] || ln -s "$VENV" "$HOME/bh-venv"
  [ -n "$vllm_py" ] && { [ -e "$HOME/vllm-venv" ] || ln -s "$VLLM_VENV" "$HOME/vllm-venv"; }
  # one environment file, sourced by the launcher; `source ~/.blackhole/env.sh` gives a shell the same
  cat >"$BH_HOME/env.sh" <<EOF
# written by the BlackHole-Harness installer; rerun it to rewrite
export BLACKHOLE_HOME="$BH_HOME"
export BLACKHOLE_SRC="$SRC"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# every compile cache in one folder, which blackhole down/up carries between VMs (cold vLLM start:
# ~7 min on a B200, warm: ~50 s)
export VLLM_CACHE_ROOT="$BH_HOME/cache/compile/vllm" TORCHINDUCTOR_CACHE_DIR="$BH_HOME/cache/compile/inductor"
export TRITON_CACHE_DIR="$BH_HOME/cache/compile/triton" FLASHINFER_WORKSPACE_BASE="$BH_HOME/cache/compile"
$( [ -n "$vllm_py" ] && printf 'export BLACKHOLE_VLLM_PY=%s REAPER_VLLM_PY=%s\n' "$vllm_py" "$vllm_py" )
$( [ -d "$BH_HOME/llama.cpp/build/bin" ] && printf 'export BLACKHOLE_LLAMACPP=%s\n' "$BH_HOME/llama.cpp" )
$( [ "${GPU_CC%%.*}" -ge 10 ] 2>/dev/null && printf 'export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0\n' )
case ":\$PATH:" in *":$VENV/bin:"*) ;; *) export PATH="$VENV/bin:\$PATH" ;; esac
EOF
  cat >"$bin/blackhole" <<EOF
#!/usr/bin/env bash
. "$BH_HOME/env.sh"
for cmd in blackhole bh reaper; do
  [ -x "$VENV/bin/\$cmd" ] && exec "$VENV/bin/\$cmd" "\$@"
done
echo "no harness command in $VENV/bin; rerun the installer" >&2; exit 1
EOF
  chmod +x "$bin/blackhole"
  ln -sf "$bin/blackhole" "$bin/bh"
  # new login shells: ~/.profile (Ubuntu's adds ~/.local/bin only when it existed at login) and
  # ~/.bashrc. A plain `ssh host blackhole` reads neither past .bashrc's interactive check, so with
  # sudo the commands also go in /usr/local/bin, which every shell has on PATH.
  local rc
  for rc in "$HOME/.profile" "$HOME/.bashrc"; do
    grep -qs '# BlackHole-Harness' "$rc" || \
      printf '\n# BlackHole-Harness\ncase ":$PATH:" in *":$HOME/.local/bin:"*) ;; *) PATH="$HOME/.local/bin:$PATH" ;; esac\n' >>"$rc"
  done
  GLOBAL=""
  if [ -w /usr/local/bin ] || { [ "$(id -u)" != 0 ] && sudo -n true 2>/dev/null; }; then
    local s=""; [ -w /usr/local/bin ] || s="sudo"
    if $s ln -sf "$bin/blackhole" /usr/local/bin/blackhole && $s ln -sf "$bin/blackhole" /usr/local/bin/bh; then
      GLOBAL=/usr/local/bin
    fi
  fi
  ok "blackhole and bh in $bin${GLOBAL:+ and $GLOBAL}"
}

done_msg() {
  # without /usr/local/bin, the full path works in any shell, here and over ssh
  local cmd="blackhole"
  [ -n "${GLOBAL:-}" ] || cmd="~/.local/bin/blackhole"
  printf '\n%s  installed.%s  From your laptop:\n\n' "$W" "$N"
  printf '    %sssh -t %s@%s %s%s\n\n' "$O" "$(id -un)" "$(vm_ip)" "$cmd" "$N"
  printf '  or here, in this shell: %s%s%s\n' "$O" "$cmd" "$N"
  printf '  update later: run the same install command again\n'
  [ "$DRY" = 1 ] && printf '%s  (dry run: nothing was changed)%s\n' "$D" "$N"
  return 0
}

# the address this SSH session came in on, when it is a public one; else a placeholder
vm_ip() {
  local ip
  ip=$(echo "${SSH_CONNECTION:-}" | awk '{print $3}')
  case "$ip" in ""|10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[01].*|127.*|*:*) echo "<vm-ip>" ;; *) echo "$ip" ;; esac
}

# everything above is definitions: a half-downloaded `curl | bash` runs nothing
main "$@"
