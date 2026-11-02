#!/usr/bin/env bash
# SWE-bench Pro V2 (Scale AI, 642 tasks, Harbor format) on one GPU VM with a local model.
#
#   bash swebench-pro.sh MODEL [--tasks smoke|hard|all|N] [-n CONCURRENT] [--name NAME] [--port P] [--check]
#
# MODEL is a BlackHole MLX HEX export (converted once to a vLLM blackhole-hex folder next to
# it), a vLLM-ready folder, or a Hugging Face id. Run it inside tmux; rerunning the same
# command resumes. Everything lands in ~/.blackhole/swebench/runs/NAME:
#   vllm.log, jobs/NAME (agent run), jobs/NAME-regrade (official score), results.json
# --check only runs the checks (Docker, disk, vLLM, the model folder) and exits 0 or 1.
# Docker missing: installed with sudo when sudo needs no password; not in the docker group
# yet: added, and the script re-runs itself under `sg docker`, so no new login is needed.
#
# How it scores: mini-swe-agent with Scale's locked prompt (v2/tooling/configs/
# mini_textbased.yaml) runs on this machine and only executes bash inside the task's
# container, which has no network. Its git diff is then re-graded on a fresh container
# (patch_replay), which is the number to report. 50-minute budget per task.
set -euo pipefail

ARGS=("$@")
MODEL=""; TASKS=smoke; N=""; NAME=""; PORT=${PORT:-8000}; SERVED=${SERVED:-local-model}; CHECK=0
MAXLEN=${MAXLEN:-}; MAX_TOKENS=${MAX_TOKENS:-16384}; MIN_FREE_GB=${MIN_FREE_GB:-150}
MIN_START_GB=${MIN_START_GB:-60}; [ -n "${REASONING_PARSER+x}" ] && REASONING=$REASONING_PARSER; SBP_REF=${SBP_REF:-v2.0.0}; VLLM_ARGS=${VLLM_ARGS:-}
while [ $# -gt 0 ]; do
  case "$1" in
    --tasks) TASKS="$2"; shift ;;
    -n|--concurrent) N="$2"; shift ;;
    --name) NAME="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    --check) CHECK=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    -*) echo "unknown option $1"; exit 2 ;;
    *) MODEL="$1" ;;
  esac
  shift
done
[ -n "$MODEL" ] || { sed -n '2,20p' "$0"; exit 2; }
case "$TASKS" in smoke|hard|all|[0-9]*) ;; *) echo "--tasks takes smoke, hard, all or a number, not $TASKS"; exit 2 ;; esac

BH_HOME=${BLACKHOLE_HOME:-$HOME/.blackhole}
SW=$BH_HOME/swebench
HERE=$(cd "$(dirname "$0")" && pwd)
SELF="$HERE/$(basename "$0")"
say() { printf '\033[38;5;208m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31mxx\033[0m %s\n' "$*" >&2; exit 1; }
mkdir -p "$SW"

# ---------- Docker: usable by this user, installing it or joining the group when sudo allows
docker_ok() { docker info >/dev/null 2>&1; }
if ! docker_ok; then
  NOSUDO="Docker is not usable and sudo asks for a password here. Run once: curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker \$USER  then log out and back in (or run: newgrp docker)"
  if ! command -v docker >/dev/null; then
    sudo -n true 2>/dev/null || die "$NOSUDO"
    say "installing Docker (log: $SW/docker-install.log)"
    curl -fsSL https://get.docker.com | sudo sh >"$SW/docker-install.log" 2>&1 \
      || die "the Docker install failed, see $SW/docker-install.log"
  fi
  if ! sudo -n docker info >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo systemctl start docker >/dev/null 2>&1 || sudo service docker start >/dev/null 2>&1 || true
  fi
  if ! docker_ok; then
    if ! id -nG "$USER" | tr ' ' '\n' | grep -qx docker; then
      sudo -n true 2>/dev/null || die "$NOSUDO"
      say "adding $USER to the docker group"
      sudo usermod -aG docker "$USER"
    fi
    if [ -z "${BH_SWEBENCH_SG:-}" ] && command -v sg >/dev/null; then
      say "re-running under the docker group (sg docker), no new login needed"
      exec sg docker -c "BH_SWEBENCH_SG=1 bash $(printf '%q ' "$SELF" "${ARGS[@]}")"
    fi
    docker_ok || die "Docker still does not answer (docker info). Check: sudo systemctl status docker; then log out and back in so the docker group applies"
  fi
fi
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
if [ -z "$N" ]; then
  cpus=$(nproc); mem=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
  N=$(( cpus - 4 )); [ $(( (mem - 40) / 5 )) -lt "$N" ] && N=$(( (mem - 40) / 5 ))
  [ "$N" -gt 32 ] && N=32; [ "$N" -lt 1 ] && N=1
fi
DOCKER_ROOT=$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)
free_gb() { local d=$DOCKER_ROOT; while [ ! -e "$d" ] && [ "$d" != / ]; do d=$(dirname "$d"); done; df -BG --output=avail "$d" | tail -1 | tr -dc 0-9; }
FREE=$(free_gb)
say "$N tasks at a time, $FREE GB free for docker images ($DOCKER_ROOT)"
[ "$FREE" -ge "$MIN_START_GB" ] || die "only $FREE GB free where Docker keeps images ($DOCKER_ROOT); SWE-bench Pro needs at least $MIN_START_GB GB (each task image is 1 to 5 GB, $N run at once). Free some space (docker system prune -af, old job folders) and rerun"

# ---------- the model: an MLX HEX export -> a checked vLLM folder; a plain MLX folder is refused
SERVE=$MODEL
mkind() {
  python3 - "$1" <<'PY'
import json, os, sys
d = sys.argv[1]
try:
    c = json.load(open(os.path.join(d, "config.json")))
except (OSError, ValueError):
    print("other"); sys.exit()
q = c.get("quantization") or {}
if c.get("hex"):
    print("hex")
elif (isinstance(q, dict) and "group_size" in q) or any(os.path.exists(os.path.join(d, f)) for f in ("flint_mlx.py", "qwen_mlx.py")):
    print("mlx")
else:
    print("hf")
PY
}
KIND=$(mkind "$MODEL")
[ "$KIND" != other ] || [ ! -d "$MODEL" ] || die "$MODEL has no readable config.json"
[ "$KIND" != mlx ] || die "$MODEL is an MLX folder without HEX experts; vLLM cannot serve it. Benchmark the int4 (GPTQ) or healed folder of the job instead, or build with mlx_hex on"
VLLM=""
for v in "$BH_HOME/vllm/bin/vllm" "$HOME/vllm-venv/bin/vllm"; do [ -x "$v" ] && { VLLM=$v; break; }; done
if [ "$CHECK" = 1 ]; then
  [ -n "$VLLM" ] || die "no vLLM found; run the BlackHole installer first: curl -fsSL https://blackholeharness.dev/install.sh | bash"
  [ "$KIND" != hex ] || [ -x "$BH_HOME/venv/bin/python" ] || die "no pipeline venv at $BH_HOME/venv to convert the HEX export; run the BlackHole installer"
  say "checks passed: Docker works, $FREE GB free, vLLM at $VLLM, model $MODEL ($KIND)"
  exit 0
fi
if [ "$KIND" = hex ]; then
  SERVE="${MODEL%/}-vllm"
  if [ ! -f "$SERVE/.verified" ]; then
    say "converting the MLX HEX export for vLLM into $SERVE (checks every layer)"
    rm -rf "$SERVE.tmp"
    "$BH_HOME/venv/bin/python" -m blackhole_runtime.verify "$MODEL" "$SERVE.tmp" \
      || { rm -rf "$SERVE.tmp"; die "the HEX conversion or its layer check failed (output above); nothing was served"; }
    rm -rf "$SERVE"; mv "$SERVE.tmp" "$SERVE"; date -u +%FT%TZ >"$SERVE/.verified"
  fi
fi
if [ -z "$MAXLEN" ] && [ -f "$SERVE/config.json" ]; then
  # the model's own context, capped at 256k (Scale's runs use up to 256k; more only costs KV cache)
  MAXLEN=$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); t=c.get("text_config") or c; m=t.get("max_position_embeddings") or c.get("max_position_embeddings") or 262144; print(min(int(m), 262144))' "$SERVE/config.json")
fi
if [ -z "${REASONING+x}" ]; then
  # vLLM's reasoning parser by model family (thinking kept out of the answer the agent reads); none when unknown
  REASONING=$(python3 -c 'import json,sys
try: c=json.load(open(sys.argv[1]))
except Exception: c={}
t=((c.get("text_config") or {}).get("model_type") or c.get("model_type") or "").lower()
print("qwen3" if t.startswith(("qwen3", "qwen4")) else "deepseek_r1" if t.startswith("deepseek") else "openai_gptoss" if t.startswith("gpt_oss") else "glm45" if t.startswith("glm4_moe") else "")' "$SERVE/config.json" 2>/dev/null || true)
fi
[ -n "$NAME" ] || NAME="$(basename "${MODEL%/}")-$TASKS"
RUN=$SW/runs/$NAME
mkdir -p "$RUN/jobs"
SRC_ID=$(cd "$(dirname "$SERVE")" 2>/dev/null && echo "$(pwd -P)/$(basename "$SERVE")" || echo "$SERVE")
if [ -f "$RUN/model.txt" ] && [ "$(cat "$RUN/model.txt")" != "$SRC_ID" ]; then
  die "run $NAME was started on $(cat "$RUN/model.txt"), not $SRC_ID; give this model its own --name (or delete $RUN)"
fi
echo "$SRC_ID" >"$RUN/model.txt"

# ---------- Harbor + mini-swe-agent (Python >= 3.12, own venv) and the SWE-bench Pro tasks
if ! command -v uv >/dev/null; then
  [ -x "$HOME/.local/bin/uv" ] || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -x "$SW/venv/bin/harbor" ]; then
  say "installing Harbor and mini-swe-agent in $SW/venv"
  uv venv -q -p 3.12 "$SW/venv"
  uv pip install -q -p "$SW/venv/bin/python" 'harbor>=0.22' mini-swe-agent pyyaml
fi
SBP=$SW/SWE-bench_Pro-os
if [ ! -d "$SBP/v2/tasks" ]; then
  say "fetching SWE-bench Pro $SBP_REF"
  rm -rf "$SBP"
  git clone -q --depth 1 --branch "$SBP_REF" https://github.com/scaleapi/SWE-bench_Pro-os "$SBP"
fi
HPY=$SW/venv/bin/python

# ---------- serve the model (or reuse a server on the port that already serves this very model)
VLLM_PID=""; PRUNE_PID=""; CHILD=""
cleanup() {
  if [ -n "$CHILD" ]; then kill "$CHILD" 2>/dev/null || true; fi
  if [ -n "$PRUNE_PID" ]; then pkill -P "$PRUNE_PID" 2>/dev/null || true; kill "$PRUNE_PID" 2>/dev/null || true; fi
  if [ -n "$VLLM_PID" ]; then say "stopping vLLM"; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# a long step in the background + wait, so a stop signal (Ctrl-C, the app's Stop) ends it at once
run_bg() { local rc=0; "$@" & CHILD=$!; wait "$CHILD" || rc=$?; CHILD=""; return "$rc"; }
free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
API=http://127.0.0.1:$PORT/v1
if curl -sf "$API/models" >/dev/null 2>&1; then
  read -r SID SROOT < <(curl -sf "$API/models" | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"][0]; print(d["id"], d.get("root") or d["id"])')
  if [ "$SROOT" = "$SERVE" ] || [ "$SROOT" = "$SRC_ID" ]; then
    SERVED=$SID; say "using the server already on port $PORT ($SERVED)"
  else
    PORT=$(free_port); API=http://127.0.0.1:$PORT/v1
    say "port in use by another model ($SROOT); serving this one on port $PORT"
  fi
fi
if [ -z "$(curl -sf "$API/models" 2>/dev/null)" ]; then
  [ -n "$VLLM" ] || die "no vLLM found; run the BlackHole installer first: curl -fsSL https://blackholeharness.dev/install.sh | bash"
  say "starting vLLM on port $PORT (log: $RUN/vllm.log)"
  LENARG=(); [ -n "$MAXLEN" ] && LENARG=(--max-model-len "$MAXLEN")
  [ -n "$REASONING" ] && LENARG+=(--reasoning-parser "$REASONING")
  # shellcheck disable=SC2086
  "$VLLM" serve "$SERVE" --served-model-name "$SERVED" --host 127.0.0.1 --port "$PORT" \
    "${LENARG[@]}" --max-num-seqs $(( N * 2 )) --enable-prefix-caching \
    --gpu-memory-utilization 0.90 $VLLM_ARGS >"$RUN/vllm.log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 360); do
    curl -sf "$API/models" >/dev/null 2>&1 && break
    kill -0 "$VLLM_PID" 2>/dev/null || { tail -30 "$RUN/vllm.log"; VLLM_PID=""; die "vLLM exited, see $RUN/vllm.log"; }
    sleep 5
  done
  curl -sf "$API/models" >/dev/null || die "vLLM did not come up in 30 minutes, see $RUN/vllm.log"
fi
say "model answers: $(curl -s "$API/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the word ok.\"}],\"max_tokens\":2048}" \
  | python3 -c 'import json,sys; m=json.load(sys.stdin)["choices"][0]["message"]; print(repr((m.get("content") or "")[:60]))' || echo "(no answer, see $RUN/vllm.log)")"

# ---------- agent config and task list
CFG=$RUN/mini.yaml
"$HPY" "$HERE/swebench_pro.py" config "$SBP/v2/tooling/configs/mini_textbased.yaml" "$CFG" --api-base "$API" --max-tokens "$MAX_TOKENS" >/dev/null
IDS=$RUN/ids.txt
say "$("$HPY" "$HERE/swebench_pro.py" ids "$SBP/v2/tasks" "$TASKS" "$IDS") tasks ($TASKS)"
INC=()
while read -r i; do INC+=(-i "*$i"); done <"$IDS"

# ---------- keep the disk from filling with task images (each 1 to 5 GB, 642 of them)
( while sleep 300; do
    if [ "$(free_gb)" -lt "$MIN_FREE_GB" ]; then docker image prune -af >/dev/null 2>&1; docker builder prune -af >/dev/null 2>&1; fi
  done ) >/dev/null 2>&1 </dev/null &
PRUNE_PID=$!

# ---------- 1. agent run, 2. re-grade on fresh containers, 3. score
export PYTHONPATH="$SBP/v2/tooling${PYTHONPATH:+:$PYTHONPATH}"
H="$SW/venv/bin/harbor"
cd "$RUN"
if [ -f "jobs/$NAME/config.json" ]; then
  say "resuming jobs/$NAME"
  run_bg "$H" job resume -p "jobs/$NAME"
else
  say "agent run: jobs/$NAME"
  run_bg "$H" run -p "$SBP/v2/tasks" "${INC[@]}" -e docker -n "$N" -r 1 -y -q \
    -a host_mini_swe:HostMiniSweAgent -m "openai/$SERVED" --ak config_file="$CFG" \
    -o jobs --job-name "$NAME"
fi
"$HPY" "$HERE/swebench_pro.py" score "jobs/$NAME" --out "$RUN/agent-results.json"
if [ -n "$VLLM_PID" ]; then say "stopping vLLM (the re-grade does not need the model)"; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; VLLM_PID=""; fi
if [ -f "jobs/$NAME-regrade/config.json" ]; then
  run_bg "$H" job resume -p "jobs/$NAME-regrade"
else
  say "re-grading every patch on a fresh container: jobs/$NAME-regrade"
  run_bg "$H" run -p "$SBP/v2/tasks" "${INC[@]}" -e docker -n "$N" -r 1 -y -q \
    -a patch_replay:PatchReplayAgent -m replay --ak source_job="$RUN/jobs/$NAME" \
    -o jobs --job-name "$NAME-regrade"
fi
say "SWE-bench Pro V2 ($TASKS), official re-graded score:"
"$HPY" "$HERE/swebench_pro.py" score "jobs/$NAME-regrade" --out "$RUN/results.json"
say "per task: $RUN/results.json, summary: $RUN/results.summary.json"
