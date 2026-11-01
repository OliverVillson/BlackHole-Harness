#!/usr/bin/env bash
# SWE-bench Pro V2 (Scale AI, 642 tasks, Harbor format) on one GPU VM with a local model.
#
#   bash swebench-pro.sh MODEL [--tasks smoke|hard|all|N] [-n CONCURRENT] [--name NAME]
#
# MODEL is a BlackHole MLX HEX export (converted once to a vLLM blackhole-hex folder next to
# it), a vLLM-ready folder, or a Hugging Face id. Run it inside tmux; rerunning the same
# command resumes. Everything lands in ~/.blackhole/swebench/runs/NAME:
#   vllm.log, jobs/NAME (agent run), jobs/NAME-regrade (official score), results.json
#
# How it scores: mini-swe-agent with Scale's locked prompt (v2/tooling/configs/
# mini_textbased.yaml) runs on this machine and only executes bash inside the task's
# container, which has no network. Its git diff is then re-graded on a fresh container
# (patch_replay), which is the number to report. 50-minute budget per task.
set -euo pipefail

MODEL=""; TASKS=smoke; N=""; NAME=""; PORT=${PORT:-8000}; SERVED=${SERVED:-local-model}
MAXLEN=${MAXLEN:-262144}; MAX_TOKENS=${MAX_TOKENS:-16384}; MIN_FREE_GB=${MIN_FREE_GB:-150}
SBP_REF=${SBP_REF:-v2.0.0}; VLLM_ARGS=${VLLM_ARGS:-}
while [ $# -gt 0 ]; do
  case "$1" in
    --tasks) TASKS="$2"; shift ;;
    -n|--concurrent) N="$2"; shift ;;
    --name) NAME="$2"; shift ;;
    --port) PORT="$2"; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    -*) echo "unknown option $1"; exit 2 ;;
    *) MODEL="$1" ;;
  esac
  shift
done
[ -n "$MODEL" ] || { sed -n '2,16p' "$0"; exit 2; }

BH_HOME=${BLACKHOLE_HOME:-$HOME/.blackhole}
SW=$BH_HOME/swebench
HERE=$(cd "$(dirname "$0")" && pwd)
say() { printf '\033[38;5;208m==>\033[0m %s\n' "$*"; }
die() { printf '\033[31mxx\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- checks
docker info >/dev/null 2>&1 || die "docker is not usable. Install it with: curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker \$USER  (then log out and back in)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
if [ -z "$N" ]; then
  cpus=$(nproc); mem=$(awk '/MemTotal/ {print int($2/1048576)}' /proc/meminfo)
  N=$(( cpus - 4 )); [ $(( (mem - 40) / 5 )) -lt "$N" ] && N=$(( (mem - 40) / 5 ))
  [ "$N" -gt 32 ] && N=32; [ "$N" -lt 1 ] && N=1
fi
free_gb() { df -BG --output=avail "$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /)" | tail -1 | tr -dc 0-9; }
say "$N tasks at a time, $(free_gb) GB free for docker images"

# ---------- Harbor + mini-swe-agent (Python >= 3.12, own venv) and the SWE-bench Pro tasks
mkdir -p "$SW"
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
  git clone -q --depth 1 --branch "$SBP_REF" https://github.com/scaleapi/SWE-bench_Pro-os "$SBP"
fi
HPY=$SW/venv/bin/python

# ---------- the model: MLX HEX export -> vLLM folder, then vllm serve
SERVE=$MODEL
if [ -f "$MODEL/config.json" ] && python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); sys.exit(0 if c.get("hex") else 1)' "$MODEL/config.json"; then
  SERVE="${MODEL%/}-vllm"
  if [ ! -f "$SERVE/config.json" ]; then
    say "converting the MLX HEX export for vLLM into $SERVE (checks every layer)"
    "$BH_HOME/venv/bin/python" -m blackhole_runtime.verify "$MODEL" "$SERVE"
  fi
fi
[ -n "$NAME" ] || NAME="$(basename "${MODEL%/}")-$TASKS"
RUN=$SW/runs/$NAME
mkdir -p "$RUN/jobs"
API=http://127.0.0.1:$PORT/v1
VLLM_PID=""; PRUNE_PID=""
cleanup() {
  if [ -n "$PRUNE_PID" ]; then kill "$PRUNE_PID" 2>/dev/null || true; fi
  if [ -n "$VLLM_PID" ]; then say "stopping vLLM"; kill "$VLLM_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT
if curl -sf "$API/models" >/dev/null 2>&1; then
  SERVED=$(curl -sf "$API/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
  say "using the server already on port $PORT ($SERVED)"
else
  VLLM=""
  for v in "$BH_HOME/vllm/bin/vllm" "$HOME/vllm-venv/bin/vllm"; do [ -x "$v" ] && { VLLM=$v; break; }; done
  [ -n "$VLLM" ] || die "no vLLM found; run the BlackHole installer first: curl -fsSL https://blackholeharness.dev/install.sh | bash"
  say "starting vLLM on port $PORT (log: $RUN/vllm.log)"
  # shellcheck disable=SC2086
  "$VLLM" serve "$SERVE" --served-model-name "$SERVED" --host 127.0.0.1 --port "$PORT" \
    --max-model-len "$MAXLEN" --max-num-seqs $(( N * 2 )) --enable-prefix-caching \
    --reasoning-parser qwen3 --gpu-memory-utilization 0.90 $VLLM_ARGS >"$RUN/vllm.log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 360); do
    curl -sf "$API/models" >/dev/null 2>&1 && break
    kill -0 "$VLLM_PID" 2>/dev/null || { tail -30 "$RUN/vllm.log"; die "vLLM exited, see $RUN/vllm.log"; }
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
  done ) &
PRUNE_PID=$!

# ---------- 1. agent run, 2. re-grade on fresh containers, 3. score
export PYTHONPATH="$SBP/v2/tooling${PYTHONPATH:+:$PYTHONPATH}"
H="$SW/venv/bin/harbor"
cd "$RUN"
if [ -f "jobs/$NAME/config.json" ]; then
  say "resuming jobs/$NAME"
  "$H" job resume -p "jobs/$NAME"
else
  say "agent run: jobs/$NAME"
  "$H" run -p "$SBP/v2/tasks" "${INC[@]}" -e docker -n "$N" -r 1 -y -q \
    -a host_mini_swe:HostMiniSweAgent -m "openai/$SERVED" --ak config_file="$CFG" \
    -o jobs --job-name "$NAME"
fi
"$HPY" "$HERE/swebench_pro.py" score "jobs/$NAME" --out "$RUN/agent-results.json"
if [ -f "jobs/$NAME-regrade/config.json" ]; then
  "$H" job resume -p "jobs/$NAME-regrade"
else
  say "re-grading every patch on a fresh container: jobs/$NAME-regrade"
  "$H" run -p "$SBP/v2/tasks" "${INC[@]}" -e docker -n "$N" -r 1 -y -q \
    -a patch_replay:PatchReplayAgent -m replay --ak source_job="$RUN/jobs/$NAME" \
    -o jobs --job-name "$NAME-regrade"
fi
say "SWE-bench Pro V2 ($TASKS), official re-graded score:"
"$HPY" "$HERE/swebench_pro.py" score "jobs/$NAME-regrade" --out "$RUN/results.json"
say "per task: $RUN/results.json, summary: $RUN/results.summary.json"
