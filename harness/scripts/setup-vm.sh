#!/usr/bin/env bash
# Kept so older notes still work: the one installer is install.sh at the repo root (the same file
# `blackhole up` runs and https://blackholeharness.dev/install.sh serves). Safe to rerun (it updates).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$here/install.sh" --vllm yes "$@"
