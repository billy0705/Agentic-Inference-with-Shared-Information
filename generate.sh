#!/bin/bash -l
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: bash generate.sh [server-config.yaml]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
CONFIG_PATH="${1:-server.yaml.example}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec uv run python -m multi_agent_sync.generate "$CONFIG_PATH"
fi

if command -v sbatch >/dev/null 2>&1 && [[ "${RUN_LOCAL:-0}" != "1" ]]; then
  slurm_output="$(uv run python -m multi_agent_sync.generate "$CONFIG_PATH" --print-slurm-args)"
  mapfile -t slurm_args <<< "$slurm_output"
  exec sbatch "${slurm_args[@]}" "$0" "$CONFIG_PATH"
fi

evaluation_matrix="$(uv run python -m multi_agent_sync.generate "$CONFIG_PATH" --print-evaluation-matrix)"
while IFS=$'\t' read -r benchmark methods limit; do
  uv run evaluation \
    --benchmark "$benchmark" \
    --methods "$methods" \
    --limit "$limit" \
    --server-config "$CONFIG_PATH"
done <<< "$evaluation_matrix"
