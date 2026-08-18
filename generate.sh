#!/bin/bash -l
set -euo pipefail
if [[ $# -gt 1 ]]; then
  echo "Usage: bash generate.sh [server-config.yaml]" >&2
  exit 2
fi
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$PROJECT_ROOT"
export HF_TOKEN=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
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export NO_PROXY="${NO_PROXY:-$no_proxy}"
CONFIG_PATH="${1:-server.yaml.example}"
eval "$(uv run python -m multi_agent_sync.generate "$CONFIG_PATH" --print-env)"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export http_proxy="${http_proxy:-http://proxy.rrze.uni-erlangen.de:80}"
  export https_proxy="${https_proxy:-http://proxy.rrze.uni-erlangen.de:80}"
  export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
  export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
  export HF_HOME="${HF_HOME:-$PROJECT_ROOT}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TMPDIR:-/tmp}/.triton}"
  export OUTLINES_CACHE_DIR="${OUTLINES_CACHE_DIR:-${TMPDIR:-/tmp}}"
  exec uv run python -m multi_agent_sync.generate "$CONFIG_PATH"
fi

if command -v sbatch >/dev/null 2>&1 && [[ "${RUN_LOCAL:-0}" != "1" ]]; then
  slurm_output="$(uv run python -m multi_agent_sync.generate "$CONFIG_PATH" --print-slurm-args)"
  mapfile -t slurm_args <<< "$slurm_output"
  exec sbatch "${slurm_args[@]}" "$0" "$CONFIG_PATH"
fi

evaluation_matrix="$(uv run python -m multi_agent_sync.generate "$CONFIG_PATH" --print-evaluation-matrix)"
while IFS=$'\t' read -r benchmark methods limit output_dir resume_run max_steps single_agent_min_steps single_agent_max_steps debate_rounds; do
  evaluation_args=(
    --benchmark "$benchmark"
    --methods "$methods"
    --limit "$limit"
    --server-config "$CONFIG_PATH"
  )
  if [[ "${max_steps:-}" != "-" && -n "${max_steps:-}" ]]; then
    evaluation_args+=(--max-steps "$max_steps")
  fi
  if [[ "${single_agent_min_steps:-}" != "-" && -n "${single_agent_min_steps:-}" ]]; then
    evaluation_args+=(--single-agent-min-steps "$single_agent_min_steps")
  fi
  if [[ "${single_agent_max_steps:-}" != "-" && -n "${single_agent_max_steps:-}" ]]; then
    evaluation_args+=(--single-agent-max-steps "$single_agent_max_steps")
  fi
  if [[ "${debate_rounds:-}" != "-" && -n "${debate_rounds:-}" ]]; then
    evaluation_args+=(--debate-rounds "$debate_rounds")
  fi
  if [[ "$output_dir" != "-" ]]; then
    evaluation_args+=(--output-dir "$output_dir")
  fi
  if [[ "$resume_run" != "-" ]]; then
    evaluation_args+=(--resume-run "$resume_run")
  fi
  uv run evaluation "${evaluation_args[@]}"
done <<< "$evaluation_matrix"
