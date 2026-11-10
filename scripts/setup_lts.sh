#!/usr/bin/env bash
# Set up the LTS baselines (Learning to Share, Fioresi et al., ICML 2026;
# https://github.com/UCF-CRCV/autogen-lts) in the project environment.
#
#   scripts/setup_lts.sh                 # submodule + `uv sync --inexact --extra lts` + DT checkpoint
#   SKIP_WEIGHTS=1 scripts/setup_lts.sh
#
# `--inexact` keeps packages that are already installed (e.g. the private
# vllm-server-launcher from the `vllm-server` extra); a plain `uv sync` would
# remove every extra that is not named. Extra uv flags are passed through, e.g.
# `scripts/setup_lts.sh --extra vllm-server` to (re)install the launcher as well.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${LTS_WEIGHTS_DIR:-$ROOT/third_party/lts-weights}"

cd "$ROOT"
if [ ! -d third_party/autogen-lts/python/packages/autogen-agentchat ]; then
  # The fork's Git LFS files are only images, and some are missing on the LFS
  # server (404), which breaks checkout. Skip LFS downloads; no code is in LFS.
  GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1 third_party/autogen-lts
fi

uv sync --inexact --extra lts "$@"

if [ "${SKIP_WEIGHTS:-0}" != "1" ]; then
  # Released decision-transformer checkpoint (~47 MB). The Qwen/Qwen3-0.6B base
  # model is fetched from Hugging Face on first use by DecisionTransformer itself.
  uv run --inexact --extra lts "$@" python - "$WEIGHTS_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download

path = hf_hub_download("jfioresi/lts-decision-transformer", "checkpoint.pt", local_dir=sys.argv[1])
print(f"Decision transformer checkpoint: {path}")
PY
fi

uv run --inexact --extra lts "$@" python -c "import autogen_agentchat.teams as t; assert hasattr(t, 'MagenticMemoryGroupChat'); print('LTS environment OK')"
