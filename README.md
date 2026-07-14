# Multi-Agent Sync Prototype

Local Python prototype for running a LangGraph-based multi-agent workflow and benchmark evaluations.

The project intentionally does not use AutoGen.

## Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Run the Multi-Agent CLI](#run-the-multi-agent-cli)
- [Run Evaluations](#run-evaluations)
- [Lean / MA-ProofBench / OlymMATH-LEAN](#lean--ma-proofbench--olymmath-lean)
- [SWE-bench Verified](#swe-bench-verified)
- [Artifacts](#artifacts)
- [Tests](#tests)

## Requirements

- Python 3.12+
- `uv`
- A model backend:
  - OpenAI-compatible API server, defaulting to `http://localhost:8000/v1`
  - or Ollama for local models
- Docker, only if using Docker workspace features, Lean agent workspace feedback, or SWE-bench workspace repair

## Setup

From the repository root:

```bash
cd /Users/billy/Desktop/UTN/Autogen-multiagents
uv sync
```

If `uv sync` is not available in your `uv` version, this also works:

```bash
uv run python -m pytest --version
```

That command will create the environment from `pyproject.toml` / `uv.lock` before running.

## Environment Variables

### OpenAI-compatible provider

Used by default.

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_MODEL=openai/gpt-oss-120b
```

Notes:

- `OPENAI_BASE_URL` defaults to `http://localhost:8000/v1`.
- `OPENAI_MODEL` defaults to `openai/gpt-oss-120b` for the main CLI.
- Evaluation defaults to `--model auto`, which checks `${OPENAI_BASE_URL}/models` and uses the first model id.
- `OPENAI_MODEL_LOOKUP_TIMEOUT` controls evaluation auto-detection timeout in seconds. Default: `2`.

Example:

```bash
export OPENAI_MODEL_LOOKUP_TIMEOUT=5
```

### Ollama provider

Used when passing `--local-model`.

```bash
export OLLAMA_MODEL=qwen3:4b
```

Example Ollama setup:

```bash
ollama pull qwen3:4b
```

### Hugging Face datasets

Some datasets are downloaded from Hugging Face and cached under `data/`.

```bash
export HF_TOKEN=<your-token>
```

`HF_TOKEN` is required for gated GPQA access unless you provide a local `--data-file`.

### Lean / Kimina

Required for verifier-based Lean scoring.

```bash
export KIMINA_CLIENT_PATH=/path/to/MA-ProofBench/kimina-lean-server/client
```

Set this only if `kimina_client` is not importable in the current environment.

## Run the Multi-Agent CLI

Basic run:

```bash
uv run python -m multi_agent_sync "Build a prototype chess website"
```

Equivalent console script:

```bash
uv run multi-agent-sync "Build a prototype chess website"
```

Use a specific OpenAI-compatible model:

```bash
uv run python -m multi_agent_sync --model openai/gpt-oss-120b "Build a prototype chess website"
```

Use Ollama:

```bash
uv run python -m multi_agent_sync --local-model --model qwen3:4b "Build a prototype chess website"
```

Use dynamic subagents:

```bash
uv run python -m multi_agent_sync --subagent-mode dynamic "Build a prototype chess website"
```

Limit runtime and agent steps:

```bash
uv run python -m multi_agent_sync \
  --max-steps 3 \
  --total-runtime-timeout 600 \
  "Build a prototype chess website"
```

Enable Docker workspace tools for eligible agents:

```bash
uv run python -m multi_agent_sync \
  --docker-workspace \
  --workspace-image python:3.12 \
  --workspace-source . \
  "Fix the failing tests in this project"
```

Useful CLI flags:

- `--model <model>`: model name for the selected provider
- `--local-model`: use Ollama instead of the OpenAI-compatible provider
- `--subagent-mode fixed|dynamic`: choose fixed registered agents or dynamic subagents
- `--max-steps <n>`: maximum inference steps per agent
- `--total-runtime-timeout <seconds>`: total multi-agent runtime limit
- `--runs-dir <path>`: where CLI run artifacts are written
- `--no-color`: disable colored terminal output
- `--docker-workspace`: enable Docker bash workspace tools

## Run Evaluations

The evaluation command is:

```bash
uv run evaluation --benchmark <benchmark> --methods <methods> --limit <n>
```

Supported benchmarks:

- `gpqa`
- `gsm8k`
- `chess`
- `mmlu_pro`
- `ma_proofbench`
- `olymmath`
- `olymmath_lean`
- `swe_bench_verified`

Supported methods:

- `multiagent_streaming`
- `multiagent_no_streaming`
- `multiagent_dynamic_streaming`
- `multiagent_dynamic_no_streaming`
- `multiagent_debate`
- `majority_vote`
- `single_agent`
- `plain_llm`
- `multiagent` legacy alias for `multiagent_streaming`

Run a small GPQA evaluation:

```bash
uv run evaluation \
  --benchmark gpqa \
  --methods multiagent_streaming,multiagent_no_streaming,plain_llm \
  --limit 10
```

Run with Ollama:

```bash
uv run evaluation \
  --benchmark gsm8k \
  --methods multiagent_streaming,plain_llm \
  --local-model \
  --model qwen3:4b \
  --limit 10
```

Run fixed and dynamic multi-agent variants together:

```bash
uv run evaluation \
  --benchmark gpqa \
  --methods multiagent_streaming,multiagent_no_streaming,multiagent_dynamic_streaming,multiagent_dynamic_no_streaming,plain_llm \
  --limit 10
```

Run common benchmarks:

```bash
uv run evaluation --benchmark gsm8k --methods multiagent_streaming,multiagent_no_streaming,plain_llm --limit 10
uv run evaluation --benchmark chess --methods multiagent_streaming,single_agent,plain_llm --limit 10
uv run evaluation --benchmark mmlu_pro --methods multiagent_streaming,multiagent_no_streaming,plain_llm --limit 10
uv run evaluation --benchmark olymmath --olymmath-subset en-hard --methods multiagent_streaming,plain_llm --limit 10
```

Use a local benchmark file:

```bash
uv run evaluation \
  --benchmark gpqa \
  --methods multiagent_streaming,plain_llm \
  --data-file /path/to/gpqa.csv \
  --limit 10
```

Useful evaluation flags:

- `--model <model>`: model name, or `auto` for OpenAI-compatible model detection
- `--local-model`: use Ollama
- `--output-dir <path>`: output root, default `output`
- `--output <file-or-path>`: CSV output name/path
- `--save-json-traces` / `--no-save-json-traces`: write per-example traces
- `--max-steps <n>`: max steps for multi-agent and single-agent methods
- `--total-runtime-timeout <seconds>`: per-example multi-agent timeout
- `--synthesis-timeout <seconds>`: synthesizer timeout
- `--seed <n>`: answer shuffle seed

## Lean / MA-ProofBench / OlymMATH-LEAN

Lean benchmarks use Kimina Lean Server for verification.

### 1. Start Kimina Lean Server

In a separate directory:

```bash
git clone https://github.com/OpenBMB/MA-ProofBench.git
cd MA-ProofBench/kimina-lean-server

cp .env.template .env
bash setup.sh
pip install -r requirements.txt
pip install .
prisma generate
python -m server
```

The evaluation commands below assume the server is available at `127.0.0.1:8001`.

If the Python client is not importable:

```bash
export KIMINA_CLIENT_PATH=/path/to/MA-ProofBench/kimina-lean-server/client
```

### 2. Run MA-ProofBench

```bash
uv run evaluation \
  --benchmark ma_proofbench \
  --methods plain_llm \
  --limit 10 \
  --kimina-host 127.0.0.1 \
  --kimina-port 8001
```

Run one difficulty level:

```bash
uv run evaluation \
  --benchmark ma_proofbench \
  --ma-proofbench-level level1 \
  --methods multiagent_streaming,plain_llm \
  --limit 10 \
  --kimina-host 127.0.0.1 \
  --kimina-port 8001
```

Enable Lean agent workspace feedback:

```bash
uv run evaluation \
  --benchmark ma_proofbench \
  --methods multiagent_streaming \
  --lean-agent-workspace \
  --workspace-image python:3.12 \
  --limit 5 \
  --kimina-host 127.0.0.1 \
  --kimina-port 8001
```

### 3. Run OlymMATH-LEAN

```bash
uv run evaluation \
  --benchmark olymmath_lean \
  --methods plain_llm \
  --limit 10 \
  --kimina-host 127.0.0.1 \
  --kimina-port 8001
```

Lean-specific flags:

- `--lean-timeout <seconds>`: verifier timeout per candidate, default `60`
- `--kimina-host <host>`: default `127.0.0.1`
- `--kimina-port <port>`: default `8001`
- `--kimina-max-workers <n>`: Kimina workers per request
- `--lean-agent-workspace`: enable Docker workspace editing plus verifier feedback for Lean multi-agent methods

## SWE-bench Verified

SWE-bench can be run in two stages:

1. Generate prediction patches.
2. Optionally run the official SWE-bench harness to score them.

### 1. Install SWE-bench harness dependencies

The official harness is not listed as a core dependency in this project. Install it in the environment you will use for evaluation:

```bash
uv pip install swebench
```

Docker must be running for the official harness.

### 2. Generate prediction patches

Simple patch generation:

```bash
uv run evaluation \
  --benchmark swe_bench_verified \
  --methods plain_llm \
  --limit 1
```

Run multi-agent repair with a Docker workspace:

```bash
uv run evaluation \
  --benchmark swe_bench_verified \
  --methods multiagent_streaming \
  --swebench-agent-workspace \
  --workspace-image python:3.12 \
  --workspace-command-timeout 120 \
  --limit 1
```

This writes prediction JSONL files under the run output directory, for example:

```text
output/swe_bench_verified/<model>/<run_id>/predictions_multiagent_streaming.jsonl
```

### 3. Run the official harness

Run patch generation and official evaluation in one command:

```bash
uv run evaluation \
  --benchmark swe_bench_verified \
  --methods multiagent_streaming \
  --swebench-agent-workspace \
  --swebench-run-harness \
  --swebench-max-workers 1 \
  --limit 1
```

Evaluate specific instance ids:

```bash
uv run evaluation \
  --benchmark swe_bench_verified \
  --methods multiagent_streaming \
  --swebench-agent-workspace \
  --swebench-run-harness \
  --swebench-instance-ids astropy__astropy-12907 \
  --limit 1
```

On Mac ARM, you may need to force local image builds by passing an empty namespace:

```bash
uv run evaluation \
  --benchmark swe_bench_verified \
  --methods multiagent_streaming \
  --swebench-agent-workspace \
  --swebench-run-harness \
  --swebench-namespace "" \
  --limit 1
```

SWE-bench flags:

- `--swebench-agent-workspace`: enable Docker workspace editing and git diff export
- `--swebench-run-harness`: run `swebench.harness.run_evaluation`
- `--swebench-max-workers <n>`: official harness worker count
- `--swebench-run-id <id>`: run id passed to the official harness
- `--swebench-namespace <namespace>`: Docker image namespace for the official harness
- `--swebench-instance-ids <ids>`: comma-separated or space-separated instance ids

Local SWE-bench data files can be `.csv`, `.jsonl`, or `.ndjson` and must include:

- `repo`
- `instance_id`
- `base_commit`
- `problem_statement`

Example:

```bash
uv run evaluation \
  --benchmark swe_bench_verified \
  --methods plain_llm \
  --data-file /path/to/swe_bench_verified.jsonl \
  --limit 1
```

## Artifacts

### Multi-agent CLI artifacts

CLI runs write to `runs/<timestamp>/` by default:

```text
runs/<timestamp>/
  task.txt
  final_answer.md
  event_log.jsonl
  agent_traces.json
  orchestrator_plan.json
```

Change the root directory:

```bash
uv run python -m multi_agent_sync --runs-dir my-runs "Build a prototype chess website"
```

### Evaluation artifacts

Evaluation runs write to:

```text
output/<benchmark>/<model>/<run_id>/
```

Typical files:

```text
<benchmark>_results.csv
correctness.csv
run_config.json
summary.json
examples/*.json
predictions_<method>.jsonl
harness_<method>.json
```

Compare traces after a run:

```bash
uv run python scripts/compare_eval_traces.py output/<benchmark>/<model>/<run_id>
```

Summarize CSV results:

```bash
uv run python scripts/summarize_csv_results.py output/<benchmark>/<model>/<run_id>/<benchmark>_results.csv
```

## Tests

Run all tests:

```bash
uv run pytest
```

Run a focused test file:

```bash
uv run pytest tests/test_evaluation.py
```

Tests use fake LLMs and do not require a model server.

## Current Limitations

- The event streamer is in-memory and process-local.
- External event streamer backends are not implemented yet.
- There is no web UI.
- The CLI and evaluations require the selected model provider to be available unless tests are using fake LLMs.
