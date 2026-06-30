# Multi-Agent Sync Prototype

This is a local Python prototype for runtime synchronization between concurrent agents. A LangGraph workflow owns the high-level lifecycle, while agents exchange internal findings through a Kafka-like event streamer while they are still running.

The prototype intentionally does not use AutoGen.

## Why LangGraph

LangGraph is used for the workflow lifecycle:

1. The orchestrator asks the LLM to inspect the task and create a structured plan.
2. The orchestrator always creates a multi-agent plan.
3. The runtime starts only the selected concurrent workers.
4. The synthesizer combines worker outputs and the event log into a final answer.

The graph shape is:

```text
START -> orchestrator -> run_multi_agent_runtime -> synthesizer -> END
```

This follows the orchestrator-worker style conceptually: the orchestrator assigns work, workers execute concurrently, and the final node synthesizes results.

## Model-Based Orchestration

The orchestrator no longer uses a fixed keyword classifier or a fixed task-type-to-agent mapping. It asks the configured LLM to return a strict JSON plan with:

- `mode`: `multi_agent`
- `subagent_mode`: `fixed` or `dynamic`
- `task_type`: a short free-text label generated for this task
- `task_summary`
- `reason`
- `selected_agents`: a small list of registered agents with concrete subtasks
- `collaboration_protocol`: event types to share and whether reactive steps are enabled

The runtime constructs only the agents listed in `selected_agents`. In fixed mode, every selected name must exist in `AGENT_REGISTRY`, and every plan includes `CriticAgent` or `VerifierAgent`.

By default the runtime uses fixed subagents:

```bash
uv run python -m multi_agent_sync --subagent-mode fixed "Build a prototype chess website"
```

Dynamic subagents can be enabled with:

```bash
uv run python -m multi_agent_sync --subagent-mode dynamic "Build a prototype chess website"
```

In dynamic mode, the Orchestrator freely names and describes the subagents for the current task. Each dynamic subagent can include a role, description, rules, concrete subtask, expected output, and `critical_debate` flag. Dynamic multi-agent plans must include at least two subagents and at least one critical debate subagent. Invalid dynamic model output falls back to `TaskWorker` and `CriticalDebateAgent`.

The console stream prints the dynamic subagent plan and shows live message routing as `source -> target`, including `broadcast` events.

The orchestrator validates model output before runtime execution:

- invented agent names are removed
- `ArchitectAgent` is never required or selected
- direct mode is disabled and falls back to a deterministic multi-agent plan
- multi-agent plans with no valid workers fall back to a small deterministic agent set
- fixed multi-agent plans always include `CriticAgent` or `VerifierAgent`
- dynamic multi-agent plans always include a critical debate subagent
- selected agents are limited to at most four

Examples:

- Simple Q&A: a small multi-agent set with critique or verification.
- Calculation: `SolverAgent`, `VerifierAgent`.
- Coding or debugging task: usually `CodingAgent`, `CriticAgent`, and `VerifierAgent`.
- Research or comparison task: usually `ResearchAgent`, `SolverAgent`, and `CriticAgent`.
- Philosophy, proof, or reasoning task: usually `SolverAgent`, `CriticAgent`, and sometimes `VerifierAgent`.

## Model Providers

The default provider is an OpenAI-compatible API endpoint at `http://localhost:8000/v1` using `langchain-openai`. The default API model is `openai/gpt-oss-120b`, or `OPENAI_MODEL` when set.

Use a different API model with:

```bash
uv run python -m multi_agent_sync --model openai/gpt-oss-120b "Build a prototype chess website"
```

Use a local Ollama model instead with:

```bash
uv run python -m multi_agent_sync --local-model --model llama3.2 "Build a prototype chess website"
```

## Internal Finding Synchronization

Each agent has its own assigned subtask and runs several local LLM steps. After each step, it may publish a useful finding. Other agents subscribe to the event streamer and can include those findings in later prompts.

Example multi-agent flow:

1. `ResearchAgent` publishes that real-time chess needs move synchronization.
2. `CodingAgent` receives that event during its own run and may use it in a later step.
3. `CriticAgent` receives both findings and may publish a critique about server-side move validation.

Agents do not call each other directly. All agent-to-agent communication goes through `EventStreamer`.

## Why EventStreamer Exists

LangGraph state is useful for lifecycle state, but reducer-style graph state alone is not enough for live runtime sharing between workers. If workers only merge state after completion, they cannot react to each other's discoveries while still working.

`EventStreamer` gives the runtime a separate communication channel:

- append-only event log
- per-event-type subscriptions
- subscribe-all hooks for terminal streaming
- async dispatch to multiple subscribers
- bounded draining and handler timeouts

## Event Model

Events are structured Pydantic models with:

- `event_id`
- `run_id`
- `source`
- `target`
- `event_type`
- `content`
- `confidence`
- `metadata`
- `timestamp`

Supported event types include `task_started`, `plan_created`, `agent_started`, `finding`, `message_received`, `question`, `warning`, `critique`, `error`, `agent_done`, and `final_summary`.

## In-Memory Streamer

`InMemoryEventStreamer` is the first local Kafka-like implementation. It stores every event in an append-only list, then dispatches callbacks for the event type and callbacks registered with `subscribe_all`.

The implementation uses asyncio tasks, catches failing handlers, applies handler timeouts, and provides `drain()` so callers can wait for currently pending callback tasks without looping forever.

## Kafka Extension

`KafkaEventStreamer` is a stub for future extension. The intended mapping is:

```text
finding    -> agent.finding
warning    -> agent.warning
critique   -> agent.critique
agent_done -> agent.done
```

To replace the local streamer later, implement the same `EventStreamer` interface with Kafka producers/consumers. The graph and agents should not need structural changes because they depend on the interface, not the in-memory implementation.

Redis Streams or NATS can follow the same pattern: map `event_type` to a stream/topic/subject, preserve `run_id`, and keep `event_id` for deduplication.

## Agents

The prototype includes five possible workers:

- `ResearchAgent`: investigates architecture, assumptions, options, and constraints.
- `CodingAgent`: plans modules, APIs, dependencies, and implementation shape.
- `CriticAgent`: reviews findings for risks, race conditions, missing cases, and safety issues.
- `SolverAgent`: solves math, science, physics, calculation, and direct reasoning problems.
- `VerifierAgent`: checks numerical correctness, unit conversion, assumptions, contradictions, and overclaiming.

Each agent:

- subscribes to relevant events
- ignores its own events
- tracks `seen_event_ids`
- avoids duplicate event processing
- limits steps with `max_steps`
- limits runtime with `max_runtime_seconds`
- limits observed events with `max_events_per_agent`
- publishes `agent_started`, findings or critiques, and `agent_done`

## Traces

`TraceLogger` is separate from `EventStreamer`.

`EventStreamer` is for runtime communication between agents. `TraceLogger` is for debugging and experiment analysis. It records each agent step with:

- inbox events available to the step
- full prompt
- raw LLM response
- parsed output
- events published by that step
- timing data

Full prompts and raw responses are not published to the event stream.

## Run Artifacts

Each CLI run saves artifacts under:

```text
runs/<timestamp>/
  task.txt
  final_answer.md
  event_log.jsonl
  agent_traces.json
  orchestrator_plan.json
```

## Console Streaming

The console streamer subscribes to all events and prints them while the run is active:

```text
[0000.10] [task_started] coordinator: Build a prototype chess website
[0001.20] [plan_created] orchestrator: mode=multi_agent, task_type=software prototype task, selected_agents=ResearchAgent,CodingAgent,CriticAgent
[0001.30] [agent_started] ResearchAgent: Researching architecture choices
[0002.10] [finding] ResearchAgent: Real-time chess needs move synchronization
[0002.30] [message_received] CodingAgent: received finding from ResearchAgent
[0003.50] [critique] CriticAgent: Need server-side move validation
[0005.00] [final_summary] Synthesizer: Final answer generated.
```

Events are not buffered until the end; they stream as subscribers receive them.

## Setup

The project was originally created with:

```bash
uv init multi-agent-sync-prototype
cd multi-agent-sync-prototype
uv add langgraph langchain langchain-ollama pydantic rich pytest pytest-asyncio
```

The project files now live at the repository root:

```bash
cd /Users/billy/Desktop/UTN/Autogen-multiagents
```

Run from that root directory:

```bash
uv run python -m multi_agent_sync "Build a prototype chess website"
```

Optional arguments:

```bash
uv run python -m multi_agent_sync --local-model --model qwen3:4b --max-steps 3 --no-color "Build a prototype chess website"
```

Run tests:

```bash
uv run pytest
```

Tests use fake LLMs, so they do not require an Ollama server.

## Evaluation

Run GPQA-Diamond against the multi-agent workflow with agent-to-agent message streaming, the same workflow without agent-to-agent message streaming, and a direct LLM baseline:

```bash
uv run evaluation --benchmark gpqa --methods multiagent_streaming,multiagent_no_streaming,plain_llm --limit 10
```

Run GSM8K test-set math word problems with the same methods:

```bash
uv run evaluation --benchmark gsm8k --methods multiagent_streaming,multiagent_no_streaming,plain_llm --limit 10
```

Run MMLU-Pro test-set multiple-choice questions with the same methods:

```bash
uv run evaluation --benchmark mmlu_pro --methods multiagent_streaming,multiagent_no_streaming,plain_llm --limit 10
```

Run MA-ProofBench Lean theorem-proving problems with the same methods:

```bash
uv run evaluation --benchmark ma_proofbench --methods multiagent_streaming,multiagent_no_streaming,plain_llm --limit 10
```

Run OlymMATH natural-language Olympiad problems with answer-key scoring:

```bash
uv run evaluation --benchmark olymmath --olymmath-subset en-hard --methods multiagent_streaming,multiagent_no_streaming,plain_llm --limit 10
```

Run OlymMATH-LEAN theorem-proving problems with verifier-based scoring:

```bash
uv run evaluation --benchmark olymmath_lean --methods plain_llm --limit 10 --kimina-host 127.0.0.1 --kimina-port 8001
```

To compare fixed subagents, dynamic subagents, and the direct baseline in one run:

```bash
uv run evaluation --benchmark gpqa --methods multiagent_streaming,multiagent_no_streaming,multiagent_dynamic_streaming,multiagent_dynamic_no_streaming,plain_llm --limit 10
```

The benchmark writes per-question CSV results with a unique run id in the filename, such as `output/gpqa_diamond_results_20260624T130000Z_ab12cd34.csv`, and prints accuracy, invalid-answer rate, total tokens, average tokens, total time, and average time for each method. The results CSV, summary JSON, and correctness matrix CSV are updated after each completed question-method run, so partial progress is inspectable while a benchmark is still running. `multiagent` remains as a legacy alias for fixed `multiagent_streaming`. Use `--output result.csv` to write `output/result.csv`, or `--output-dir other-output` to change the results directory. Use `--local-model --model <ollama-model>` to evaluate with Ollama instead of the default OpenAI-compatible API provider.

The correctness matrix is written inside the run trace folder as `output/json_traces/<run_id>/correctness.csv`. Rows are task ids, columns are method names, and each cell is `T`, `F`, or blank if that method has not finished that task yet.

JSON traces are saved by default under `output/json_traces/<run_id>/`. Each per-example method trace stores the resolved model name, settings, question, answer choices, prompt, raw output, token usage, selected subagents, messages sent by agents, a compact workflow, and the full method trace. It does not duplicate the entire benchmark row. Use `--no-save-json-traces` to disable JSON artifacts.

To compare where methods disagree after a run:

```bash
uv run python scripts/compare_eval_traces.py output/json_traces/<run_id>
```

GPQA on Hugging Face is gated. If the Hub-backed load fails because the dataset requires authentication, the evaluator automatically falls back to `data/gpqa_diamond.csv` when that file exists. Authenticate with an account that has dataset access, set `HF_TOKEN`, or pass your own local GPQA-style file with:

```bash
uv run evaluation --benchmark gpqa --methods multiagent_streaming,multiagent_no_streaming,plain_llm --limit 10 --data-file /path/to/gpqa.csv
```

Local `.csv`, `.jsonl`, and `.ndjson` files must include `Question`, `Correct Answer`, `Incorrect Answer 1`, `Incorrect Answer 2`, and `Incorrect Answer 3`.

GSM8K uses the public Hugging Face dataset `openai/gsm8k`, config `main`, split `test`. Rows contain `question` and `answer`; the gold answer is the final numeric value after the `####` marker in `answer`. The evaluator normalizes equivalent numeric formatting, so values like `10`, `10.0`, and `$10.00` score the same. To use a local GSM8K-shaped file:

```bash
uv run evaluation --benchmark gsm8k --methods multiagent_streaming,multiagent_no_streaming,plain_llm --limit 10 --data-file /path/to/gsm8k.jsonl
```

Local GSM8K `.csv`, `.jsonl`, and `.ndjson` files must include `question` and `answer`.

MMLU-Pro uses the public Hugging Face dataset `TIGER-Lab/MMLU-Pro`, split `test` only. Rows contain `question`, `options`, `answer`, and optionally `answer_index`, `category`, `cot_content`, `question_id`, and `src`. The evaluator keeps the dataset option order, labels the available options from `A` through at most `J`, and scores against the `answer` letter. To use a local MMLU-Pro-shaped file:

```bash
uv run evaluation --benchmark mmlu_pro --methods multiagent_streaming,multiagent_no_streaming,plain_llm --limit 10 --data-file /path/to/mmlu_pro.jsonl
```

Local MMLU-Pro `.csv`, `.jsonl`, and `.ndjson` files must include `question`, `options`, and `answer`. For CSV files, `options` must be a JSON list string.

MA-ProofBench uses the public Hugging Face dataset `openbmb/MA-ProofBench`, split `test`. Rows contain `id`, `split`, `informal_statement`, `formal_statement`, `header`, `topic`, `tag`, and `version`. The default level is `all`; use `--ma-proofbench-level level1` or `--ma-proofbench-level level2` to run one tier. The evaluator preserves dataset order and defaults to `--attempts 1`.

MA-ProofBench scoring is verifier-based. The evaluator extracts the final Lean code block, merges the dataset `header`, rejects outputs containing `sorry`, checks that the target theorem statement was not changed, and then verifies the proof. A proof is correct only when verification reports a complete proof with no errors and no sorries.

By default, MA-ProofBench uses Kimina Lean Server, matching the upstream benchmark workflow. Start the server first:

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

Then run the benchmark from this repository:

```bash
uv run evaluation --benchmark ma_proofbench --methods plain_llm --limit 10 --kimina-host 127.0.0.1 --kimina-port 8001
```

If `kimina_client` is not importable, install the Kimina Lean Server client package or set `KIMINA_CLIENT_PATH` to its client directory before running evaluation.

Local MA-ProofBench `.csv`, `.jsonl`, and `.ndjson` files must include `id`, `split`, `informal_statement`, `formal_statement`, `header`, `topic`, `tag`, and `version`.

OlymMATH uses the public Hugging Face dataset `RUC-AIBOX/OlymMATH`. The natural-language benchmark supports `--olymmath-subset en-easy`, `en-hard`, `zh-easy`, and `zh-hard`, corresponding to the upstream JSONL files `OlymMATH-EN-EASY.jsonl`, `OlymMATH-EN-HARD.jsonl`, `OlymMATH-ZH-EASY.jsonl`, and `OlymMATH-ZH-HARD.jsonl`. Rows contain `problem`, `answer`, `subject`, and `unique_id`. The evaluator prompts for a final answer in `Final Answer: <answer>` format, normalizes common LaTeX answer forms, and scores against the released answer key.

The OlymMATH paper reports rule-based answer evaluation for EASY/HARD and formal verification for LEAN. It does not provide one universal natural-language solver prompt; for Lean, the appendix prompt is for generating formalizations during benchmark construction, while model evaluation uses theorem-proving model prompt templates. This repository therefore uses local prompts that match the benchmark contracts: final answer extraction for EASY/HARD, and complete Lean code generation for LEAN.

OlymMATH-LEAN loads the upstream `OlymMATH-LEAN.jsonl` subset. Rows contain `unique_id`, `subject`, `formal_statement`, `formal_statement_raw`, `formal_proof`, `en_informal`, `zh_informal`, and natural-language proof fields. Scoring reuses the Lean verifier workflow: extract a Lean code block, reject `sorry`, ensure the theorem statement is unchanged, then verify with Kimina Lean Server. To use local OlymMATH files:

```bash
uv run evaluation --benchmark olymmath --olymmath-subset en-hard --methods plain_llm --limit 10 --data-file /path/to/OlymMATH-EN-HARD.jsonl
uv run evaluation --benchmark olymmath_lean --methods plain_llm --limit 10 --data-file /path/to/OlymMATH-LEAN.jsonl
```

## Current Limitations

- The streamer is in-memory and process-local.
- There is no real Kafka, Redis Streams, or NATS integration yet.
- There is no web UI.
- The orchestrator depends on the configured LLM and falls back deterministically when the model plan is invalid.
- Terminal streaming is best-effort and depends on async callback scheduling.
- The final CLI demo requires the selected model provider to be running.
