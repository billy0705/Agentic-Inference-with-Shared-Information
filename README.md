# Multi-Agent Sync Prototype

This is a local Python prototype for runtime synchronization between concurrent agents. A LangGraph workflow owns the high-level lifecycle, while agents exchange internal findings through a Kafka-like event streamer while they are still running.

The prototype intentionally does not use AutoGen.

## Why LangGraph

LangGraph is used for the workflow lifecycle:

1. The orchestrator creates a plan and worker assignments.
2. The multi-agent runtime starts concurrent workers.
3. The synthesizer combines worker outputs and the event log into a final answer.

The graph shape is:

```text
START -> orchestrator -> run_multi_agent_runtime -> synthesizer -> END
```

This follows the orchestrator-worker style conceptually: the orchestrator assigns work, workers execute concurrently, and the final node synthesizes results.

## Why Ollama and qwen3:4b

The demo uses Ollama through `langchain-ollama` so inference stays local. The default model is `qwen3:4b`, which keeps the prototype lightweight enough for local experiments while still giving agents multiple reasoning steps.

Override the model with either:

```bash
OLLAMA_MODEL=llama3.2 uv run python -m multi_agent_sync "Build a prototype chess website"
```

or:

```bash
uv run python -m multi_agent_sync --model llama3.2 "Build a prototype chess website"
```

## Internal Finding Synchronization

Each agent has its own assigned subtask and runs several local LLM steps. After each step, it may publish a useful finding. Other agents subscribe to the event streamer and can include those findings in later prompts.

Example flow:

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

The prototype includes three workers:

- `ResearchAgent`: investigates architecture, assumptions, options, and constraints.
- `CodingAgent`: plans modules, APIs, dependencies, and implementation shape.
- `CriticAgent`: reviews findings for risks, race conditions, missing cases, and safety issues.

Each agent:

- subscribes to relevant events
- ignores its own events
- tracks `seen_event_ids`
- avoids duplicate event processing
- limits steps with `max_steps`
- limits runtime with `max_runtime_seconds`
- limits observed events with `max_events_per_agent`
- publishes `agent_started`, findings or critiques, and `agent_done`

## Console Streaming

The console streamer subscribes to all events and prints them while the run is active:

```text
[0000.10] [task_started] coordinator: Build a prototype chess website
[0001.20] [plan_created] orchestrator: Created 3 subtasks
[0001.30] [agent_started] ResearchAgent: Researching architecture choices
[0002.10] [finding] ResearchAgent: Real-time chess needs move synchronization
[0002.30] [message_received] CodingAgent: received finding from ResearchAgent
[0003.50] [critique] CriticAgent: Need server-side move validation
[0005.00] [final_summary] Synthesizer: ...
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
uv run python -m multi_agent_sync --model qwen3:4b --max-steps 3 --no-color "Build a prototype chess website"
```

Run tests:

```bash
uv run pytest
```

Tests use fake LLMs, so they do not require an Ollama server.

## Current Limitations

- The streamer is in-memory and process-local.
- There is no real Kafka, Redis Streams, or NATS integration yet.
- There is no web UI.
- The orchestrator uses a fixed three-agent assignment pattern.
- LLM output parsing is intentionally simple.
- Terminal streaming is best-effort and depends on async callback scheduling.
- The final CLI demo requires Ollama to be running and the selected model to be available locally.
