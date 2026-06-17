from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.events.streamer import EventStreamer


class GraphState(TypedDict):
    task: str
    run_id: str
    mode: str
    task_type: str
    reason: str
    plan: list[str]
    selected_agents: list[dict[str, Any]]
    assignments: list[dict[str, Any]]
    orchestrator_plan: dict[str, Any]
    event_log: list[AgentEvent]
    agent_outputs: dict[str, str]
    agent_traces: dict[str, Any]
    final_answer: str
    event_streamer: NotRequired[EventStreamer]
    trace_logger: NotRequired[Any]
    llm: NotRequired[Any]
    max_steps_per_agent: NotRequired[int]
    total_runtime_timeout: NotRequired[float]
    synthesis_timeout: NotRequired[float]
    stream_to_console: NotRequired[bool]
    no_color: NotRequired[bool]
