from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.events.streamer import EventStreamer


class GraphState(TypedDict):
    task: str
    run_id: str
    mode: str
    subagent_mode: str
    task_type: str
    reason: str
    plan: list[str]
    selected_agents: list[dict[str, Any]]
    assignments: list[dict[str, Any]]
    orchestrator_plan: dict[str, Any]
    event_log: list[AgentEvent]
    agent_outputs: dict[str, str]
    agent_traces: dict[str, Any]
    synthesizer_mode: NotRequired[str]
    synthesizer_trace: NotRequired[dict[str, Any]]
    direct_trace: NotRequired[dict[str, Any]]
    final_answer: str
    event_streamer: NotRequired[EventStreamer]
    trace_logger: NotRequired[Any]
    llm: NotRequired[Any]
    max_steps_per_agent: NotRequired[int]
    min_dynamic_subagents: NotRequired[int]
    max_dynamic_subagents: NotRequired[int]
    allow_agent_early_stop: NotRequired[bool]
    think_mode: NotRequired[bool]
    full_trace_sharing: NotRequired[bool]
    agent_runtime_timeout: NotRequired[float]
    total_runtime_timeout: NotRequired[float]
    synthesis_timeout: NotRequired[float]
    enable_agent_message_streaming: NotRequired[bool]
    stream_to_console: NotRequired[bool]
    no_color: NotRequired[bool]
    enable_workspace_tools: NotRequired[bool]
    docker_workspace: NotRequired[Any]
    feedback_tool: NotRequired[Any]
    final_guard_tool: NotRequired[Any]
    benchmark: NotRequired[str]
