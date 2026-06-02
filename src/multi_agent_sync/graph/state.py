from __future__ import annotations

from typing import Any, NotRequired, TypedDict

from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.events.streamer import EventStreamer


class GraphState(TypedDict):
    task: str
    run_id: str
    plan: list[str]
    assignments: dict[str, str]
    event_log: list[AgentEvent]
    agent_outputs: dict[str, str]
    final_answer: str
    event_streamer: NotRequired[EventStreamer]
    llm: NotRequired[Any]
    max_steps_per_agent: NotRequired[int]
    total_runtime_timeout: NotRequired[float]
    synthesis_timeout: NotRequired[float]
    stream_to_console: NotRequired[bool]
    no_color: NotRequired[bool]
