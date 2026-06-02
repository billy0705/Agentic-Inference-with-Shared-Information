from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from multi_agent_sync.events.event import AgentEvent


@dataclass
class AgentStepTrace:
    agent_name: str
    step: int
    inbox_events: list[dict[str, Any]]
    prompt: str
    raw_response: str
    parsed_output: dict[str, Any]
    published_events: list[dict[str, Any]]
    started_at: float
    ended_at: float
    duration_seconds: float


@dataclass
class AgentConversationTrace:
    agent_name: str
    assignment: dict[str, Any]
    steps: list[AgentStepTrace] = field(default_factory=list)
    final_output: str | None = None


class TraceLogger:
    def __init__(self) -> None:
        self._traces: dict[str, AgentConversationTrace] = {}

    def start_agent(self, agent_name: str, assignment: dict[str, Any]) -> None:
        self._traces.setdefault(
            agent_name,
            AgentConversationTrace(agent_name=agent_name, assignment=dict(assignment)),
        )

    def log_step(
        self,
        agent_name: str,
        step: int,
        inbox_events: list[AgentEvent],
        prompt: str,
        raw_response: str,
        parsed_output: dict[str, Any],
        published_events: list[AgentEvent],
        started_at: float,
        ended_at: float,
    ) -> None:
        if agent_name not in self._traces:
            self.start_agent(agent_name, {"agent_name": agent_name})

        self._traces[agent_name].steps.append(
            AgentStepTrace(
                agent_name=agent_name,
                step=step,
                inbox_events=[event.model_dump(mode="json") for event in inbox_events],
                prompt=prompt,
                raw_response=raw_response,
                parsed_output=dict(parsed_output),
                published_events=[event.model_dump(mode="json") for event in published_events],
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=max(0.0, ended_at - started_at),
            )
        )

    def finish_agent(self, agent_name: str, final_output: str) -> None:
        if agent_name not in self._traces:
            self.start_agent(agent_name, {"agent_name": agent_name})
        self._traces[agent_name].final_output = final_output

    def export(self) -> dict[str, dict[str, Any]]:
        return {agent_name: asdict(trace) for agent_name, trace in self._traces.items()}
