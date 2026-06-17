from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from multi_agent_sync.events.event import AgentEvent


@dataclass
class AgentEventTrace:
    agent_name: str
    event_id: str
    event_type: str
    source: str
    target: str | None
    content: str
    received_at: float
    accepted: bool
    ignored_reason: str | None
    inbox_size_after: int
    agent_status: str
    used_in_step: int | None = None
    used_at: float | None = None


@dataclass
class AgentStepTrace:
    agent_name: str
    step: int
    inbox_events: list[dict[str, Any]]
    used_event_ids: list[str]
    prompt: str
    raw_response: str
    parsed_output: dict[str, Any]
    published_events: list[dict[str, Any]]
    started_at: float
    ended_at: float
    duration_seconds: float
    is_reactive: bool = False
    reactive_reason: str | None = None


@dataclass
class AgentConversationTrace:
    agent_name: str
    assignment: dict[str, Any]
    event_receipts: list[AgentEventTrace] = field(default_factory=list)
    steps: list[AgentStepTrace] = field(default_factory=list)
    final_output: str | None = None


class TraceLogger:
    def __init__(self) -> None:
        self._traces: dict[str, AgentConversationTrace] = {}
        self._pending_used_events: dict[tuple[str, str], tuple[int, float]] = {}

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
        used_event_ids: list[str],
        prompt: str,
        raw_response: str,
        parsed_output: dict[str, Any],
        published_events: list[AgentEvent],
        started_at: float,
        ended_at: float,
        is_reactive: bool = False,
        reactive_reason: str | None = None,
    ) -> None:
        if agent_name not in self._traces:
            self.start_agent(agent_name, {"agent_name": agent_name})

        self._traces[agent_name].steps.append(
            AgentStepTrace(
                agent_name=agent_name,
                step=step,
                inbox_events=[event.model_dump(mode="json") for event in inbox_events],
                used_event_ids=list(used_event_ids),
                prompt=prompt,
                raw_response=raw_response,
                parsed_output=dict(parsed_output),
                published_events=[event.model_dump(mode="json") for event in published_events],
                started_at=started_at,
                ended_at=ended_at,
                duration_seconds=max(0.0, ended_at - started_at),
                is_reactive=is_reactive,
                reactive_reason=reactive_reason,
            )
        )

    def log_event_received(
        self,
        agent_name: str,
        event: AgentEvent,
        received_at: float,
        accepted: bool,
        ignored_reason: str | None,
        inbox_size_after: int,
        agent_status: str,
    ) -> None:
        if agent_name not in self._traces:
            self.start_agent(agent_name, {"agent_name": agent_name})

        receipt = AgentEventTrace(
            agent_name=agent_name,
            event_id=event.event_id,
            event_type=event.event_type,
            source=event.source,
            target=event.target,
            content=event.content,
            received_at=received_at,
            accepted=accepted,
            ignored_reason=ignored_reason,
            inbox_size_after=inbox_size_after,
            agent_status=agent_status,
        )
        pending_used = self._pending_used_events.pop((agent_name, event.event_id), None)
        if accepted and pending_used is not None:
            receipt.used_in_step, receipt.used_at = pending_used
        self._traces[agent_name].event_receipts.append(receipt)

    def mark_event_used(self, agent_name: str, event_id: str, step: int, used_at: float) -> None:
        if agent_name not in self._traces:
            return
        for receipt in self._traces[agent_name].event_receipts:
            if receipt.event_id == event_id and receipt.accepted:
                receipt.used_in_step = step
                receipt.used_at = used_at
                return
        self._pending_used_events[(agent_name, event_id)] = (step, used_at)

    def finish_agent(self, agent_name: str, final_output: str) -> None:
        if agent_name not in self._traces:
            self.start_agent(agent_name, {"agent_name": agent_name})
        self._traces[agent_name].final_output = final_output

    def export(self) -> dict[str, dict[str, Any]]:
        return {agent_name: self._export_trace(trace) for agent_name, trace in self._traces.items()}

    def _export_trace(self, trace: AgentConversationTrace) -> dict[str, Any]:
        data = asdict(trace)
        data["unused_received_events"] = [
            {
                "event_id": receipt.event_id,
                "event_type": receipt.event_type,
                "source": receipt.source,
                "received_at": receipt.received_at,
                "reason": self._unused_reason(trace, receipt),
            }
            for receipt in trace.event_receipts
            if receipt.accepted and receipt.used_in_step is None
        ]
        return data

    def _unused_reason(self, trace: AgentConversationTrace, receipt: AgentEventTrace) -> str:
        later_steps = [step for step in trace.steps if step.started_at > receipt.received_at]
        if not later_steps:
            return "received_after_last_step_started_no_later_step"
        return "received_but_filtered_from_prompt"
