from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

from multi_agent_sync.events.event import AgentEvent, EventType
from multi_agent_sync.events.streamer import EventStreamer


@dataclass
class StepResult:
    summary: str = ""
    share_finding: str = ""
    confidence: float = 0.7
    local_notes: str = ""


@dataclass(kw_only=True)
class BaseAgent:
    name: str
    role: str
    run_id: str
    task: str
    assigned_subtask: str
    llm: Any
    event_streamer: EventStreamer
    max_steps: int = 3
    max_runtime_seconds: float = 30.0
    max_events_per_agent: int = 50
    step_delay_seconds: float = 0.2
    seen_event_ids: set[str] = field(default_factory=set)
    inbox: list[AgentEvent] = field(default_factory=list)
    observed_events: list[AgentEvent] = field(default_factory=list)
    local_notes: list[str] = field(default_factory=list)
    local_output: str = ""
    is_done: bool = False
    _subscribed: bool = False

    def subscribe(self) -> None:
        if self._subscribed:
            return
        for event_type in ("finding", "warning", "critique", "question"):
            self.event_streamer.subscribe(event_type, self.handle_event)
        self._subscribed = True

    async def handle_event(self, event: AgentEvent) -> None:
        if self.is_done:
            return
        if event.run_id != self.run_id:
            return
        if event.source == self.name:
            return
        if event.target not in (None, "broadcast", self.name):
            return
        if event.event_id in self.seen_event_ids:
            return
        if len(self.observed_events) >= self.max_events_per_agent:
            return

        self.seen_event_ids.add(event.event_id)
        self.inbox.append(event)
        self.observed_events.append(event)

        if event.event_type in {"finding", "warning", "critique", "question"}:
            await self.publish_event(
                "message_received",
                f"received {event.event_type} from {event.source}",
                target=event.source,
                confidence=1.0,
                metadata={"received_event_id": event.event_id},
            )

    async def run(self) -> str:
        self.subscribe()
        started_at = time.monotonic()
        await self.publish_event("agent_started", self.assigned_subtask)

        for step_index in range(1, self.max_steps + 1):
            if time.monotonic() - started_at > self.max_runtime_seconds:
                await self.publish_warning("Agent runtime limit reached before all steps completed.")
                break

            result = await self.run_step(step_index)
            if result.summary:
                self.local_notes.append(result.summary)
            if result.local_notes:
                self.local_notes.append(result.local_notes)
            if result.share_finding:
                await self.publish_finding(result.share_finding, confidence=result.confidence, step_index=step_index)
            await self.event_streamer.drain(timeout=0.5)
            await asyncio.sleep(self.step_delay_seconds)

        self.local_output = "\n".join(self.local_notes).strip()
        self.is_done = True
        await self.publish_done(self.local_output)
        return self.local_output

    async def run_step(self, step_index: int) -> StepResult:
        prompt = await self.build_prompt(step_index)
        response = await self.llm.ainvoke(prompt)
        content = getattr(response, "content", str(response))
        return self.parse_step_response(content)

    async def build_prompt(self, step_index: int) -> str:
        relevant_events = await self.get_relevant_events(limit=8)
        event_lines = [
            f"- [{event.event_type}] {event.source}: {event.content}"
            for event in relevant_events
            if event.source != self.name
        ]
        notes = "\n".join(f"- {note}" for note in self.local_notes[-8:]) or "- None yet."
        events = "\n".join(event_lines) or "- No relevant external findings yet."
        return f"""
You are {self.name}.

Overall user task:
{self.task}

Agent role:
{self.role}

Assigned subtask:
{self.assigned_subtask}

Current step:
{step_index} of {self.max_steps}

Local notes so far:
{notes}

Recent relevant events from other agents:
{events}

Respond with concise summaries only. Do not reveal private chain-of-thought.
Use this exact format:

SUMMARY:
<brief reasoning summary>

SHARE_FINDING:
<one finding useful to other agents, or leave blank>

CONFIDENCE:
<number from 0.0 to 1.0>

LOCAL_NOTES:
<short private working notes summary for later steps>
""".strip()

    async def publish_event(
        self,
        event_type: EventType,
        content: str,
        target: str | None = "broadcast",
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentEvent:
        event = AgentEvent(
            run_id=self.run_id,
            source=self.name,
            target=target,
            event_type=event_type,
            content=content,
            confidence=confidence,
            metadata=metadata or {},
        )
        await self.event_streamer.publish(event)
        return event

    async def publish_finding(self, content: str, confidence: float = 0.7, step_index: int | None = None) -> AgentEvent:
        return await self.publish_event(
            "finding",
            content,
            confidence=confidence,
            metadata={"step_index": step_index} if step_index is not None else {},
        )

    async def publish_warning(self, content: str, confidence: float = 0.7) -> AgentEvent:
        return await self.publish_event("warning", content, confidence=confidence)

    async def publish_done(self, output: str) -> AgentEvent:
        return await self.publish_event(
            "agent_done",
            output or "Agent completed without local notes.",
            confidence=1.0,
            metadata={"observed_events": len(self.observed_events)},
        )

    async def get_relevant_events(self, limit: int = 10) -> list[AgentEvent]:
        recent_events = await self.event_streamer.get_recent_events(self.run_id, limit=limit * 2)
        merged: dict[str, AgentEvent] = {}
        for event in [*recent_events, *self.inbox]:
            if event.source == self.name:
                continue
            if event.event_type not in {"finding", "warning", "critique", "question"}:
                continue
            merged[event.event_id] = event
        return list(merged.values())[-limit:]

    def parse_step_response(self, content: str) -> StepResult:
        sections: dict[str, list[str]] = {
            "SUMMARY": [],
            "SHARE_FINDING": [],
            "CONFIDENCE": [],
            "LOCAL_NOTES": [],
        }
        current_key: str | None = None
        for raw_line in content.splitlines():
            line = raw_line.strip()
            upper = line.rstrip(":").upper()
            if upper in sections and line.endswith(":"):
                current_key = upper
                continue
            if current_key is not None:
                sections[current_key].append(raw_line)

        confidence = 0.7
        confidence_text = "\n".join(sections["CONFIDENCE"]).strip()
        with contextlib.suppress(ValueError):
            confidence = max(0.0, min(1.0, float(confidence_text)))

        return StepResult(
            summary="\n".join(sections["SUMMARY"]).strip() or content.strip(),
            share_finding="\n".join(sections["SHARE_FINDING"]).strip(),
            confidence=confidence,
            local_notes="\n".join(sections["LOCAL_NOTES"]).strip(),
        )
