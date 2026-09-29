import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from muffin.events.event import AgentEvent, EventType
from muffin.events.streamer import EventStreamer
from muffin.prompts import render_prompt
from muffin.tracing.token_usage import extract_token_usage


MAX_LOCAL_NOTES_CHARS = 500
DEFAULT_AGENT_RUNTIME_TIMEOUT_SECONDS = 600.0


def compact_local_notes(notes: str) -> str:
    normalized = notes.strip()
    if len(normalized) <= MAX_LOCAL_NOTES_CHARS:
        return normalized
    return f"{normalized[: MAX_LOCAL_NOTES_CHARS - 3].rstrip()}..."


@dataclass
class StepResult:
    summary: str = ""
    share_finding: str = ""
    local_notes: str = ""
    status: str = "continue"

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "share_finding": self.share_finding,
            "local_notes": self.local_notes,
            "status": self.status,
        }


@dataclass(kw_only=True)
class BaseAgent:
    name: str
    role: str
    run_id: str
    task: str
    assigned_subtask: str
    llm: Any
    event_streamer: EventStreamer
    assignment: dict[str, Any] | None = None
    description: str = ""
    rules: list[str] = field(default_factory=list)
    critical_debate: bool = False
    trace_logger: Any | None = None
    max_steps: int = 3
    max_runtime_seconds: float = DEFAULT_AGENT_RUNTIME_TIMEOUT_SECONDS
    max_events_per_agent: int = 50
    step_delay_seconds: float = 0.2
    allow_agent_early_stop: bool = False
    think_mode: bool = True
    reactive_steps_enabled: bool = True
    max_reactive_steps: int = 1
    reactive_event_types: set[str] = field(default_factory=lambda: {"finding", "critique", "warning"})
    seen_event_ids: set[str] = field(default_factory=set)
    used_event_ids: set[str] = field(default_factory=set)
    inbox: list[AgentEvent] = field(default_factory=list)
    observed_events: list[AgentEvent] = field(default_factory=list)
    local_notes: list[str] = field(default_factory=list)
    output_summaries: list[str] = field(default_factory=list)
    local_output: str = ""
    is_done: bool = False
    reactive_steps_used: int = 0
    _subscribed: bool = False
    _last_step_trace_data: dict[str, Any] = field(default_factory=dict)

    def subscribe(self) -> None:
        if self._subscribed:
            return
        for event_type in ("finding", "warning", "critique", "question"):
            self.event_streamer.subscribe(event_type, self.handle_event)
        self._subscribed = True

    async def handle_event(self, event: AgentEvent) -> None:
        received_at = time.time()
        accepted = False
        ignored_reason: str | None = None
        supported_event_types = {"finding", "warning", "critique", "question"}

        if self.is_done:
            ignored_reason = "agent_done"
        elif event.run_id != self.run_id:
            ignored_reason = "different_run_id"
        elif event.source == self.name:
            ignored_reason = "self_event"
        elif event.target not in (None, "broadcast", self.name):
            ignored_reason = "targeted_to_other_agent"
        elif event.event_id in self.seen_event_ids:
            ignored_reason = "duplicate_event"
        elif len(self.observed_events) >= self.max_events_per_agent:
            ignored_reason = "max_events_per_agent_reached"
        elif event.event_type not in supported_event_types:
            ignored_reason = "unsupported_event_type"
        else:
            accepted = True
            self.seen_event_ids.add(event.event_id)
            self.inbox.append(event)
            self.observed_events.append(event)

        if self.trace_logger is not None:
            self.trace_logger.log_event_received(
                agent_name=self.name,
                event=event,
                received_at=received_at,
                accepted=accepted,
                ignored_reason=ignored_reason,
                inbox_size_after=len(self.inbox),
                agent_status=self.status,
            )

        if accepted:
            await self.publish_event(
                "message_received",
                f"received {event.event_type} from {event.source}",
                target=event.source,
                confidence=1.0,
                metadata={"received_event_id": event.event_id},
            )

    @property
    def status(self) -> str:
        return "done" if self.is_done else "running"

    async def run(self) -> str:
        self.subscribe()
        if self.trace_logger is not None:
            self.trace_logger.start_agent(self.name, self.assignment or self._default_assignment())
        started_at = time.monotonic()

        last_step_index = 0
        final_response_received = False
        step_index = 1
        while step_index <= self.max_steps:
            if time.monotonic() - started_at > self.max_runtime_seconds:
                await self.publish_warning("Agent runtime limit reached before all steps completed.")
                break

            result = await self.run_step_and_record(step_index)
            last_step_index = step_index
            if result.status == "final" and (self.allow_agent_early_stop or step_index >= self.max_steps):
                final_response_received = True
                break
            step_index += 1

        if not final_response_received:
            await self.maybe_run_reactive_steps(started_at, last_step_index)

        self.local_output = "\n".join(self.output_summaries[-self.max_steps :]).strip() or "\n".join(self.local_notes).strip()
        self.is_done = True
        await self.publish_done(self.local_output)
        if self.trace_logger is not None:
            self.trace_logger.finish_agent(self.name, self.local_output)
        return self.local_output

    async def run_step_and_record(
        self,
        step_index: int,
        *,
        is_reactive: bool = False,
        reactive_reason: str | None = None,
        forced_events: list[AgentEvent] | None = None,
    ) -> StepResult:
        result = await self.run_step(
            step_index,
            is_reactive=is_reactive,
            reactive_reason=reactive_reason,
            forced_events=forced_events,
        )
        if result.summary:
            self.output_summaries.append(result.summary)
        if result.local_notes:
            self.local_notes.append(compact_local_notes(result.local_notes))
        published_events: list[AgentEvent] = []
        if result.share_finding:
            published_events.append(
                await self.publish_finding(result.share_finding, step_index=step_index)
            )
        self._log_step_trace(step_index, result, published_events, is_reactive=is_reactive, reactive_reason=reactive_reason)
        await self.event_streamer.drain(timeout=0.5)
        await asyncio.sleep(self.step_delay_seconds)
        return result

    async def run_step(
        self,
        step_index: int,
        *,
        is_reactive: bool = False,
        reactive_reason: str | None = None,
        forced_events: list[AgentEvent] | None = None,
    ) -> StepResult:
        started_at = time.time()
        relevant_events = forced_events if forced_events is not None else await self.get_relevant_events(limit=8)
        used_event_ids = [event.event_id for event in relevant_events]
        used_at = time.time()
        for event_id in used_event_ids:
            self.used_event_ids.add(event_id)
            if self.trace_logger is not None:
                self.trace_logger.mark_event_used(
                    agent_name=self.name,
                    event_id=event_id,
                    step=step_index,
                    used_at=used_at,
                )
        prompt = await self.build_prompt(
            step_index,
            relevant_events=relevant_events,
            is_reactive=is_reactive,
            reactive_reason=reactive_reason,
        )
        response = await self.llm.ainvoke(prompt)
        content = getattr(response, "content", str(response))
        token_usage = extract_token_usage(response)
        result = self.parse_step_response(content)
        ended_at = time.time()
        self._last_step_trace_data = {
            "inbox_events": relevant_events,
            "used_event_ids": used_event_ids,
            "prompt": prompt,
            "raw_response": content,
            "token_usage": token_usage.as_dict(),
            "started_at": started_at,
            "ended_at": ended_at,
        }
        return result

    async def maybe_run_reactive_steps(self, started_at: float, last_step_index: int) -> int:
        while self.reactive_steps_enabled and self.reactive_steps_used < self.max_reactive_steps:
            if time.monotonic() - started_at > self.max_runtime_seconds:
                break
            unused_events = self.get_unused_important_events()
            if not unused_events:
                break

            self.reactive_steps_used += 1
            last_step_index += 1
            await self.run_step_and_record(
                last_step_index,
                is_reactive=True,
                reactive_reason="important_unused_events_received",
                forced_events=unused_events,
            )
        return last_step_index

    def get_unused_important_events(self) -> list[AgentEvent]:
        return [
            event
            for event in self.observed_events
            if event.event_type in self.reactive_event_types and event.event_id not in self.used_event_ids
        ]

    async def build_prompt(
        self,
        step_index: int,
        relevant_events: list[AgentEvent] | None = None,
        *,
        is_reactive: bool = False,
        reactive_reason: str | None = None,
    ) -> str:
        if relevant_events is None:
            relevant_events = await self.get_relevant_events(limit=8)
        event_lines = [
            f"- [{event.event_type}] {event.source}: {event.content}"
            for event in relevant_events
            if event.source != self.name
        ]
        notes = "\n".join(f"- {note}" for note in self.local_notes[-8:]) or "- None yet."
        events = "\n".join(event_lines) or "- No shared findings yet."
        expected_output = str((self.assignment or {}).get("expected_output") or "").strip()
        return render_prompt(
            "agents/step.j2",
            agent_name=self.name,
            think_mode=self.think_mode,
            task=self.task,
            role=self.role,
            description=self.description,
            rules=self.rules,
            critical_debate=self.critical_debate,
            assigned_subtask=self.assigned_subtask,
            expected_output=expected_output,
            is_reactive=is_reactive,
            reactive_reason=reactive_reason or "important_unused_events_received",
            step_index=step_index,
            max_steps=self.max_steps,
            allow_agent_early_stop=self.allow_agent_early_stop,
            notes=notes,
            events=events,
        )

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
            "FINAL": [],
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

        final = "\n".join(sections["FINAL"]).strip()
        summary = "\n".join(sections["SUMMARY"]).strip() or final or content.strip()
        status = "final" if final else "continue"

        return StepResult(
            summary=summary,
            share_finding="\n".join(sections["SHARE_FINDING"]).strip(),
            local_notes="\n".join(sections["LOCAL_NOTES"]).strip(),
            status=status,
        )

    def _default_assignment(self) -> dict[str, Any]:
        return {
            "agent_name": self.name,
            "task": self.assigned_subtask,
            "max_steps": self.max_steps,
            "allow_agent_early_stop": self.allow_agent_early_stop,
            "think_mode": self.think_mode,
        }

    def _log_step_trace(
        self,
        step_index: int,
        result: StepResult,
        published_events: list[AgentEvent],
        *,
        is_reactive: bool = False,
        reactive_reason: str | None = None,
    ) -> None:
        if self.trace_logger is None:
            return
        self.trace_logger.log_step(
            agent_name=self.name,
            step=step_index,
            inbox_events=self._last_step_trace_data.get("inbox_events", []),
            used_event_ids=self._last_step_trace_data.get("used_event_ids", []),
            prompt=self._last_step_trace_data.get("prompt", ""),
            raw_response=self._last_step_trace_data.get("raw_response", ""),
            token_usage=self._last_step_trace_data.get("token_usage", {}),
            parsed_output=result.as_dict(),
            published_events=published_events,
            started_at=self._last_step_trace_data.get("started_at", time.time()),
            ended_at=self._last_step_trace_data.get("ended_at", time.time()),
            is_reactive=is_reactive,
            reactive_reason=reactive_reason,
        )
