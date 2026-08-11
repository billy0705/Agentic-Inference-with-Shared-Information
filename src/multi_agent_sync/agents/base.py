from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from multi_agent_sync.events.event import AgentEvent, EventType
from multi_agent_sync.events.streamer import EventStreamer
from multi_agent_sync.prompts import render_prompt
from multi_agent_sync.token_usage import extract_token_usage


MAX_LOCAL_NOTES_CHARS = 500


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
    tool_action: dict[str, Any] | None = None
    tool_result: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "share_finding": self.share_finding,
            "local_notes": self.local_notes,
            "status": self.status,
            "tool_action": self.tool_action,
            "tool_result": self.tool_result,
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
    max_runtime_seconds: float = 180.0
    max_events_per_agent: int = 50
    step_delay_seconds: float = 0.2
    enable_message_streaming: bool = True
    bash_tool: Any | None = None
    feedback_tool: Any | None = None
    final_guard_tool: Any | None = None
    workspace_access: str = "none"
    allow_agent_early_stop: bool = False
    tool_observations: list[dict[str, Any]] = field(default_factory=list)
    final_guard_retry_steps: int = 2
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
        if not self.enable_message_streaming:
            return
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
        final_guard_extra_steps = 0
        while step_index <= self.max_steps + final_guard_extra_steps:
            if time.monotonic() - started_at > self.max_runtime_seconds:
                await self.publish_warning("Agent runtime limit reached before all steps completed.")
                break

            result = await self.run_step_and_record(step_index)
            last_step_index = step_index
            if result.status == "final" and (self.allow_agent_early_stop or step_index >= self.max_steps):
                final_response_received = True
                break
            if self.is_final_guard_retry(result) and final_guard_extra_steps == 0:
                final_guard_extra_steps = self.final_guard_retry_steps
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
        result = (
            await self.parse_and_run_tool_response(content)
            if self.has_workspace_tool
            else self.parse_step_response(content)
        )
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

    @property
    def has_workspace_tool(self) -> bool:
        return self.bash_tool is not None and self.workspace_access != "none"

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
        if self.has_workspace_tool:
            return render_prompt(
                "agents/tool_step.j2",
                agent_name=self.name,
                task=self.task,
                role=self.role,
                description=self.description,
                rules=self.rules,
                critical_debate=self.critical_debate,
                assigned_subtask=self.assigned_subtask,
                expected_output=expected_output,
                workspace_access=self.workspace_access,
                is_reactive=is_reactive,
                reactive_reason=reactive_reason or "important_unused_events_received",
                step_index=step_index,
                max_steps=self.max_steps,
                allow_agent_early_stop=self.allow_agent_early_stop,
                notes=notes,
                events=events,
                tool_observations=self.format_tool_observations(),
                feedback_tool_name=getattr(self.feedback_tool, "name", "") if self.feedback_tool is not None else "",
            )
        return render_prompt(
            "agents/step.j2",
            agent_name=self.name,
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

    def format_tool_observations(self) -> str:
        if not self.tool_observations:
            return "- No bash observations yet."
        lines: list[str] = []
        for observation in self.tool_observations[-6:]:
            if observation.get("command") is not None:
                lines.append(f"- command: {observation.get('command', '')}")
                lines.append(f"  exit_code: {observation.get('exit_code')}")
            else:
                lines.append(f"- tool: {observation.get('tool', '')}")
                if observation.get("path"):
                    lines.append(f"  path: {observation.get('path')}")
                if observation.get("passed") is not None:
                    lines.append(f"  passed: {observation.get('passed')}")
                if observation.get("backend"):
                    lines.append(f"  backend: {observation.get('backend')}")
                if observation.get("returncode") is not None:
                    lines.append(f"  returncode: {observation.get('returncode')}")
                if observation.get("error"):
                    lines.append(f"  error: {observation.get('error')}")
                if observation.get("message"):
                    lines.append(f"  message: {observation.get('message')}")
            stdout = str(observation.get("stdout") or "").strip()
            stderr = str(observation.get("stderr") or "").strip()
            verifier_output = str(observation.get("verifier_output") or "").strip()
            if stdout:
                lines.append(f"  stdout: {stdout}")
            if stderr:
                lines.append(f"  stderr: {stderr}")
            if verifier_output:
                lines.append(f"  verifier_output: {verifier_output}")
            if observation.get("timed_out"):
                lines.append("  timed_out: true")
        return "\n".join(lines)

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
            "ACTION": [],
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
        action = self.parse_tool_action("\n".join(sections["ACTION"]).strip())
        summary = "\n".join(sections["SUMMARY"]).strip() or final or content.strip()
        status = "final" if final else "continue"

        return StepResult(
            summary=summary,
            share_finding="\n".join(sections["SHARE_FINDING"]).strip(),
            local_notes="\n".join(sections["LOCAL_NOTES"]).strip(),
            status=status,
            tool_action=action,
        )

    async def parse_and_run_tool_response(self, content: str) -> StepResult:
        parsed = self.parse_step_response(content)
        if parsed.status == "final":
            return await self.check_final_response(parsed)
        if not parsed.tool_action:
            return parsed

        if parsed.tool_action.get("tool") == "verify_candidate":
            if self.feedback_tool is None:
                return StepResult(
                    summary="Lean verifier feedback tool was requested but is not configured.",
                    share_finding=parsed.share_finding,
                    local_notes=parsed.local_notes,
                    status="continue",
                    tool_action=parsed.tool_action,
                    tool_result={"error": "missing_feedback_tool"},
                )
            result_payload = await self.feedback_tool.run(path=parsed.tool_action.get("path"))
            self.tool_observations.append(result_payload)
            return StepResult(
                summary=parsed.summary or "Ran Lean verifier feedback tool.",
                share_finding=parsed.share_finding,
                local_notes=parsed.local_notes,
                status="continue",
                tool_action=parsed.tool_action,
                tool_result=result_payload,
            )

        if parsed.tool_action.get("tool") != "bash":
            return StepResult(
                summary=f"Unsupported tool requested: {parsed.tool_action.get('tool')}",
                share_finding=parsed.share_finding,
                local_notes=parsed.local_notes,
                status="continue",
                tool_action=parsed.tool_action,
                tool_result={"error": "unsupported_tool"},
            )

        command = str(parsed.tool_action.get("command") or "").strip()
        if not command:
            return StepResult(
                summary="Bash tool action was missing a command.",
                share_finding=parsed.share_finding,
                local_notes=parsed.local_notes,
                status="continue",
                tool_action=parsed.tool_action,
                tool_result={"error": "missing_command"},
            )

        result = await self.bash_tool.run(command)
        result_payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        self.tool_observations.append(result_payload)
        return StepResult(
            summary=parsed.summary or f"Ran bash command in Docker: {command}",
            share_finding=parsed.share_finding,
            local_notes=parsed.local_notes,
            status="continue",
            tool_action=parsed.tool_action,
            tool_result=result_payload,
        )

    async def check_final_response(self, parsed: StepResult) -> StepResult:
        if self.final_guard_tool is None or not self.has_workspace_tool:
            return parsed

        result_payload = await self.final_guard_tool.run(final_response=parsed.summary)
        if result_payload.get("passed"):
            return parsed

        self.tool_observations.append(result_payload)
        guard_name = result_payload.get("tool") or getattr(self.final_guard_tool, "name", "final_guard")
        message = str(result_payload.get("message") or "Final response rejected by workspace guard.")
        return StepResult(
            summary=f"{guard_name} rejected FINAL: {message}",
            share_finding=parsed.share_finding,
            local_notes=parsed.local_notes,
            status="continue",
            tool_action={"tool": guard_name, "reason": "final_response_guard"},
            tool_result=result_payload,
        )

    def is_final_guard_retry(self, result: StepResult) -> bool:
        return bool(result.tool_result and result.tool_result.get("error") == "final_guard_failed")

    def parse_tool_action(self, text: str) -> dict[str, Any] | None:
        if not text:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return {"tool": "parse_error", "raw": text}
        return value if isinstance(value, dict) else {"tool": "parse_error", "raw": text}

    def _default_assignment(self) -> dict[str, Any]:
        return {
            "agent_name": self.name,
            "task": self.assigned_subtask,
            "max_steps": self.max_steps,
            "allow_agent_early_stop": self.allow_agent_early_stop,
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
