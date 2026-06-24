from __future__ import annotations

import time

from rich.console import Console

from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.events.streamer import EventStreamer


class ConsoleEventStreamer:
    def __init__(self, streamer: EventStreamer, no_color: bool = False) -> None:
        self._console = Console(no_color=no_color)
        self._started_at = time.monotonic()
        streamer.subscribe_all(self.handle_event)

    async def handle_event(self, event: AgentEvent) -> None:
        elapsed = time.monotonic() - self._started_at
        self._console.print(format_console_event(event, elapsed), markup=False)


def format_console_event(event: AgentEvent, elapsed: float) -> str:
    target = event.target or "broadcast"
    content = f": {event.content}" if event.content else ""
    rendered = f"[{elapsed:07.2f}] [{event.event_type}] {event.source} -> {target}{content}"
    dynamic_details = format_dynamic_plan_details(event)
    if dynamic_details:
        rendered = f"{rendered}\n{dynamic_details}"
    return rendered


def format_dynamic_plan_details(event: AgentEvent) -> str:
    if event.event_type != "plan_created":
        return ""
    metadata = event.metadata or {}
    if metadata.get("subagent_mode") != "dynamic":
        return ""

    selected_agents = metadata.get("selected_agents")
    if not isinstance(selected_agents, list) or not selected_agents:
        return ""

    lines = ["Dynamic subagents:"]
    for agent in selected_agents:
        if not isinstance(agent, dict):
            continue
        rules = agent.get("rules") if isinstance(agent.get("rules"), list) else []
        rules_text = "; ".join(str(rule) for rule in rules if str(rule).strip()) or "none"
        lines.extend(
            [
                f"- name: {agent.get('name', '')}",
                f"  role: {agent.get('role', '')}",
                f"  description: {agent.get('description', '')}",
                f"  rules: {rules_text}",
                f"  subtask: {agent.get('subtask', '')}",
                f"  expected_output: {agent.get('expected_output', '')}",
                f"  critical_debate: {agent.get('critical_debate', False)}",
            ]
        )
    return "\n".join(lines)
