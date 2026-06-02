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
        content = f": {event.content}" if event.content else ""
        self._console.print(f"[{elapsed:07.2f}] [{event.event_type}] {event.source}{content}", markup=False)
