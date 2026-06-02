from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict

from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.events.streamer import EventHandler, EventStreamer


class InMemoryEventStreamer(EventStreamer):
    """Async append-only event streamer with Kafka-like publish/subscribe semantics."""

    def __init__(self, handler_timeout: float = 2.0) -> None:
        self._event_log: list[AgentEvent] = []
        self._subscribers: dict[str, list[EventHandler]] = defaultdict(list)
        self._all_subscribers: list[EventHandler] = []
        self._pending: set[asyncio.Task[None]] = set()
        self._lock = asyncio.Lock()
        self._handler_timeout = handler_timeout

    async def publish(self, event: AgentEvent) -> None:
        async with self._lock:
            self._event_log.append(event)
            handlers = [
                *self._subscribers.get(event.event_type, []),
                *self._all_subscribers,
            ]

        for handler in handlers:
            task = asyncio.create_task(self._safe_handle(handler, event))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        if handler not in self._all_subscribers:
            self._all_subscribers.append(handler)

    async def get_events(
        self,
        run_id: str | None = None,
        event_type: str | None = None,
        source: str | None = None,
    ) -> list[AgentEvent]:
        async with self._lock:
            events = list(self._event_log)

        if run_id is not None:
            events = [event for event in events if event.run_id == run_id]
        if event_type is not None:
            events = [event for event in events if event.event_type == event_type]
        if source is not None:
            events = [event for event in events if event.source == source]
        return events

    async def get_recent_events(
        self,
        run_id: str,
        limit: int = 20,
    ) -> list[AgentEvent]:
        events = await self.get_events(run_id=run_id)
        return events[-limit:]

    async def drain(self, timeout: float | None = None) -> None:
        pending = [task for task in self._pending if not task.done()]
        if not pending:
            return

        done, still_pending = await asyncio.wait(pending, timeout=timeout)
        for task in done:
            with contextlib.suppress(Exception, asyncio.CancelledError):
                task.result()
        for task in still_pending:
            task.cancel()

    async def _safe_handle(self, handler: EventHandler, event: AgentEvent) -> None:
        try:
            await asyncio.wait_for(handler(event), timeout=self._handler_timeout)
        except (Exception, asyncio.TimeoutError):
            return
