from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

from multi_agent_sync.events.event import AgentEvent

EventHandler = Callable[[AgentEvent], Awaitable[None]]


class EventStreamer(ABC):
    @abstractmethod
    async def publish(self, event: AgentEvent) -> None:
        raise NotImplementedError

    @abstractmethod
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        raise NotImplementedError

    @abstractmethod
    def subscribe_all(self, handler: EventHandler) -> None:
        raise NotImplementedError

    @abstractmethod
    async def get_events(
        self,
        run_id: str | None = None,
        event_type: str | None = None,
        source: str | None = None,
    ) -> list[AgentEvent]:
        raise NotImplementedError

    @abstractmethod
    async def get_recent_events(
        self,
        run_id: str,
        limit: int = 20,
    ) -> list[AgentEvent]:
        raise NotImplementedError

    @abstractmethod
    async def drain(self, timeout: float | None = None) -> None:
        raise NotImplementedError
