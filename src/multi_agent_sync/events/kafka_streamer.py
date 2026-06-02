from __future__ import annotations

from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.events.streamer import EventHandler, EventStreamer

TOPIC_MAP = {
    "finding": "agent.finding",
    "warning": "agent.warning",
    "critique": "agent.critique",
    "agent_done": "agent.done",
}


class KafkaEventStreamer(EventStreamer):
    """Future Kafka-backed implementation.

    Event types should map to topics such as:
    finding -> agent.finding
    warning -> agent.warning
    critique -> agent.critique
    agent_done -> agent.done
    """

    _message = "KafkaEventStreamer is a future extension. Use InMemoryEventStreamer for the prototype."

    async def publish(self, event: AgentEvent) -> None:
        raise NotImplementedError(self._message)

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        raise NotImplementedError(self._message)

    def subscribe_all(self, handler: EventHandler) -> None:
        raise NotImplementedError(self._message)

    async def get_events(
        self,
        run_id: str | None = None,
        event_type: str | None = None,
        source: str | None = None,
    ) -> list[AgentEvent]:
        raise NotImplementedError(self._message)

    async def get_recent_events(
        self,
        run_id: str,
        limit: int = 20,
    ) -> list[AgentEvent]:
        raise NotImplementedError(self._message)

    async def drain(self, timeout: float | None = None) -> None:
        raise NotImplementedError(self._message)
