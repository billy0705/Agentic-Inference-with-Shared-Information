import asyncio

import pytest

from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.events.in_memory_streamer import InMemoryEventStreamer


@pytest.mark.asyncio
async def test_publish_one_event_and_subscriber_receives_it():
    streamer = InMemoryEventStreamer()
    received: list[AgentEvent] = []

    async def handler(event: AgentEvent) -> None:
        received.append(event)

    streamer.subscribe("finding", handler)
    event = AgentEvent(run_id="run-1", source="ResearchAgent", event_type="finding", content="Use WebSocket sync")

    await streamer.publish(event)
    await streamer.drain(timeout=1)

    assert received == [event]


@pytest.mark.asyncio
async def test_multiple_subscribers_receive_same_event():
    streamer = InMemoryEventStreamer()
    first: list[str] = []
    second: list[str] = []

    async def first_handler(event: AgentEvent) -> None:
        first.append(event.event_id)

    async def second_handler(event: AgentEvent) -> None:
        second.append(event.event_id)

    streamer.subscribe("finding", first_handler)
    streamer.subscribe("finding", second_handler)
    event = AgentEvent(run_id="run-1", source="ResearchAgent", event_type="finding", content="Synchronize moves")

    await streamer.publish(event)
    await streamer.drain(timeout=1)

    assert first == [event.event_id]
    assert second == [event.event_id]


@pytest.mark.asyncio
async def test_subscribe_all_receives_every_event():
    streamer = InMemoryEventStreamer()
    received_types: list[str] = []

    async def all_handler(event: AgentEvent) -> None:
        received_types.append(event.event_type)

    streamer.subscribe_all(all_handler)

    await streamer.publish(AgentEvent(run_id="run-1", source="coordinator", event_type="task_started", content="Task"))
    await streamer.publish(AgentEvent(run_id="run-1", source="ResearchAgent", event_type="finding", content="Finding"))
    await streamer.drain(timeout=1)

    assert received_types == ["task_started", "finding"]


@pytest.mark.asyncio
async def test_event_log_filters_events():
    streamer = InMemoryEventStreamer()
    finding = AgentEvent(run_id="run-1", source="ResearchAgent", event_type="finding", content="Finding")
    warning = AgentEvent(run_id="run-1", source="CriticAgent", event_type="warning", content="Warning")
    other_run = AgentEvent(run_id="run-2", source="ResearchAgent", event_type="finding", content="Other run")

    await streamer.publish(finding)
    await streamer.publish(warning)
    await streamer.publish(other_run)

    assert await streamer.get_events(run_id="run-1") == [finding, warning]
    assert await streamer.get_events(event_type="finding") == [finding, other_run]
    assert await streamer.get_events(source="CriticAgent") == [warning]
    assert await streamer.get_recent_events("run-1", limit=1) == [warning]


@pytest.mark.asyncio
async def test_drain_does_not_hang_when_handler_is_slow_or_bad():
    streamer = InMemoryEventStreamer(handler_timeout=0.05)
    completed = asyncio.Event()

    async def slow_handler(event: AgentEvent) -> None:
        await asyncio.sleep(10)
        completed.set()

    async def bad_handler(event: AgentEvent) -> None:
        raise RuntimeError("subscriber failed")

    streamer.subscribe("finding", slow_handler)
    streamer.subscribe("finding", bad_handler)

    await streamer.publish(AgentEvent(run_id="run-1", source="ResearchAgent", event_type="finding", content="Finding"))
    await streamer.drain(timeout=0.2)

    assert not completed.is_set()
