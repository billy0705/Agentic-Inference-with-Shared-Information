from __future__ import annotations

import asyncio

from multi_agent_sync.agents.coding_agent import CodingAgent
from multi_agent_sync.agents.critic_agent import CriticAgent
from multi_agent_sync.agents.research_agent import ResearchAgent
from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.events.in_memory_streamer import InMemoryEventStreamer
from multi_agent_sync.graph.state import GraphState
from multi_agent_sync.llm import get_llm
from multi_agent_sync.orchestrator.orchestrator import create_assignments, create_plan


async def orchestrator_node(state: GraphState) -> GraphState:
    streamer = state.get("event_streamer") or InMemoryEventStreamer()
    task = state["task"]
    plan = create_plan(task)
    assignments = create_assignments(task)

    await streamer.publish(
        AgentEvent(
            run_id=state["run_id"],
            source="coordinator",
            event_type="task_started",
            content=task,
        )
    )
    await streamer.publish(
        AgentEvent(
            run_id=state["run_id"],
            source="orchestrator",
            event_type="plan_created",
            content=f"Created {len(assignments)} subtasks",
            metadata={"assignments": assignments},
        )
    )
    await streamer.drain(timeout=1)

    return {
        **state,
        "event_streamer": streamer,
        "plan": plan,
        "assignments": assignments,
    }


async def run_multi_agent_runtime_node(state: GraphState) -> GraphState:
    streamer = state.get("event_streamer") or InMemoryEventStreamer()
    llm = state.get("llm") or get_llm()
    max_steps = state.get("max_steps_per_agent", 3)

    agents = [
        ResearchAgent(
            run_id=state["run_id"],
            task=state["task"],
            assigned_subtask=state["assignments"]["ResearchAgent"],
            llm=llm,
            event_streamer=streamer,
            max_steps=max_steps,
        ),
        CodingAgent(
            run_id=state["run_id"],
            task=state["task"],
            assigned_subtask=state["assignments"]["CodingAgent"],
            llm=llm,
            event_streamer=streamer,
            max_steps=max_steps,
        ),
        CriticAgent(
            run_id=state["run_id"],
            task=state["task"],
            assigned_subtask=state["assignments"]["CriticAgent"],
            llm=llm,
            event_streamer=streamer,
            max_steps=max_steps,
        ),
    ]

    for agent in agents:
        agent.subscribe()

    try:
        outputs = await asyncio.wait_for(
            asyncio.gather(*(agent.run() for agent in agents)),
            timeout=state.get("total_runtime_timeout", 90.0),
        )
    except asyncio.TimeoutError:
        for agent in agents:
            agent.is_done = True
        await streamer.publish(
            AgentEvent(
                run_id=state["run_id"],
                source="orchestrator",
                event_type="warning",
                content="Total runtime timeout reached before all agents completed.",
            )
        )
        outputs = [agent.local_output for agent in agents]

    await streamer.drain(timeout=2)
    event_log = await streamer.get_events(run_id=state["run_id"])

    return {
        **state,
        "event_streamer": streamer,
        "event_log": event_log,
        "agent_outputs": {agent.name: output for agent, output in zip(agents, outputs, strict=False)},
    }


async def synthesizer_node(state: GraphState) -> GraphState:
    streamer = state.get("event_streamer") or InMemoryEventStreamer()
    llm = state.get("llm") or get_llm()
    event_lines = "\n".join(
        f"- [{event.event_type}] {event.source}: {event.content}"
        for event in state.get("event_log", [])
        if event.event_type in {"finding", "warning", "critique", "agent_done"}
    )
    output_lines = "\n".join(f"- {name}: {output}" for name, output in state.get("agent_outputs", {}).items())
    prompt = f"""
You are the Synthesizer for a LangGraph multi-agent prototype.

Create the final answer for this user task:
{state["task"]}

Agent outputs:
{output_lines or "- No agent outputs collected."}

Runtime event log:
{event_lines or "- No runtime events collected."}

Write a concise final answer that combines the useful findings, implementation direction, and critique.
""".strip()
    try:
        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=state.get("synthesis_timeout", 60.0))
        final_answer = getattr(response, "content", str(response)).strip()
    except asyncio.TimeoutError:
        final_answer = build_fallback_summary(state)
        await streamer.publish(
            AgentEvent(
                run_id=state["run_id"],
                source="Synthesizer",
                event_type="warning",
                content="Synthesis timed out; using deterministic fallback summary.",
            )
        )

    await streamer.publish(
        AgentEvent(
            run_id=state["run_id"],
            source="Synthesizer",
            event_type="final_summary",
            content=final_answer,
        )
    )
    await streamer.drain(timeout=2)
    event_log = await streamer.get_events(run_id=state["run_id"])

    return {
        **state,
        "event_streamer": streamer,
        "event_log": event_log,
        "final_answer": final_answer,
    }


def build_fallback_summary(state: GraphState) -> str:
    outputs = state.get("agent_outputs", {})
    parts = [f"Synthesis timed out; fallback summary for task: {state['task']}."]
    for name, output in outputs.items():
        if output:
            parts.append(f"{name}: {output}")
    if len(parts) == 1:
        parts.append("No agent outputs were available before synthesis timed out.")
    return "\n".join(parts)
