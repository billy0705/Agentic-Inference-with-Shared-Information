from __future__ import annotations

import asyncio

from multi_agent_sync.agents.registry import AGENT_REGISTRY
from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.events.in_memory_streamer import InMemoryEventStreamer
from multi_agent_sync.graph.state import GraphState
from multi_agent_sync.llm import get_llm
from multi_agent_sync.orchestrator.orchestrator import create_model_based_plan, selected_agents_to_assignments
from multi_agent_sync.tracing.trace import TraceLogger


async def orchestrator_node(state: GraphState) -> GraphState:
    streamer = state.get("event_streamer") or InMemoryEventStreamer()
    task = state["task"]
    llm = state.get("llm") or get_llm()
    orchestrator_plan = await create_model_based_plan(task, llm, AGENT_REGISTRY)
    selected_agent_specs = orchestrator_plan["selected_agents"]
    assignments = selected_agents_to_assignments(orchestrator_plan, max_steps=state.get("max_steps_per_agent", 3))
    selected_agents = ",".join(agent["name"] for agent in selected_agent_specs) if selected_agent_specs else "none"

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
            content=(
                f"mode={orchestrator_plan['mode']}, task_type={orchestrator_plan['task_type']}, "
                f"selected_agents={selected_agents}"
            ),
            metadata=orchestrator_plan,
        )
    )
    await streamer.drain(timeout=1)

    return {
        **state,
        "event_streamer": streamer,
        "mode": orchestrator_plan["mode"],
        "task_type": orchestrator_plan["task_type"],
        "reason": orchestrator_plan["reason"],
        "plan": [agent["subtask"] for agent in selected_agent_specs],
        "selected_agents": selected_agent_specs,
        "assignments": assignments,
        "orchestrator_plan": orchestrator_plan,
    }


def route_after_orchestrator(state: GraphState) -> str:
    if state.get("mode") == "direct" or not state.get("selected_agents"):
        return "direct_answer"
    return "run_multi_agent_runtime"


async def direct_answer_node(state: GraphState) -> GraphState:
    streamer = state.get("event_streamer") or InMemoryEventStreamer()
    llm = state.get("llm") or get_llm()
    prompt = f"""
Answer the user task directly with one concise response.

User task:
{state["task"]}

If the task asks you to choose from provided options but the options are missing,
explicitly say that the options are missing and that you cannot choose one of them.
""".strip()
    try:
        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=state.get("synthesis_timeout", 60.0))
        final_answer = getattr(response, "content", str(response)).strip()
    except asyncio.TimeoutError:
        final_answer = "Direct answer timed out before the LLM returned a response."

    final_answer = apply_missing_options_guard(state["task"], final_answer)
    await streamer.publish(
        AgentEvent(
            run_id=state["run_id"],
            source="Synthesizer",
            event_type="final_summary",
            content="Final answer generated.",
        )
    )
    await streamer.drain(timeout=2)

    return {
        **state,
        "event_streamer": streamer,
        "event_log": await streamer.get_events(run_id=state["run_id"]),
        "agent_outputs": {},
        "agent_traces": {},
        "final_answer": final_answer,
    }


async def run_multi_agent_runtime_node(state: GraphState) -> GraphState:
    streamer = state.get("event_streamer") or InMemoryEventStreamer()
    llm = state.get("llm") or get_llm()
    max_steps = state.get("max_steps_per_agent", 3)
    enable_agent_message_streaming = state.get("enable_agent_message_streaming", True)
    trace_logger = state.get("trace_logger") or TraceLogger()

    agents = []
    assignments = selected_agents_to_assignments(state["orchestrator_plan"], max_steps=max_steps)
    for assignment in assignments:
        agent_name = assignment["agent_name"]
        agent_class = AGENT_REGISTRY.get(agent_name)
        if agent_class is None:
            await streamer.publish(
                AgentEvent(
                    run_id=state["run_id"],
                    source="orchestrator",
                    event_type="warning",
                    content=f"Unknown agent assignment skipped: {agent_name}",
                )
            )
            continue
        agent_kwargs = {
            "run_id": state["run_id"],
            "task": state["task"],
            "assigned_subtask": assignment["task"],
            "llm": llm,
            "event_streamer": streamer,
            "assignment": assignment,
            "trace_logger": trace_logger,
            "max_steps": assignment.get("max_steps", max_steps),
            "enable_message_streaming": enable_agent_message_streaming,
        }
        for reactive_field in ("reactive_steps_enabled", "max_reactive_steps", "reactive_event_types"):
            if reactive_field in assignment:
                agent_kwargs[reactive_field] = assignment[reactive_field]
        agents.append(agent_class(**agent_kwargs))

    for agent in agents:
        agent.subscribe()

    if agents:
        try:
            outputs = await asyncio.wait_for(
                asyncio.gather(*(agent.run() for agent in agents)),
                timeout=state.get("total_runtime_timeout", 600.0),
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
    else:
        outputs = []

    await streamer.drain(timeout=2)
    event_log = await streamer.get_events(run_id=state["run_id"])

    return {
        **state,
        "event_streamer": streamer,
        "trace_logger": trace_logger,
        "event_log": event_log,
        "agent_outputs": {agent.name: output for agent, output in zip(agents, outputs, strict=False)},
        "agent_traces": trace_logger.export(),
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
    final_answer = apply_missing_options_guard(state["task"], final_answer)

    await streamer.publish(
        AgentEvent(
            run_id=state["run_id"],
            source="Synthesizer",
            event_type="final_summary",
            content="Final answer generated.",
        )
    )
    await streamer.drain(timeout=2)
    event_log = await streamer.get_events(run_id=state["run_id"])

    return {
        **state,
        "event_streamer": streamer,
        "event_log": event_log,
        "agent_traces": state.get("agent_traces", {}),
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


def apply_missing_options_guard(task: str, final_answer: str) -> str:
    message = "The options are missing, so I cannot choose one of them."
    if asks_for_missing_options(task) and message not in final_answer:
        return f"{message}\n\n{final_answer}"
    return final_answer


def asks_for_missing_options(task: str) -> bool:
    normalized = task.lower()
    asks_from_options = "which one of the following" in normalized or "following options" in normalized
    if not asks_from_options:
        return False

    option_markers = ("a)", "b)", "c)", "1)", "2)", "- ")
    return not any(marker in normalized for marker in option_markers)
