from __future__ import annotations

import asyncio

from multi_agent_sync.agents.registry import AGENT_REGISTRY
from multi_agent_sync.agents.dynamic_agent import DynamicAgent
from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.events.in_memory_streamer import InMemoryEventStreamer
from multi_agent_sync.graph.state import GraphState
from multi_agent_sync.llm import get_llm
from multi_agent_sync.orchestrator.orchestrator import create_model_based_plan, selected_agents_to_assignments
from multi_agent_sync.prompts import render_prompt
from multi_agent_sync.tools.bash import BashTool
from multi_agent_sync.tracing.trace import TraceLogger


async def orchestrator_node(state: GraphState) -> GraphState:
    streamer = state.get("event_streamer") or InMemoryEventStreamer()
    task = state["task"]
    llm = state.get("llm") or get_llm()
    subagent_mode = state.get("subagent_mode", "fixed")
    orchestrator_plan = await create_model_based_plan(task, llm, AGENT_REGISTRY, subagent_mode=subagent_mode)
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
        "subagent_mode": orchestrator_plan["subagent_mode"],
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
    prompt = render_prompt("graph/direct_answer.j2", task=state["task"])
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
    subagent_mode = state.get("subagent_mode", "fixed")
    enable_workspace_tools = bool(state.get("enable_workspace_tools", False))
    docker_workspace = state.get("docker_workspace")
    feedback_tool = state.get("feedback_tool")
    if enable_workspace_tools and docker_workspace is None:
        raise RuntimeError("Workspace tools were enabled, but no DockerWorkspace was provided.")

    agents = []
    assignments = apply_workspace_access_policy(
        selected_agents_to_assignments(state["orchestrator_plan"], max_steps=max_steps),
        subagent_mode=subagent_mode,
        enable_workspace_tools=enable_workspace_tools,
    )
    for assignment in assignments:
        agent_name = assignment["agent_name"]
        agent_class = DynamicAgent if subagent_mode == "dynamic" else AGENT_REGISTRY.get(agent_name)
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
            "workspace_access": assignment.get("workspace_access", "none"),
        }
        if enable_workspace_tools and assignment.get("workspace_access") != "none":
            agent_kwargs["bash_tool"] = BashTool(docker_workspace)
            if feedback_tool is not None:
                agent_kwargs["feedback_tool"] = feedback_tool
        if subagent_mode == "dynamic":
            agent_kwargs.update(
                {
                    "name": agent_name,
                    "role": assignment.get("role") or "Dynamic task subagent.",
                    "description": assignment.get("description", ""),
                    "rules": assignment.get("rules", []),
                    "critical_debate": assignment.get("critical_debate", False),
                }
            )
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


def apply_workspace_access_policy(
    assignments: list[dict[str, object]],
    *,
    subagent_mode: str,
    enable_workspace_tools: bool,
) -> list[dict[str, object]]:
    normalized = [dict(assignment) for assignment in assignments]
    if not enable_workspace_tools:
        for assignment in normalized:
            assignment["workspace_access"] = "none"
        return normalized

    if subagent_mode == "fixed":
        writer_name = "CodingAgent" if any(assignment.get("agent_name") == "CodingAgent" for assignment in normalized) else None
        if writer_name is None and normalized:
            writer_name = str(normalized[0].get("agent_name"))
        for assignment in normalized:
            assignment["workspace_access"] = "write" if assignment.get("agent_name") == writer_name else "none"
        return normalized

    writer_granted = False
    for assignment in normalized:
        requested = str(assignment.get("workspace_access") or "none").lower()
        if requested not in {"none", "read", "write"}:
            requested = "none"
        if assignment.get("critical_debate") and requested == "write":
            requested = "read"
        if requested == "write":
            if writer_granted:
                requested = "read"
            else:
                writer_granted = True
        assignment["workspace_access"] = requested
    return normalized


async def synthesizer_node(state: GraphState) -> GraphState:
    streamer = state.get("event_streamer") or InMemoryEventStreamer()
    llm = state.get("llm") or get_llm()
    event_lines = "\n".join(
        f"- [{event.event_type}] {event.source}: {event.content}"
        for event in state.get("event_log", [])
        if event.event_type in {"finding", "warning", "critique", "agent_done"}
    )
    output_lines = "\n".join(f"- {name}: {output}" for name, output in state.get("agent_outputs", {}).items())
    prompt = render_prompt(
        "graph/synthesizer.j2",
        task=state["task"],
        output_lines=output_lines,
        event_lines=event_lines,
    )
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
