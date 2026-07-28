from __future__ import annotations

import asyncio
import re
from typing import Any
from uuid import uuid4

from multi_agent_sync.agents.dynamic_agent import DynamicAgent
from multi_agent_sync.events.in_memory_streamer import InMemoryEventStreamer
from multi_agent_sync.graph.nodes import (
    apply_missing_options_guard,
    apply_workspace_access_policy,
    answer_extractor_for_benchmark,
    build_agent_last_summaries,
    build_candidate_aggregation,
    build_fallback_summary,
    candidate_from_answer_choice,
    deterministic_synthesized_answer,
    format_agent_summary_lines,
    format_candidate_final_answer,
    format_candidate_aggregation,
)
from multi_agent_sync.llm import get_llm
from multi_agent_sync.orchestrator.orchestrator import (
    create_dynamic_fallback_plan,
    extract_json_object,
    normalize_collaboration_protocol,
    normalize_dynamic_selected_agents,
    selected_agents_to_assignments,
)
from multi_agent_sync.prompts import render_prompt
from multi_agent_sync.streaming.console import ConsoleEventStreamer
from multi_agent_sync.token_usage import extract_token_usage
from multi_agent_sync.tools.bash import BashTool
from multi_agent_sync.tracing.trace import TraceLogger


MAX_DYNAMIC_ORCHESTRATION_AGENTS_PER_ROUND = 4


async def run_dynamic_orchestration_workflow(
    *,
    task: str,
    llm: Any | None = None,
    benchmark: str = "",
    max_steps_per_agent: int = 3,
    max_orchestrator_rounds: int = 3,
    total_runtime_timeout: float = 600.0,
    synthesis_timeout: float = 60.0,
    enable_agent_message_streaming: bool = True,
    stream_to_console: bool = True,
    no_color: bool = False,
    enable_workspace_tools: bool = False,
    docker_workspace: Any | None = None,
    feedback_tool: Any | None = None,
    final_guard_tool: Any | None = None,
) -> dict[str, Any]:
    model = llm or get_llm()
    streamer = InMemoryEventStreamer()
    if stream_to_console:
        ConsoleEventStreamer(streamer, no_color=no_color)
    run_id = str(uuid4())
    rounds: list[dict[str, Any]] = []
    all_assignments: list[dict[str, Any]] = []
    all_agent_outputs: dict[str, str] = {}
    all_agent_traces: dict[str, Any] = {}
    final_answer = ""
    forced_final = False

    if enable_workspace_tools and docker_workspace is None:
        raise RuntimeError("Workspace tools were enabled, but no DockerWorkspace was provided.")

    max_rounds = max(1, int(max_orchestrator_rounds or 1))
    for round_index in range(1, max_rounds + 1):
        prior_state = {
            "task": task,
            "benchmark": benchmark,
            "assignments": all_assignments,
            "agent_outputs": all_agent_outputs,
            "agent_traces": all_agent_traces,
            "synthesizer_mode": "summarize_outputs",
        }
        agent_last_summaries = build_agent_last_summaries(prior_state)
        candidate_aggregation = build_candidate_aggregation(prior_state, agent_last_summaries)
        prompt = build_dynamic_orchestration_prompt(
            task=task,
            round_index=round_index,
            max_rounds=max_rounds,
            rounds=rounds,
            agent_last_summaries=agent_last_summaries,
            candidate_aggregation=candidate_aggregation,
            force_final=round_index == max_rounds and bool(rounds),
        )
        decision, raw_response, token_usage, timed_out = await request_orchestration_decision(
            model,
            prompt,
            timeout=synthesis_timeout,
            task=task,
        )
        round_trace: dict[str, Any] = {
            "round": round_index,
            "prompt": prompt,
            "raw_response": raw_response,
            "token_usage": token_usage,
            "timed_out": timed_out,
            "decision": decision,
            "candidate_aggregation_before": candidate_aggregation,
            "agent_last_summaries_before": agent_last_summaries,
        }

        if decision["action"] == "final":
            final_answer = normalize_orchestrator_final_answer(
                benchmark,
                str(decision.get("final_answer") or "").strip(),
            )
            decision["final_answer"] = final_answer
            round_trace["final_answer"] = final_answer
            rounds.append(round_trace)
            break

        if round_index == max_rounds and rounds:
            forced_final = True
            final_answer = fallback_final_answer(task, benchmark, candidate_aggregation, prior_state, raw_response)
            round_trace["forced_final"] = True
            round_trace["final_answer"] = final_answer
            rounds.append(round_trace)
            break

        round_state = await run_dynamic_agent_round(
            task=task,
            run_id=run_id,
            llm=model,
            streamer=streamer,
            decision=decision,
            round_index=round_index,
            max_steps_per_agent=max_steps_per_agent,
            total_runtime_timeout=total_runtime_timeout,
            enable_agent_message_streaming=enable_agent_message_streaming,
            enable_workspace_tools=enable_workspace_tools,
            docker_workspace=docker_workspace,
            feedback_tool=feedback_tool,
            final_guard_tool=final_guard_tool,
        )
        round_assignments = round_state["assignments"]
        round_outputs = round_state["agent_outputs"]
        round_traces = round_state["agent_traces"]
        all_assignments.extend(round_assignments)
        for agent_name, output in round_outputs.items():
            scoped_name = scoped_agent_name(round_index, agent_name, all_agent_outputs)
            all_agent_outputs[scoped_name] = output
        for agent_name, trace in round_traces.items():
            scoped_name = scoped_agent_name(round_index, agent_name, all_agent_traces)
            all_agent_traces[scoped_name] = trace

        post_round_state = {
            "task": task,
            "benchmark": benchmark,
            "assignments": round_assignments,
            "agent_outputs": round_outputs,
            "agent_traces": round_traces,
            "synthesizer_mode": "summarize_outputs",
        }
        round_trace.update(
            {
                "selected_agents": decision.get("selected_agents", []),
                "assignments": round_assignments,
                "agent_outputs": round_outputs,
                "agent_traces": round_traces,
                "event_log": round_state["event_log"],
                "candidate_aggregation": build_candidate_aggregation(
                    post_round_state,
                    build_agent_last_summaries(post_round_state),
                ),
            }
        )
        rounds.append(round_trace)

    if not final_answer:
        forced_final = True
        fallback_state = {
            "task": task,
            "benchmark": benchmark,
            "assignments": all_assignments,
            "agent_outputs": all_agent_outputs,
            "agent_traces": all_agent_traces,
            "synthesizer_mode": "summarize_outputs",
        }
        final_answer = fallback_final_answer(
            task,
            benchmark,
            build_candidate_aggregation(fallback_state, build_agent_last_summaries(fallback_state)),
            fallback_state,
            "Dynamic orchestration reached the round limit without a final decision.",
        )

    final_answer = apply_missing_options_guard(task, final_answer)
    await streamer.drain(timeout=2)
    event_log = await streamer.get_events(run_id=run_id)
    trace = {
        "method": "dynamic_orchestration",
        "rounds": rounds,
        "forced_final": forced_final,
        "max_orchestrator_rounds": max_rounds,
        "final_answer": final_answer,
    }
    return {
        "method": "dynamic_orchestration",
        "run_id": run_id,
        "mode": "dynamic_orchestration",
        "subagent_mode": "dynamic",
        "task": task,
        "benchmark": benchmark,
        "event_log": event_log,
        "agent_outputs": all_agent_outputs,
        "agent_traces": all_agent_traces,
        "assignments": all_assignments,
        "selected_agents": [agent for round_trace in rounds for agent in round_trace.get("selected_agents", [])],
        "orchestrator_rounds": rounds,
        "dynamic_orchestration_trace": trace,
        "final_answer": final_answer,
    }


def build_dynamic_orchestration_prompt(
    *,
    task: str,
    round_index: int,
    max_rounds: int,
    rounds: list[dict[str, Any]],
    agent_last_summaries: dict[str, str],
    candidate_aggregation: dict[str, Any],
    force_final: bool,
) -> str:
    return render_prompt(
        "orchestrator/dynamic_orchestration_step.j2",
        task=task,
        round_index=round_index,
        max_rounds=max_rounds,
        history=format_orchestration_history(rounds),
        agent_summary_lines=format_agent_summary_lines(agent_last_summaries),
        candidate_lines=format_candidate_aggregation(candidate_aggregation),
        force_final=force_final,
    )


async def request_orchestration_decision(
    llm: Any,
    prompt: str,
    *,
    timeout: float,
    task: str,
) -> tuple[dict[str, Any], str, dict[str, int | None], bool]:
    try:
        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=timeout)
        raw_response = getattr(response, "content", str(response)).strip()
        token_usage = extract_token_usage(response).as_dict()
        return parse_orchestration_decision(raw_response, task), raw_response, token_usage, False
    except asyncio.TimeoutError:
        return (
            {
                "action": "final",
                "reason": "The orchestration controller timed out.",
                "final_answer": "Dynamic orchestration timed out before a final answer was produced.",
            },
            "Dynamic orchestration timed out before a final answer was produced.",
            {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            True,
        )
    except Exception as exc:
        fallback = create_dynamic_fallback_plan(task, reason=f"Dynamic orchestration decision was invalid: {exc}")
        return (
            {
                "action": "run_agents",
                "reason": fallback["reason"],
                "round_goal": fallback["task_summary"],
                "selected_agents": fallback["selected_agents"],
                "collaboration_protocol": fallback["collaboration_protocol"],
            },
            str(exc),
            {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            False,
        )


def parse_orchestration_decision(content: str, task: str) -> dict[str, Any]:
    raw_decision = extract_orchestration_json_object(content)
    action = str(raw_decision.get("action") or "").strip().lower()
    if action == "final":
        final_answer = str(raw_decision.get("final_answer") or "").strip()
        if not final_answer:
            raise ValueError("final action requires final_answer")
        return {
            "action": "final",
            "reason": str(raw_decision.get("reason") or "The orchestrator selected final.").strip(),
            "final_answer": final_answer,
        }
    if action != "run_agents":
        raise ValueError("dynamic orchestration action must be run_agents or final")

    selected_agents = normalize_dynamic_selected_agents(raw_decision.get("selected_agents"))[
        :MAX_DYNAMIC_ORCHESTRATION_AGENTS_PER_ROUND
    ]
    if not selected_agents:
        selected_agents = create_dynamic_fallback_plan(task, reason="No valid dynamic agents were selected.")["selected_agents"]
    return {
        "action": "run_agents",
        "reason": str(raw_decision.get("reason") or "The orchestrator selected another agent round.").strip(),
        "round_goal": str(raw_decision.get("round_goal") or "Resolve the task with dynamic subagents.").strip(),
        "selected_agents": selected_agents,
        "collaboration_protocol": normalize_collaboration_protocol(raw_decision.get("collaboration_protocol")),
    }


def extract_orchestration_json_object(content: str) -> dict[str, Any]:
    try:
        return extract_json_object(content)
    except ValueError:
        repaired = repair_invalid_json_backslash_escapes(content)
        if repaired == content:
            raise
        return extract_json_object(repaired)


def repair_invalid_json_backslash_escapes(content: str) -> str:
    return re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", content)


def normalize_orchestrator_final_answer(benchmark: str, final_answer: str) -> str:
    stripped = final_answer.strip()
    extractor = answer_extractor_for_benchmark(benchmark)
    if extractor is None or not stripped:
        return stripped
    if extractor(stripped) is not None:
        return stripped
    candidate = candidate_from_answer_choice(benchmark, stripped, extractor)
    if candidate is None:
        return stripped
    return format_candidate_final_answer(benchmark, candidate)


async def run_dynamic_agent_round(
    *,
    task: str,
    run_id: str,
    llm: Any,
    streamer: InMemoryEventStreamer,
    decision: dict[str, Any],
    round_index: int,
    max_steps_per_agent: int,
    total_runtime_timeout: float,
    enable_agent_message_streaming: bool,
    enable_workspace_tools: bool,
    docker_workspace: Any | None,
    feedback_tool: Any | None,
    final_guard_tool: Any | None,
) -> dict[str, Any]:
    trace_logger = TraceLogger()
    plan = {
        "mode": "multi_agent",
        "subagent_mode": "dynamic",
        "task_type": "dynamic orchestration round",
        "task_summary": decision.get("round_goal", ""),
        "reason": decision.get("reason", ""),
        "selected_agents": decision.get("selected_agents", []),
        "collaboration_protocol": decision.get("collaboration_protocol"),
    }
    assignments = apply_workspace_access_policy(
        selected_agents_to_assignments(plan, max_steps=max_steps_per_agent),
        subagent_mode="dynamic",
        enable_workspace_tools=enable_workspace_tools,
    )
    agents = []
    for assignment in assignments:
        agent_name = str(assignment["agent_name"])
        agent_kwargs = {
            "name": agent_name,
            "role": assignment.get("role") or "Dynamic orchestration subagent.",
            "description": assignment.get("description", ""),
            "rules": assignment.get("rules", []),
            "critical_debate": bool(assignment.get("critical_debate", False)),
            "run_id": run_id,
            "task": task,
            "assigned_subtask": assignment["task"],
            "llm": llm,
            "event_streamer": streamer,
            "assignment": {**assignment, "orchestration_round": round_index},
            "trace_logger": trace_logger,
            "max_steps": int(assignment.get("max_steps", max_steps_per_agent)),
            "enable_message_streaming": enable_agent_message_streaming,
            "workspace_access": assignment.get("workspace_access", "none"),
        }
        if enable_workspace_tools and assignment.get("workspace_access") != "none":
            agent_kwargs["bash_tool"] = BashTool(docker_workspace)
            if feedback_tool is not None:
                agent_kwargs["feedback_tool"] = feedback_tool
            if final_guard_tool is not None:
                agent_kwargs["final_guard_tool"] = final_guard_tool
        for reactive_field in ("reactive_steps_enabled", "max_reactive_steps", "reactive_event_types"):
            if reactive_field in assignment:
                agent_kwargs[reactive_field] = assignment[reactive_field]
        agents.append(DynamicAgent(**agent_kwargs))

    for agent in agents:
        agent.subscribe()

    if agents:
        try:
            outputs = await asyncio.wait_for(
                asyncio.gather(*(agent.run() for agent in agents)),
                timeout=total_runtime_timeout,
            )
        except asyncio.TimeoutError:
            for agent in agents:
                agent.is_done = True
            outputs = [agent.local_output for agent in agents]
    else:
        outputs = []

    await streamer.drain(timeout=2)
    return {
        "assignments": assignments,
        "agent_outputs": {agent.name: output for agent, output in zip(agents, outputs, strict=False)},
        "agent_traces": trace_logger.export(),
        "event_log": await streamer.get_events(run_id=run_id),
    }


def fallback_final_answer(
    task: str,
    benchmark: str,
    aggregation: dict[str, Any],
    state: dict[str, Any],
    reason: str,
) -> str:
    deterministic = deterministic_synthesized_answer(
        {"benchmark": benchmark, "synthesizer_mode": "summarize_outputs"},
        aggregation,
    )
    if deterministic:
        return deterministic
    selected_candidate = aggregation.get("selected_candidate")
    if selected_candidate is not None:
        return format_candidate_final_answer(benchmark, str(selected_candidate))
    fallback = build_fallback_summary({"task": task, **state})
    return f"{reason}\n\n{fallback}".strip()


def format_orchestration_history(rounds: list[dict[str, Any]]) -> str:
    if not rounds:
        return ""
    lines: list[str] = []
    for round_trace in rounds[-4:]:
        decision = round_trace.get("decision", {})
        lines.append(
            f"- Round {round_trace.get('round')}: action={decision.get('action')}; "
            f"reason={decision.get('reason', '')}; final={round_trace.get('final_answer', '')}"
        )
        aggregation = round_trace.get("candidate_aggregation")
        if isinstance(aggregation, dict):
            lines.append(
                f"  candidate_rule={aggregation.get('selection_rule')}; "
                f"strength={aggregation.get('consensus_strength')}; counts={aggregation.get('counts')}"
            )
    return "\n".join(lines)


def scoped_agent_name(round_index: int, agent_name: str, existing: dict[str, Any]) -> str:
    if agent_name not in existing:
        return agent_name
    return f"round_{round_index}_{agent_name}"
