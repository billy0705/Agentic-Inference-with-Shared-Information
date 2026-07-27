from __future__ import annotations

import asyncio
import re
from collections import Counter
from collections.abc import Callable
from typing import Any

from multi_agent_sync.agents.registry import AGENT_REGISTRY
from multi_agent_sync.agents.dynamic_agent import DynamicAgent
from multi_agent_sync.debate_prompts import build_debate_round_prompt, debate_prompt_style, render_debate_context
from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.events.in_memory_streamer import InMemoryEventStreamer
from multi_agent_sync.graph.state import GraphState
from multi_agent_sync.llm import get_llm
from multi_agent_sync.orchestrator.orchestrator import create_fallback_plan, create_model_based_plan, selected_agents_to_assignments
from multi_agent_sync.prompts import render_prompt
from multi_agent_sync.token_usage import extract_token_usage
from multi_agent_sync.tools.bash import BashTool
from multi_agent_sync.tracing.trace import TraceLogger


async def orchestrator_node(state: GraphState) -> GraphState:
    streamer = state.get("event_streamer") or InMemoryEventStreamer()
    task = state["task"]
    llm = state.get("llm") or get_llm()
    subagent_mode = state.get("subagent_mode", "fixed")
    orchestrator_plan = await create_model_based_plan(task, llm, AGENT_REGISTRY, subagent_mode=subagent_mode)
    if state.get("enable_workspace_tools") and (
        orchestrator_plan.get("mode") == "direct" or not orchestrator_plan.get("selected_agents")
    ):
        orchestrator_plan = create_fallback_plan(
            task,
            AGENT_REGISTRY,
            reason="Workspace tools are enabled, so the workflow must run agents with Docker bash access.",
            subagent_mode=subagent_mode,
        )
    selected_agent_specs = orchestrator_plan["selected_agents"]
    assignments = selected_agents_to_assignments(orchestrator_plan, max_steps=state.get("max_steps_per_agent", 3))
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
    direct_trace = {
        "mode": "dynamic_direct_debate" if state.get("subagent_mode") == "dynamic" else "direct",
        "steps": [],
        "final_answer": "",
    }
    try:
        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=state.get("synthesis_timeout", 60.0))
        final_answer = getattr(response, "content", str(response)).strip()
        direct_trace["steps"].append(
            {
                "step": 1,
                "kind": "direct_answer",
                "prompt": prompt,
                "raw_response": final_answer,
                "token_usage": extract_token_usage(response).as_dict(),
                "timed_out": False,
            }
        )
    except asyncio.TimeoutError:
        final_answer = "Direct answer timed out before the LLM returned a response."
        direct_trace["steps"].append(
            {
                "step": 1,
                "kind": "direct_answer",
                "prompt": prompt,
                "raw_response": final_answer,
                "token_usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                "timed_out": True,
            }
        )

    if state.get("subagent_mode") == "dynamic" and not direct_trace["steps"][0]["timed_out"]:
        debate_prompt = build_dynamic_direct_debate_prompt(
            task=state["task"],
            first_prompt=prompt,
            first_answer=final_answer,
            benchmark=str(state.get("benchmark", "") or ""),
        )
        try:
            debate_response = await asyncio.wait_for(llm.ainvoke(debate_prompt), timeout=state.get("synthesis_timeout", 60.0))
            final_answer = getattr(debate_response, "content", str(debate_response)).strip()
            direct_trace["steps"].append(
                {
                    "step": 2,
                    "kind": "debate_revision",
                    "prompt": debate_prompt,
                    "raw_response": final_answer,
                    "token_usage": extract_token_usage(debate_response).as_dict(),
                    "timed_out": False,
                }
            )
        except asyncio.TimeoutError:
            direct_trace["steps"].append(
                {
                    "step": 2,
                    "kind": "debate_revision",
                    "prompt": debate_prompt,
                    "raw_response": "Dynamic direct debate revision timed out; using the first direct answer.",
                    "token_usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    "timed_out": True,
                }
            )

    final_answer = apply_missing_options_guard(state["task"], final_answer)
    direct_trace["final_answer"] = final_answer
    return {
        **state,
        "event_streamer": streamer,
        "event_log": await streamer.get_events(run_id=state["run_id"]),
        "agent_outputs": {},
        "agent_traces": {},
        "direct_trace": direct_trace,
        "final_answer": final_answer,
    }


def build_dynamic_direct_debate_prompt(*, task: str, first_prompt: str, first_answer: str, benchmark: str) -> str:
    prompt_style = debate_prompt_style(benchmark)
    first_context = [
        {"role": "user", "content": first_prompt},
        {"role": "assistant", "content": first_answer},
    ]
    round_prompt = build_debate_round_prompt([first_context], task, 1, prompt_style)
    return render_debate_context([*first_context, {"role": "user", "content": round_prompt}])


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
    final_guard_tool = state.get("final_guard_tool")
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
            if final_guard_tool is not None:
                agent_kwargs["final_guard_tool"] = final_guard_tool
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
    agent_last_summaries = build_agent_last_summaries(state)
    candidate_aggregation = build_candidate_aggregation(state, agent_last_summaries)
    output_lines = format_agent_summary_lines(agent_last_summaries)
    candidate_lines = format_candidate_aggregation(candidate_aggregation)
    synthesizer_mode = state.get("synthesizer_mode", "generic")
    template_name = "graph/synthesizer_summarize_outputs.j2" if synthesizer_mode == "summarize_outputs" else "graph/synthesizer.j2"
    prompt = render_prompt(
        template_name,
        task=state["task"],
        output_lines=output_lines,
        candidate_lines=candidate_lines,
    )
    raw_response = ""
    token_usage = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    timed_out = False
    try:
        response = await asyncio.wait_for(llm.ainvoke(prompt), timeout=state.get("synthesis_timeout", 60.0))
        raw_response = getattr(response, "content", str(response)).strip()
        token_usage = extract_token_usage(response).as_dict()
        final_answer = raw_response if is_valid_summarizer_answer(state, raw_response) else (
            deterministic_synthesized_answer(state, candidate_aggregation) or raw_response
        )
    except asyncio.TimeoutError:
        timed_out = True
        final_answer = deterministic_synthesized_answer(state, candidate_aggregation) or build_fallback_summary(state)
        raw_response = final_answer
        await streamer.publish(
            AgentEvent(
                run_id=state["run_id"],
                source="Summarizer",
                event_type="warning",
                content="Summarizer timed out; using deterministic fallback summary.",
            )
        )
    final_answer = apply_missing_options_guard(state["task"], final_answer)

    await streamer.drain(timeout=2)
    event_log = await streamer.get_events(run_id=state["run_id"])

    next_state = {
        **state,
        "event_streamer": streamer,
        "event_log": event_log,
        "agent_traces": state.get("agent_traces", {}),
        "final_answer": final_answer,
    }
    if synthesizer_mode == "summarize_outputs":
        next_state["synthesizer_trace"] = {
            "mode": synthesizer_mode,
            "prompt": prompt,
            "raw_response": raw_response,
            "final_answer": final_answer,
            "token_usage": token_usage,
            "timed_out": timed_out,
            "candidate_aggregation": candidate_aggregation,
            "agent_last_summaries": agent_last_summaries,
        }
    return next_state


def build_agent_last_summaries(state: GraphState) -> dict[str, str]:
    traces = state.get("agent_traces", {})
    outputs = state.get("agent_outputs", {})
    names: list[str] = []
    for assignment in state.get("assignments", []):
        if isinstance(assignment, dict) and assignment.get("agent_name"):
            names.append(str(assignment["agent_name"]))
    names.extend(str(name) for name in outputs)
    if isinstance(traces, dict):
        names.extend(str(name) for name in traces)

    summaries: dict[str, str] = {}
    for name in dict.fromkeys(names):
        summary = last_summary_from_trace(traces.get(name) if isinstance(traces, dict) else None)
        if not summary:
            summary = str(outputs.get(name) or "").strip() if isinstance(outputs, dict) else ""
        if summary:
            summaries[name] = summary
    return summaries


def last_summary_from_trace(trace: Any) -> str:
    if not isinstance(trace, dict):
        return ""
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return ""
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        parsed_output = step.get("parsed_output")
        if not isinstance(parsed_output, dict):
            continue
        summary = str(parsed_output.get("summary") or "").strip()
        if summary:
            return summary
    return ""


def format_agent_summary_lines(agent_last_summaries: dict[str, str]) -> str:
    return "\n".join(f"- {name}: {summary}" for name, summary in agent_last_summaries.items())


def build_candidate_aggregation(state: GraphState, agent_last_summaries: dict[str, str] | None = None) -> dict[str, Any]:
    benchmark = str(state.get("benchmark", "") or "")
    extractor = answer_extractor_for_benchmark(benchmark)
    summary_texts = agent_last_summaries or build_agent_last_summaries(state)
    traces = state.get("agent_traces", {})
    assignments = {
        str(assignment.get("agent_name")): assignment
        for assignment in state.get("assignments", [])
        if isinstance(assignment, dict) and assignment.get("agent_name")
    }
    candidates: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    if extractor is None:
        return {
            "benchmark": benchmark,
            "candidates": candidates,
            "counts": {},
            "selected_candidate": None,
            "selection_rule": "no_benchmark_extractor",
        }

    for agent_name, output in summary_texts.items():
        output_text = str(output or "")
        answer_choice = last_answer_choice_from_trace(traces.get(agent_name) if isinstance(traces, dict) else None)
        candidate = candidate_from_answer_choice(benchmark, answer_choice, extractor)
        candidate_source = "answer_choice" if candidate is not None else "summary"
        if candidate is None:
            candidate = extractor(output_text)
        assignment = assignments.get(agent_name, {})
        candidate_row = {
            "agent_name": agent_name,
            "candidate": candidate,
            "candidate_source": candidate_source if candidate is not None else "none",
            "role": str(assignment.get("role") or ""),
            "description": str(assignment.get("description") or ""),
            "subtask": str(assignment.get("task") or ""),
            "expected_output": str(assignment.get("expected_output") or ""),
            "critical_debate": bool(assignment.get("critical_debate", False)),
            "reason_excerpt": compact_text(output_text, limit=360),
        }
        candidates.append(candidate_row)
        if candidate is not None:
            counts[str(candidate)] += 1

    selected_candidate = None
    selection_rule = "no_extractable_candidates"
    if counts:
        most_common = counts.most_common()
        top_candidate, top_count = most_common[0]
        runner_up_count = most_common[1][1] if len(most_common) > 1 else 0
        if top_count >= 2 and top_count > runner_up_count:
            selected_candidate = top_candidate
            selection_rule = "majority_vote"
        elif len(most_common) == 1:
            selected_candidate = top_candidate
            selection_rule = "unanimous_single_candidate"
        else:
            selection_rule = "tie_requires_role_aware_synthesis"

    return {
        "benchmark": benchmark,
        "candidates": candidates,
        "counts": dict(counts),
        "selected_candidate": selected_candidate,
        "selection_rule": selection_rule,
    }


def last_answer_choice_from_trace(trace: Any) -> str | None:
    if not isinstance(trace, dict):
        return None
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return None
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        parsed_output = step.get("parsed_output")
        if not isinstance(parsed_output, dict):
            continue
        answer_choice = extract_answer_choice_from_notes(str(parsed_output.get("local_notes") or ""))
        if answer_choice:
            return answer_choice
    return None


def extract_answer_choice_from_notes(notes: str) -> str | None:
    match = re.search(r"(?im)^\s*ANSWER_CHOICE\s*:\s*(.+?)\s*$", notes)
    if not match:
        return None
    answer_choice = match.group(1).strip()
    if answer_choice.lower() in {"", "none", "n/a", "undecided", "unknown"}:
        return None
    return answer_choice


def candidate_from_answer_choice(benchmark: str, answer_choice: str | None, extractor: Callable[[str], str | None]) -> str | None:
    if not answer_choice:
        return None
    return extractor(format_candidate_final_answer(benchmark, answer_choice)) or extractor(answer_choice)


def answer_extractor_for_benchmark(benchmark: str) -> Callable[[str], str | None] | None:
    normalized = benchmark.lower()
    if normalized == "gpqa":
        from multi_agent_sync.evaluation.benchmarks.gpqa import extract_answer

        return extract_answer
    if normalized == "mmlu_pro":
        from multi_agent_sync.evaluation.benchmarks.mmlu_pro import extract_answer

        return extract_answer
    if normalized == "gsm8k":
        from multi_agent_sync.evaluation.benchmarks.gsm8k import extract_answer

        return extract_answer
    if normalized == "chess":
        from multi_agent_sync.evaluation.benchmarks.chess import extract_answer

        return extract_answer
    if normalized == "olymmath":
        from multi_agent_sync.evaluation.benchmarks.olymmath import extract_answer

        return extract_answer
    return None


def deterministic_synthesized_answer(state: GraphState, aggregation: dict[str, Any]) -> str | None:
    if state.get("synthesizer_mode") != "summarize_outputs":
        return None
    if aggregation.get("selection_rule") != "majority_vote":
        return None
    selected_candidate = aggregation.get("selected_candidate")
    if selected_candidate is None:
        return None
    return format_candidate_final_answer(str(state.get("benchmark", "") or ""), str(selected_candidate))


def is_valid_summarizer_answer(state: GraphState, response: str) -> bool:
    benchmark = str(state.get("benchmark", "") or "")
    extractor = answer_extractor_for_benchmark(benchmark)
    if extractor is None:
        return True
    return extractor(response) is not None


def format_candidate_final_answer(benchmark: str, candidate: str) -> str:
    normalized = benchmark.lower()
    if normalized == "chess":
        return f"({candidate})"
    if normalized in {"gsm8k", "olymmath"}:
        return f"\\boxed{{{candidate}}}"
    return f"Final Answer: {candidate}"


def format_candidate_aggregation(aggregation: dict[str, Any]) -> str:
    candidates = aggregation.get("candidates") or []
    if not candidates:
        return "- No benchmark answer candidates were extracted."

    lines = [
        f"- selection_rule: {aggregation.get('selection_rule')}",
        f"- selected_candidate: {aggregation.get('selected_candidate')}",
        f"- counts: {aggregation.get('counts')}",
        "- agent candidates:",
    ]
    for candidate in candidates:
        lines.append(
            "  - "
            f"agent: {candidate.get('agent_name')}; "
            f"candidate: {candidate.get('candidate')}; "
            f"role: {candidate.get('role') or 'unspecified'}; "
            f"critical_debate: {candidate.get('critical_debate')}; "
            f"subtask: {candidate.get('subtask') or 'unspecified'}; "
            f"description: {candidate.get('description') or 'unspecified'}; "
            f"expected_output: {candidate.get('expected_output') or 'unspecified'}; "
            f"reason_excerpt: {candidate.get('reason_excerpt') or 'empty'}"
        )
    return "\n".join(lines)


def compact_text(text: str, *, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: max(0, limit - 3)].rstrip()}..."


def build_fallback_summary(state: GraphState) -> str:
    outputs = state.get("agent_outputs", {})
    parts = [f"Summarizer timed out; fallback summary for task: {state['task']}."]
    for name, output in outputs.items():
        if output:
            parts.append(f"{name}: {output}")
    if len(parts) == 1:
        parts.append("No agent outputs were available before the summarizer timed out.")
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
