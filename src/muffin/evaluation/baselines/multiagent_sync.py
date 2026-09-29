import argparse
from typing import Any

from muffin.agents.base import DEFAULT_AGENT_RUNTIME_TIMEOUT_SECONDS
from muffin.graph.workflow import run_workflow


async def run_multiagent(
    prompt: str,
    llm: Any,
    args: argparse.Namespace,
) -> tuple[str, int, dict[str, Any]]:
    state = await run_workflow(
        task=prompt,
        llm=llm,
        max_steps_per_agent=args.max_steps,
        min_dynamic_subagents=getattr(args, "min_dynamic_subagents", 3),
        max_dynamic_subagents=getattr(args, "max_dynamic_subagents", 3),
        total_runtime_timeout=args.total_runtime_timeout,
        agent_runtime_timeout=getattr(args, "agent_runtime_timeout", DEFAULT_AGENT_RUNTIME_TIMEOUT_SECONDS),
        synthesis_timeout=args.synthesis_timeout,
        allow_agent_early_stop=getattr(args, "allow_agent_early_stop", False),
        think_mode=getattr(args, "think_mode", True),
        stream_to_console=False,
        no_color=True,
        benchmark=str(getattr(args, "benchmark", "") or ""),
    )
    return state["final_answer"], 0, extract_workflow_trace(state)


def extract_workflow_trace(state: dict[str, Any]) -> dict[str, Any]:
    trace_keys = [
        "method",
        "run_id",
        "mode",
        "task_type",
        "reason",
        "plan",
        "selected_agents",
        "assignments",
        "allow_agent_early_stop",
        "think_mode",
        "agent_runtime_timeout",
        "orchestrator_plan",
        "event_log",
        "agent_outputs",
        "agent_traces",
        "synthesizer_trace",
        "direct_trace",
        "final_answer",
    ]
    return {key: state.get(key) for key in trace_keys if key in state}
