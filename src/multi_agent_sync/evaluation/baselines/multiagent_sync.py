from __future__ import annotations

import argparse
from typing import Any

from multi_agent_sync.graph.workflow import run_workflow


async def run_multiagent(
    prompt: str,
    llm: Any,
    args: argparse.Namespace,
    *,
    enable_agent_message_streaming: bool = True,
    subagent_mode: str = "fixed",
) -> tuple[str, int, dict[str, Any]]:
    state = await run_workflow(
        task=prompt,
        llm=llm,
        subagent_mode=subagent_mode,
        max_steps_per_agent=args.max_steps,
        total_runtime_timeout=args.total_runtime_timeout,
        synthesis_timeout=args.synthesis_timeout,
        enable_agent_message_streaming=enable_agent_message_streaming,
        stream_to_console=False,
        no_color=True,
    )
    return state["final_answer"], 0, extract_workflow_trace(state)


def extract_workflow_trace(state: dict[str, Any]) -> dict[str, Any]:
    trace_keys = [
        "run_id",
        "mode",
        "subagent_mode",
        "task_type",
        "reason",
        "plan",
        "selected_agents",
        "assignments",
        "orchestrator_plan",
        "event_log",
        "agent_outputs",
        "agent_traces",
        "final_answer",
    ]
    return {key: state.get(key) for key in trace_keys if key in state}
