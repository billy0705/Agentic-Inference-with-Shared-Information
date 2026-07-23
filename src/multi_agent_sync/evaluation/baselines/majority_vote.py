from __future__ import annotations

import argparse
from typing import Any

from multi_agent_sync.evaluation.baselines.common import extract_answer_with_args, most_frequent_present_answer
from multi_agent_sync.evaluation.baselines.single_agent import run_single_agent


MAJORITY_VOTE_AGENT_COUNT = 3


async def run_majority_vote(prompt: str, llm: Any, args: argparse.Namespace) -> tuple[str, int, dict[str, Any]]:
    agent_runs: list[dict[str, Any]] = []
    agent_outputs: dict[str, str] = {}
    agent_traces: dict[str, dict[str, Any]] = {}
    raw_outputs: list[str] = []
    parsed_answers: list[str | None] = []

    for agent_index in range(MAJORITY_VOTE_AGENT_COUNT):
        agent_name = f"VoterAgent{agent_index + 1}"
        raw_output, returncode, trace = await run_single_agent(prompt, llm, args)
        parsed_answer = parsed_answer_from_single_agent_trace(trace) or extract_answer_with_args(raw_output, args)
        raw_outputs.append(raw_output)
        parsed_answers.append(parsed_answer)
        agent_outputs[agent_name] = raw_output
        agent_traces[agent_name] = {
            "assignment": {
                "method": "majority_vote",
                "vote": agent_index + 1,
                "parsed_answer": parsed_answer,
                "returncode": returncode,
            },
            "steps": trace.get("steps", []) if isinstance(trace, dict) else [],
            "event_receipts": [],
            "unused_received_events": [],
            "final_output": raw_output,
            "stopped_reason": trace.get("stopped_reason") if isinstance(trace, dict) else None,
        }
        agent_runs.append(
            {
                "agent": agent_index + 1,
                "agent_name": agent_name,
                "raw_output": raw_output,
                "returncode": returncode,
                "parsed_answer": parsed_answer,
                "trace": trace,
            }
        )

    voted_answer = most_frequent_present_answer(parsed_answers)
    fallback_used = voted_answer is None
    final_output = f"Final Answer: {voted_answer}" if voted_answer is not None else raw_outputs[0]

    return final_output, 0, {
        "method": "majority_vote",
        "prompt": prompt,
        "agents": MAJORITY_VOTE_AGENT_COUNT,
        "agent_runs": agent_runs,
        "agent_outputs": agent_outputs,
        "agent_traces": agent_traces,
        "raw_outputs": raw_outputs,
        "parsed_answers": parsed_answers,
        "voted_answer": voted_answer,
        "fallback_used": fallback_used,
        "raw_output": final_output,
    }


def parsed_answer_from_single_agent_trace(trace: dict[str, Any]) -> str | None:
    if not isinstance(trace, dict):
        return None
    steps = trace.get("steps")
    if not isinstance(steps, list):
        return None
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        parsed_answer = step.get("parsed_answer")
        if parsed_answer is not None:
            return str(parsed_answer)
    return None
