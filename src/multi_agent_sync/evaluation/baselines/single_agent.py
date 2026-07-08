from __future__ import annotations

import argparse
import json
import re
from typing import Any

from multi_agent_sync.prompts import render_prompt


async def run_single_agent(prompt: str, llm: Any, args: argparse.Namespace) -> tuple[str, int, dict[str, Any]]:
    max_steps = max(1, int(getattr(args, "max_steps", 3)))
    answer_extractor = getattr(args, "answer_extractor", None)
    if not callable(answer_extractor):
        answer_extractor = lambda text: text.strip() if text.strip() else None

    steps: list[dict[str, Any]] = []
    raw_output = ""
    stopped_reason = "max_steps"

    for step_index in range(1, max_steps + 1):
        step_prompt = render_prompt(
            "evaluation/single_agent_step.j2",
            benchmark_prompt=prompt,
            previous_steps=steps,
        )
        response = await llm.ainvoke(step_prompt)
        content = getattr(response, "content", str(response)).strip()
        parsed_payload = parse_single_agent_payload(content)
        status = parsed_payload.get("status")
        final_answer = str(parsed_payload.get("final_answer") or "").strip()
        notes = str(parsed_payload.get("notes") or "").strip()
        candidate_output = final_answer or content
        parsed_answer = answer_extractor(candidate_output)
        raw_output = candidate_output

        step_trace = {
            "step": step_index,
            "status": status,
            "output": candidate_output,
            "notes": notes,
            "parsed_answer": parsed_answer,
            "raw_response": content,
        }
        steps.append(step_trace)

        if status == "final" and parsed_answer is not None:
            stopped_reason = "final_answer_parseable"
            break

    return raw_output, 0, {
        "method": "single_agent",
        "prompt": prompt,
        "raw_output": raw_output,
        "steps": steps,
        "stopped_reason": stopped_reason,
    }


def parse_single_agent_payload(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", stripped):
            try:
                payload, _ = decoder.raw_decode(stripped[match.start() :])
                break
            except json.JSONDecodeError:
                continue
        else:
            return {"status": "continue", "final_answer": "", "notes": "Model did not return JSON."}
    if not isinstance(payload, dict):
        return {"status": "continue", "final_answer": "", "notes": "Model did not return a JSON object."}

    status = str(payload.get("status") or "continue").strip().lower()
    if status not in {"continue", "final"}:
        status = "continue"
    return {
        "status": status,
        "final_answer": payload.get("final_answer") or "",
        "notes": payload.get("notes") or "",
    }
