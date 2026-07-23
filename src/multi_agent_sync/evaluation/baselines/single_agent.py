from __future__ import annotations

import argparse
import json
import re
from typing import Any

from multi_agent_sync.prompts import render_prompt
from multi_agent_sync.token_usage import extract_token_usage


DEFAULT_SINGLE_AGENT_MIN_STEPS = 3
DEFAULT_SINGLE_AGENT_MAX_STEPS = 7


async def run_single_agent(prompt: str, llm: Any, args: argparse.Namespace) -> tuple[str, int, dict[str, Any]]:
    min_steps, max_steps = resolve_single_agent_step_limits(args)
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
            previous_step=steps[-1] if steps else None,
        )
        response = await llm.ainvoke(step_prompt)
        token_usage = extract_token_usage(response)
        content = getattr(response, "content", str(response)).strip()
        parsed_payload = parse_single_agent_payload(content)
        status = parsed_payload.get("status")
        final_answer = str(parsed_payload.get("final_answer") or "").strip()
        final_reason = str(parsed_payload.get("final_reason") or "").strip()
        notes = str(parsed_payload.get("notes") or "").strip()
        candidate_output = final_answer or content
        parsed_answer = answer_extractor(candidate_output)
        if parsed_answer is None and final_answer:
            formatted_output = format_answer_with_args(final_answer, args)
            formatted_answer = answer_extractor(formatted_output)
            if formatted_answer is not None:
                candidate_output = formatted_output
                parsed_answer = formatted_answer
        human_output = format_single_agent_output(candidate_output, final_reason)
        raw_output = human_output
        final_blocked_reason = final_block_reason(
            step_index=step_index,
            min_steps=min_steps,
            status=status,
            parsed_answer=parsed_answer,
            final_reason=final_reason,
        )
        final_accepted = status == "final" and parsed_answer is not None and final_blocked_reason is None

        step_trace = {
            "step": step_index,
            "status": status,
            "prompt": step_prompt,
            "token_usage": token_usage.as_dict(),
            "output": candidate_output,
            "final_reason": final_reason,
            "final_accepted": final_accepted,
            "final_blocked_reason": final_blocked_reason,
            "notes": notes,
            "parsed_answer": parsed_answer,
            "raw_response": content,
        }
        steps.append(step_trace)

        if final_accepted:
            stopped_reason = "final_answer_parseable"
            break

    return raw_output, 0, {
        "method": "single_agent",
        "prompt": prompt,
        "raw_output": raw_output,
        "single_agent_min_steps": min_steps,
        "single_agent_max_steps": max_steps,
        "steps": steps,
        "stopped_reason": stopped_reason,
    }


def resolve_single_agent_step_limits(args: argparse.Namespace) -> tuple[int, int]:
    min_steps = _positive_int_or_default(
        getattr(args, "single_agent_min_steps", None),
        DEFAULT_SINGLE_AGENT_MIN_STEPS,
    )
    max_steps = _positive_int_or_default(
        getattr(args, "single_agent_max_steps", None),
        DEFAULT_SINGLE_AGENT_MAX_STEPS,
    )
    if max_steps < min_steps:
        max_steps = min_steps
    return min_steps, max_steps


def format_single_agent_output(output: str, reason: str) -> str:
    output = output.strip()
    reason = reason.strip()
    if reason:
        return f"Reason: {reason}\n{output}".strip()
    return output


def format_answer_with_args(answer: str, args: argparse.Namespace) -> str:
    answer_formatter = getattr(args, "answer_formatter", None)
    if callable(answer_formatter):
        return str(answer_formatter(answer)).strip()
    return answer


def _positive_int_or_default(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def final_block_reason(
    *,
    step_index: int,
    min_steps: int,
    status: Any,
    parsed_answer: str | None,
    final_reason: str,
) -> str | None:
    if status != "final" or parsed_answer is None:
        return None
    if step_index < min_steps:
        return "minimum_steps_not_reached"
    if not is_substantive_final_reason(final_reason):
        return "final_reason_required"
    return None


def is_substantive_final_reason(final_reason: str) -> bool:
    normalized = re.sub(r"[^a-z0-9\s]", " ", final_reason.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    confidence_only_patterns = (
        r"^(confidence|confident)(\s+[0-9]+)*\s*%?$",
        r"^(high|medium|low)\s+confidence$",
        r"^[0-9.]+\s*%?$",
    )
    return not any(re.fullmatch(pattern, normalized) for pattern in confidence_only_patterns)


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
            jsonish_payload = parse_jsonish_single_agent_payload(stripped)
            if jsonish_payload is not None:
                return jsonish_payload
            return {"status": "continue", "final_answer": "", "notes": "Model did not return JSON."}
    if not isinstance(payload, dict):
        return {"status": "continue", "final_answer": "", "notes": "Model did not return a JSON object."}

    status = str(payload.get("status") or "continue").strip().lower()
    if status not in {"continue", "final"}:
        status = "continue"
    return {
        "status": status,
        "final_answer": payload.get("final_answer") or "",
        "final_reason": payload.get("final_reason") or "",
        "notes": payload.get("notes") or "",
    }


def parse_jsonish_single_agent_payload(text: str) -> dict[str, Any] | None:
    status = extract_jsonish_string_field(text, "status")
    final_answer = extract_jsonish_string_field(text, "final_answer")
    final_reason = extract_jsonish_string_field(text, "final_reason")
    notes = extract_jsonish_string_field(text, "notes")
    if status is None and final_answer is None and final_reason is None and notes is None:
        return None

    normalized_status = str(status or "continue").strip().lower()
    if normalized_status not in {"continue", "final"}:
        normalized_status = "continue"
    return {
        "status": normalized_status,
        "final_answer": final_answer or "",
        "final_reason": final_reason or "",
        "notes": notes or "Recovered from JSON-like response.",
    }


def extract_jsonish_string_field(text: str, key: str) -> str | None:
    pattern = rf'"{re.escape(key)}"\s*:\s*"(?P<value>.*?)"(?=\s*,\s*"[^"]+"\s*:|\s*}})'
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return None
    return match.group("value").strip()
