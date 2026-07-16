from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.graph.state import GraphState
from multi_agent_sync.sync_report import build_sync_report


def save_run_artifacts(state: GraphState, root_dir: str | Path = "runs") -> Path:
    run_dir = Path(root_dir) / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    agent_traces = state.get("agent_traces", {})
    token_usage_by_step = build_token_usage_by_step(agent_traces)

    (run_dir / "task.txt").write_text(state["task"])
    (run_dir / "final_answer.md").write_text(state["final_answer"])
    _write_event_log(run_dir / "event_log.jsonl", state.get("event_log", []))
    (run_dir / "agent_traces.json").write_text(json.dumps(agent_traces, indent=2))
    (run_dir / "token_usage_by_step.json").write_text(json.dumps(token_usage_by_step, indent=2))
    (run_dir / "token_usage_by_step.txt").write_text(format_token_usage_by_step(token_usage_by_step))
    (run_dir / "orchestrator_plan.json").write_text(json.dumps(state.get("orchestrator_plan", {}), indent=2))
    sync_report = build_sync_report(state.get("event_log", []), agent_traces)
    (run_dir / "sync_report.json").write_text(json.dumps(sync_report, indent=2))

    return run_dir


def build_token_usage_by_step(agent_traces: dict[str, Any]) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for agent_name, trace in agent_traces.items():
        if not isinstance(trace, dict):
            continue
        steps = trace.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            token_usage = step.get("token_usage") if isinstance(step.get("token_usage"), dict) else {}
            report.append(
                {
                    "agent": agent_name,
                    "step": step.get("step"),
                    "prompt_tokens": _coerce_optional_int(token_usage.get("prompt_tokens")),
                    "completion_tokens": _coerce_optional_int(token_usage.get("completion_tokens")),
                    "total_tokens": _coerce_optional_int(token_usage.get("total_tokens")),
                }
            )
    return report


def format_token_usage_by_step(report: list[dict[str, Any]]) -> str:
    lines = ["Token usage by step"]
    if not report:
        lines.append("- No agent steps recorded.")
        return "\n".join(lines)

    for item in report:
        lines.append(
            f"- {item.get('agent', 'unknown')} step {_format_token_value(item.get('step'))}: "
            f"prompt={_format_token_value(item.get('prompt_tokens'))}, "
            f"completion={_format_token_value(item.get('completion_tokens'))}, "
            f"total={_format_token_value(item.get('total_tokens'))}"
        )
    return "\n".join(lines)


def _write_event_log(path: Path, event_log: list[AgentEvent]) -> None:
    lines = [json.dumps(event.model_dump(mode="json")) for event in event_log]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_token_value(value: Any) -> str:
    coerced = _coerce_optional_int(value)
    return str(coerced) if coerced is not None else "unknown"


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
