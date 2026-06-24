#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_trace_files(path: Path) -> list[dict[str, Any]]:
    candidates = path.rglob("*.json") if path.is_dir() else [path]
    traces: list[dict[str, Any]] = []
    for candidate in sorted(candidates):
        if candidate.name in {"run_config.json", "summary.json"}:
            continue
        try:
            trace = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "index" in trace and "method" in trace:
            trace["_path"] = str(candidate)
            traces.append(trace)
    return traces


def find_method_differences(traces: list[dict[str, Any]], methods: list[str] | None = None) -> list[dict[str, Any]]:
    allowed_methods = set(methods or [])
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for trace in traces:
        method = str(trace.get("method", ""))
        if allowed_methods and method not in allowed_methods:
            continue
        grouped[int(trace["index"])][method] = trace

    differences: list[dict[str, Any]] = []
    for index, by_method in sorted(grouped.items()):
        if len(by_method) < 2:
            continue
        predictions = {trace.get("pred") for trace in by_method.values()}
        correctness = {trace.get("correct") for trace in by_method.values()}
        if len(predictions) <= 1 and len(correctness) <= 1:
            continue
        first_trace = next(iter(by_method.values()))
        differences.append(
            {
                "index": index,
                "gold": first_trace.get("gold"),
                "methods": [summarize_trace(trace) for _, trace in sorted(by_method.items())],
            }
        )
    return differences


def summarize_trace(trace: dict[str, Any]) -> dict[str, Any]:
    method_trace = trace.get("method_trace") if isinstance(trace.get("method_trace"), dict) else {}
    orchestrator_plan = method_trace.get("orchestrator_plan") if isinstance(method_trace.get("orchestrator_plan"), dict) else {}
    selected_agents = orchestrator_plan.get("selected_agents") or method_trace.get("selected_agents") or []
    agent_names = [str(agent.get("name")) for agent in selected_agents if isinstance(agent, dict) and agent.get("name")]

    event_log = method_trace.get("event_log") if isinstance(method_trace.get("event_log"), list) else []
    event_counts = Counter(str(event.get("event_type")) for event in event_log if isinstance(event, dict) and event.get("event_type"))

    agent_traces = method_trace.get("agent_traces") if isinstance(method_trace.get("agent_traces"), dict) else {}
    used_events = 0
    accepted_receipts = 0
    for agent_trace in agent_traces.values():
        if not isinstance(agent_trace, dict):
            continue
        for step in agent_trace.get("steps", []):
            if isinstance(step, dict):
                used_events += len(step.get("used_event_ids", []) or [])
        for receipt in agent_trace.get("event_receipts", []):
            if isinstance(receipt, dict) and receipt.get("accepted") is True:
                accepted_receipts += 1

    return {
        "method": trace.get("method"),
        "pred": trace.get("pred"),
        "correct": trace.get("correct"),
        "elapsed_seconds": trace.get("elapsed_seconds"),
        "total_tokens": (trace.get("token_usage") or {}).get("total_tokens"),
        "agents": agent_names,
        "event_counts": dict(sorted(event_counts.items())),
        "used_events": used_events,
        "accepted_receipts": accepted_receipts,
        "path": trace.get("_path", ""),
    }


def format_differences(differences: list[dict[str, Any]]) -> str:
    if not differences:
        return "No method differences found."

    lines: list[str] = []
    for difference in differences:
        lines.append(f"Index {difference['index']} | gold={difference.get('gold')}")
        for method in difference["methods"]:
            agents = ",".join(method["agents"]) or "none"
            event_counts = ",".join(f"{key}:{value}" for key, value in method["event_counts"].items()) or "none"
            lines.append(
                "  "
                f"{method['method']}: pred={method['pred']} correct={method['correct']} "
                f"tokens={method['total_tokens']} elapsed={method['elapsed_seconds']} "
                f"agents={agents} events={event_counts} used_events={method['used_events']} "
                f"accepted_receipts={method['accepted_receipts']}"
            )
            if method["path"]:
                lines.append(f"    trace={method['path']}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare evaluation JSON traces across methods.")
    parser.add_argument("trace_path", type=Path, help="Path to a JSON trace file, examples directory, or run trace directory.")
    parser.add_argument("--methods", nargs="*", default=None, help="Optional method names to compare.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    traces = load_trace_files(args.trace_path)
    differences = find_method_differences(traces, methods=args.methods)
    print(format_differences(differences))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
