from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from multi_agent_sync.evaluation.types import BenchmarkSpec
from multi_agent_sync.graph.workflow import run_workflow


DEFAULT_OUTPUT_DIR = Path("output")
VALID_METHODS = {
    "multiagent",
    "multiagent_streaming",
    "multiagent_no_streaming",
    "multiagent_dynamic_streaming",
    "multiagent_dynamic_no_streaming",
    "plain_llm",
}


@dataclass
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens = add_optional_ints(self.prompt_tokens, other.prompt_tokens)
        self.completion_tokens = add_optional_ints(self.completion_tokens, other.completion_tokens)
        self.total_tokens = add_optional_ints(self.total_tokens, other.total_tokens)


@dataclass
class RunResult:
    raw_output: str
    returncode: int
    elapsed_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    trace: dict[str, Any] | None = None


class MeteredLLM:
    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self.usage = TokenUsage()

    async def ainvoke(self, prompt: str) -> Any:
        response = await self._llm.ainvoke(prompt)
        self.usage.add(extract_token_usage(response))
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def add_optional_ints(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_token_usage(response: Any) -> TokenUsage:
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}

    prompt_tokens = coerce_int(
        usage_metadata.get("input_tokens")
        or usage_metadata.get("prompt_tokens")
        or token_usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
    )
    completion_tokens = coerce_int(
        usage_metadata.get("output_tokens")
        or usage_metadata.get("completion_tokens")
        or token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
    )
    total_tokens = coerce_int(usage_metadata.get("total_tokens") or token_usage.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def parse_methods(methods: str) -> list[str]:
    parsed = [method.strip() for method in methods.split(",") if method.strip()]
    if not parsed:
        raise ValueError("At least one method is required.")

    unknown = [method for method in parsed if method not in VALID_METHODS]
    if unknown:
        supported = ", ".join(sorted(VALID_METHODS))
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}. Supported methods: {supported}.")

    deduped: list[str] = []
    for method in parsed:
        if method not in deduped:
            deduped.append(method)
    return deduped


def create_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def resolve_output_path(benchmark: BenchmarkSpec, args: argparse.Namespace, run_id: str | None = None) -> Path:
    output_dir = Path(args.output_dir)
    if args.output is None:
        default_output = Path(benchmark.default_output_filename)
        suffix = default_output.suffix or ".csv"
        return output_dir / f"{default_output.stem}_{run_id or create_run_id()}{suffix}"

    output_path = Path(args.output)
    if output_path.is_absolute() or output_path.parent != Path("."):
        return output_path
    return output_dir / output_path


def resolve_correctness_matrix_path(trace_root: Path) -> Path:
    return trace_root / "correctness.csv"


def resolve_json_trace_root(args: argparse.Namespace, run_id: str) -> Path:
    return Path(args.output_dir) / "json_traces" / run_id


def resolve_example_trace_path(args: argparse.Namespace, run_id: str, index: int, method: str) -> Path:
    return resolve_json_trace_root(args, run_id) / "examples" / f"{index:04d}_{safe_filename(method)}.json"


def safe_filename(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return normalized.strip("._") or "trace"


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [to_jsonable(item) for item in value]
    return str(value)


def build_run_config(
    benchmark: BenchmarkSpec,
    args: argparse.Namespace,
    methods: list[str],
    *,
    run_id: str,
    output_path: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "benchmark": benchmark.name,
        "benchmark_display_name": benchmark.display_name,
        "methods": methods,
        "output_path": str(output_path),
        "settings": {
            "model": args.model,
            "resolved_model": resolve_model_name(args),
            "local_model": args.local_model,
            "limit": args.limit,
            "max_steps": args.max_steps,
            "total_runtime_timeout": args.total_runtime_timeout,
            "synthesis_timeout": args.synthesis_timeout,
            "seed": args.seed,
            "data_file": args.data_file,
            "save_json_traces": args.save_json_traces,
        },
    }


def build_method_settings(method: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": method,
        "model": args.model,
        "resolved_model": resolve_model_name(args),
        "local_model": args.local_model,
        "subagent_mode": method_subagent_mode(method),
        "message_streaming": method_message_streaming(method),
        "max_steps": args.max_steps,
        "total_runtime_timeout": args.total_runtime_timeout,
        "synthesis_timeout": args.synthesis_timeout,
        "save_json_traces": args.save_json_traces,
    }


def resolve_model_name(args: argparse.Namespace) -> str:
    if args.model:
        return args.model
    if args.local_model:
        return os.getenv("OLLAMA_MODEL", "qwen3:4b")
    return os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b")


def method_subagent_mode(method: str) -> str:
    if method == "plain_llm":
        return "none"
    return "dynamic" if "dynamic" in method else "fixed"


def method_message_streaming(method: str) -> bool | None:
    if method == "plain_llm":
        return None
    return "no_streaming" not in method


def build_question_context(row: dict[str, Any]) -> dict[str, Any]:
    incorrect_answers = [
        row[field]
        for field in ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
        if row.get(field)
    ]
    options: dict[str, Any] = {}
    if row.get("Correct Answer") is not None:
        options["correct"] = row.get("Correct Answer")
    if incorrect_answers:
        options["incorrect"] = incorrect_answers
    return {
        "question": row.get("Question", ""),
        "options": options,
    }


def build_multiagent_debug(method_trace: dict[str, Any]) -> dict[str, Any]:
    selected_agents = extract_selected_subagents(method_trace)
    return {
        "subagents": selected_agents,
        "messages": extract_agent_messages(method_trace),
        "workflow": build_debug_workflow(selected_agents, method_trace),
    }


def extract_selected_subagents(method_trace: dict[str, Any]) -> list[dict[str, Any]]:
    orchestrator_plan = method_trace.get("orchestrator_plan")
    if not isinstance(orchestrator_plan, dict):
        return []
    selected_agents = orchestrator_plan.get("selected_agents")
    if not isinstance(selected_agents, list):
        return []

    subagents: list[dict[str, Any]] = []
    for agent in selected_agents:
        if not isinstance(agent, dict) or not agent.get("name"):
            continue
        compact_agent = {
            key: agent[key]
            for key in ("name", "role", "description", "subtask", "expected_output", "critical_debate")
            if key in agent
        }
        subagents.append(compact_agent)
    return subagents


def extract_agent_messages(method_trace: dict[str, Any]) -> list[dict[str, Any]]:
    event_log = method_trace.get("event_log")
    if not isinstance(event_log, list):
        return []

    messages: list[dict[str, Any]] = []
    for event in event_log:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        if event_type not in {"finding", "critique", "warning", "question", "message_received", "agent_done"}:
            continue
        messages.append(
            {
                "source": event.get("source"),
                "target": event.get("target") or "broadcast",
                "event_type": event_type,
                "content": event.get("content", ""),
                "confidence": event.get("confidence"),
            }
        )
    return messages


def build_debug_workflow(selected_agents: list[dict[str, Any]], method_trace: dict[str, Any]) -> list[dict[str, str]]:
    if not selected_agents:
        if method_trace.get("prompt") is not None:
            return [
                {"node": "plain_llm", "description": "Answered the benchmark prompt directly."},
            ]
        return []

    workflow = [{"node": "orchestrator", "description": "Created plan and selected subagents."}]
    workflow.extend(
        {
            "node": str(agent["name"]),
            "description": "Ran subagent and published/received messages.",
        }
        for agent in selected_agents
    )
    workflow.append({"node": "synthesizer", "description": "Combined subagent outputs and event log into final answer."})
    return workflow


async def run_plain_llm(prompt: str, llm: Any) -> tuple[str, int]:
    response = await llm.ainvoke(prompt)
    return getattr(response, "content", str(response)), 0


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


async def run_method(method: str, prompt: str, llm: Any, args: argparse.Namespace) -> RunResult:
    metered_llm = MeteredLLM(llm)
    started_at = time.perf_counter()
    trace: dict[str, Any] | None = None
    if method == "plain_llm":
        raw_output, returncode = await run_plain_llm(prompt, metered_llm)
        trace = {"prompt": prompt, "raw_output": raw_output}
    elif method in {"multiagent", "multiagent_streaming"}:
        raw_output, returncode, trace = await run_multiagent(
            prompt,
            metered_llm,
            args,
            enable_agent_message_streaming=True,
            subagent_mode="fixed",
        )
    elif method == "multiagent_no_streaming":
        raw_output, returncode, trace = await run_multiagent(
            prompt,
            metered_llm,
            args,
            enable_agent_message_streaming=False,
            subagent_mode="fixed",
        )
    elif method == "multiagent_dynamic_streaming":
        raw_output, returncode, trace = await run_multiagent(
            prompt,
            metered_llm,
            args,
            enable_agent_message_streaming=True,
            subagent_mode="dynamic",
        )
    elif method == "multiagent_dynamic_no_streaming":
        raw_output, returncode, trace = await run_multiagent(
            prompt,
            metered_llm,
            args,
            enable_agent_message_streaming=False,
            subagent_mode="dynamic",
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed_seconds = time.perf_counter() - started_at
    return RunResult(
        raw_output=raw_output,
        returncode=returncode,
        elapsed_seconds=elapsed_seconds,
        prompt_tokens=metered_llm.usage.prompt_tokens,
        completion_tokens=metered_llm.usage.completion_tokens,
        total_tokens=metered_llm.usage.total_tokens,
        trace=trace,
    )


def progress(items: Any, desc: str) -> Any:
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return items
    return tqdm(items, desc=desc)


def write_results_csv(output_path: Path, results: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark",
                "method",
                "index",
                "gold",
                "pred",
                "correct",
                "returncode",
                "error",
                "elapsed_seconds",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "raw_output",
                "json_trace_path",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def write_correctness_matrix_csv(trace_root: Path, results: list[dict[str, Any]], methods: list[str]) -> None:
    matrix_path = resolve_correctness_matrix_path(trace_root)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, dict[str, str]] = defaultdict(dict)
    for result in results:
        grouped[int(result["index"])][str(result["method"])] = "T" if result.get("correct") else "F"

    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", *methods])
        writer.writeheader()
        for task_id in sorted(grouped):
            row = {"task_id": str(task_id)}
            row.update({method: grouped[task_id].get(method, "") for method in methods})
            writer.writerow(row)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["method"]].append(result)

    summary: dict[str, dict[str, float | int]] = {}
    for method, method_results in grouped.items():
        total = len(method_results)
        correct_count = sum(1 for result in method_results if result["correct"])
        invalid_count = sum(1 for result in method_results if result["pred"] is None)
        elapsed_seconds = sum(float(result.get("elapsed_seconds") or 0.0) for result in method_results)
        prompt_tokens = sum(int(result.get("prompt_tokens") or 0) for result in method_results)
        completion_tokens = sum(int(result.get("completion_tokens") or 0) for result in method_results)
        total_tokens = sum(int(result.get("total_tokens") or 0) for result in method_results)
        summary[method] = {
            "total": total,
            "correct": correct_count,
            "accuracy": correct_count / total if total else 0.0,
            "invalid": invalid_count,
            "invalid_rate": invalid_count / total if total else 0.0,
            "elapsed_seconds": elapsed_seconds,
            "avg_elapsed_seconds": elapsed_seconds / total if total else 0.0,
            "prompt_tokens": prompt_tokens,
            "avg_prompt_tokens": prompt_tokens / total if total else 0.0,
            "completion_tokens": completion_tokens,
            "avg_completion_tokens": completion_tokens / total if total else 0.0,
            "total_tokens": total_tokens,
            "avg_total_tokens": total_tokens / total if total else 0.0,
        }
    return summary


def build_method_averages(summary: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float]]:
    return {
        method: {
            "correct_avg": float(stats["accuracy"]),
            "time_avg_seconds": float(stats["avg_elapsed_seconds"]),
            "total_token_avg": float(stats["avg_total_tokens"]),
        }
        for method, stats in summary.items()
    }


def print_result(result: dict[str, Any]) -> None:
    print()
    print("=" * 80)
    print(f"Benchmark: {result['benchmark']}")
    print(f"Method: {result['method']}")
    print(f"Question: {result['index']}")
    print(f"Gold: {result['gold']}")
    print(f"Pred: {result['pred']}")
    print(f"Correct: {result['correct']}")
    print(f"Elapsed seconds: {result['elapsed_seconds']:.4f}")
    if result.get("total_tokens") is not None:
        print(
            f"Tokens: prompt={result.get('prompt_tokens')}, "
            f"completion={result.get('completion_tokens')}, total={result.get('total_tokens')}"
        )
    if result["error"]:
        print(f"Error: {result['error']}")
    print("=" * 80)


def print_summary(benchmark: BenchmarkSpec, summary: dict[str, dict[str, float | int]], output_path: Path) -> None:
    print()
    print("Benchmark finished.")
    print(f"Benchmark: {benchmark.display_name}")
    for method, stats in summary.items():
        print()
        print(f"Method: {method}")
        print(f"Total: {stats['total']}")
        print(f"Correct: {stats['correct']}")
        print(f"Accuracy: {stats['accuracy']:.4f}")
        print(f"Invalid answers: {stats['invalid']}")
        print(f"Invalid rate: {stats['invalid_rate']:.4f}")
        print(f"Total time: {stats['elapsed_seconds']:.4f}s")
        print(f"Avg time: {stats['avg_elapsed_seconds']:.4f}s")
        print(f"Prompt tokens: {stats['prompt_tokens']}")
        print(f"Avg prompt tokens: {stats['avg_prompt_tokens']:.2f}")
        print(f"Completion tokens: {stats['completion_tokens']}")
        print(f"Avg completion tokens: {stats['avg_completion_tokens']:.2f}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Avg total tokens: {stats['avg_total_tokens']:.2f}")
    print()
    print(f"Saved results to: {output_path}")
