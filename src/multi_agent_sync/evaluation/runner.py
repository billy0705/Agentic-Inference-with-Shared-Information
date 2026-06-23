from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from multi_agent_sync.evaluation.types import BenchmarkSpec
from multi_agent_sync.graph.workflow import run_workflow


DEFAULT_OUTPUT_DIR = Path("output")
VALID_METHODS = {"multiagent", "multiagent_streaming", "multiagent_no_streaming", "plain_llm"}


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


def resolve_output_path(benchmark: BenchmarkSpec, args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir)
    if args.output is None:
        return output_dir / benchmark.default_output_filename

    output_path = Path(args.output)
    if output_path.is_absolute() or output_path.parent != Path("."):
        return output_path
    return output_dir / output_path


async def run_plain_llm(prompt: str, llm: Any) -> tuple[str, int]:
    response = await llm.ainvoke(prompt)
    return getattr(response, "content", str(response)), 0


async def run_multiagent(
    prompt: str,
    llm: Any,
    args: argparse.Namespace,
    *,
    enable_agent_message_streaming: bool = True,
) -> tuple[str, int]:
    state = await run_workflow(
        task=prompt,
        llm=llm,
        max_steps_per_agent=args.max_steps,
        total_runtime_timeout=args.total_runtime_timeout,
        synthesis_timeout=args.synthesis_timeout,
        enable_agent_message_streaming=enable_agent_message_streaming,
        stream_to_console=False,
        no_color=True,
    )
    return state["final_answer"], 0


async def run_method(method: str, prompt: str, llm: Any, args: argparse.Namespace) -> tuple[str, int]:
    if method == "plain_llm":
        return await run_plain_llm(prompt, llm)
    if method in {"multiagent", "multiagent_streaming"}:
        return await run_multiagent(prompt, llm, args, enable_agent_message_streaming=True)
    if method == "multiagent_no_streaming":
        return await run_multiagent(prompt, llm, args, enable_agent_message_streaming=False)
    raise ValueError(f"Unknown method: {method}")


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
            fieldnames=["benchmark", "method", "index", "gold", "pred", "correct", "returncode", "error", "raw_output"],
        )
        writer.writeheader()
        writer.writerows(results)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["method"]].append(result)

    summary: dict[str, dict[str, float | int]] = {}
    for method, method_results in grouped.items():
        total = len(method_results)
        correct_count = sum(1 for result in method_results if result["correct"])
        invalid_count = sum(1 for result in method_results if result["pred"] is None)
        summary[method] = {
            "total": total,
            "correct": correct_count,
            "accuracy": correct_count / total if total else 0.0,
            "invalid": invalid_count,
            "invalid_rate": invalid_count / total if total else 0.0,
        }
    return summary


def print_result(result: dict[str, Any]) -> None:
    print()
    print("=" * 80)
    print(f"Benchmark: {result['benchmark']}")
    print(f"Method: {result['method']}")
    print(f"Question: {result['index']}")
    print(f"Gold: {result['gold']}")
    print(f"Pred: {result['pred']}")
    print(f"Correct: {result['correct']}")
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
    print()
    print(f"Saved results to: {output_path}")
