from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    subagent_mode: str = "fixed",
) -> tuple[str, int]:
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
    return state["final_answer"], 0


async def run_method(method: str, prompt: str, llm: Any, args: argparse.Namespace) -> RunResult:
    metered_llm = MeteredLLM(llm)
    started_at = time.perf_counter()
    if method == "plain_llm":
        raw_output, returncode = await run_plain_llm(prompt, metered_llm)
    elif method in {"multiagent", "multiagent_streaming"}:
        raw_output, returncode = await run_multiagent(
            prompt,
            metered_llm,
            args,
            enable_agent_message_streaming=True,
            subagent_mode="fixed",
        )
    elif method == "multiagent_no_streaming":
        raw_output, returncode = await run_multiagent(
            prompt,
            metered_llm,
            args,
            enable_agent_message_streaming=False,
            subagent_mode="fixed",
        )
    elif method == "multiagent_dynamic_streaming":
        raw_output, returncode = await run_multiagent(
            prompt,
            metered_llm,
            args,
            enable_agent_message_streaming=True,
            subagent_mode="dynamic",
        )
    elif method == "multiagent_dynamic_no_streaming":
        raw_output, returncode = await run_multiagent(
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
            ],
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
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
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
        print(f"Completion tokens: {stats['completion_tokens']}")
        print(f"Total tokens: {stats['total_tokens']}")
    print()
    print(f"Saved results to: {output_path}")
