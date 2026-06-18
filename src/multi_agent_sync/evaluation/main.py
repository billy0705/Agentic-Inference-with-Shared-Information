from __future__ import annotations

import argparse
import asyncio
import random
from typing import Any

from multi_agent_sync.evaluation import gpqa
from multi_agent_sync.evaluation import runner
from multi_agent_sync.evaluation.types import BenchmarkSpec
from multi_agent_sync.llm import get_llm


DEFAULT_LIMIT = 10
RANDOM_SEED = 42


def get_benchmarks() -> dict[str, BenchmarkSpec]:
    benchmark = gpqa.build_benchmark()
    return {benchmark.name: benchmark}


def build_parser() -> argparse.ArgumentParser:
    benchmarks = get_benchmarks()
    parser = argparse.ArgumentParser(description="Run local multi_agent_sync evaluation benchmarks.")
    parser.add_argument("--benchmark", default="gpqa", choices=sorted(benchmarks), help="Benchmark to run.")
    parser.add_argument(
        "--methods",
        default="multiagent,plain_llm",
        help="Comma-separated methods to compare. Supported: multiagent, plain_llm.",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of examples to evaluate. Use 0 for full split.")
    parser.add_argument("--output-dir", default=str(runner.DEFAULT_OUTPUT_DIR), help="Directory for benchmark result CSV files.")
    parser.add_argument("--output", default=None, help="Optional CSV filename or path for per-example results.")
    parser.add_argument(
        "--data-file",
        default=None,
        help="Optional local benchmark data file. For GPQA, use a GPQA-style CSV or JSONL file.",
    )
    parser.add_argument("--model", default=None, help="Model name to pass to the selected provider.")
    parser.add_argument(
        "--local-model",
        action="store_true",
        help="Use local Ollama instead of the OpenAI-compatible API provider.",
    )
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum inference steps per agent for multiagent runs.")
    parser.add_argument(
        "--total-runtime-timeout",
        type=float,
        default=600.0,
        help="Maximum total runtime per multiagent example, in seconds.",
    )
    parser.add_argument("--synthesis-timeout", type=float, default=60.0, help="Maximum synthesizer runtime, in seconds.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for answer shuffling.")
    return parser


async def run_evaluation(args: argparse.Namespace) -> list[dict[str, Any]]:
    benchmark = get_benchmarks()[args.benchmark]
    methods = runner.parse_methods(args.methods)
    items = benchmark.load_items(args)
    rng = random.Random(args.seed)
    llm = get_llm(args.model, openai=not args.local_model)
    output_path = runner.resolve_output_path(benchmark, args)
    results: list[dict[str, Any]] = []

    for idx, row in enumerate(runner.progress(items, desc=f"Benchmarking {benchmark.display_name}")):
        prompt, gold = benchmark.build_prompt(dict(row), rng)
        for method in methods:
            try:
                raw_output, returncode = await runner.run_method(method, prompt, llm, args)
                error = ""
            except Exception as exc:
                raw_output = ""
                returncode = 1
                error = str(exc)

            pred = benchmark.extract_answer(raw_output)
            result = {
                "benchmark": benchmark.name,
                "method": method,
                "index": idx,
                "gold": gold,
                "pred": pred,
                "correct": pred == gold,
                "returncode": returncode,
                "error": error,
                "raw_output": raw_output,
            }
            results.append(result)
            runner.print_result(result)

    runner.write_results_csv(output_path, results)
    runner.print_summary(benchmark, runner.summarize_results(results), output_path)
    return results


async def async_main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        await run_evaluation(args)
    except RuntimeError as exc:
        parser.exit(1, f"error: {exc}\n")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
