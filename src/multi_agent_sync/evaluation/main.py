from __future__ import annotations

import argparse
import asyncio
import random
import time
from typing import Any

from multi_agent_sync.evaluation import gpqa, gsm8k, ma_proofbench, mmlu_pro, olymmath
from multi_agent_sync.evaluation import runner
from multi_agent_sync.evaluation.types import BenchmarkSpec
from multi_agent_sync.llm import get_llm


DEFAULT_LIMIT = 10
RANDOM_SEED = 42
DEFAULT_METHODS = "multiagent_streaming,multiagent_no_streaming,plain_llm"
EVALUATION_MAX_TOKENS = 16384


def get_benchmarks() -> dict[str, BenchmarkSpec]:
    benchmarks = [
        gpqa.build_benchmark(),
        gsm8k.build_benchmark(),
        ma_proofbench.build_benchmark(),
        mmlu_pro.build_benchmark(),
        olymmath.build_benchmark(),
        olymmath.build_lean_benchmark(),
    ]
    return {benchmark.name: benchmark for benchmark in benchmarks}


def build_parser() -> argparse.ArgumentParser:
    benchmarks = get_benchmarks()
    parser = argparse.ArgumentParser(description="Run local multi_agent_sync evaluation benchmarks.")
    parser.add_argument("--benchmark", default="gpqa", choices=sorted(benchmarks), help="Benchmark to run.")
    parser.add_argument(
        "--methods",
        default=DEFAULT_METHODS,
        help=(
            "Comma-separated methods to compare. Supported: multiagent_streaming, "
            "multiagent_no_streaming, multiagent_dynamic_streaming, "
            "multiagent_dynamic_no_streaming, plain_llm. Legacy alias: multiagent."
        ),
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of examples to evaluate. Use 0 for full split.")
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Number of candidate attempts per example. Currently defaults to Pass@1-style evaluation.",
    )
    parser.add_argument("--output-dir", default=str(runner.DEFAULT_OUTPUT_DIR), help="Directory for benchmark result CSV files.")
    parser.add_argument("--output", default=None, help="Optional CSV filename or path for per-example results.")
    parser.add_argument(
        "--data-file",
        default=None,
        help="Optional local benchmark data file. Use benchmark-shaped CSV, JSONL, or NDJSON rows.",
    )
    parser.add_argument(
        "--save-json-traces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save per-example JSON traces and run config artifacts. Enabled by default; use --no-save-json-traces to disable.",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="Model name to pass to the selected provider. Defaults to auto for OpenAI-compatible evaluation.",
    )
    parser.add_argument(
        "--ma-proofbench-level",
        choices=["all", "level1", "level2"],
        default="all",
        help="MA-ProofBench difficulty tier to run. Defaults to all and preserves dataset order.",
    )
    parser.add_argument(
        "--olymmath-subset",
        choices=["en-easy", "en-hard", "zh-easy", "zh-hard"],
        default="en-easy",
        help="OlymMATH natural-language subset to run. Ignored by --benchmark olymmath_lean.",
    )
    parser.add_argument(
        "--lean-timeout",
        type=float,
        default=60.0,
        help="Maximum Lean verifier runtime per MA-ProofBench candidate, in seconds.",
    )
    parser.add_argument(
        "--kimina-host",
        default="127.0.0.1",
        help="Kimina Lean Server host for MA-ProofBench verification.",
    )
    parser.add_argument(
        "--kimina-port",
        type=int,
        default=8001,
        help="Kimina Lean Server port for MA-ProofBench verification.",
    )
    parser.add_argument(
        "--kimina-max-workers",
        type=int,
        default=1,
        help="Maximum Kimina Lean Server workers per verification request.",
    )
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
    resolved_model = runner.resolve_model_name(args)
    setattr(args, "resolved_model", resolved_model)
    llm = get_llm(resolved_model, openai=not args.local_model, max_tokens=EVALUATION_MAX_TOKENS)
    run_id = runner.create_run_id()
    output_path = runner.resolve_output_path(benchmark, args, run_id=run_id)
    run_config = runner.build_run_config(benchmark, args, methods, run_id=run_id, output_path=output_path)
    if args.save_json_traces:
        runner.write_json_file(runner.resolve_json_trace_root(args, run_id) / "run_config.json", run_config)
    results: list[dict[str, Any]] = []

    def flush_incremental_artifacts() -> None:
        runner.write_results_csv(output_path, results)
        trace_root = runner.resolve_json_trace_root(args, run_id)
        runner.write_correctness_matrix_csv(trace_root, results, methods)
        summary = runner.summarize_results(results)
        if args.save_json_traces:
            runner.write_json_file(
                trace_root / "summary.json",
                {
                    "run_config": run_config,
                    "summary": summary,
                    "method_averages": runner.build_method_averages(summary),
                    "results": results,
                },
            )

    for idx, row in enumerate(runner.progress(items, desc=f"Benchmarking {benchmark.display_name}")):
        row_dict = dict(row)
        prompt, gold = benchmark.build_prompt(row_dict, rng)
        question_context = runner.build_question_context(row_dict)
        for method in methods:
            started_at = time.perf_counter()
            method_trace: dict[str, Any] = {}
            try:
                run_result = await runner.run_method(method, prompt, llm, args)
                raw_output = run_result.raw_output
                returncode = run_result.returncode
                elapsed_seconds = run_result.elapsed_seconds
                prompt_tokens = run_result.prompt_tokens
                completion_tokens = run_result.completion_tokens
                total_tokens = run_result.total_tokens
                method_trace = run_result.trace or {}
                error = ""
            except Exception as exc:
                raw_output = ""
                returncode = 1
                elapsed_seconds = time.perf_counter() - started_at
                prompt_tokens = None
                completion_tokens = None
                total_tokens = None
                error = str(exc)
                method_trace = {"error": error}

            score = runner.score_benchmark_response(benchmark, row_dict, raw_output, gold, args)
            result_error = error or score.error
            result_pred = score.pred
            result_metadata = dict(score.metadata)
            artifact_method_trace = dict(method_trace)
            result = {
                "benchmark": benchmark.name,
                "method": method,
                "index": idx,
                "gold": gold,
                "pred": result_pred,
                "correct": score.correct,
                "returncode": returncode,
                "error": result_error,
                "elapsed_seconds": elapsed_seconds,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "raw_output": raw_output,
                "score_metadata": result_metadata,
                "json_trace_path": "",
            }
            if args.save_json_traces:
                trace_path = runner.resolve_example_trace_path(args, run_id, idx, method)
                result["json_trace_path"] = str(trace_path)
                runner.write_json_file(
                    trace_path,
                    {
                        "run_id": run_id,
                        "benchmark": benchmark.name,
                        "benchmark_display_name": benchmark.display_name,
                        "method": method,
                        "index": idx,
                        "gold": gold,
                        "pred": result_pred,
                        "correct": score.correct,
                        "returncode": returncode,
                        "error": result_error,
                        "elapsed_seconds": elapsed_seconds,
                        "token_usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                        },
                        "settings": runner.build_method_settings(method, args),
                        "question": question_context["question"],
                        "options": question_context["options"],
                        "gold_answer": question_context.get("gold_answer", gold),
                        "score_metadata": result_metadata,
                        "prompt": prompt,
                        "raw_output": raw_output,
                        "multiagent_debug": runner.build_multiagent_debug(artifact_method_trace),
                        "method_trace": artifact_method_trace,
                    },
                )
            results.append(result)
            flush_incremental_artifacts()
            runner.print_result(result)

    summary = runner.summarize_results(results)
    runner.print_summary(benchmark, summary, output_path)
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
