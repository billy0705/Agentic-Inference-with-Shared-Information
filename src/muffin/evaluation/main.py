import argparse
import asyncio
import random
import time
from pathlib import Path
from typing import Any

from muffin.evaluation import runner
from muffin.evaluation.benchmarks import chess, gpqa, mmlu_pro, olymmath
from muffin.evaluation.resume_helpers import load_resume_results, resolve_resume_output_path
from muffin.evaluation.types import BenchmarkSpec
from muffin.llm import get_llm

def get_benchmarks() -> dict[str, BenchmarkSpec]:
    benchmarks = [
        gpqa.build_benchmark(),
        chess.build_benchmark(),
        mmlu_pro.build_benchmark(),
        olymmath.build_benchmark(),
    ]
    return {benchmark.name: benchmark for benchmark in benchmarks}


def build_parser() -> argparse.ArgumentParser:
    benchmarks = get_benchmarks()
    parser = argparse.ArgumentParser(
        description="Run Muffin evaluation benchmarks through an OpenAI-compatible API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark", required=True, choices=sorted(benchmarks), help="Benchmark to run.")
    parser.add_argument(
        "--methods",
        required=True,
        help=(
            "Comma-separated methods to compare. Supported: "
            "multiagent_dynamic_streaming, multiagent_debate, majority_vote, single_agent."
        ),
    )
    parser.add_argument("--limit", type=int, default=100, help="Number of examples to evaluate; use 0 for the full split.")
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Number of candidate attempts per example.",
    )
    parser.add_argument("--output-dir", default="output", help="Directory for benchmark result CSV files.")
    parser.add_argument("--output", default=None, help="Optional CSV filename or path for per-example results.")
    parser.add_argument(
        "--resume-run",
        default=None,
        help="Existing run output directory to continue by skipping completed JSON traces.",
    )
    parser.add_argument(
        "--data-file",
        default=None,
        help="Optional local benchmark data file. Use benchmark-shaped CSV, JSONL, or NDJSON rows.",
    )
    parser.add_argument(
        "--save-json-traces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable per-example JSON traces and run config artifacts.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use. If omitted, detect it from the API's /models endpoint.",
    )
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help=(
            "OpenAI-compatible API URL. If omitted, use OPENAI_BASE_URL or "
            "http://localhost:8000/v1."
        ),
    )
    parser.add_argument(
        "--olymmath-subset",
        choices=["en-easy", "en-hard"],
        help="OlymMATH natural-language subset to run; required with --benchmark olymmath.",
    )
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum inference steps per agent for multiagent runs.")
    parser.add_argument(
        "--allow-agent-early-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow agents to stop before --max-steps.",
    )
    parser.add_argument(
        "--think-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable the <|think|> prompt prefix.",
    )
    parser.add_argument(
        "--min-dynamic-subagents",
        type=int,
        default=3,
        help="Minimum dynamic subagents to create.",
    )
    parser.add_argument(
        "--max-dynamic-subagents",
        type=int,
        default=3,
        help="Maximum dynamic subagents to create.",
    )
    parser.add_argument(
        "--single-agent-min-steps",
        type=int,
        default=5,
        help="Minimum inference steps for single_agent and majority_vote single-agent voters.",
    )
    parser.add_argument(
        "--single-agent-max-steps",
        type=int,
        default=5,
        help="Maximum inference steps for single_agent and majority_vote single-agent voters.",
    )
    parser.add_argument(
        "--debate-rounds",
        type=int,
        default=5,
        help="Number of rounds for multiagent_debate.",
    )
    parser.add_argument(
        "--total-runtime-timeout",
        type=float,
        default=1800,
        help="Maximum total runtime per multiagent example, in seconds.",
    )
    parser.add_argument(
        "--agent-runtime-timeout",
        type=float,
        default=600,
        help="Maximum runtime per multiagent subagent, in seconds.",
    )
    parser.add_argument("--synthesis-timeout", type=float, default=60, help="Maximum summarizer runtime, in seconds.")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Maximum completion tokens per LLM response.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for answer shuffling.")
    return parser


async def run_evaluation(args: argparse.Namespace) -> list[dict[str, Any]]:
    benchmark = get_benchmarks()[args.benchmark]
    if benchmark.name == "olymmath" and args.olymmath_subset is None:
        raise ValueError("--olymmath-subset is required with --benchmark olymmath")
    methods = runner.parse_methods(args.methods)
    return await run_evaluation_body(args, benchmark, methods)


async def run_evaluation_body(args: argparse.Namespace, benchmark: BenchmarkSpec, methods: list[str]) -> list[dict[str, Any]]:
    setattr(args, "answer_extractor", benchmark.extract_answer)
    setattr(args, "answer_formatter", benchmark.format_answer)
    items = benchmark.load_items(args)
    rng = random.Random(args.seed)
    resolved_model = runner.resolve_model_name(args)
    setattr(args, "resolved_model", resolved_model)
    llm = get_llm(
        resolved_model,
        max_tokens=args.max_tokens,
        base_url=getattr(args, "openai_base_url", None),
    )
    resume_root = Path(args.resume_run) if getattr(args, "resume_run", None) else None
    run_id = resume_root.name if resume_root is not None else runner.create_run_id()
    output_path = resolve_resume_output_path(resume_root, benchmark, args, run_id)
    trace_root = resume_root or runner.resolve_json_trace_root(args, run_id)
    run_config = runner.build_run_config(benchmark, args, methods, run_id=run_id, output_path=output_path)
    if args.save_json_traces:
        runner.write_json_file(trace_root / "run_config.json", run_config)
    results = load_resume_results(resume_root, benchmark.name, methods) if resume_root is not None else []
    completed = {(int(result["index"]), str(result["method"])) for result in results if not result.get("error")}

    def flush_incremental_artifacts() -> None:
        runner.write_results_csv(output_path, results)
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
            if (idx, method) in completed:
                continue
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
                trace_path = trace_root / "examples" / f"{idx:04d}_{runner.safe_filename(method)}.json"
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
            completed.add((idx, method))
            flush_incremental_artifacts()
            runner.print_result(result)

    flush_incremental_artifacts()
    summary = runner.summarize_results(results)
    runner.print_summary(benchmark, summary, output_path)
    return results

async def async_main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        await run_evaluation(args)
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
