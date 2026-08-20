from __future__ import annotations

import argparse
import asyncio
import random
import time
from pathlib import Path
from typing import Any

from multi_agent_sync.evaluation import runner
from multi_agent_sync.agents.base import DEFAULT_AGENT_RUNTIME_TIMEOUT_SECONDS
from multi_agent_sync.evaluation.benchmarks import chess, gpqa, gsm8k, hotpotqa, ma_proofbench, mmlu_pro, olymmath, swe_bench_verified
from multi_agent_sync.evaluation import swebench_harness
from multi_agent_sync.evaluation.resume_helpers import resolve_resume_output_path, load_resume_results
from multi_agent_sync.evaluation.kimina_docker import (
    DEFAULT_KIMINA_DOCKER_CONTAINER,
    DEFAULT_KIMINA_DOCKER_IMAGE,
    DEFAULT_KIMINA_CONTAINER_PORT,
    KiminaDockerServer,
)
from multi_agent_sync.evaluation.types import BenchmarkSpec
from multi_agent_sync.llm import get_llm
from multi_agent_sync.vllm_server import (
    add_spinup_server_arguments,
    resolve_server_config,
    should_spinup_server,
    spinup_server,
)

DEFAULT_LIMIT = 0
RANDOM_SEED = 42
DEFAULT_METHODS = "multiagent_streaming,multiagent_no_streaming,plain_llm"
EVALUATION_MAX_TOKENS = 16384
LEAN_VERIFIER_BENCHMARKS = {"ma_proofbench", "olymmath_lean"}


def get_benchmarks() -> dict[str, BenchmarkSpec]:
    benchmarks = [
        hotpotqa.build_benchmark(),
        gpqa.build_benchmark(),
        gsm8k.build_benchmark(),
        chess.build_benchmark(),
        ma_proofbench.build_benchmark(),
        mmlu_pro.build_benchmark(),
        olymmath.build_benchmark(),
        olymmath.build_lean_benchmark(),
        swe_bench_verified.build_benchmark(),
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
            "multiagent_dynamic_no_streaming, dynamic_orchestration, multiagent_debate, "
            "majority_vote, single_agent, plain_llm. "
            "Legacy alias: multiagent."
        ),
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of examples to evaluate. Defaults to 0 for full split.")
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Number of candidate attempts per example. Currently defaults to Pass@1-style evaluation.",
    )
    parser.add_argument("--output-dir", default=str(runner.DEFAULT_OUTPUT_DIR), help="Directory for benchmark result CSV files.")
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
        "--kimina-docker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Start a Kimina Lean Server Docker container for Lean benchmarks if the configured host/port is unavailable. Enabled by default for Lean benchmarks.",
    )
    parser.add_argument(
        "--kimina-docker-image",
        default=DEFAULT_KIMINA_DOCKER_IMAGE,
        help="Docker image used when --kimina-docker is enabled.",
    )
    parser.add_argument(
        "--kimina-docker-container",
        default=DEFAULT_KIMINA_DOCKER_CONTAINER,
        help="Container name used when --kimina-docker is enabled.",
    )
    parser.add_argument(
        "--kimina-docker-startup-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for the Kimina Docker server to become reachable.",
    )
    parser.add_argument(
        "--kimina-docker-cleanup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop and remove a Kimina Docker container started by this evaluation run when the run exits.",
    )
    parser.add_argument(
        "--lean-agent-workspace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Docker workspace editing plus Kimina feedback for Lean multi-agent methods.",
    )
    parser.add_argument(
        "--swebench-agent-workspace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Docker workspace editing and git diff export for SWE-bench multi-agent methods.",
    )
    parser.add_argument(
        "--swebench-run-harness",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the official SWE-bench Docker harness after writing prediction JSONL files.",
    )
    parser.add_argument(
        "--swebench-max-workers",
        type=int,
        default=1,
        help="Maximum workers for official SWE-bench harness evaluation.",
    )
    parser.add_argument(
        "--swebench-run-id",
        default=None,
        help="Optional run_id passed to the official SWE-bench harness.",
    )
    parser.add_argument(
        "--swebench-namespace",
        default=None,
        help="Optional Docker image namespace passed to SWE-bench harness. Use empty string on Mac ARM to build locally.",
    )
    parser.add_argument(
        "--swebench-instance-ids",
        default="",
        help="Optional comma-separated instance ids passed to SWE-bench harness.",
    )
    parser.add_argument(
        "--workspace-image",
        default="python:3.12",
        help="Docker image for benchmark agent workspaces when enabled.",
    )
    parser.add_argument(
        "--workspace-command-timeout",
        type=float,
        default=60.0,
        help="Per-command timeout in seconds for Docker benchmark workspaces.",
    )
    parser.add_argument(
        "--workspace-output-limit",
        type=int,
        default=12000,
        help="Maximum stdout/stderr characters retained per Docker workspace command.",
    )
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum inference steps per agent for multiagent runs.")
    parser.add_argument(
        "--max-orchestrator-rounds",
        type=int,
        default=3,
        help="Maximum controller rounds for dynamic_orchestration.",
    )
    parser.add_argument(
        "--allow-agent-early-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow multiagent subagents to stop before --max-steps when they return FINAL. Disabled by default.",
    )
    parser.add_argument(
        "--single-agent-min-steps",
        type=int,
        default=3,
        help="Minimum inference steps for single_agent and majority_vote single-agent voters.",
    )
    parser.add_argument(
        "--single-agent-max-steps",
        type=int,
        default=7,
        help="Maximum inference steps for single_agent and majority_vote single-agent voters.",
    )
    parser.add_argument(
        "--debate-rounds",
        type=int,
        default=3,
        help="Number of rounds for multiagent_debate. Defaults to 3.",
    )
    parser.add_argument(
        "--total-runtime-timeout",
        type=float,
        default=1800.0,
        help="Maximum total runtime per multiagent example, in seconds.",
    )
    parser.add_argument(
        "--agent-runtime-timeout",
        type=float,
        default=DEFAULT_AGENT_RUNTIME_TIMEOUT_SECONDS,
        help=f"Maximum runtime per multiagent subagent, in seconds. Defaults to {DEFAULT_AGENT_RUNTIME_TIMEOUT_SECONDS:g}.",
    )
    parser.add_argument("--synthesis-timeout", type=float, default=60.0, help="Maximum summarizer runtime, in seconds.")
    parser.add_argument("--max-tokens", type=int, default=EVALUATION_MAX_TOKENS, help="Maximum completion tokens per LLM response.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for answer shuffling.")
    add_spinup_server_arguments(parser)
    return parser


async def run_evaluation(args: argparse.Namespace) -> list[dict[str, Any]]:
    benchmark = get_benchmarks()[args.benchmark]
    methods = runner.parse_methods(args.methods)
    kimina_server = await ensure_kimina_server_ready(benchmark, args)
    try:
        return await run_evaluation_body(args, benchmark, methods)
    finally:
        if kimina_server is not None and getattr(args, "kimina_docker_cleanup", False):
            await kimina_server.cleanup()


async def run_evaluation_body(args: argparse.Namespace, benchmark: BenchmarkSpec, methods: list[str]) -> list[dict[str, Any]]:
    setattr(args, "answer_extractor", benchmark.extract_answer)
    setattr(args, "answer_formatter", benchmark.format_answer)
    items = benchmark.load_items(args)
    rng = random.Random(args.seed)
    resolved_model = runner.resolve_model_name(args)
    setattr(args, "resolved_model", resolved_model)
    llm = get_llm(resolved_model, openai=not getattr(args, "local_model", False), max_tokens=args.max_tokens)
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
                workflow_config = build_method_workflow_config(benchmark, row_dict, method, args)
                if workflow_config is None:
                    run_result = await runner.run_method(method, prompt, llm, args)
                else:
                    run_result = await runner.run_method(method, prompt, llm, args, workflow_config=workflow_config)
                raw_output = run_result.raw_output
                returncode = run_result.returncode
                elapsed_seconds = run_result.elapsed_seconds
                prompt_tokens = run_result.prompt_tokens
                completion_tokens = run_result.completion_tokens
                total_tokens = run_result.total_tokens
                method_trace = run_result.trace or {}
                error = ""
                workspace_trace = method_trace.get("workspace")
                if returncode != 0 and isinstance(workspace_trace, dict) and workspace_trace.get("export_error"):
                    error = str(workspace_trace["export_error"])
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

    postprocess_swebench_predictions(benchmark, results, methods, args, run_id)
    flush_incremental_artifacts()
    summary = runner.summarize_results(results)
    runner.print_summary(benchmark, summary, output_path)
    return results





async def ensure_kimina_server_ready(benchmark: BenchmarkSpec, args: argparse.Namespace) -> KiminaDockerServer | None:
    if benchmark.name not in LEAN_VERIFIER_BENCHMARKS:
        return None

    server = build_kimina_docker_server(args)
    if getattr(args, "kimina_docker", False):
        await server.ensure_running()
        return server

    if await server.is_available():
        return None

    raise RuntimeError(
        f"Kimina Lean Server is not reachable at http://{server.host}:{server.host_port}. "
        f"Start it manually, or rerun with --kimina-docker --kimina-port {server.host_port}. "
        "Preflight stopped before any LLM calls."
    )


def build_kimina_docker_server(args: argparse.Namespace) -> KiminaDockerServer:
    server = KiminaDockerServer(
        host=str(getattr(args, "kimina_host", "127.0.0.1")),
        host_port=int(getattr(args, "kimina_port", 8001)),
        image=str(getattr(args, "kimina_docker_image", DEFAULT_KIMINA_DOCKER_IMAGE)),
        container_name=str(getattr(args, "kimina_docker_container", DEFAULT_KIMINA_DOCKER_CONTAINER)),
        container_port=DEFAULT_KIMINA_CONTAINER_PORT,
        startup_timeout=float(getattr(args, "kimina_docker_startup_timeout", 120.0) or 120.0),
    )
    return server


def postprocess_swebench_predictions(
    benchmark: BenchmarkSpec,
    results: list[dict[str, Any]],
    methods: list[str],
    args: argparse.Namespace,
    run_id: str,
) -> None:
    if benchmark.name != "swe_bench_verified":
        return

    trace_root = runner.resolve_json_trace_root(args, run_id)
    model_name = runner.resolve_model_name(args)
    for method in methods:
        predictions = swebench_harness.build_predictions(results, method=method, model_name_or_path=model_name)
        if not predictions:
            continue
        predictions_path = trace_root / f"predictions_{runner.safe_filename(method)}.jsonl"
        swebench_harness.write_predictions_jsonl(predictions_path, predictions)
        if getattr(args, "swebench_run_harness", False):
            command = swebench_harness.build_run_evaluation_command(predictions_path, args)
            artifact_path = trace_root / f"harness_{runner.safe_filename(method)}.json"
            payload = swebench_harness.run_official_evaluation(command, artifact_path)
            report, report_path = swebench_harness.load_report_from_payload(payload, artifact_path)
            apply_swebench_harness_results(
                results,
                method=method,
                report=report,
                report_path=report_path,
                artifact_path=artifact_path,
                stdout=str(payload.get("stdout") or ""),
            )


def apply_swebench_harness_results(
    results: list[dict[str, Any]],
    *,
    method: str,
    report: dict[str, Any] | None,
    report_path: Any,
    artifact_path: Any,
    stdout: str,
) -> None:
    for result in results:
        if result.get("benchmark") != "swe_bench_verified" or result.get("method") != method:
            continue
        metadata = result.get("score_metadata")
        if not isinstance(metadata, dict):
            continue
        metadata["official_evaluation"] = "report_missing" if report is None else "not_submitted"
        metadata["harness_artifact_path"] = str(artifact_path)
        if report_path is not None:
            metadata["harness_report_path"] = str(report_path)
        if report is None:
            result["correct"] = False
            result["error"] = "Official SWE-bench harness ran, but no report JSON was found."
            result["score_metadata"] = metadata
            update_swebench_trace_result(result)
            continue

        instance_id = str(metadata.get("instance_id") or "")
        status = swebench_harness.instance_official_status(report, instance_id)
        metadata["official_evaluation"] = status or "not_submitted"
        result["score_metadata"] = metadata
        if status == "resolved":
            result["correct"] = True
            result["error"] = ""
            result["returncode"] = 0
        elif status == "unresolved":
            result["correct"] = False
            result["error"] = "Official SWE-bench harness marked this instance unresolved."
            result["returncode"] = 0
        elif status == "empty_patch":
            result["correct"] = False
            result["pred"] = None
            result["error"] = "Official SWE-bench harness received an empty patch."
            result["returncode"] = 1
        elif status == "error":
            detail = swebench_harness.extract_instance_error(stdout, instance_id)
            result["correct"] = False
            result["error"] = detail or "Official SWE-bench harness reported an evaluation error."
            result["returncode"] = 1
        else:
            result["correct"] = False
            result["error"] = "Official SWE-bench harness did not submit this instance."
            result["returncode"] = 1
        update_swebench_trace_result(result)


def update_swebench_trace_result(result: dict[str, Any]) -> None:
    trace_path = str(result.get("json_trace_path") or "")
    if not trace_path:
        return
    path = Path(trace_path)
    if not path.exists():
        return
    payload = runner.to_jsonable(result)
    trace_payload = runner.json.loads(path.read_text(encoding="utf-8"))
    if isinstance(trace_payload, dict):
        for key in ("pred", "correct", "returncode", "error", "score_metadata"):
            trace_payload[key] = payload.get(key)
        path.write_text(runner.json.dumps(trace_payload, indent=2, sort_keys=True), encoding="utf-8")


def build_method_workflow_config(benchmark: BenchmarkSpec, row: dict[str, Any], method: str, args: argparse.Namespace):
    if method in {"plain_llm", "single_agent"}:
        return None
    workspace_enabled = (
        getattr(args, "lean_agent_workspace", False)
        or (benchmark.name == "swe_bench_verified" and getattr(args, "swebench_agent_workspace", False))
    )
    if not workspace_enabled:
        return None
    if benchmark.build_workflow_config is None:
        return None
    return benchmark.build_workflow_config(row, args)


async def async_main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    vllm_server = None
    try:
        if should_spinup_server(args):
            vllm_server = spinup_server(resolve_server_config(args), timeout=args.server_startup_timeout)
            args.model = "auto"
        await run_evaluation(args)
    except RuntimeError as exc:
        parser.exit(1, f"error: {exc}\n")
    finally:
        if vllm_server is not None:
            vllm_server.stop()


def main(argv: list[str] | None = None) -> None:
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main()
