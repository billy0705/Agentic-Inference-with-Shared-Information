from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multi_agent_sync.evaluation import main as evaluation
from multi_agent_sync.evaluation import runner
from multi_agent_sync.vllm_server import spinup_server


@dataclass(frozen=True)
class ExperimentConfig:
    benchmarks: tuple[str, ...]
    methods: tuple[str, ...]
    limit: int
    resume_run: Path | None


@dataclass(frozen=True)
class HelmaConfig:
    job_name: str
    nodes: int
    gres: str
    partition: str
    output: str
    time: str

    def slurm_arguments(self) -> tuple[str, ...]:
        return (
            f"--job-name={self.job_name}",
            f"--nodes={self.nodes}",
            f"--gres={self.gres}",
            f"--partition={self.partition}",
            f"--output={self.output}",
            f"--time={self.time}",
        )


def get_config_manager() -> Any:
    try:
        from vllm_server_launcher import ConfigManager
    except ImportError as exc:
        raise RuntimeError(
            "Automatic vLLM startup requires the optional dependency. "
            "Run: uv sync --extra vllm-server"
        ) from exc
    return ConfigManager


def _required_string(section: dict[str, Any], section_name: str, key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{section_name}.{key}' must be a non-empty string")
    return value.strip()


def load_helma_config(manager: Any) -> HelmaConfig:
    helma = manager.section("helma")
    nodes = helma.get("nodes")
    if not isinstance(nodes, int) or isinstance(nodes, bool) or nodes <= 0:
        raise ValueError("'helma.nodes' must be a positive integer")

    return HelmaConfig(
        job_name=_required_string(helma, "helma", "job_name"),
        nodes=nodes,
        gres=_required_string(helma, "helma", "gres"),
        partition=_required_string(helma, "helma", "partition"),
        output=_required_string(helma, "helma", "output"),
        time=_required_string(helma, "helma", "time"),
    )


def load_experiment_config(manager: Any) -> ExperimentConfig:
    expt = manager.section("expt")

    benchmarks = expt.get("benchmark")
    if not isinstance(benchmarks, list) or not benchmarks or not all(
        isinstance(value, str) and value.strip() for value in benchmarks
    ):
        raise ValueError("'expt.benchmark' must be a non-empty list of benchmark names")
    normalized_benchmarks = tuple(dict.fromkeys(value.strip() for value in benchmarks))
    unknown_benchmarks = sorted(set(normalized_benchmarks) - set(evaluation.get_benchmarks()))
    if unknown_benchmarks:
        raise ValueError(f"Unknown benchmark(s): {', '.join(unknown_benchmarks)}")

    methods = expt.get("methods")
    if not isinstance(methods, list) or not methods or not all(
        isinstance(value, str) and value.strip() for value in methods
    ):
        raise ValueError("'expt.methods' must be a non-empty list of method names")
    normalized_methods = tuple(runner.parse_methods(",".join(methods)))

    limit = expt.get("limit", 0)
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError("'expt.limit' must be a non-negative integer")

    configured_resume_run = expt.get("resume_run")
    if configured_resume_run is None:
        resume_run = None
    elif not isinstance(configured_resume_run, str) or not configured_resume_run.strip():
        raise ValueError("'expt.resume_run' must be a non-empty path or null")
    else:
        resume_run = Path(configured_resume_run).expanduser()
        if not resume_run.is_absolute():
            resume_run = manager.path.parent / resume_run
        resume_run = resume_run.resolve()
        if len(normalized_benchmarks) != 1:
            raise ValueError("'expt.resume_run' can only be used with one configured benchmark")

    return ExperimentConfig(
        benchmarks=normalized_benchmarks,
        methods=normalized_methods,
        limit=limit,
        resume_run=resume_run,
    )


async def run_experiments(config_path: str | Path) -> None:
    manager = get_config_manager()(config_path)
    experiment = load_experiment_config(manager)
    methods = ",".join(experiment.methods)

    server = spinup_server(manager, timeout=float(manager.config.engine_ready_timeout))
    try:
        for index, benchmark in enumerate(experiment.benchmarks, start=1):
            print(
                f"Running benchmark {index}/{len(experiment.benchmarks)}: {benchmark} "
                f"with {methods}",
                file=sys.stderr,
                flush=True,
            )
            args = evaluation.build_parser().parse_args(
                [
                    "--benchmark",
                    benchmark,
                    "--methods",
                    methods,
                    "--limit",
                    str(experiment.limit),
                    *(["--resume-run", str(experiment.resume_run)] if experiment.resume_run else []),
                ]
            )
            await evaluation.run_evaluation(args)
    finally:
        server.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run configured evaluations with one vLLM server.")
    parser.add_argument("config", type=Path, help="Combined vLLM and experiment YAML configuration")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--print-slurm-args",
        action="store_true",
        help="Print one configured sbatch argument per line without running an experiment.",
    )
    output_mode.add_argument(
        "--print-evaluation-matrix",
        action="store_true",
        help="Print tab-separated benchmark, methods, and limit rows for local execution.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.print_slurm_args:
            manager = get_config_manager()(args.config)
            print("\n".join(load_helma_config(manager).slurm_arguments()))
            return
        if args.print_evaluation_matrix:
            manager = get_config_manager()(args.config)
            experiment = load_experiment_config(manager)
            methods = ",".join(experiment.methods)
            for benchmark in experiment.benchmarks:
                resume_run = str(experiment.resume_run) if experiment.resume_run else "-"
                print(f"{benchmark}\t{methods}\t{experiment.limit}\t{resume_run}")
            return
        asyncio.run(run_experiments(args.config))
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
