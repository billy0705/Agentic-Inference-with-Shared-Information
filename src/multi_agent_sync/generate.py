from __future__ import annotations

import argparse
import asyncio
import os
import shlex
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
    repeats: int
    max_steps: int | None
    think_mode: bool | None
    min_dynamic_subagents: int | None
    max_dynamic_subagents: int | None
    single_agent_min_steps: int | None
    single_agent_max_steps: int | None
    debate_rounds: int | None
    max_tokens: int | None
    resume_run: Path | None
    benchmark_data_dir: Path | None
    output_dir: Path | None


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


def _optional_path(section: dict[str, Any], section_name: str, key: str, base_path: Path) -> Path | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{section_name}.{key}' must be a non-empty path or null")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = base_path / path
    return path.resolve()


def _optional_positive_int(section: dict[str, Any], section_name: str, key: str) -> int | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'{section_name}.{key}' must be a positive integer")
    return value


def _optional_bool(section: dict[str, Any], section_name: str, key: str) -> bool | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"'{section_name}.{key}' must be a boolean")
    return value


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

    repeats = expt.get("repeats", 1)
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ValueError("'expt.repeats' must be a positive integer")

    max_steps = _optional_positive_int(expt, "expt", "max_steps")
    think_mode = _optional_bool(expt, "expt", "think_mode")
    min_dynamic_subagents = _optional_positive_int(expt, "expt", "min_dynamic_subagents")
    max_dynamic_subagents = _optional_positive_int(expt, "expt", "max_dynamic_subagents")
    single_agent_min_steps = _optional_positive_int(expt, "expt", "single_agent_min_steps")
    single_agent_max_steps = _optional_positive_int(expt, "expt", "single_agent_max_steps")
    debate_rounds = _optional_positive_int(expt, "expt", "debate_rounds")
    max_tokens = _optional_positive_int(expt, "expt", "max_tokens")
    if (
        min_dynamic_subagents is not None
        and max_dynamic_subagents is not None
        and min_dynamic_subagents > max_dynamic_subagents
    ):
        raise ValueError("'expt.min_dynamic_subagents' must be less than or equal to 'expt.max_dynamic_subagents'")

    if (
        single_agent_min_steps is not None
        and single_agent_max_steps is not None
        and single_agent_min_steps > single_agent_max_steps
    ):
        raise ValueError("'expt.single_agent_min_steps' must be less than or equal to 'expt.single_agent_max_steps'")

    resume_run = _optional_path(expt, "expt", "resume_run", manager.path.parent)
    if resume_run is not None and len(normalized_benchmarks) != 1:
        raise ValueError("'expt.resume_run' can only be used with one configured benchmark")
    benchmark_data_dir = _optional_path(expt, "expt", "benchmark_data_dir", manager.path.parent)
    output_dir = _optional_path(expt, "expt", "output_dir", manager.path.parent)
    if output_dir is None:
        output_dir = (manager.path.parent / runner.DEFAULT_OUTPUT_DIR).resolve()

    return ExperimentConfig(
        benchmarks=normalized_benchmarks,
        methods=normalized_methods,
        limit=limit,
        repeats=repeats,
        max_steps=max_steps,
        think_mode=think_mode,
        min_dynamic_subagents=min_dynamic_subagents,
        max_dynamic_subagents=max_dynamic_subagents,
        single_agent_min_steps=single_agent_min_steps,
        single_agent_max_steps=single_agent_max_steps,
        debate_rounds=debate_rounds,
        max_tokens=max_tokens,
        resume_run=resume_run,
        benchmark_data_dir=benchmark_data_dir,
        output_dir=output_dir,
    )


def environment_values(experiment: ExperimentConfig) -> dict[str, str]:
    values = {}
    if experiment.benchmark_data_dir is not None:
        values["BENCHMARK_DATA_DIR"] = str(experiment.benchmark_data_dir)
    return values


def configure_environment(experiment: ExperimentConfig) -> None:
    os.environ.update(environment_values(experiment))


def shell_environment_exports(experiment: ExperimentConfig) -> tuple[str, ...]:
    return tuple(f"export {key}={shlex.quote(value)}" for key, value in environment_values(experiment).items())


def server_ready_timeout(manager: Any) -> float:
    vllm = manager.section("vllm")
    timeout = vllm.get("engine_ready_timeout", 1800)
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("'vllm.engine_ready_timeout' must be a positive number")
    return float(timeout)


async def run_experiments(config_path: str | Path) -> None:
    manager = get_config_manager()(config_path)
    experiment = load_experiment_config(manager)
    configure_environment(experiment)
    methods = ",".join(experiment.methods)

    server = spinup_server(manager, timeout=server_ready_timeout(manager))
    try:
        for index, benchmark in enumerate(experiment.benchmarks, start=1):
            for repeat in range(1, experiment.repeats + 1):
                print(
                    f"Running benchmark {index}/{len(experiment.benchmarks)} "
                    f"repeat {repeat}/{experiment.repeats}: {benchmark} with {methods}",
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
                        *(["--max-steps", str(experiment.max_steps)] if experiment.max_steps is not None else []),
                        *(["--think-mode"] if experiment.think_mode is True else []),
                        *(["--no-think-mode"] if experiment.think_mode is False else []),
                        *(
                            ["--min-dynamic-subagents", str(experiment.min_dynamic_subagents)]
                            if experiment.min_dynamic_subagents is not None
                            else []
                        ),
                        *(
                            ["--max-dynamic-subagents", str(experiment.max_dynamic_subagents)]
                            if experiment.max_dynamic_subagents is not None
                            else []
                        ),
                        *(
                            ["--single-agent-min-steps", str(experiment.single_agent_min_steps)]
                            if experiment.single_agent_min_steps is not None
                            else []
                        ),
                        *(
                            ["--single-agent-max-steps", str(experiment.single_agent_max_steps)]
                            if experiment.single_agent_max_steps is not None
                            else []
                        ),
                        *(
                            ["--debate-rounds", str(experiment.debate_rounds)]
                            if experiment.debate_rounds is not None
                            else []
                        ),
                        *(
                            ["--max-tokens", str(experiment.max_tokens)]
                            if experiment.max_tokens is not None
                            else []
                        ),
                        *(["--output-dir", str(experiment.output_dir)] if experiment.output_dir else []),
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
    output_mode.add_argument(
        "--print-env",
        action="store_true",
        help="Print shell exports for environment values configured in YAML.",
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
                for _ in range(experiment.repeats):
                    output_dir = str(experiment.output_dir) if experiment.output_dir else "-"
                    resume_run = str(experiment.resume_run) if experiment.resume_run else "-"
                    max_steps = str(experiment.max_steps) if experiment.max_steps is not None else "-"
                    think_mode = str(experiment.think_mode).lower() if experiment.think_mode is not None else "-"
                    min_dynamic_subagents = (
                        str(experiment.min_dynamic_subagents)
                        if experiment.min_dynamic_subagents is not None
                        else "-"
                    )
                    max_dynamic_subagents = (
                        str(experiment.max_dynamic_subagents)
                        if experiment.max_dynamic_subagents is not None
                        else "-"
                    )
                    single_agent_min_steps = (
                        str(experiment.single_agent_min_steps) if experiment.single_agent_min_steps is not None else "-"
                    )
                    single_agent_max_steps = (
                        str(experiment.single_agent_max_steps) if experiment.single_agent_max_steps is not None else "-"
                    )
                    debate_rounds = str(experiment.debate_rounds) if experiment.debate_rounds is not None else "-"
                    max_tokens = str(experiment.max_tokens) if experiment.max_tokens is not None else "-"
                    print(
                        f"{benchmark}\t{methods}\t{experiment.limit}\t{output_dir}\t{resume_run}\t"
                        f"{max_steps}\t{min_dynamic_subagents}\t{max_dynamic_subagents}\t"
                        f"{think_mode}\t"
                        f"{single_agent_min_steps}\t{single_agent_max_steps}\t{debate_rounds}\t{max_tokens}"
                    )
            return
        if args.print_env:
            manager = get_config_manager()(args.config)
            print("\n".join(shell_environment_exports(load_experiment_config(manager))))
            return
        asyncio.run(run_experiments(args.config))
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
