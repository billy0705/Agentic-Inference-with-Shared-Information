#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("output")
DEFAULT_OUTPUT_DIR = Path("figures")

METHOD_ORDER = [
    "plain_llm",
    "single_agent",
    "majority_vote",
    "multiagent_debate",
    "multiagent",
    "multiagent_no_streaming",
    "multiagent_streaming",
    "multiagent_dynamic_no_streaming",
    "multiagent_dynamic_streaming",
    "dynamic_orchestration",
]

LABEL_ORDER = ["before", "previous", "current", "after"]
LABEL_MARKERS = {
    "before": "*",
    "previous": "*",
    "current": "o",
    "after": "o",
}


@dataclass(frozen=True)
class MethodResult:
    benchmark: str
    method: str
    accuracy: float
    avg_total_tokens: float
    run_dir: Path
    label: str = ""

    @property
    def display_name(self) -> str:
        if not self.label:
            return self.method
        return f"{self.method} ({self.label})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot latest benchmark results.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON file with explicit runs to plot.",
    )
    return parser.parse_args()


def collect_latest_results(output_root: Path) -> list[MethodResult]:
    paths_by_benchmark: dict[str, list[Path]] = {}
    for summary_path in output_root.glob("*/*/*/summary.json"):
        benchmark = infer_benchmark(summary_path)
        paths_by_benchmark.setdefault(benchmark, []).append(summary_path)

    records: list[MethodResult] = []
    for benchmark, summary_paths in sorted(paths_by_benchmark.items()):
        selected_model = latest_model_name(summary_paths)
        latest_by_method: dict[str, MethodResult] = {}
        latest_run_by_method: dict[str, str] = {}
        for summary_path in sorted(summary_paths):
            if model_name(summary_path) != selected_model:
                continue
            run_key = run_recency_key(summary_path)
            for record in load_summary(summary_path, benchmark):
                if run_key > latest_run_by_method.get(record.method, ""):
                    latest_by_method[record.method] = record
                    latest_run_by_method[record.method] = run_key
        records.extend(latest_by_method.values())

    return sorted(records, key=lambda row: (row.benchmark, method_sort_key(row.method)))


def collect_configured_results(output_root: Path, config_path: Path) -> list[MethodResult]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs = config.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return collect_latest_results(output_root)

    records: list[MethodResult] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        summary_path = resolve_config_summary_path(output_root, run)
        benchmark = str(run.get("benchmark") or infer_benchmark(summary_path))
        methods = run.get("methods")
        label = str(run.get("label") or "")
        allowed_methods = set(methods) if isinstance(methods, list) else None
        for record in load_summary(summary_path, benchmark):
            if allowed_methods is None or record.method in allowed_methods:
                records.append(
                    MethodResult(
                        benchmark=record.benchmark,
                        method=record.method,
                        accuracy=record.accuracy,
                        avg_total_tokens=record.avg_total_tokens,
                        run_dir=record.run_dir,
                        label=label,
                    )
                )

    return sorted(records, key=lambda row: (row.benchmark, method_sort_key(row.method), label_sort_key(row.label)))


def resolve_config_summary_path(output_root: Path, run: dict[str, object]) -> Path:
    summary_path = run.get("summary_path")
    if isinstance(summary_path, str) and summary_path:
        path = Path(summary_path)
        resolved_path = path if path.is_absolute() else path
        if not resolved_path.exists():
            raise FileNotFoundError(f"Configured summary not found: {resolved_path}")
        return resolved_path

    benchmark = run.get("benchmark")
    model = run.get("model")
    run_id = run.get("run_id")
    if not all(isinstance(value, str) and value for value in (benchmark, model, run_id)):
        raise ValueError("Each configured run needs summary_path or benchmark/model/run_id.")

    path = output_root / str(benchmark) / str(model) / str(run_id) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Configured summary not found: {path}")
    return path


def latest_model_name(summary_paths: list[Path]) -> str:
    latest_path = max(summary_paths, key=run_recency_key)
    return model_name(latest_path)


def model_name(summary_path: Path) -> str:
    return summary_path.parents[1].name


def run_recency_key(summary_path: Path) -> str:
    return f"{summary_path.parent.name} {summary_path.parent}"


def infer_benchmark(summary_path: Path) -> str:
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return summary_path.parents[2].name

    run_config = data.get("run_config")
    if isinstance(run_config, dict) and run_config.get("benchmark"):
        return str(run_config["benchmark"])

    results = data.get("results")
    if isinstance(results, list):
        for result in results:
            if isinstance(result, dict) and result.get("benchmark"):
                return str(result["benchmark"])

    return summary_path.parents[2].name


def load_summary(summary_path: Path, benchmark: str) -> list[MethodResult]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    records: list[MethodResult] = []
    for method, stats in summary.items():
        if not isinstance(stats, dict):
            continue
        if stats.get("accuracy") is None or stats.get("avg_total_tokens") is None:
            continue
        records.append(
            MethodResult(
                benchmark=benchmark,
                method=method,
                accuracy=float(stats["accuracy"]),
                avg_total_tokens=float(stats["avg_total_tokens"]),
                run_dir=summary_path.parent,
            )
        )
    return records


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def label_sort_key(label: str) -> tuple[int, str]:
    normalized = label.lower()
    try:
        return (LABEL_ORDER.index(normalized), normalized)
    except ValueError:
        return (len(LABEL_ORDER), normalized)


def marker_for_label(label: str) -> str:
    return LABEL_MARKERS.get(label.lower(), "o")


def method_color_map(records: list[MethodResult]) -> dict[str, str]:
    methods = sorted({record.method for record in records}, key=method_sort_key)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {method: color_cycle[index % len(color_cycle)] for index, method in enumerate(methods)}


def compute_pareto_front(records: list[MethodResult]) -> list[MethodResult]:
    best_accuracy = -math.inf
    front: list[MethodResult] = []
    for record in sorted(records, key=lambda row: (row.avg_total_tokens, -row.accuracy, row.display_name)):
        if record.accuracy > best_accuracy:
            front.append(record)
            best_accuracy = record.accuracy
    return sorted(front, key=lambda row: row.avg_total_tokens, reverse=True)


def bar_plot_records(records: list[MethodResult]) -> list[MethodResult]:
    selected: dict[tuple[str, str], MethodResult] = {}
    for record in records:
        key = (record.benchmark, record.method)
        current = selected.get(key)
        if current is None or bar_label_priority(record.label) > bar_label_priority(current.label):
            selected[key] = record
    return sorted(selected.values(), key=lambda row: (row.benchmark, method_sort_key(row.method)))


def bar_label_priority(label: str) -> int:
    normalized = label.lower()
    if normalized in {"before", "previous"}:
        return 0
    if normalized in {"", "current", "after"}:
        return 2
    return 1


def build_accuracy_bar_plot(records: list[MethodResult]) -> tuple[plt.Figure, plt.Axes]:
    records = bar_plot_records(records)
    benchmarks = sorted({row.benchmark for row in records})
    methods = sorted({row.method for row in records}, key=method_sort_key)
    values = {(row.benchmark, row.method): row.accuracy for row in records}

    width = 0.8 / max(len(methods), 1)
    x_positions = list(range(len(benchmarks)))

    fig, ax = plt.subplots(figsize=(max(8, len(benchmarks) * 1.5), 5))
    for method_index, method in enumerate(methods):
        offsets = [x - 0.4 + width / 2 + method_index * width for x in x_positions]
        accuracies = [values.get((benchmark, method), 0.0) for benchmark in benchmarks]
        ax.bar(offsets, accuracies, width=width, label=method.replace("_", " "))

    ax.set_title("Benchmark accuracy by method")
    ax.set_xlabel("Benchmark")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(benchmarks)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return fig, ax


def write_accuracy_bar_plot(records: list[MethodResult], output_path: Path) -> None:
    fig, _ = build_accuracy_bar_plot(records)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_pareto_plot(records: list[MethodResult]) -> tuple[plt.Figure, plt.Axes]:
    records = sorted(records, key=lambda row: (method_sort_key(row.method), label_sort_key(row.label)))
    front = compute_pareto_front(records)
    colors = method_color_map(records)

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in records:
        ax.scatter(
            row.avg_total_tokens,
            row.accuracy,
            s=70,
            color=colors[row.method],
            marker=marker_for_label(row.label),
            label=row.display_name.replace("_", " "),
        )

    if len(front) > 1:
        ax.plot(
            [row.avg_total_tokens for row in front],
            [row.accuracy for row in front],
            linestyle="--",
            color="black",
            linewidth=1,
            label="Pareto front",
        )

    benchmark = records[0].benchmark if records else "benchmark"
    ax.set_title(f"Accuracy vs token usage: {benchmark}")
    ax.set_xlabel("Average total tokens (lower is better; axis reversed)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return fig, ax


def write_pareto_plot(records: list[MethodResult], output_path: Path) -> None:
    fig, _ = build_pareto_plot(records)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_plots(records: list[MethodResult], output_dir: Path) -> list[Path]:
    if not records:
        raise ValueError("No benchmark summaries found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / "accuracy_grouped_bar.png"]
    write_accuracy_bar_plot(records, paths[0])

    for benchmark in sorted({row.benchmark for row in records}):
        benchmark_records = [row for row in records if row.benchmark == benchmark]
        path = output_dir / f"pareto_{benchmark}.png"
        write_pareto_plot(benchmark_records, path)
        paths.append(path)

    return paths


def main() -> int:
    args = parse_args()
    records = collect_configured_results(args.input_dir, args.config) if args.config else collect_latest_results(args.input_dir)
    paths = write_plots(records, args.output_dir)

    print(f"Loaded {len(records)} method results from {len({row.benchmark for row in records})} benchmarks.")
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
