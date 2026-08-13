#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_INPUT_DIR = Path("output")
DEFAULT_OUTPUT_PATH = Path("figures/round_token_accuracy.png")
DEFAULT_MAX_ROUNDS = 3
EXCLUDED_METHODS = {"dynamic_orchestration"}

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
]

METHOD_MARKERS = {
    "plain_llm": "X",
    "single_agent": "o",
    "majority_vote": "s",
    "multiagent_debate": "^",
    "multiagent": "D",
    "multiagent_no_streaming": "P",
    "multiagent_streaming": "v",
    "multiagent_dynamic_no_streaming": "h",
    "multiagent_dynamic_streaming": "*",
}

ROUND_COLORS = [
    "#d62728",  # round 1: red
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#7f7f7f",
]


@dataclass(frozen=True)
class RunSpec:
    summary_path: Path
    methods: tuple[str, ...] | None = None
    label: str = ""


@dataclass(frozen=True)
class RunIdentity:
    benchmark: str
    model: str
    run_id: str


@dataclass(frozen=True)
class RoundExample:
    benchmark: str
    model: str
    run_id: str
    run_label: str
    method: str
    round: int
    total_tokens: float
    correct: bool


@dataclass(frozen=True)
class RoundPoint:
    method: str
    series: str
    round: int
    accuracy: float
    avg_total_tokens: float
    examples: int
    benchmarks: tuple[str, ...]
    models: tuple[str, ...]
    run_labels: tuple[str, ...]

    @property
    def display_method(self) -> str:
        return self.method.replace("_", " ")

    @property
    def display_series(self) -> str:
        if not self.series:
            return self.display_method
        return f"{self.display_method} ({self.series})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot round-level accuracy against token usage from output example traces. "
            "dynamic_orchestration is ignored by design."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Optional CSV path for the plotted points. Defaults to output path with .csv suffix.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional JSON config using the same runs format as plot_eval_results.py.",
    )
    parser.add_argument(
        "--summary-path",
        action="append",
        type=Path,
        default=[],
        help="Specific summary.json to include. Can be passed multiple times.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help='Benchmark filter. Defaults to chess. Can be repeated; use "--benchmark all" for all benchmarks.',
    )
    parser.add_argument("--model", action="append", default=[], help="Model directory filter. Can be repeated.")
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated method filter. dynamic_orchestration is always ignored.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=DEFAULT_MAX_ROUNDS,
        help="Maximum round number to plot. Defaults to 3. Use 0 for all available rounds.",
    )
    parser.add_argument(
        "--token-mode",
        choices=("cumulative", "round"),
        default="cumulative",
        help="Use cumulative tokens through each round or only tokens spent in that round.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Write method labels next to points.",
    )
    return parser.parse_args()


def collect_run_specs(output_root: Path, args: argparse.Namespace) -> list[RunSpec]:
    if args.summary_path:
        return [RunSpec(path) for path in args.summary_path]
    if args.config:
        return collect_configured_run_specs(output_root, args.config)
    return collect_latest_run_specs(output_root)


def collect_configured_run_specs(output_root: Path, config_path: Path) -> list[RunSpec]:
    config = read_json(config_path)
    runs = config.get("runs", [])
    if not isinstance(runs, list) or not runs:
        return collect_latest_run_specs(output_root)

    specs: list[RunSpec] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        summary_path = resolve_config_summary_path(output_root, run)
        methods = tuple(str(method) for method in run.get("methods", []) if str(method))
        methods = tuple(method for method in methods if method not in EXCLUDED_METHODS)
        if run.get("methods") and not methods:
            continue
        specs.append(
            RunSpec(
                summary_path=summary_path,
                methods=methods or None,
                label=str(run.get("label") or ""),
            )
        )
    return specs


def collect_latest_run_specs(output_root: Path) -> list[RunSpec]:
    summary_paths = sorted(output_root.glob("*/*/*/summary.json"))
    by_benchmark: dict[str, list[Path]] = defaultdict(list)
    for summary_path in summary_paths:
        by_benchmark[infer_identity(summary_path).benchmark].append(summary_path)

    specs: list[RunSpec] = []
    for paths in by_benchmark.values():
        latest_model = infer_identity(max(paths, key=run_recency_key)).model
        latest_by_method: dict[str, Path] = {}
        latest_key_by_method: dict[str, str] = {}
        for summary_path in paths:
            if infer_identity(summary_path).model != latest_model:
                continue
            for method in methods_in_run(summary_path.parent):
                if method in EXCLUDED_METHODS:
                    continue
                recency_key = run_recency_key(summary_path)
                if recency_key > latest_key_by_method.get(method, ""):
                    latest_by_method[method] = summary_path
                    latest_key_by_method[method] = recency_key
        specs.extend(RunSpec(path, methods=(method,)) for method, path in sorted(latest_by_method.items()))
    return specs


def resolve_config_summary_path(output_root: Path, run: dict[str, object]) -> Path:
    summary_path = run.get("summary_path")
    if isinstance(summary_path, str) and summary_path:
        path = Path(summary_path)
        if not path.is_absolute() and not path.exists():
            path = output_root.parent / path
        if not path.exists():
            raise FileNotFoundError(f"Configured summary not found: {summary_path}")
        return path

    benchmark = run.get("benchmark")
    model = run.get("model")
    run_id = run.get("run_id")
    if not all(isinstance(value, str) and value for value in (benchmark, model, run_id)):
        raise ValueError("Each configured run needs summary_path or benchmark/model/run_id.")

    path = output_root / str(benchmark) / str(model) / str(run_id) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Configured summary not found: {path}")
    return path


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def infer_identity(summary_path: Path) -> RunIdentity:
    summary = read_json(summary_path)
    run_config_path = summary_path.parent / "run_config.json"
    run_config = read_json(run_config_path) if run_config_path.exists() else {}
    config = run_config or summary.get("run_config") or {}
    settings = config.get("settings") if isinstance(config, dict) else {}
    settings = settings if isinstance(settings, dict) else {}
    parts = summary_path.parent.parts
    return RunIdentity(
        benchmark=str(config.get("benchmark") or summary.get("benchmark") or parts[-3]),
        model=str(parts[-2] if len(parts) >= 2 else settings.get("model") or "unknown"),
        run_id=str(config.get("run_id") or parts[-1]),
    )


def methods_in_run(run_dir: Path) -> set[str]:
    methods: set[str] = set()
    examples_dir = run_dir / "examples"
    if not examples_dir.exists():
        return methods
    for path in examples_dir.glob("*.json"):
        match = re.match(r"^\d+_(.+)\.json$", path.name)
        if match:
            methods.add(match.group(1))
    return methods


def run_recency_key(summary_path: Path) -> str:
    return f"{summary_path.parent.name} {summary_path.parent}"


def collect_round_examples(
    specs: list[RunSpec],
    *,
    benchmarks: set[str],
    models: set[str],
    methods: set[str],
    max_rounds: int,
    token_mode: str,
) -> list[RoundExample]:
    rows: list[RoundExample] = []
    seen: set[tuple[Path, str, str]] = set()
    for spec in specs:
        identity = infer_identity(spec.summary_path)
        if benchmarks and identity.benchmark not in benchmarks:
            continue
        if models and identity.model not in models:
            continue
        allowed_methods = set(spec.methods or ())
        if methods:
            allowed_methods = allowed_methods & methods if allowed_methods else set(methods)
        if not allowed_methods:
            allowed_methods = set(methods_in_run(spec.summary_path.parent))
        allowed_methods -= EXCLUDED_METHODS

        for method in sorted(allowed_methods, key=method_sort_key):
            key = (spec.summary_path, method, spec.label)
            if key in seen:
                continue
            seen.add(key)
            rows.extend(
                collect_method_round_examples(
                    spec.summary_path.parent,
                    identity,
                    method,
                    spec.label,
                    max_rounds=max_rounds,
                    token_mode=token_mode,
                )
            )
    return rows


def collect_method_round_examples(
    run_dir: Path,
    identity: RunIdentity,
    method: str,
    label: str,
    *,
    max_rounds: int,
    token_mode: str,
) -> list[RoundExample]:
    examples_dir = run_dir / "examples"
    if not examples_dir.exists():
        return []

    rows: list[RoundExample] = []
    for path in sorted(examples_dir.glob(f"*_{method}.json")):
        trace = read_json(path)
        if not trace:
            continue
        method_trace = trace.get("method_trace")
        if not isinstance(method_trace, dict):
            continue
        round_rows = extract_example_rounds(trace, method_trace, method, token_mode=token_mode)
        for round_number, tokens, correct in round_rows:
            if max_rounds > 0 and round_number > max_rounds:
                continue
            if tokens is None:
                continue
            rows.append(
                RoundExample(
                    benchmark=identity.benchmark,
                    model=identity.model,
                    run_id=identity.run_id,
                    run_label=label,
                    method=method,
                    round=round_number,
                    total_tokens=tokens,
                    correct=correct,
                )
            )
    return rows


def extract_example_rounds(
    trace: dict[str, Any],
    method_trace: dict[str, Any],
    method: str,
    *,
    token_mode: str,
) -> list[tuple[int, float | None, bool]]:
    if method == "single_agent":
        return extract_step_rounds(trace, method_trace.get("steps"), token_mode=token_mode)

    agent_step_rows = extract_agent_step_rounds(trace, method_trace, token_mode=token_mode)
    if agent_step_rows:
        return agent_step_rows

    debate_rounds = extract_debate_rounds(trace, method_trace)
    if debate_rounds:
        return debate_rounds

    return extract_final_round(trace, method_trace)


def extract_step_rounds(
    trace: dict[str, Any],
    steps: object,
    *,
    token_mode: str,
) -> list[tuple[int, float | None, bool]]:
    if not isinstance(steps, list):
        return []

    cumulative = 0.0
    rows: list[tuple[int, float | None, bool]] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        round_number = int_or_default(step.get("step"), index)
        round_tokens = token_total(step.get("token_usage"))
        if round_tokens is not None:
            cumulative += round_tokens
        candidate = candidate_from_step(step)
        rows.append(
            (
                round_number,
                cumulative if token_mode == "cumulative" and round_tokens is not None else round_tokens,
                answer_matches(candidate, trace),
            )
        )
    return rows


def extract_agent_step_rounds(
    trace: dict[str, Any],
    method_trace: dict[str, Any],
    *,
    token_mode: str,
) -> list[tuple[int, float | None, bool]]:
    agent_traces = method_trace.get("agent_traces")
    if not isinstance(agent_traces, dict):
        return []

    by_round: dict[int, list[tuple[str, float | None]]] = defaultdict(list)
    for agent_trace in agent_traces.values():
        if not isinstance(agent_trace, dict):
            continue
        steps = agent_trace.get("steps")
        if not isinstance(steps, list):
            continue
        for index, step in enumerate(steps, start=1):
            if not isinstance(step, dict):
                continue
            round_number = int_or_default(step.get("step"), index)
            candidate = candidate_from_step(step)
            if not candidate:
                candidate = candidate_from_text(agent_trace.get("final_output", ""))
            by_round[round_number].append((candidate, token_total(step.get("token_usage"))))

    if not by_round:
        return []

    cumulative = 0.0
    rows: list[tuple[int, float | None, bool]] = []
    for round_number in sorted(by_round):
        candidates_and_tokens = by_round[round_number]
        round_tokens_values = [tokens for _, tokens in candidates_and_tokens if tokens is not None]
        round_tokens = sum(round_tokens_values) if round_tokens_values else None
        if round_tokens is not None:
            cumulative += round_tokens
        candidate = majority_candidate(candidate for candidate, _ in candidates_and_tokens)
        rows.append(
            (
                round_number,
                cumulative if token_mode == "cumulative" and round_tokens is not None else round_tokens,
                answer_matches(candidate, trace),
            )
        )
    return rows


def extract_debate_rounds(
    trace: dict[str, Any],
    method_trace: dict[str, Any],
) -> list[tuple[int, float | None, bool]]:
    rounds = method_trace.get("round_traces")
    if not isinstance(rounds, list):
        return []
    rows: list[tuple[int, float | None, bool]] = []
    for index, round_trace in enumerate(rounds, start=1):
        if not isinstance(round_trace, dict):
            continue
        responses = round_trace.get("agent_responses")
        if not isinstance(responses, list):
            continue
        candidate = majority_candidate(candidate_from_text(response.get("response", "")) for response in responses if isinstance(response, dict))
        tokens = token_total(round_trace.get("token_usage"))
        rows.append((int_or_default(round_trace.get("round"), index), tokens, answer_matches(candidate, trace)))
    return rows


def extract_final_round(trace: dict[str, Any], method_trace: dict[str, Any]) -> list[tuple[int, float | None, bool]]:
    tokens = (
        token_total(method_trace.get("token_usage"))
        or number_or_none(trace.get("total_tokens"))
        or number_or_none(trace.get("result", {}).get("total_tokens") if isinstance(trace.get("result"), dict) else None)
    )
    correct = bool(trace.get("correct"))
    return [(1, tokens, correct)] if tokens is not None else []


def aggregate_points(rows: list[RoundExample]) -> list[RoundPoint]:
    buckets: dict[tuple[str, str, int], list[RoundExample]] = defaultdict(list)
    for row in rows:
        buckets[(row.method, row.run_label, row.round)].append(row)

    points: list[RoundPoint] = []
    for (method, series, round_number), examples in buckets.items():
        points.append(
            RoundPoint(
                method=method,
                series=series,
                round=round_number,
                accuracy=sum(1 for example in examples if example.correct) / len(examples),
                avg_total_tokens=sum(example.total_tokens for example in examples) / len(examples),
                examples=len(examples),
                benchmarks=tuple(sorted({example.benchmark for example in examples})),
                models=tuple(sorted({example.model for example in examples})),
                run_labels=tuple(sorted({example.run_label for example in examples if example.run_label})),
            )
        )
    return sorted(points, key=lambda point: (method_sort_key(point.method), point.series, point.round))


def candidate_from_step(step: dict[str, Any]) -> str:
    for key in ("parsed_answer", "answer_choice", "final_answer", "output"):
        value = step.get(key)
        if value not in (None, ""):
            return clean_candidate(value)
    parsed = step.get("parsed_output")
    if isinstance(parsed, dict):
        for key in ("parsed_answer", "answer_choice", "final_answer"):
            value = parsed.get(key)
            if value not in (None, ""):
                return clean_candidate(value)
        for key in ("local_notes", "summary", "output"):
            value = candidate_from_text(parsed.get(key, ""))
            if value:
                return value
    return candidate_from_text(step.get("raw_response", ""))


def candidate_from_text(value: object) -> str:
    text = str(value or "")
    if not text.strip():
        return ""

    patterns = [
        r"(?im)^\s*ANSWER_CHOICE\s*:\s*(.+?)\s*$",
        r"(?im)^\s*Final\s+Answer\s*:\s*(.+?)\s*$",
        r"\\boxed\{([^}]+)\}",
        r'"final_answer"\s*:\s*"([^"]+)"',
        r"\(([a-h][1-8])\)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return clean_candidate(matches[-1])
    return ""


def clean_candidate(value: object) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^Final\s+Answer\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    boxed = re.match(r"^\\boxed\{(.+)\}$", text, flags=re.IGNORECASE)
    if boxed:
        text = boxed.group(1).strip()
    wrapped = re.match(r"^\((.+)\)$", text)
    if wrapped:
        text = wrapped.group(1).strip()
    return re.sub(r'^[("*\s]+|[)"*\s.]+$', "", text).strip()


def comparable_answer(value: object) -> str:
    return clean_candidate(value).lower()


def accepted_answers(trace: dict[str, Any]) -> set[str]:
    metadata = trace.get("score_metadata")
    valid_targets = metadata.get("valid_targets") if isinstance(metadata, dict) else None
    answers = valid_targets if isinstance(valid_targets, list) and valid_targets else [trace.get("gold")]
    return {comparable_answer(answer) for answer in answers if comparable_answer(answer)}


def answer_matches(candidate: object, trace: dict[str, Any]) -> bool:
    normalized = comparable_answer(candidate)
    if not normalized:
        return False
    accepted = accepted_answers(trace)
    if accepted:
        return normalized in accepted
    gold = comparable_answer(trace.get("gold"))
    return bool(gold and normalized == gold)


def majority_candidate(candidates: object) -> str:
    counts = Counter(clean_candidate(candidate) for candidate in candidates if clean_candidate(candidate))
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def token_total(value: object) -> float | None:
    if not isinstance(value, dict):
        return None
    return number_or_none(value.get("total_tokens"))


def number_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def int_or_default(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def marker_for_method(method: str) -> str:
    if method in METHOD_MARKERS:
        return METHOD_MARKERS[method]
    markers = ["o", "s", "^", "D", "P", "v", "h", "X", "*"]
    return markers[abs(hash(method)) % len(markers)]


def color_for_round(round_number: int) -> str:
    return ROUND_COLORS[(round_number - 1) % len(ROUND_COLORS)]


def write_plot(points: list[RoundPoint], output_path: Path, *, annotate: bool, token_mode: str) -> None:
    if not points:
        raise ValueError("No round-level records found.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 6))

    points_by_series: dict[tuple[str, str], list[RoundPoint]] = defaultdict(list)
    for point in points:
        points_by_series[(point.method, point.series)].append(point)
    for series_points in points_by_series.values():
        ordered_points = sorted(series_points, key=lambda point: point.round)
        if len(ordered_points) < 2:
            continue
        ax.plot(
            [point.avg_total_tokens for point in ordered_points],
            [point.accuracy for point in ordered_points],
            color="#4b5563",
            linewidth=1.2,
            alpha=0.55,
            zorder=1,
        )

    for point in points:
        ax.scatter(
            point.avg_total_tokens,
            point.accuracy,
            s=170 if point.method == "multiagent_dynamic_streaming" else 90,
            marker=marker_for_method(point.method),
            color=color_for_round(point.round),
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            zorder=2,
        )
        if annotate:
            label = point.display_series
            ax.annotate(
                label,
                (point.avg_total_tokens, point.accuracy),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
                alpha=0.85,
            )

    methods = sorted({point.method for point in points}, key=method_sort_key)
    rounds = sorted({point.round for point in points})
    method_handles = [
        Line2D(
            [0],
            [0],
            marker=marker_for_method(method),
            color="black",
            markerfacecolor="white",
            linestyle="None",
            markersize=9,
            label=method.replace("_", " "),
        )
        for method in methods
    ]
    round_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color_for_round(round_number),
            linestyle="None",
            markersize=9,
            label=f"round {round_number}",
        )
        for round_number in rounds
    ]

    method_legend = ax.legend(handles=method_handles, title="Method marker", fontsize=8, loc="upper right")
    ax.add_artist(method_legend)
    ax.legend(handles=round_handles, title="Round color", fontsize=8, loc="lower right")

    benchmark_title = ", ".join(sorted({benchmark for point in points for benchmark in point.benchmarks}))
    ax.set_title(f"Round Accuracy vs Token Usage: {benchmark_title}" if benchmark_title else "Round Accuracy vs Token Usage")
    ax.set_xlabel(f"Average {'cumulative ' if token_mode == 'cumulative' else ''}tokens")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_csv(points: list[RoundPoint], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "method",
                "series",
                "round",
                "accuracy",
                "avg_total_tokens",
                "examples",
                "benchmarks",
                "models",
                "run_labels",
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "method": point.method,
                    "series": point.series,
                    "round": point.round,
                    "accuracy": f"{point.accuracy:.6f}",
                    "avg_total_tokens": f"{point.avg_total_tokens:.2f}",
                    "examples": point.examples,
                    "benchmarks": ";".join(point.benchmarks),
                    "models": ";".join(point.models),
                    "run_labels": ";".join(point.run_labels),
                }
            )


def main() -> int:
    args = parse_args()
    method_filter = {method.strip() for method in args.methods.split(",") if method.strip()} - EXCLUDED_METHODS
    benchmark_filter = benchmark_filters(args.benchmark)
    specs = collect_run_specs(args.input_dir, args)
    round_examples = collect_round_examples(
        specs,
        benchmarks=benchmark_filter,
        models=set(args.model),
        methods=method_filter,
        max_rounds=args.max_rounds,
        token_mode=args.token_mode,
    )
    points = aggregate_points(round_examples)
    csv_output = args.csv_output or args.output_path.with_suffix(".csv")
    write_plot(points, args.output_path, annotate=args.annotate, token_mode=args.token_mode)
    write_csv(points, csv_output)

    print(f"Loaded {len(round_examples)} round examples into {len(points)} plotted points.")
    print(args.output_path)
    print(csv_output)
    return 0


def benchmark_filters(values: list[str]) -> set[str]:
    normalized = {value.strip() for value in values if value.strip()}
    if not normalized:
        return {"chess"}
    if any(value.lower() == "all" for value in normalized):
        return set()
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
