"""Debate vs MUFFIN (multiagent_dynamic_streaming): input / output / total tokens.

Each metric gets its own panel and y-axis, so small values (e.g. debate's output)
stay visible next to large ones (debate's input).

Usage (one figure from one or more runs of the same model / benchmark):
    .venv/bin/python scripts/plot_token_compare_debate_muffin.py <run_dir> [<run_dir> ...]
        [--out-dir DIR] [--stem NAME]

Batch over a whole results tree: see scripts/plot_token_compare_debate_muffin_all.py.

Definitions
  * Per task, tokens of a step are summed over all agents.
  * Only tasks that actually ran agent steps are averaged: drops debate tasks that
    errored (e.g. context overflow) and MUFFIN tasks the orchestrator routed to a single
    agent. For the 10-step OlymMATH gemma-4-E4B-it run this gives 95 / 98 tasks and
    matches round 10 of figures/4b_round_token_accuracy_benchmarks/
    olymmath_10_steps_round_token_accuracy.csv (696491.09 vs 157747.71 -> 77.4% fewer).
  * Several runs: mean per run first, then the unweighted mean over runs
    (each run counts equally, regardless of how many tasks it has).
  * Reactive step is hidden in the per-step panels but counted in the totals.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_step_token_usage import bucket_sort_key, step_buckets  # noqa: E402

METHODS = {
    "multiagent_debate": ("Debate", "#9ca3af"),
    "multiagent_dynamic_streaming": ("MUFFIN", "#1f4e79"),
}
DEB, MUF = list(METHODS)
METRICS = ("input", "output", "total")
KFMT = FuncFormatter(lambda v, _: f"{v / 1000:g}k")

# means[method][bucket] = (input, output) mean tokens per task
Means = dict[str, dict[str, tuple[float, float]]]


def run_means(run_dir: Path) -> tuple[Means, dict[str, int]]:
    """Per-task mean (input, output) per step bucket for one run."""
    sums: dict[str, dict[str, list[int]]] = {m: defaultdict(lambda: [0, 0]) for m in METHODS}
    n: dict[str, int] = defaultdict(int)
    for f in sorted((run_dir / "examples").glob("*.json")):
        ex = json.loads(f.read_text())
        m = ex.get("method")
        if m not in METHODS:
            continue
        b_ex = step_buckets(ex)
        if not any(b.isdigit() for b in b_ex):  # no agent steps -> errored / routed to single
            continue
        n[m] += 1
        for b, (p, c) in b_ex.items():
            sums[m][b][0] += p
            sums[m][b][1] += c
    means: Means = {
        m: {b: (v[0] / n[m], v[1] / n[m]) for b, v in sums[m].items()} for m in METHODS if n[m]
    }
    return means, dict(n)


def average_runs(per_run: list[Means]) -> tuple[Means, dict[str, int]]:
    """Unweighted mean over runs (per method, over the runs that have that method)."""
    out: Means = {}
    n_runs: dict[str, int] = {}
    for m in METHODS:
        runs = [r[m] for r in per_run if m in r]
        if not runs:
            continue
        n_runs[m] = len(runs)
        buckets = set().union(*[r.keys() for r in runs])
        out[m] = {
            b: (sum(r.get(b, (0, 0))[0] for r in runs) / len(runs),
                sum(r.get(b, (0, 0))[1] for r in runs) / len(runs))
            for b in buckets
        }
    return out, n_runs


def value(means: Means, m: str, b: str, metric: str) -> float:
    i, o = means.get(m, {}).get(b, (0.0, 0.0))
    return {"input": i, "output": o, "total": i + o}[metric]


def overall(means: Means, m: str, metric: str) -> float:
    return sum(value(means, m, b, metric) for b in means.get(m, {}))


def all_buckets(means: Means) -> list[str]:
    return sorted(set().union(*[means[m].keys() for m in means]), key=bucket_sort_key)


def write_csv(means: Means, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step"] + [f"{METHODS[m][0].lower()}_mean_{mt}_tokens" for mt in METRICS for m in METHODS])
        for b in all_buckets(means) + ["ALL"]:
            w.writerow([b] + [
                f"{overall(means, m, mt) if b == 'ALL' else value(means, m, b, mt):.2f}"
                for mt in METRICS for m in METHODS
            ])


def plot(means: Means, title: str, out_stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    width = 0.4
    plot_buckets = [b for b in all_buckets(means) if b != "React."]
    x = list(range(len(plot_buckets)))
    for ax, metric in zip(axes.flat[:3], METRICS):
        for k, (m, (name, color)) in enumerate(METHODS.items()):
            vals = [value(means, m, b, metric) for b in plot_buckets]
            ax.bar([xi + (k - 0.5) * width for xi in x], vals, width=width, color=color, label=name)
        ax.set_xticks(x, plot_buckets, fontsize=9)
        ax.set_title(f"{metric.capitalize()} tokens per step", fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean tokens per task (all agents)")

    ax = axes.flat[3]
    for k, (m, (name, color)) in enumerate(METHODS.items()):
        vals = [overall(means, m, mt) for mt in METRICS]
        bars = ax.bar([i + (k - 0.5) * width for i in range(3)], vals, width=width, color=color, label=name)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v / 1000:.1f}k",
                    ha="center", va="bottom", fontsize=8.5)
    ymax = max(overall(means, m, "total") for m in METHODS)
    for i, mt in enumerate(METRICS):
        d, u = overall(means, DEB, mt), overall(means, MUF, mt)
        pct = (d - u) / d * 100 if d else 0.0
        txt = f"MUFFIN {pct:.1f}% fewer" if pct >= 0 else f"MUFFIN {-pct:.1f}% more"
        ax.text(i, ymax * 1.12, txt, ha="center", fontsize=9, color="#374151",
                fontweight="bold" if mt == "total" else "normal")
    ax.set_xticks(range(3), [m.capitalize() for m in METRICS])
    ax.set_ylim(0, ymax * 1.2)
    ax.set_title("Total per task (sum over all steps)", fontsize=11)
    ax.set_ylabel("Mean tokens per task")

    for ax in axes.flat:
        ax.yaxis.set_major_formatter(KFMT)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_stem.parent / f"{out_stem.name}.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", type=Path, nargs="+")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--stem", default="token_compare_debate_muffin")
    args = ap.parse_args()

    cfg = json.loads((args.run_dirs[0] / "run_config.json").read_text())
    model = cfg["settings"]["resolved_model"].split("/")[-1]
    bench = cfg.get("benchmark_display_name", cfg["benchmark"])
    out_dir = args.out_dir or Path("figures") / f"step_token_usage_{cfg['benchmark']}_{model}_{cfg['run_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_run = [run_means(rd)[0] for rd in args.run_dirs]
    means, n_runs = average_runs(per_run)
    write_csv(means, out_dir / f"{args.stem}.csv")
    plot(means, f"Debate vs MUFFIN token usage — {bench}, {model}", out_dir / args.stem)
    d, u = overall(means, DEB, "total"), overall(means, MUF, "total")
    print(f"{model}: runs={n_runs} debate={d:.2f} muffin={u:.2f} reduction={(1 - u / d) * 100:.2f}%")


if __name__ == "__main__":
    main()
