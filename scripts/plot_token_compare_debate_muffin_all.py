"""Debate vs MUFFIN token comparison for every benchmark x model in a results tree.

Usage:
    .venv/bin/python scripts/plot_token_compare_debate_muffin_all.py \
        [--root output_final/5-repeats] [--out figures/token_compare_debate_muffin]

Expected layout: <root>/<benchmark>/<model>/<run_id>/{run_config.json,examples/*.json}

Output layout:
    <out>/<benchmark>/<model>.{png,pdf,csv}       one figure per benchmark x model
    <out>/<benchmark>/_average.{png,pdf,csv}       mean over the models of that benchmark
    <out>/overall/all.{png,pdf,csv}                mean over every benchmark x model
    <out>/overall/summary.csv                      one row per benchmark x model (+ averages)
    <out>/overall/reduction_total_tokens.{png,pdf} % fewer total tokens, benchmark x model
    <out>/overall/used_runs.json                   runs and task counts behind every number

Averages over models / benchmarks are unweighted means of the per-(benchmark, model)
means; see plot_token_compare_debate_muffin.py for the per-run definitions.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_token_compare_debate_muffin import (  # noqa: E402
    DEB, METRICS, MUF, Means, average_runs, overall, plot, run_means, write_csv,
)

BENCH_ORDER = ["olymmath", "olymmath-hard", "gpqa", "mmlu_pro", "chess"]
MODEL_ORDER = ["gemma-4-E4B-it", "gemma-4-31B-it", "Qwen3.5-4B", "Qwen3.8-27B"]
DISPLAY = {"olymmath": "OlymMATH-E", "olymmath-hard": "OlymMATH-H"}  # paper naming
YMIN = -60.0  # clip for the reduction plot
MODEL_COLORS = ["#9ec5e8", "#1f4e79", "#f4b183", "#c55a11"]


def order(items, ref):
    return sorted(items, key=lambda x: (ref.index(x) if x in ref else len(ref), x))


def reduction(means: Means, metric: str) -> float:
    d, u = overall(means, DEB, metric), overall(means, MUF, metric)
    return (d - u) / d * 100 if d else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("output_final/5-repeats"))
    ap.add_argument("--out", type=Path, default=Path("figures/token_compare_debate_muffin"))
    args = ap.parse_args()

    cells: dict[tuple[str, str], Means] = {}
    names: dict[str, str] = {}
    used: dict[str, dict[str, list[dict]]] = {}
    summary_rows = []

    for bdir in order([p for p in args.root.iterdir() if p.is_dir()], [*map(args.root.joinpath, BENCH_ORDER)]):
        bench = bdir.name
        for mdir in order([p for p in bdir.iterdir() if p.is_dir()], [*map(bdir.joinpath, MODEL_ORDER)]):
            model = mdir.name
            runs = sorted(p for p in mdir.iterdir() if (p / "run_config.json").exists())
            per_run, info = [], []
            for rd in runs:
                cfg = json.loads((rd / "run_config.json").read_text())
                names[bench] = DISPLAY.get(bench, cfg.get("benchmark_display_name", bench))
                means, n = run_means(rd)
                if means:
                    per_run.append(means)
                info.append({"run": str(rd), "n_tasks": n})
            means, n_runs = average_runs(per_run)
            used.setdefault(bench, {})[model] = info
            if DEB not in means or MUF not in means:
                print(f"skip {bench}/{model}: missing a method ({n_runs})")
                continue
            cells[(bench, model)] = means
            out_stem = args.out / bench / model
            write_csv(means, args.out / bench / f"{model}.csv")
            plot(means, f"Debate vs MUFFIN token usage — {names[bench]}, {model}", out_stem)
            row = {
                "benchmark": bench, "model": model,
                "debate_runs": n_runs.get(DEB, 0), "muffin_runs": n_runs.get(MUF, 0),
                "debate_tasks": sum(i["n_tasks"].get(DEB, 0) for i in info),
                "muffin_tasks": sum(i["n_tasks"].get(MUF, 0) for i in info),
            }
            for mt in METRICS:
                row[f"debate_{mt}"] = overall(means, DEB, mt)
                row[f"muffin_{mt}"] = overall(means, MUF, mt)
                row[f"reduction_{mt}_pct"] = reduction(means, mt)
            summary_rows.append(row)
            print(f"{bench:14s} {model:16s} runs D/M={row['debate_runs']}/{row['muffin_runs']} "
                  f"total {row['debate_total']:>10.0f} vs {row['muffin_total']:>10.0f} "
                  f"-> {row['reduction_total_pct']:.1f}% fewer")

    def mean_of(cs: list[Means]) -> Means:
        # each (benchmark, model) cell counts once, same as averaging runs
        return average_runs(cs)[0]

    def avg_row(label_b: str, label_m: str, cs: list[Means], rows: list[dict]) -> dict:
        m = mean_of(cs)
        r = {"benchmark": label_b, "model": label_m,
             "debate_runs": sum(x["debate_runs"] for x in rows),
             "muffin_runs": sum(x["muffin_runs"] for x in rows),
             "debate_tasks": sum(x["debate_tasks"] for x in rows),
             "muffin_tasks": sum(x["muffin_tasks"] for x in rows)}
        for mt in METRICS:
            r[f"debate_{mt}"] = overall(m, DEB, mt)
            r[f"muffin_{mt}"] = overall(m, MUF, mt)
            r[f"reduction_{mt}_pct"] = reduction(m, mt)
        return r

    benches = order({b for b, _ in cells}, BENCH_ORDER)
    models = order({m for _, m in cells}, MODEL_ORDER)

    avg_rows = []
    for b in benches:
        cs = [cells[(b, m)] for m in models if (b, m) in cells]
        rows = [r for r in summary_rows if r["benchmark"] == b]
        m_avg = mean_of(cs)
        write_csv(m_avg, args.out / b / "_average.csv")
        plot(m_avg, f"Debate vs MUFFIN token usage — {names[b]}, average over {len(cs)} models",
             args.out / b / "_average")
        avg_rows.append(avg_row(b, "AVERAGE", cs, rows))

    all_means = mean_of(list(cells.values()))
    (args.out / "overall").mkdir(parents=True, exist_ok=True)
    write_csv(all_means, args.out / "overall" / "all.csv")
    plot(all_means, f"Debate vs MUFFIN token usage — average over {len(benches)} benchmarks × "
         f"{len(models)} models", args.out / "overall" / "all")
    avg_rows.append(avg_row("ALL", "AVERAGE", list(cells.values()), summary_rows))

    # summary.csv
    fields = list(summary_rows[0].keys())
    with (args.out / "overall" / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in summary_rows + avg_rows:
            w.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v) for k, v in r.items()})
    (args.out / "overall" / "used_runs.json").write_text(json.dumps(used, indent=2))

    # reduction plot: % fewer total tokens per benchmark x model (+ benchmark average)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    groups = benches + ["ALL"]
    width = 0.8 / (len(models) + 1)
    for k, m in enumerate(models + ["Average"]):
        xs, ys = [], []
        for gi, b in enumerate(groups):
            if m == "Average":
                r = next(r for r in avg_rows if r["benchmark"] == b)
            else:
                r = next((r for r in summary_rows if r["benchmark"] == b and r["model"] == m), None)
            if r is None:
                continue
            xs.append(gi + (k - len(models) / 2) * width)
            ys.append(r["reduction_total_pct"])
        color = MODEL_COLORS[k % len(MODEL_COLORS)] if m != "Average" else "#374151"
        bars = ax.bar(xs, [max(y, YMIN) for y in ys], width=width, color=color, label=m)
        for bar, y in zip(bars, ys):
            cx = bar.get_x() + bar.get_width() / 2
            if y >= 0:
                ax.text(cx, y + 1, f"{y:.0f}", ha="center", va="bottom", fontsize=7)
            else:  # MUFFIN used MORE tokens; bars below YMIN are clipped
                ax.text(cx, max(y, YMIN) - 2, f"{y:.0f}", ha="center", va="top", fontsize=7,
                        color="#b91c1c", fontweight="bold")
    ax.set_xticks(range(len(groups)), [names.get(b, "All benchmarks") for b in groups])
    ax.set_ylabel("MUFFIN: % fewer total tokens\nthan multi-agent debate")
    ax.set_title("Negative = MUFFIN uses more tokens than debate (bars clipped at "
                 f"{YMIN:.0f}%, true value labelled)", fontsize=9, loc="left", pad=28)
    lo = min(0.0, *[r["reduction_total_pct"] for r in summary_rows + avg_rows])
    ax.set_ylim(YMIN - 12 if lo < 0 else 0, 105)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(ncol=len(models) + 1, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / "overall" / f"reduction_total_tokens.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
