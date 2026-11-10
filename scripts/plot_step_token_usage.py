"""Per-step token usage (stacked: output bottom, input top) for one run directory.

Usage:
    .venv/bin/python scripts/plot_step_token_usage.py <run_dir> [--out-dir figures/<name>]

For every method, tokens of each step are summed over all agents of a task, then
averaged over all tasks in the run (tasks that errored / have no such step count as 0,
so the bars of one method add up to its mean total tokens per task).

multiagent_dynamic_streaming also gets:
  * "Orch."     : orchestrator planning = task total - agent steps - synthesizer
                  (the orchestrator's usage is not stored separately in the trace)
  * "React."    : reactive agent steps (is_reactive == True)
  * "Synth."    : synthesizer call
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METHOD_ORDER = ["single_agent", "majority_vote", "multiagent_debate", "multiagent_dynamic_streaming"]
METHOD_TITLE = {
    "single_agent": "Single agent",
    "majority_vote": "Majority vote",
    "multiagent_debate": "Multi-agent debate",
    "multiagent_dynamic_streaming": "Dynamic streaming (ours)",
}
OUTPUT_COLOR = "#1f4e79"
INPUT_COLOR = "#9ec5e8"


def usage(d: dict | None) -> tuple[int, int]:
    d = d or {}
    return int(d.get("prompt_tokens") or 0), int(d.get("completion_tokens") or 0)


def step_buckets(example: dict) -> dict[str, tuple[int, int]]:
    """Return {bucket_label: (input, output)} summed over agents for one task."""
    mt = example.get("method_trace") or {}
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    if "agent_traces" in mt:
        traces = mt["agent_traces"].values()
    elif "steps" in mt:
        traces = [{"steps": mt["steps"]}]
    else:
        traces = []
    agent_in = agent_out = 0
    for t in traces:
        for s in t.get("steps", []):
            p, c = usage(s.get("token_usage"))
            label = "React." if s.get("is_reactive") else str(s["step"])
            out[label][0] += p
            out[label][1] += c
            agent_in += p
            agent_out += c

    if example.get("method") == "multiagent_dynamic_streaming" and mt:
        sp, sc = usage((mt.get("synthesizer_trace") or {}).get("token_usage"))
        out["Synth."][0] += sp
        out["Synth."][1] += sc
        tp, tc = usage(example.get("token_usage"))
        out["Orch."][0] += max(tp - agent_in - sp, 0)
        out["Orch."][1] += max(tc - agent_out - sc, 0)
    return {k: (v[0], v[1]) for k, v in out.items()}


def bucket_sort_key(label: str) -> tuple[int, int]:
    if label == "Orch.":
        return (0, 0)
    if label.isdigit():
        return (1, int(label))
    if label == "React.":
        return (2, 0)
    return (3, 0)  # Synth.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    run_dir: Path = args.run_dir
    cfg = json.loads((run_dir / "run_config.json").read_text())
    model = cfg["settings"]["resolved_model"].split("/")[-1]
    bench = cfg.get("benchmark_display_name", cfg["benchmark"])
    out_dir = args.out_dir or Path("figures") / f"step_token_usage_{cfg['benchmark']}_{model}_{cfg['run_id']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # method -> bucket -> [sum_in, sum_out, n_tasks_with_bucket]
    agg: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    n_tasks: dict[str, int] = defaultdict(int)
    for f in sorted((run_dir / "examples").glob("*.json")):
        ex = json.loads(f.read_text())
        m = ex["method"]
        n_tasks[m] += 1
        for b, (p, c) in step_buckets(ex).items():
            a = agg[m][b]
            a[0] += p
            a[1] += c
            a[2] += 1

    methods = [m for m in METHOD_ORDER if m in agg] + sorted(set(agg) - set(METHOD_ORDER))

    # CSV
    csv_path = out_dir / "step_token_usage.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "method", "step", "n_tasks", "n_tasks_with_step",
            "mean_input_tokens", "mean_output_tokens", "mean_total_tokens",
            "sum_input_tokens", "sum_output_tokens", "sum_total_tokens",
        ])
        for m in methods:
            n = n_tasks[m]
            for b in sorted(agg[m], key=bucket_sort_key):
                si, so, k = agg[m][b]
                w.writerow([m, b, n, k, f"{si / n:.1f}", f"{so / n:.1f}", f"{(si + so) / n:.1f}", si, so, si + so])
            ti = sum(v[0] for v in agg[m].values())
            to = sum(v[1] for v in agg[m].values())
            w.writerow([m, "ALL", n, n, f"{ti / n:.1f}", f"{to / n:.1f}", f"{(ti + to) / n:.1f}", ti, to, ti + to])

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for ax, m in zip(axes.flat, methods):
        n = n_tasks[m]
        labels = sorted(agg[m], key=bucket_sort_key)
        ins = [agg[m][b][0] / n for b in labels]
        outs = [agg[m][b][1] / n for b in labels]
        x = range(len(labels))
        ax.bar(x, outs, color=OUTPUT_COLOR, label="Output tokens", width=0.75)
        ax.bar(x, ins, bottom=outs, color=INPUT_COLOR, label="Input tokens", width=0.75)
        tops = [i + o for i, o in zip(ins, outs)]
        ymax = max(tops) if tops else 1
        for xi, t in zip(x, tops):
            ax.text(xi, t + ymax * 0.01, f"{t / 1000:.1f}k", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(list(x), labels, fontsize=9)
        ax.set_ylim(0, ymax * 1.12)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        total = sum(tops)
        ax.set_title(f"{METHOD_TITLE.get(m, m)}  (total {total / 1000:.1f}k / task)", fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean tokens per task (all agents)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(methods):]:
        ax.axis("off")
    handles, lbls = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles[::-1], lbls[::-1], loc="upper right", ncol=2, frameon=False)
    fig.suptitle(f"Per-step token usage — {bench}, {model}, n={max(n_tasks.values())} tasks", fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"step_token_usage.{ext}", dpi=200)
    print(f"wrote {csv_path}\nwrote {out_dir / 'step_token_usage.png'}")


if __name__ == "__main__":
    main()
