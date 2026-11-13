"""Mean input / output tokens per task for every benchmark x model x method (Table 1 layout).

Usage:
    .venv/bin/python scripts/token_usage_table.py [--root output_final/5-repeats]
        [--out figures/token_usage_table]

Per task, tokens are the task's total `token_usage` (all agents, orchestrator, synthesizer).
Tasks without recorded usage (errored, e.g. debate context overflow) are skipped.
All tasks of all runs of a (benchmark, model, method) cell are pooled, so partial /
resumed runs are handled without giving a 9-task run the same weight as a 100-task run.

Writes <out>/token_usage.csv (long format) and <out>/token_usage_table.tex.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

BENCHES = [("chess", "Chess"), ("gpqa", "GPQA"), ("mmlu_pro", "MMLU-Pro"),
           ("olymmath", "OlymMATH-E"), ("olymmath-hard", "OlymMATH-H")]
MODELS = ["gemma-4-E4B-it", "gemma-4-31B-it", "Qwen3.5-4B", "Qwen3.8-27B"]
METHODS = [("single_agent", "Single-Agent"), ("majority_vote", "Majority Vote"),
           ("multiagent_debate", "Multi-Agent Debate"), ("multiagent_dynamic_streaming", "Muffin")]


def k(v: float | None) -> str:
    return "--" if v is None else f"{v / 1000:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("output_final/5-repeats"))
    ap.add_argument("--out", type=Path, default=Path("figures/token_usage_table"))
    args = ap.parse_args()

    # (bench, model, method) -> [sum_in, sum_out, n_tasks, n_skipped, set(runs)]
    agg = defaultdict(lambda: [0, 0, 0, 0, set()])
    for f in args.root.glob("*/*/*/examples/*.json"):
        bench, model, run = f.parts[-5], f.parts[-4], f.parts[-3]
        ex = json.loads(f.read_text())
        a = agg[(bench, model, ex["method"])]
        tu = ex.get("token_usage") or {}
        if not tu.get("prompt_tokens"):
            a[3] += 1
            continue
        a[0] += tu["prompt_tokens"]
        a[1] += tu.get("completion_tokens") or 0
        a[2] += 1
        a[4].add(run)

    def mean(b, m, meth):
        a = agg.get((b, m, meth))
        return (a[0] / a[2], a[1] / a[2]) if a and a[2] else (None, None)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "token_usage.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "model", "method", "n_runs", "n_tasks", "n_skipped_no_usage",
                    "mean_input_tokens", "mean_output_tokens", "mean_total_tokens"])
        for b, _ in BENCHES:
            for m in MODELS:
                for meth, _ in METHODS:
                    a = agg.get((b, m, meth))
                    if not a:
                        continue
                    i, o = mean(b, m, meth)
                    w.writerow([b, m, meth, len(a[4]), a[2], a[3],
                                *(f"{x:.1f}" if x is not None else "" for x in (i, o, (i or 0) + (o or 0)))])

    # LaTeX: In / Out per benchmark, then an average block over benchmarks:
    #   Avg. In / Out / Total (thousands) and Total relative to Muffin of the same model.
    def avg(m, meth):
        vals = [mean(b, m, meth) for b, _ in BENCHES]
        vals = [v for v in vals if v[0] is not None]
        if not vals:
            return None, None
        return sum(v[0] for v in vals) / len(vals), sum(v[1] for v in vals) / len(vals)

    nb = len(BENCHES)
    lines = [
        r"\begin{table*}[H]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\caption{Mean input/output tokens per task (in thousands), summed over all LLM calls of a "
        r"task (agents, orchestrator and synthesizer). Averaged over all tasks of the runs in "
        r"\Cref{tab:hypothesis1_accuracy}. \emph{Avg.} is the unweighted mean over the five benchmarks; "
        r"\emph{$\times$Muffin} is the average total token usage relative to Muffin with the same model.}",
        r"\label{tab:token_usage}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll" + "rr" * nb + "|rrrr}",
        r"\toprule",
        r" & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for _, name in BENCHES)
        + r" & \multicolumn{4}{c}{Avg.} \\",
        "% " + " ".join(rf"\cmidrule(lr){{{3 + 2 * j}-{4 + 2 * j}}}" for j in range(nb + 1)),
        r"Model & Method & " + " & ".join(["In", "Out"] * nb)
        + r" & In & Out & Total & $\times$Muffin \\",
        r"\midrule",
    ]
    for mi, m in enumerate(MODELS):
        mi_, mo_ = avg(m, "multiagent_dynamic_streaming")
        muf_total = (mi_ or 0) + (mo_ or 0)
        for j, (meth, name) in enumerate(METHODS):
            cells = []
            for b, _ in BENCHES:
                i, o = mean(b, m, meth)
                cells += [k(i), k(o)]
            ai, ao = avg(m, meth)
            tot = None if ai is None else ai + ao
            ratio = "--" if tot is None or not muf_total else f"{tot / muf_total:.2f}$\\times$"
            cells += [k(ai), k(ao), k(tot), ratio]
            row = (m if j == 0 else "") + f" & {name} & " + " & ".join(cells) + r" \\"
            lines.append(row)
        lines.append(r"\midrule" if mi < len(MODELS) - 1 else r"\bottomrule")
    lines += [r"\end{tabular}}", r"\end{table*}"]
    (args.out / "token_usage_table.tex").write_text("\n".join(lines) + "\n")
    print((args.out / "token_usage_table.tex").read_text())


if __name__ == "__main__":
    main()
