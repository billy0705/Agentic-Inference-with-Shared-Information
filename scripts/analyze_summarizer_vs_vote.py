"""Summarizer vs. majority vote over the exact inputs the summarizer receives.

For every MUFFIN (multiagent_dynamic_streaming) trace, the summarizer is given each agent's
last-step output. method_trace.synthesizer_trace.candidate_aggregation.candidates holds the
answer each of those outputs commits to. We compare, per example:

  summarizer   trace["correct"] (benchmark scorer on the summarizer output)
  vote         plurality vote over the agents' candidates; ties broken uniformly at random
               (scored as expected value); no candidate -> wrong
  strict_vote  correct only if a strict majority of non-empty candidates agrees on a correct answer
  oracle       at least one agent candidate is correct (upper bound for pure selection)

Candidates are graded with the same benchmark scorer used for the summarizer, and grouped by
answer equivalence (OlymMATH: answers_equivalent), so formatting differences do not split votes.
Direct-route examples (no agents, no summarizer) keep trace["correct"] for every column.

Usage:
    .venv/bin/python scripts/analyze_summarizer_vs_vote.py [--root output_final/5-repeats]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from multi_agent_sync.evaluation.benchmarks import chess, gpqa, mmlu_pro, olymmath

METHOD = "multiagent_dynamic_streaming"
BENCH_ORDER = ["chess", "gpqa", "mmlu_pro", "olymmath", "olymmath-hard"]
MODEL_ORDER = ["gemma-4-E4B-it", "gemma-4-31B-it", "Qwen3.5-4B", "Qwen3.8-27B"]


def grader(benchmark: str, trace: dict):
    """Return (normalize(candidate) -> key|None, correct(key) -> bool, same(a, b) -> bool)."""
    gold = str(trace.get("gold"))
    if benchmark.startswith("olymmath"):
        gold_n = olymmath.normalize_answer_text(gold)

        def norm(c):
            return olymmath.extract_answer(f"Final Answer: {c}")

        return norm, (lambda k: olymmath.answers_equivalent(k, gold_n)), olymmath.answers_equivalent
    if benchmark == "chess":
        targets = set((trace.get("score_metadata") or {}).get("valid_targets") or [gold])
        return (lambda c: chess.extract_answer(f"Final Answer: {c}")), (lambda k: k in targets), (lambda a, b: a == b)
    module = {"gpqa": gpqa, "mmlu_pro": mmlu_pro}[benchmark]
    return (lambda c: module.extract_answer(f"Final Answer: {c}")), (lambda k: k == gold), (lambda a, b: a == b)


_CHOICE_RE = re.compile(r"(?im)^\s*ANSWER_CHOICE\s*:\s*(.+?)\s*$")
_FINAL_RE = re.compile(r"(?im)^\s*Final\s+Answer\s*:\s*(.+?)\s*$")


def candidate_from_last_step(agent_trace: dict) -> str:
    """Answer committed in the agent's last step, read after any </think> block.

    candidate_aggregation can pick up template text from inside Qwen's thinking block
    (e.g. "ANSWER_CHOICE: ..."), so we re-extract from the visible part of the response.
    """
    steps = agent_trace.get("steps") or []
    if not steps:
        return ""
    raw = str(steps[-1].get("raw_response") or "")
    visible = raw.rsplit("</think>", 1)[-1]
    for pattern in (_CHOICE_RE, _FINAL_RE):
        found = [m for m in pattern.findall(visible) if m.strip() not in {"...", "<...>"}]
        if found:
            return found[-1].strip()
    return ""


def score_example(benchmark: str, trace: dict) -> dict[str, float] | None:
    summ = float(bool(trace.get("correct")))
    mt = trace.get("method_trace") or {}
    agent_traces = mt.get("agent_traces") or {}
    if mt.get("mode") != "multi_agent" or not agent_traces:
        return {"summarizer": summ, "vote": summ, "strict_vote": summ, "oracle": summ, "multi": 0.0}
    cands = [candidate_from_last_step(a) for a in agent_traces.values()]
    norm, correct, same = grader(benchmark, trace)
    keys = []
    for raw in cands:
        k = norm(raw) if raw and raw.lower() not in {"undecided", "none", "n/a", "unknown"} else None
        if k:
            keys.append(k)
    clusters: list[list[str]] = []
    for k in keys:
        for cl in clusters:
            if same(cl[0], k):
                cl.append(k)
                break
        else:
            clusters.append([k])
    if not clusters:
        return {"summarizer": summ, "vote": 0.0, "strict_vote": 0.0, "oracle": 0.0, "multi": 1.0}
    top = max(len(cl) for cl in clusters)
    tied = [cl for cl in clusters if len(cl) == top]
    vote = sum(correct(cl[0]) for cl in tied) / len(tied)
    strict = float(top > len(keys) / 2 and correct(tied[0][0]))
    oracle = float(any(correct(cl[0]) for cl in clusters))
    return {"summarizer": summ, "vote": vote, "strict_vote": strict, "oracle": oracle, "multi": 1.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output_final/5-repeats")
    args = ap.parse_args()
    root = Path(args.root)

    # (bench, model) -> list of per-run dicts of mean scores
    per_cell: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for run in sorted(root.glob("*/*/*/")):
        files = sorted(glob.glob(str(run / "examples" / f"*_{METHOD}.json")))
        if not files:
            continue
        bench, model = run.parts[-3], run.parts[-2]
        rows = [r for f in files if (r := score_example(bench, json.load(open(f)))) is not None]
        if not rows:
            continue
        agg = {k: 100 * st.mean(r[k] for r in rows) for k in rows[0]}
        agg["n"] = len(rows)
        per_cell[(bench, model)].append(agg)

    cols = ["summarizer", "vote", "strict_vote", "oracle", "multi"]
    print(f"{'benchmark':14s} {'model':16s} runs   " + "  ".join(f"{c:>11s}" for c in cols))
    overall = defaultdict(list)
    for bench in BENCH_ORDER:
        for model in MODEL_ORDER:
            runs = per_cell.get((bench, model))
            if not runs:
                continue
            means = {c: st.mean(r[c] for r in runs) for c in cols}
            for c in cols:
                overall[c].append(means[c])
            print(f"{bench:14s} {model:16s} {len(runs):4d}   " + "  ".join(f"{means[c]:11.1f}" for c in cols))
    print(f"{'AVERAGE':14s} {'(cells)':16s} {len(overall['vote']):4d}   " + "  ".join(f"{st.mean(overall[c]):11.1f}" for c in cols))


if __name__ == "__main__":
    main()
