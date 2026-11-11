"""What does the MUFFIN orchestrator assign? Roles / strategies / subtasks per benchmark x model.

Usage:
    .venv/bin/python scripts/analyze_orchestrator_roles.py \
        [--root output_final/5-repeats] [--out figures/orchestrator_roles]

Reads method_trace.orchestrator_plan from every *_multiagent_dynamic_streaming.json under
<root>/<benchmark>/<model>/<run_id>/examples/ and writes:

    assignments.csv            one row per assigned agent (raw name / role / subtask + category)
    tasks.csv                  one row per task (routing, task_type, team composition, correct)
    routing.csv / .png         direct vs custom multi-agent plan vs fallback template
    role_categories.csv / .png share of functional role categories (custom plans only)
    top_roles.csv              most frequent raw role names per benchmark x model
    top_teams.csv              most frequent team compositions (category multiset)
    accuracy_by_routing.csv    MUFFIN accuracy split by routing outcome
    report.md                  human-readable summary of all of the above

Routing outcome per task
  * direct            orchestrator answered alone (no sub-agents)
  * custom            orchestrator designed its own team (names / roles / subtasks)
  * fallback:<reason> orchestrator output was unusable and the system fell back to the fixed
                      template TaskWorker / CriticalDebateAgent / EvidenceReviewAgent

Role categories (keyword rules on "role + name"; see CATEGORY_RULES). The agent the
orchestrator flags with critical_debate=True is always "Critic / adversary".
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BENCH_ORDER = ["olymmath", "olymmath-hard", "gpqa", "mmlu_pro", "chess"]
BENCH_NAME = {"olymmath": "OlymMATH-E", "olymmath-hard": "OlymMATH-H", "gpqa": "GPQA",
              "mmlu_pro": "MMLU-Pro", "chess": "Chess"}
MODEL_ORDER = ["gemma-4-E4B-it", "gemma-4-31B-it", "Qwen3.5-4B", "Qwen3.8-27B"]

FALLBACK_ROLE_PREFIX = "Works on the core task"

# first match wins (applied to non-critical agents)
CATEGORY_RULES: list[tuple[str, str]] = [
    ("Verifier / checker",
     r"verif|validat|check|audit|confirm|legal|consisten|error|sanity|cross|critic|advers|challeng|review|skeptic|debate"),
    ("Option analyst / eliminator",
     r"multiple.choice|\bchoice|eliminat|comparativ|distractor|candidate|selector|mcq|"
     r"\boptions?\b(?! pric| payoff)"),  # not finance "option pricing"
    ("State / problem modeler",
     r"reconstruct|state|track|setup|set-up|set up|model|formul|pars|simulat|board|position|interpret|translat|"
     r"extract|decompos|framework|structur|represent|diagram|configur"),
    ("Solver / calculator",
     r"solv|calcul|comput|execut|deriv|quantitat|enumerat|count|algebra|arithmetic|bound|estimat|"
     r"optimi|mathemat|proof|prover|integrat|evaluat|numer|kinetic|thermo"),
    ("Domain expert",
     r"expert|specialist|theor|chem|physic|biolog|mechanism|engine|rule|knowledge|domain|scientist|econom|"
     r"financ|account|law|histor|psycholog|medic|engineer|geomet|combinator|probab|analyst|strateg"),
]
CATEGORIES = ["Critic / adversary"] + [c for c, _ in CATEGORY_RULES] + ["Other"]
CAT_COLORS = ["#c55a11", "#f4b183", "#a9d18e", "#9ec5e8", "#1f4e79", "#7030a0", "#bfbfbf"]


def categorize(agent: dict) -> str:
    if agent.get("critical_debate"):
        return "Critic / adversary"
    text = f"{agent.get('role', '')} {agent.get('name', '')}".lower()
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    for cat, pat in CATEGORY_RULES:
        if re.search(pat, text):
            return cat
    return "Other"


def fallback_reason(reason: str) -> str:
    r = (reason or "").lower()
    if "invalid json" in r or "failed" in r:
        return "fallback: invalid JSON"
    if "invalid mode" in r:
        return "fallback: invalid mode"
    if "direct mode without" in r or "certain" in r:
        return "fallback: direct rejected"
    return "fallback: other"


ROUTES = ["direct", "custom", "fallback: direct rejected", "fallback: invalid mode",
          "fallback: invalid JSON", "fallback: other", "no plan"]
ROUTE_COLORS = ["#a9d18e", "#1f4e79", "#f4b183", "#c55a11", "#7f1d1d", "#bfbfbf", "#e5e7eb"]


def order(items, ref):
    return sorted(items, key=lambda x: (ref.index(x) if x in ref else len(ref), x))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def stacked_bar(ax, labels, shares: list[dict], keys, colors, title):
    left = [0.0] * len(labels)
    for k, c in zip(keys, colors):
        vals = [s.get(k, 0) * 100 for s in shares]
        if not any(vals):
            continue
        ax.barh(range(len(labels)), vals, left=left, color=c, label=k, height=0.75)
        for i, v in enumerate(vals):
            if v >= 6:
                ax.text(left[i] + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if c in ("#1f4e79", "#7030a0", "#c55a11", "#7f1d1d") else "black")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("%")
    ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("output_final/5-repeats"))
    ap.add_argument("--out", type=Path, default=Path("figures/orchestrator_roles"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    agents_rows, task_rows = [], []
    for f in sorted(args.root.glob("*/*/*/examples/*_multiagent_dynamic_streaming.json")):
        bench, model, run = f.relative_to(args.root).parts[:3]
        ex = json.loads(f.read_text())
        mt = ex.get("method_trace") or {}
        plan = mt.get("orchestrator_plan") or {}
        sel = plan.get("selected_agents") or []
        mode = mt.get("mode") or plan.get("mode")
        if not plan:
            route = "no plan"
        elif mode == "direct":
            route = "direct"
        elif any(str(a.get("role", "")).startswith(FALLBACK_ROLE_PREFIX) for a in sel):
            route = fallback_reason(plan.get("reason", ""))
        else:
            route = "custom"
        cats = [categorize(a) for a in sel] if route == "custom" else []
        task_rows.append({
            "benchmark": bench, "model": model, "run": run, "index": ex.get("index"),
            "correct": ex.get("correct"), "route": route, "n_agents": len(sel),
            "task_type": plan.get("task_type", ""),
            "team": " + ".join(sorted(cats)),
            "agent_names": " | ".join(a.get("name", "") for a in sel),
            "reason": (plan.get("reason") or "").replace("\n", " "),
        })
        for a in sel:
            agents_rows.append({
                "benchmark": bench, "model": model, "run": run, "index": ex.get("index"),
                "route": route, "task_type": plan.get("task_type", ""),
                "agent_name": a.get("name", ""), "role": a.get("role", ""),
                "critical_debate": a.get("critical_debate"),
                "category": categorize(a) if route == "custom" else "Fallback template",
                "subtask": (a.get("subtask") or "").replace("\n", " "),
                "description": (a.get("description") or "").replace("\n", " "),
            })

    write_csv(args.out / "assignments.csv", agents_rows)
    write_csv(args.out / "tasks.csv", task_rows)

    cells = order({(t["benchmark"], t["model"]) for t in task_rows},
                  [(b, m) for b in BENCH_ORDER for m in MODEL_ORDER])
    by_cell_tasks = defaultdict(list)
    for t in task_rows:
        by_cell_tasks[(t["benchmark"], t["model"])].append(t)
    by_cell_agents = defaultdict(list)
    for a in agents_rows:
        if a["route"] == "custom":
            by_cell_agents[(a["benchmark"], a["model"])].append(a)

    # routing + accuracy by routing
    routing_rows, routing_shares, acc_rows = [], [], []
    for c in cells:
        ts = by_cell_tasks[c]
        cnt = Counter(t["route"] for t in ts)
        share = {r: cnt[r] / len(ts) for r in ROUTES}
        routing_shares.append(share)
        routing_rows.append({"benchmark": c[0], "model": c[1], "n_tasks": len(ts),
                             **{f"{r}_pct": f"{share[r] * 100:.1f}" for r in ROUTES}})
        for r in ROUTES:
            sub = [t for t in ts if t["route"] == r]
            if sub:
                acc_rows.append({"benchmark": c[0], "model": c[1], "route": r, "n_tasks": len(sub),
                                 "accuracy": f"{sum(bool(t['correct']) for t in sub) / len(sub):.3f}"})
    write_csv(args.out / "routing.csv", routing_rows)
    write_csv(args.out / "accuracy_by_routing.csv", acc_rows)

    # role categories
    cat_rows, cat_shares = [], []
    for c in cells:
        ag = by_cell_agents[c]
        cnt = Counter(a["category"] for a in ag)
        n = max(len(ag), 1)
        share = {k: cnt[k] / n for k in CATEGORIES}
        cat_shares.append(share)
        cat_rows.append({"benchmark": c[0], "model": c[1], "n_agents_custom": len(ag),
                         **{f"{k}_pct": f"{share[k] * 100:.1f}" for k in CATEGORIES}})
    write_csv(args.out / "role_categories.csv", cat_rows)

    # top raw roles / teams / task types / subtasks
    top_roles, top_teams = [], []
    md_cells = {}
    for c in cells:
        ag = by_cell_agents[c]
        ts = [t for t in by_cell_tasks[c] if t["route"] == "custom"]
        noncrit = Counter(a["role"].strip() for a in ag if not a["critical_debate"])
        crit = Counter(a["role"].strip() for a in ag if a["critical_debate"])
        names = Counter(a["agent_name"] for a in ag)
        teams = Counter(t["team"] for t in ts)
        ttypes = Counter(t["task_type"].strip().lower().replace(" ", "_") for t in ts)
        for rank, (r, k) in enumerate(noncrit.most_common(10), 1):
            top_roles.append({"benchmark": c[0], "model": c[1], "slot": "worker", "rank": rank,
                              "role": r, "count": k, "pct_of_slot": f"{k / max(sum(noncrit.values()), 1) * 100:.1f}"})
        for rank, (r, k) in enumerate(crit.most_common(5), 1):
            top_roles.append({"benchmark": c[0], "model": c[1], "slot": "critic", "rank": rank,
                              "role": r, "count": k, "pct_of_slot": f"{k / max(sum(crit.values()), 1) * 100:.1f}"})
        for rank, (tm, k) in enumerate(teams.most_common(5), 1):
            top_teams.append({"benchmark": c[0], "model": c[1], "rank": rank, "team": tm, "count": k,
                              "pct": f"{k / max(len(ts), 1) * 100:.1f}"})
        md_cells[c] = dict(noncrit=noncrit, crit=crit, names=names, teams=teams, ttypes=ttypes,
                           n_custom=len(ts), distinct=len(set(a["role"].strip().lower() for a in ag)))
    write_csv(args.out / "top_roles.csv", top_roles)
    write_csv(args.out / "top_teams.csv", top_teams)

    # plots
    labels = [f"{BENCH_NAME.get(b, b)} · {m}" for b, m in cells]
    h = 0.32 * len(cells) + 1.5
    fig, ax = plt.subplots(figsize=(11, h))
    stacked_bar(ax, labels, routing_shares, ROUTES, ROUTE_COLORS,
                "Orchestrator routing outcome (% of tasks)")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06 * 20 / len(cells)))
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"routing.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, h))
    stacked_bar(ax, labels, cat_shares, CATEGORIES, CAT_COLORS,
                "Functional role of assigned agents (% of agents, custom plans only)")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.06 * 20 / len(cells)))
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"role_categories.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # report.md
    L = ["# What the MUFFIN orchestrator assigns", "",
         f"Source: `{args.root}` — {len(task_rows)} MUFFIN tasks, {len(agents_rows)} assigned agents.",
         "Regenerate with `.venv/bin/python scripts/analyze_orchestrator_roles.py`.", "",
         "## Routing (% of tasks)", "",
         "| Benchmark | Model | n | direct | custom team | fallback (direct rejected / invalid mode / invalid JSON) |",
         "|---|---|---|---|---|---|"]
    for row in routing_rows:
        fb = " / ".join(row[f"{r}_pct"] for r in ROUTES[2:5])
        L.append(f"| {BENCH_NAME.get(row['benchmark'], row['benchmark'])} | {row['model']} | {row['n_tasks']} | "
                 f"{row['direct_pct']} | {row['custom_pct']} | {fb} |")
    L += ["", "## Functional role categories (% of agents in custom teams)", "",
          "| Benchmark | Model | " + " | ".join(CATEGORIES) + " |", "|---|---|" + "---|" * len(CATEGORIES)]
    for row in cat_rows:
        L.append(f"| {BENCH_NAME.get(row['benchmark'], row['benchmark'])} | {row['model']} | "
                 + " | ".join(row[f"{k}_pct"] for k in CATEGORIES) + " |")
    L += ["", "## Per benchmark × model", ""]
    for c in cells:
        d = md_cells[c]
        if not d["n_custom"]:
            continue
        L += [f"### {BENCH_NAME.get(c[0], c[0])} · {c[1]}",
              f"{d['n_custom']} custom plans, {d['distinct']} distinct role strings.", "",
              "- **Task types:** " + ", ".join(f"`{k}` ({v})" for k, v in d["ttypes"].most_common(5)),
              "- **Worker roles:** " + ", ".join(f"{k} ({v})" for k, v in d["noncrit"].most_common(6)),
              "- **Critic role:** " + ", ".join(f"{k} ({v})" for k, v in d["crit"].most_common(3)),
              "- **Agent names:** " + ", ".join(f"`{k}` ({v})" for k, v in d["names"].most_common(6)),
              "- **Team compositions:** " + "; ".join(f"{k} ({v})" for k, v in d["teams"].most_common(3)),
              ""]
    (args.out / "report.md").write_text("\n".join(L))
    print(f"tasks={len(task_rows)} agents={len(agents_rows)} -> {args.out}")


if __name__ == "__main__":
    main()
