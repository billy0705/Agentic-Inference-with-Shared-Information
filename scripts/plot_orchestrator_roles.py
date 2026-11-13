"""Paper figure: which functional roles does the MUFFIN orchestrator assign, per task domain?

Usage:
    .venv/bin/python scripts/plot_orchestrator_roles.py [--assignments figures/orchestrator_roles/assignments.csv]
        [--out figures/orchestrator_roles]

Input: assignments.csv from scripts/analyze_orchestrator_roles.py (one row per assigned agent).
Only teams the orchestrator designed itself ("custom" plans) are used.

Every custom team has 3 agents. Exactly one of them is the Critic: the first agent the
orchestrator flags with critical_debate=True (5 / 5333 teams flag two; the second one is
classified by its role keywords like any other agent). Hence Critic = 33.3% everywhere.

Categories (one definition each). The function is read from the agent's CamelCase name, whose head
noun names the operation (see function_of):
  Critic          challenges the other agents' findings (critical_debate flag)
  Verifier        checks correctness / consistency of intermediate results
  Modeler         turns the problem into a structured representation (board state, variables, equations)
  Solver          carries out the derivation / computation
  Domain expert   contributes subject knowledge (physics, chemistry, law, ...)
  Option analyst  compares and eliminates multiple-choice options
  Other           no rule matched

Outputs:
  role_shares.csv                  % of agents per category, per benchmark x model and per benchmark
  role_shares_by_domain.{png,pdf}  one bar per benchmark (mean over models)       -> main text
  role_shares_by_model.{png,pdf}   one bar per benchmark x model, grouped by domain -> appendix
  role_examples.csv / role_examples_table.tex   most frequent concrete agent names per domain x role
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_orchestrator_roles import BENCH_NAME, BENCH_ORDER, MODEL_ORDER, categorize  # noqa: E402

RENAME = {"Critic / adversary": "Verifier", "Verifier / checker": "Verifier",
          "State / problem modeler": "Modeler", "Solver / calculator": "Solver",
          "Domain expert": "Domain expert", "Option analyst / eliminator": "Option analyst", "Other": "Other"}

# Agent names are CamelCase compounds "<subject><function>", e.g. Board|State|Reconstructor,
# Financial|Calculator, Reaction|Mechanist. The function is read from the head noun (last word).
OPTION_WORDS = re.compile(r"option|distractor|choice", re.I)
HEAD = {
    "Verifier": "verifier validator checker auditor legalizer tester verification validation legality check "
                "checking audit consistency",
    "Modeler": "reconstructor modeler modeller simulator mapper interpreter tracker parser engine transformer "
               "converter constructor architect builder formulator formalizer formaliser identifier locator dimensioner reconstruction modeling modelling "
               "simulation setup state parsing tracking mapping formulation representation",
    "Solver": "calculator solver deriver counter optimizer optimiser simplifier enumerator generator integrator "
              "evaluator estimator computer calculation computation derivation solving counting optimization "
              "enumeration searcher finder bounder expander manipulator factorizer integration differentiation",
    "Domain expert": "expert specialist analyst analyzer analyser mechanist chemist stereochemist theorist "
                     "physicist geometer biologist engineer strategist researcher scientist economist lawyer "
                     "historian psychologist consultant advisor",
}
HEAD = {w: cat for cat, ws in HEAD.items() for w in ws.split()}
GENERIC = {"agent", "worker", "unit", "module", "assistant"}


def words(name: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", name)]


# leading verb of the subtask / role (both are usually imperative sentences: "Calculate ...")
VERBS = [
    ("Option analyst", r"match|map|select|choos|eliminat|evaluat|compar|filter|assign|assess|rank"),
    ("Verifier", r"verif|validat|check|audit|confirm|test|critiqu|double.check|cross.check"),
    ("Modeler", r"model|define|set|establish|identif|build|construct|trace|formul|represent|reconstruct|"
                r"translat|pars|extract|convert|rewrite|express|map|track|simulat|describ|interpret|propos|transform"),
    ("Solver", r"calculat|comput|deriv|solv|simplif|determin|find|prov|optimi|maximi|minimi|count|"
               r"enumerat|estimat|explor|evaluat|integrat|bound|appl|perform|generat|search"),
    ("Domain expert", r"analy|assess|explain|deduc|recall|predict|review|judg|consider|examin|research|"
                     r"compar|retriev|classif|filter|assign|infer"),
]


def verb_function(text: str) -> str | None:
    m = re.match(r"\W*([A-Za-z-]+)", text or "")
    if not m:
        return None
    v = m.group(1).lower()
    for cat, pat in VERBS:
        if re.match(pat, v):
            if cat == "Option analyst" and not OPTION_WORDS.search(text):
                continue  # "match residues", "select a square": not about answer options
            return cat
    return None


def function_of(name: str, role: str = "", subtask: str = "") -> str:
    """Functional role of a non-critic agent.

    1. option / choice / distractor in the name           -> Option analyst
    2. head noun (last word; '...Agent' -> previous word) is an operation
       (Verifier, Reconstructor, Calculator, ...)            -> that function
    3. head noun is a persona (Expert, Specialist, Analyst, Chemist, ...):
       an earlier operation word wins (MoveLegalityExpert -> Verifier),
       otherwise                                             -> Domain expert
    4. the name only states a subject (PhotonMomentumAgent): leading verb of the
       subtask, then of the role ("Calculate ..." -> Solver), else Other
    """
    if OPTION_WORDS.search(name):
        return "Option analyst"
    ws = [w for w in words(name) if w not in GENERIC]
    if not ws:
        return "Other"
    head = HEAD.get(ws[-1])
    if head and head != "Domain expert":
        return head
    for w in reversed(ws[:-1]):
        if HEAD.get(w) not in (None, "Domain expert"):
            return HEAD[w]
    if head == "Domain expert":
        return head
    return verb_function(subtask) or verb_function(role) or "Other"


CATS = ["Critic", "Verifier", "Modeler", "Solver", "Domain expert", "Option analyst", "Other"]
COLORS = {  # neutral grays for the fixed / leftover slots, validated categorical hues for the rest
    "Critic": "#52514e", "Verifier": "#2a78d6", "Modeler": "#eb6834", "Solver": "#1baf7a",
    "Domain expert": "#eda100", "Option analyst": "#e87ba4", "Other": "#c9c8c2",
}
WHITE_TEXT = {"Critic", "Verifier"}
GAP = 0.6  # vertical gap between benchmark groups


def load(path: Path):
    """counts[(bench, model)][category]; names[bench][category][agent_name] (non-critic);
    n_teams[bench]."""
    teams: dict[tuple, list[dict]] = defaultdict(list)
    with path.open() as fh:
        for r in csv.DictReader(fh):
            if r["route"] == "custom":
                teams[(r["benchmark"], r["model"], r["run"], r["index"])].append(r)
    counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    names: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    n_teams: Counter = Counter()
    for (b, m, *_), agents in teams.items():
        n_teams[b] += 1
        critic_seen = False
        for a in agents:
            is_critic = a["critical_debate"] == "True" and not critic_seen
            critic_seen |= is_critic
            cat = "Critic" if is_critic else function_of(a["agent_name"], a["role"], a["subtask"])
            counts[(b, m)][cat] += 1
            if not is_critic:
                names[b][cat][a["agent_name"]] += 1
    return counts, names, n_teams


def shares(c: Counter) -> dict[str, float]:
    n = sum(c.values())
    return {k: 100 * c[k] / n for k in CATS}


def draw(ax, rows: list[tuple[float, str, dict[str, float]]], bold: set[str] = frozenset()) -> None:
    for y, _, s in rows:
        left = 0.0
        for k in CATS:
            v = s[k]
            if v <= 0:
                continue
            ax.barh(y, v, left=left, color=COLORS[k], height=0.78, edgecolor="white", linewidth=1.2)
            if v >= 5:
                ax.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if k in WHITE_TEXT else "#0b0b0b")
            left += v
    ax.set_yticks([y for y, *_ in rows], [l for _, l, _ in rows], fontsize=8.5)
    for t in ax.get_yticklabels():
        if t.get_text() in bold:
            t.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of assigned agents (%)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def legend(fig, ax) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in CATS]
    ax.legend(handles, CATS, ncol=len(CATS), frameon=False, fontsize=8.5, handlelength=1.2,
              columnspacing=1.0, loc="lower left", bbox_to_anchor=(0, 1.0))


def pct(v: float) -> str:
    return r"$<$1\%" if v < 1 else f"{v:.0f}\\%"


def save(fig, stem: Path) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(stem.parent / f"{stem.name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments", type=Path, default=Path("figures/orchestrator_roles/assignments.csv"))
    ap.add_argument("--out", type=Path, default=Path("figures/orchestrator_roles"))
    args = ap.parse_args()

    counts, names, n_teams = load(args.assignments)
    benches = [b for b in BENCH_ORDER if any(k[0] == b for k in counts)]
    cell = {k: shares(c) for k, c in counts.items()}
    bench_avg = {}  # unweighted mean over models
    for b in benches:
        ms = [cell[(b, m)] for m in MODEL_ORDER if (b, m) in cell]
        bench_avg[b] = {k: sum(s[k] for s in ms) / len(ms) for k in CATS}

    with (args.out / "role_shares.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "model", "n_agents"] + [f"{k}_pct" for k in CATS])
        for b in benches:
            for m in MODEL_ORDER:
                if (b, m) in cell:
                    w.writerow([b, m, sum(counts[(b, m)].values())] + [f"{cell[(b, m)][k]:.1f}" for k in CATS])
            w.writerow([b, "MEAN_OVER_MODELS", ""] + [f"{bench_avg[b][k]:.1f}" for k in CATS])

    # main-text figure: one bar per domain
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(benches) + 1.0))
    draw(ax, [(i, BENCH_NAME[b], bench_avg[b]) for i, b in enumerate(benches)])
    legend(fig, ax)
    save(fig, args.out / "role_shares_by_domain")

    # appendix figure: grouped by domain, one bar per model
    rows, y, group_mid = [], 0.0, []
    for b in benches:
        start = y
        for m in MODEL_ORDER:
            if (b, m) in cell:
                rows.append((y, m, cell[(b, m)]))
                y += 1
        group_mid.append((start, y - 1, BENCH_NAME[b]))
        y += GAP
    fig, ax = plt.subplots(figsize=(9, 0.3 * y + 1.2))
    draw(ax, rows)
    for y0, y1, name in group_mid:
        ax.text(-24, (y0 + y1) / 2, name, ha="right", va="center", fontsize=9.5, fontweight="bold",
                transform=ax.transData, clip_on=False)
        ax.plot([-23, -23], [y0 - 0.35, y1 + 0.35], color="#52514e", linewidth=1, clip_on=False)
    legend(fig, ax)
    save(fig, args.out / "role_shares_by_model")
    print(open(args.out / "role_shares.csv").read())

    # examples: most frequent concrete agent names per domain x function
    # (% = share of that domain's teams that contain an agent with exactly this name)
    with (args.out / "role_examples.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["benchmark", "n_teams", "n_distinct_worker_names", "function", "rank", "agent_name",
                    "count", "pct_of_teams"])
        for b in benches:
            distinct = len(set().union(*[c.keys() for c in names[b].values()]))
            for cat in CATS[1:]:
                for r, (nm, k) in enumerate(names[b][cat].most_common(5), 1):
                    w.writerow([b, n_teams[b], distinct, cat, r, nm, k, f"{100 * k / n_teams[b]:.1f}"])

    # LaTeX table: what each function looks like in each domain (top-2 names)
    tex = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{Most frequent agent names created by the orchestrator for each functional role and "
           r"domain (share of teams containing that agent in parentheses; -- if the role makes up less than 2\% of "
           r"the agents in that domain). The last column gives the number of distinct agent names (excluding the "
           r"Critic) and the number of teams.}",
           r"\label{tab:role_examples}",
           r"\resizebox{\textwidth}{!}{%",
           r"\begin{tabular}{l" + "l" * 5 + "r}", r"\toprule",
           " & ".join(["Domain", "Modeler", "Solver", "Domain expert", "Verifier", "Option analyst", r"\makecell[r]{Distinct names\\/ teams}"]) + r" \\", r"\midrule"]
    for b in benches:
        cells = []
        for cat in ["Modeler", "Solver", "Domain expert", "Verifier", "Option analyst"]:
            top = [(nm, k) for nm, k in names[b][cat].most_common(2)] if bench_avg[b][cat] >= 2 else []
            cells.append(r"\makecell[l]{" + r" \\ ".join(
                rf"\texttt{{{nm}}} ({pct(100 * k / n_teams[b])})" for nm, k in top) + "}" if top else "--")
        distinct = len(set().union(*[c.keys() for c in names[b].values()]))
        cells.append(f"{distinct:,} / {n_teams[b]:,}")
        tex.append(BENCH_NAME[b] + " & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (args.out / "role_examples_table.tex").write_text("\n".join(tex) + "\n")


if __name__ == "__main__":
    main()
