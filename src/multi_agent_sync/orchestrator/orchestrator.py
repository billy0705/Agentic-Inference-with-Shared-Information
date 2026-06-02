from __future__ import annotations


def create_plan(task: str) -> list[str]:
    return [
        f"Research architecture choices and assumptions for: {task}",
        f"Plan executable prototype modules and APIs for: {task}",
        f"Critique risks, race conditions, and missing safeguards for: {task}",
    ]


def create_assignments(task: str) -> dict[str, str]:
    plan = create_plan(task)
    return {
        "ResearchAgent": plan[0],
        "CodingAgent": plan[1],
        "CriticAgent": plan[2],
    }
