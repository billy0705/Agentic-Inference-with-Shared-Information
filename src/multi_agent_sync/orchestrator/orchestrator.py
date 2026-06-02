from __future__ import annotations


TASK_TYPES = {
    "simple_qa",
    "calculation",
    "coding_project",
    "debugging_task",
    "research_project",
    "writing_task",
    "architecture_design",
    "unknown",
}


def classify_task(task: str) -> str:
    normalized = task.lower()

    debugging_keywords = ("debug", "bug", "traceback", "failing", "failure", "error", "exception", "pytest")
    coding_keywords = (
        "build",
        "implement",
        "code",
        "prototype",
        "website",
        "api",
        "module",
        "software",
        "app",
        "cli",
        "package",
    )
    architecture_keywords = ("architecture", "design the system", "system design", "microservice", "multi-tenant", "scalable")
    calculation_keywords = (
        "calculate",
        "compute",
        "solve",
        "equation",
        "math",
        "physics",
        "force",
        "acceleration",
        "velocity",
        "threshold",
    )
    research_keywords = ("research", "investigate", "compare", "survey", "tradeoff", "trade-off", "literature")
    writing_keywords = ("write", "draft", "compose", "email", "announcement", "blog", "essay", "copy")
    simple_qa_prefixes = ("what is", "who is", "when is", "where is", "define", "explain")

    if any(keyword in normalized for keyword in debugging_keywords):
        return "debugging_task"
    if any(keyword in normalized for keyword in architecture_keywords):
        return "architecture_design"
    if any(keyword in normalized for keyword in calculation_keywords) or _contains_numbers_and_operator(normalized):
        return "calculation"
    if normalized.startswith(simple_qa_prefixes) or "which one of the following" in normalized:
        return "simple_qa"
    if any(keyword in normalized for keyword in coding_keywords):
        return "coding_project"
    if any(keyword in normalized for keyword in research_keywords):
        return "research_project"
    if any(keyword in normalized for keyword in writing_keywords):
        return "writing_task"
    return "unknown"


def create_orchestrator_plan(task: str) -> dict:
    task_type = classify_task(task)

    if task_type == "simple_qa":
        return _direct_plan(task_type, "The task looks answerable with one direct LLM response.")
    if task_type == "writing_task":
        return _direct_plan(task_type, "The task is a writing request that does not require runtime agent synchronization.")
    if task_type == "calculation":
        return {
            "mode": "multi_agent",
            "task_type": task_type,
            "reason": "Calculation/reasoning tasks benefit from a solver and an independent verifier.",
            "assignments": [
                {
                    "agent_name": "SolverAgent",
                    "task": "Solve the calculation or reasoning problem step by step at summary level.",
                    "max_steps": 2,
                },
                {
                    "agent_name": "VerifierAgent",
                    "task": "Check numerical correctness, units, assumptions, and overclaiming.",
                    "max_steps": 1,
                },
            ],
        }
    if task_type == "debugging_task":
        return {
            "mode": "multi_agent",
            "task_type": task_type,
            "reason": "Debugging needs implementation analysis plus critique of risks and missing cases.",
            "assignments": [
                {"agent_name": "CodingAgent", "task": f"Analyze the debugging task and propose a fix path: {task}", "max_steps": 3},
                {"agent_name": "CriticAgent", "task": "Review the debugging plan for risks and missing verification.", "max_steps": 2},
            ],
        }
    if task_type in {"coding_project", "architecture_design"}:
        return {
            "mode": "multi_agent",
            "task_type": task_type,
            "reason": "Software and architecture tasks need research, implementation planning, and critique.",
            "assignments": [
                {"agent_name": "ResearchAgent", "task": f"Research architecture choices and assumptions for: {task}", "max_steps": 3},
                {"agent_name": "CodingAgent", "task": f"Plan executable modules, APIs, and implementation steps for: {task}", "max_steps": 3},
                {"agent_name": "CriticAgent", "task": "Review the plan for risks, race conditions, missing cases, and safety issues.", "max_steps": 2},
            ],
        }
    if task_type == "research_project":
        return {
            "mode": "multi_agent",
            "task_type": task_type,
            "reason": "Open-ended research benefits from a research worker and a critical reviewer.",
            "assignments": [
                {"agent_name": "ResearchAgent", "task": f"Investigate background, options, and constraints for: {task}", "max_steps": 3},
                {"agent_name": "CriticAgent", "task": "Review the research findings for gaps and weak assumptions.", "max_steps": 2},
            ],
        }

    return {
        "mode": "multi_agent",
        "task_type": "unknown",
        "reason": "The task type is unclear, so use a small research-and-critique setup.",
        "assignments": [
            {"agent_name": "ResearchAgent", "task": f"Clarify assumptions and possible approaches for: {task}", "max_steps": 2},
            {"agent_name": "CriticAgent", "task": "Identify risks, missing information, and uncertainty.", "max_steps": 1},
        ],
    }


def create_plan(task: str) -> list[str]:
    return [assignment["task"] for assignment in create_orchestrator_plan(task)["assignments"]]


def create_assignments(task: str) -> list[dict]:
    return create_orchestrator_plan(task)["assignments"]


def _direct_plan(task_type: str, reason: str) -> dict:
    return {
        "mode": "direct",
        "task_type": task_type,
        "reason": reason,
        "assignments": [],
    }


def _contains_numbers_and_operator(task: str) -> bool:
    has_digit = any(character.isdigit() for character in task)
    has_operator = any(operator in task for operator in ("+", "-", "*", "/", "="))
    return has_digit and has_operator
