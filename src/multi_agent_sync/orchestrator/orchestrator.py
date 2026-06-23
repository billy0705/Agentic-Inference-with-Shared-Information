from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from multi_agent_sync.prompts import render_prompt


Mode = Literal["direct", "multi_agent"]


class SelectedAgent(TypedDict):
    name: str
    subtask: str
    expected_output: str


class CollaborationProtocol(TypedDict):
    event_types_to_share: list[str]
    reactive_steps: bool
    notes: str


class OrchestratorPlan(TypedDict):
    mode: Mode
    task_type: str
    task_summary: str
    reason: str
    selected_agents: list[SelectedAgent]
    collaboration_protocol: CollaborationProtocol


AGENT_DESCRIPTIONS = {
    "ResearchAgent": "Best for background research, assumptions, constraints, domain context, comparisons, possible viewpoints, and information gathering.",
    "SolverAgent": "Best for mathematical reasoning, logical reasoning, philosophy questions, abstract analysis, proofs, general problem solving, and non-code analytical tasks.",
    "CodingAgent": "Best for implementation plans, code structure, concrete patches, debugging, software errors, APIs, tests, and executable prototypes.",
    "CriticAgent": "Best for finding weaknesses, missing assumptions, counterarguments, risks, contradictions, edge cases, and alternative interpretations.",
    "VerifierAgent": "Best for checking correctness, validating calculations, checking consistency, testing logic, and confirming that the final solution satisfies the original task.",
}

DEFAULT_EVENT_TYPES = ["finding", "critique", "warning"]
MAX_SELECTED_AGENTS = 4

CODE_KEYWORDS = ("code", "programming", "bug", "error", "pytest", "function", "api", "implementation", "repository")
RESEARCH_KEYWORDS = ("research", "comparison", "compare", "literature", "recent work", "background", "evidence")
REASONING_KEYWORDS = ("math", "calculate", "calculation", "proof", "prove", "theorem", "logic", "philosophy", "reasoning", "argument", "theory")


async def create_model_based_plan(task: str, llm: Any, available_agents: Mapping[str, Any]) -> OrchestratorPlan:
    prompt = build_orchestrator_prompt(task, available_agents)
    try:
        response = await llm.ainvoke(prompt)
        content = getattr(response, "content", str(response))
        raw_plan = extract_json_object(content)
    except Exception as exc:
        return create_fallback_plan(task, available_agents, reason=f"Model orchestrator failed or returned invalid JSON: {exc}")

    return validate_orchestrator_plan(raw_plan, task, available_agents)


async def create_orchestrator_plan(task: str, llm: Any, available_agents: Mapping[str, Any]) -> OrchestratorPlan:
    return await create_model_based_plan(task, llm, available_agents)


def build_orchestrator_prompt(task: str, available_agents: Mapping[str, Any]) -> str:
    agent_lines = "\n".join(
        f"{index}. {name}\n    {AGENT_DESCRIPTIONS.get(name, 'Registered worker agent.')}"
        for index, name in enumerate(available_agents, start=1)
    )
    allowed_names = " | ".join(available_agents)
    return render_prompt(
        "orchestrator/model_plan.j2",
        task=task,
        agent_lines=agent_lines,
        allowed_names=allowed_names,
    )


def extract_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", stripped):
            try:
                value, _ = decoder.raw_decode(stripped[match.start() :])
                break
            except json.JSONDecodeError:
                continue
        else:
            raise

    if not isinstance(value, dict):
        raise ValueError("orchestrator response must be a JSON object")
    return value


def validate_orchestrator_plan(raw_plan: Mapping[str, Any], task: str, available_agents: Mapping[str, Any]) -> OrchestratorPlan:
    mode = raw_plan.get("mode")
    if mode not in {"direct", "multi_agent"}:
        return create_fallback_plan(task, available_agents, reason="Model orchestrator returned an invalid mode.")

    task_type = _clean_text(raw_plan.get("task_type")) or infer_fallback_task_type(task)
    if task_type.lower() == "unknown" and task.strip():
        task_type = infer_fallback_task_type(task)
    task_summary = _clean_text(raw_plan.get("task_summary")) or f"Handle the user task: {task}"
    reason = _clean_text(raw_plan.get("reason")) or "The model orchestrator selected this route."
    collaboration_protocol = normalize_collaboration_protocol(raw_plan.get("collaboration_protocol"))

    selected_agents = normalize_selected_agents(raw_plan.get("selected_agents"), available_agents)
    if mode == "direct":
        selected_agents = []
    elif not selected_agents or _is_critic_alone(selected_agents):
        return create_fallback_plan(task, available_agents, reason="Model orchestrator did not select a valid multi-agent pool.")

    return {
        "mode": mode,
        "task_type": task_type,
        "task_summary": task_summary,
        "reason": reason,
        "selected_agents": selected_agents[:MAX_SELECTED_AGENTS],
        "collaboration_protocol": collaboration_protocol,
    }


def normalize_selected_agents(value: Any, available_agents: Mapping[str, Any]) -> list[SelectedAgent]:
    if not isinstance(value, list):
        return []

    selected_agents: list[SelectedAgent] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = _clean_text(item.get("name"))
        if name not in available_agents or name in seen:
            continue
        selected_agents.append(
            {
                "name": name,
                "subtask": _clean_text(item.get("subtask")) or f"Contribute as {name} to the user task.",
                "expected_output": _clean_text(item.get("expected_output")) or "Concise findings for final synthesis.",
            }
        )
        seen.add(name)
        if len(selected_agents) == MAX_SELECTED_AGENTS:
            break
    return selected_agents


def normalize_collaboration_protocol(value: Any) -> CollaborationProtocol:
    if not isinstance(value, Mapping):
        return default_collaboration_protocol()

    raw_event_types = value.get("event_types_to_share")
    event_types = []
    if isinstance(raw_event_types, list):
        event_types = [event_type for event_type in raw_event_types if isinstance(event_type, str) and event_type.strip()]
    if not event_types:
        event_types = list(DEFAULT_EVENT_TYPES)
    reactive_steps = value.get("reactive_steps")
    if not isinstance(reactive_steps, bool):
        reactive_steps = True

    return {
        "event_types_to_share": event_types[:5],
        "reactive_steps": reactive_steps,
        "notes": _clean_text(value.get("notes")) or "Agents should share useful findings, critiques, and warnings for final synthesis.",
    }


def create_fallback_plan(task: str, available_agents: Mapping[str, Any], reason: str | None = None) -> OrchestratorPlan:
    task_type = infer_fallback_task_type(task)
    selected_names = fallback_agent_names(task)
    selected_names = [name for name in selected_names if name in available_agents][:MAX_SELECTED_AGENTS]
    if not selected_names:
        selected_names = [name for name in ("SolverAgent", "CriticAgent") if name in available_agents]
    if selected_names == ["CriticAgent"] and "SolverAgent" in available_agents:
        selected_names.insert(0, "SolverAgent")

    return {
        "mode": "multi_agent",
        "task_type": task_type,
        "task_summary": f"Handle the user task: {task}",
        "reason": reason or "Using deterministic fallback routing because model orchestration was unavailable or invalid.",
        "selected_agents": [
            {
                "name": name,
                "subtask": fallback_subtask(name, task),
                "expected_output": fallback_expected_output(name),
            }
            for name in selected_names
        ],
        "collaboration_protocol": default_collaboration_protocol(),
    }


def fallback_agent_names(task: str) -> list[str]:
    normalized = task.lower()
    if any(keyword in normalized for keyword in CODE_KEYWORDS):
        return ["CodingAgent", "CriticAgent", "VerifierAgent"]
    if any(keyword in normalized for keyword in RESEARCH_KEYWORDS):
        return ["ResearchAgent", "SolverAgent", "CriticAgent"]
    if any(keyword in normalized for keyword in REASONING_KEYWORDS):
        return ["SolverAgent", "CriticAgent", "VerifierAgent"]
    return ["SolverAgent", "CriticAgent"]


def infer_fallback_task_type(task: str) -> str:
    normalized = task.lower()
    if any(keyword in normalized for keyword in CODE_KEYWORDS):
        return "software or debugging task"
    if any(keyword in normalized for keyword in RESEARCH_KEYWORDS):
        return "research and comparison task"
    if any(keyword in normalized for keyword in REASONING_KEYWORDS):
        return "reasoning and verification task"
    if not task.strip():
        return "unknown"
    return "general reasoning task"


def fallback_subtask(agent_name: str, task: str) -> str:
    if agent_name == "ResearchAgent":
        return f"Identify background context, assumptions, comparisons, and useful evidence for: {task}"
    if agent_name == "CodingAgent":
        return f"Analyze the software task and propose concrete implementation, debugging, API, or test steps for: {task}"
    if agent_name == "CriticAgent":
        return "Identify weaknesses, missing assumptions, risks, counterarguments, and edge cases in the emerging answer."
    if agent_name == "VerifierAgent":
        return "Validate correctness, consistency, calculations, constraints, and final-answer fit against the user task."
    return f"Solve or analyze the core problem carefully and share concise reasoning findings for: {task}"


def fallback_expected_output(agent_name: str) -> str:
    if agent_name == "ResearchAgent":
        return "Concise context, assumptions, options, and evidence."
    if agent_name == "CodingAgent":
        return "Concrete implementation or debugging plan with relevant tests."
    if agent_name == "CriticAgent":
        return "Weaknesses, edge cases, counterarguments, and risks."
    if agent_name == "VerifierAgent":
        return "Correctness checks, validation notes, and any corrections."
    return "Reasoned solution steps and important findings."


def selected_agents_to_assignments(plan: OrchestratorPlan, max_steps: int = 3) -> list[dict[str, Any]]:
    protocol = plan["collaboration_protocol"]
    return [
        {
            "agent_name": agent["name"],
            "task": agent["subtask"],
            "expected_output": agent["expected_output"],
            "max_steps": max_steps,
            "reactive_steps_enabled": protocol["reactive_steps"],
            "max_reactive_steps": 1,
            "reactive_event_types": protocol["event_types_to_share"],
        }
        for agent in plan["selected_agents"]
    ]


def create_plan(task: str) -> list[str]:
    from multi_agent_sync.agents.registry import AGENT_REGISTRY

    return [agent["subtask"] for agent in create_fallback_plan(task, AGENT_REGISTRY).get("selected_agents", [])]


def create_assignments(task: str) -> list[dict[str, Any]]:
    from multi_agent_sync.agents.registry import AGENT_REGISTRY

    return selected_agents_to_assignments(create_fallback_plan(task, AGENT_REGISTRY))


def classify_task(task: str) -> str:
    return infer_fallback_task_type(task)


def default_collaboration_protocol() -> CollaborationProtocol:
    return {
        "event_types_to_share": list(DEFAULT_EVENT_TYPES),
        "reactive_steps": True,
        "notes": "Agents should share findings, critiques, and warnings, then react to useful messages from other agents when available.",
    }


def _clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _is_critic_alone(selected_agents: list[SelectedAgent]) -> bool:
    return len(selected_agents) == 1 and selected_agents[0]["name"] == "CriticAgent"
