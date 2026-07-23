from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict

from multi_agent_sync.prompts import render_prompt


Mode = Literal["multi_agent", "direct"]
SubagentMode = Literal["fixed", "dynamic"]


class SelectedAgent(TypedDict, total=False):
    name: str
    subtask: str
    expected_output: str
    role: str
    description: str
    rules: list[str]
    critical_debate: bool
    workspace_access: str


class CollaborationProtocol(TypedDict):
    event_types_to_share: list[str]
    reactive_steps: bool
    notes: str


class OrchestratorPlan(TypedDict):
    mode: Mode
    subagent_mode: SubagentMode
    task_type: str
    task_summary: str
    reason: str
    selected_agents: list[SelectedAgent]
    collaboration_protocol: CollaborationProtocol
    direct_certainty: NotRequired[str]


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
async def create_model_based_plan(
    task: str,
    llm: Any,
    available_agents: Mapping[str, Any],
    subagent_mode: SubagentMode = "fixed",
) -> OrchestratorPlan:
    prompt = (
        build_dynamic_orchestrator_prompt(task)
        if subagent_mode == "dynamic"
        else build_orchestrator_prompt(task, available_agents)
    )
    try:
        response = await llm.ainvoke(prompt)
        content = getattr(response, "content", str(response))
        raw_plan = extract_json_object(content)
    except Exception as exc:
        return create_fallback_plan(
            task,
            available_agents,
            reason=f"Model orchestrator failed or returned invalid JSON: {exc}",
            subagent_mode=subagent_mode,
        )

    return validate_orchestrator_plan(raw_plan, task, available_agents, subagent_mode=subagent_mode)


async def create_orchestrator_plan(
    task: str,
    llm: Any,
    available_agents: Mapping[str, Any],
    subagent_mode: SubagentMode = "fixed",
) -> OrchestratorPlan:
    return await create_model_based_plan(task, llm, available_agents, subagent_mode=subagent_mode)


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


def build_dynamic_orchestrator_prompt(task: str) -> str:
    return render_prompt("orchestrator/dynamic_model_plan.j2", task=task)


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


def validate_orchestrator_plan(
    raw_plan: Mapping[str, Any],
    task: str,
    available_agents: Mapping[str, Any],
    subagent_mode: SubagentMode = "fixed",
) -> OrchestratorPlan:
    mode = raw_plan.get("mode")
    if mode == "direct":
        if subagent_mode == "dynamic" and not is_truthful_dynamic_direct_plan(raw_plan, task):
            return create_fallback_plan(
                task,
                available_agents,
                reason=(
                    "Dynamic orchestrator selected direct mode without being truthfully 100% certain; "
                    "dynamic direct is only allowed when the answer is certain enough that multi-agent verification would not help. "
                    "Tasks requiring state tracking, sequence reconstruction, validity checks, legality checks, or hidden context must use dynamic multi-agent review."
                ),
                subagent_mode=subagent_mode,
            )
        return create_direct_plan(raw_plan, task, subagent_mode=subagent_mode)
    if mode != "multi_agent":
        return create_fallback_plan(
            task,
            available_agents,
            reason="Model orchestrator returned an invalid mode.",
            subagent_mode=subagent_mode,
        )

    task_type = _clean_text(raw_plan.get("task_type")) or infer_fallback_task_type(task)
    if task_type.lower() == "unknown" and task.strip():
        task_type = infer_fallback_task_type(task)
    task_summary = _clean_text(raw_plan.get("task_summary")) or f"Handle the user task: {task}"
    reason = _clean_text(raw_plan.get("reason")) or "The model orchestrator selected this route."
    collaboration_protocol = normalize_collaboration_protocol(raw_plan.get("collaboration_protocol"))

    selected_agents = (
        normalize_dynamic_selected_agents(raw_plan.get("selected_agents"))
        if subagent_mode == "dynamic"
        else normalize_selected_agents(raw_plan.get("selected_agents"), available_agents)
    )
    if subagent_mode == "dynamic" and not is_valid_dynamic_agent_pool(selected_agents):
        return create_fallback_plan(
            task,
            available_agents,
            reason="Model orchestrator did not select a valid dynamic multi-agent pool.",
            subagent_mode=subagent_mode,
        )
    elif subagent_mode == "fixed" and (not selected_agents or _is_critic_alone(selected_agents)):
        return create_fallback_plan(
            task,
            available_agents,
            reason="Model orchestrator did not select a valid multi-agent pool.",
            subagent_mode=subagent_mode,
        )
    elif subagent_mode == "fixed":
        selected_agents = ensure_fixed_review_agent(selected_agents, available_agents)

    return {
        "mode": "multi_agent",
        "subagent_mode": subagent_mode,
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


def create_direct_plan(raw_plan: Mapping[str, Any], task: str, subagent_mode: SubagentMode) -> OrchestratorPlan:
    task_type = _clean_text(raw_plan.get("task_type")) or infer_fallback_task_type(task)
    task_summary = _clean_text(raw_plan.get("task_summary")) or f"Answer the user task directly: {task}"
    reason = _clean_text(raw_plan.get("reason")) or (
        "The orchestrator selected direct mode because it judged one direct response sufficient and no subagent "
        "specialization, independent verification, or debate was needed."
    )
    return {
        "mode": "direct",
        "subagent_mode": subagent_mode,
        "task_type": task_type,
        "task_summary": task_summary,
        "reason": reason,
        "selected_agents": [],
        "collaboration_protocol": normalize_collaboration_protocol(raw_plan.get("collaboration_protocol")),
        "direct_certainty": _clean_text(raw_plan.get("direct_certainty")),
    }


def is_truthful_dynamic_direct_plan(raw_plan: Mapping[str, Any], task: str = "") -> bool:
    direct_certainty = _clean_text(raw_plan.get("direct_certainty")).lower()
    reason = _clean_text(raw_plan.get("reason")).lower()
    if direct_certainty != "100_percent":
        return False
    vague_reason_markers = ("confident", "confidence", "likely", "probably", "seems", "enough")
    if any(marker in reason for marker in vague_reason_markers):
        return False
    if requires_state_validity_review(task):
        return False
    evidence_markers = ("explicit", "provided", "deterministic", "given", "verbatim", "lookup", "trivial")
    no_verification_markers = ("no decomposition", "no verification", "verification would not", "debate would not", "multi-agent")
    return any(marker in reason for marker in evidence_markers) and any(marker in reason for marker in no_verification_markers)


def requires_state_validity_review(task: str) -> bool:
    normalized = task.lower()
    state_markers = (
        "current state",
        "current board",
        "board state",
        "state tracking",
        "sequence of",
        "sequence",
        "move sequence",
        "game sequence",
        "game",
        "after the moves",
        "after the sequence",
        "given the game",
        "in-progress",
        "piece at",
    )
    validity_markers = (
        "valid",
        "legal",
        "legality",
        "destination square",
        "execute next",
        "next move",
        "complete the notation",
    )
    return any(marker in normalized for marker in state_markers) and any(marker in normalized for marker in validity_markers)


def ensure_fixed_review_agent(selected_agents: list[SelectedAgent], available_agents: Mapping[str, Any]) -> list[SelectedAgent]:
    if any(agent["name"] in {"CriticAgent", "VerifierAgent"} for agent in selected_agents):
        return selected_agents

    review_agent_name = "VerifierAgent" if "VerifierAgent" in available_agents else "CriticAgent"
    if review_agent_name not in available_agents:
        return selected_agents

    review_agent: SelectedAgent = {
        "name": review_agent_name,
        "subtask": fallback_subtask(review_agent_name, ""),
        "expected_output": fallback_expected_output(review_agent_name),
    }
    if len(selected_agents) < MAX_SELECTED_AGENTS:
        return [*selected_agents, review_agent]
    return [*selected_agents[:-1], review_agent]


def normalize_dynamic_selected_agents(value: Any) -> list[SelectedAgent]:
    if not isinstance(value, list):
        return []

    selected_agents: list[SelectedAgent] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            continue
        name = sanitize_dynamic_agent_name(item.get("name"))
        if not name or name in seen:
            continue
        rules = normalize_rules(item.get("rules"))
        critical_debate = bool(item.get("critical_debate"))
        selected_agents.append(
            {
                "name": name,
                "role": _clean_text(item.get("role")) or dynamic_default_role(critical_debate),
                "description": _clean_text(item.get("description")) or dynamic_default_description(critical_debate),
                "rules": rules or dynamic_default_rules(critical_debate),
                "subtask": _clean_text(item.get("subtask")) or dynamic_default_subtask(name, critical_debate),
                "expected_output": _clean_text(item.get("expected_output")) or dynamic_default_expected_output(critical_debate),
                "critical_debate": critical_debate,
                "workspace_access": normalize_workspace_access(item.get("workspace_access")),
            }
        )
        seen.add(name)
        if len(selected_agents) == MAX_SELECTED_AGENTS:
            break
    return selected_agents


def sanitize_dynamic_agent_name(value: Any) -> str:
    raw_name = _clean_text(value)
    if not raw_name:
        return ""
    tokens = re.findall(r"[A-Za-z0-9]+", raw_name)
    if not tokens:
        return ""
    name = "".join(token[:1].upper() + token[1:] for token in tokens)
    if not name[0].isalpha():
        name = f"Agent{name}"
    return name[:64]


def normalize_rules(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    rules = [_clean_text(rule) for rule in value]
    return [rule for rule in rules if rule][:8]


def normalize_workspace_access(value: Any) -> str:
    access = _clean_text(value).lower()
    return access if access in {"none", "read", "write"} else "none"


def is_valid_dynamic_agent_pool(selected_agents: list[SelectedAgent]) -> bool:
    return len(selected_agents) >= 2 and any(agent.get("critical_debate") for agent in selected_agents)


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


def create_fallback_plan(
    task: str,
    available_agents: Mapping[str, Any],
    reason: str | None = None,
    subagent_mode: SubagentMode = "fixed",
) -> OrchestratorPlan:
    if subagent_mode == "dynamic":
        return create_dynamic_fallback_plan(task, reason=reason)

    task_type = infer_fallback_task_type(task)
    selected_names = fallback_agent_names(task)
    selected_names = [name for name in selected_names if name in available_agents][:MAX_SELECTED_AGENTS]
    if not selected_names:
        selected_names = [name for name in ("SolverAgent", "CriticAgent") if name in available_agents]
    if selected_names == ["CriticAgent"] and "SolverAgent" in available_agents:
        selected_names.insert(0, "SolverAgent")

    return {
        "mode": "multi_agent",
        "subagent_mode": "fixed",
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


def create_dynamic_fallback_plan(task: str, reason: str | None = None) -> OrchestratorPlan:
    return {
        "mode": "multi_agent",
        "subagent_mode": "dynamic",
        "task_type": infer_fallback_task_type(task),
        "task_summary": f"Handle the user task with dynamic subagents: {task}",
        "reason": reason or "Using deterministic dynamic fallback because model orchestration was unavailable or invalid.",
        "selected_agents": [
            {
                "name": "TaskWorker",
                "role": dynamic_default_role(False),
                "description": dynamic_default_description(False),
                "rules": dynamic_default_rules(False),
                "subtask": dynamic_default_subtask("TaskWorker", False),
                "expected_output": dynamic_default_expected_output(False),
                "critical_debate": False,
                "workspace_access": "write" if any(keyword in task.lower() for keyword in CODE_KEYWORDS) else "none",
            },
            {
                "name": "CriticalDebateAgent",
                "role": dynamic_default_role(True),
                "description": dynamic_default_description(True),
                "rules": dynamic_default_rules(True),
                "subtask": dynamic_default_subtask("CriticalDebateAgent", True),
                "expected_output": dynamic_default_expected_output(True),
                "critical_debate": True,
                "workspace_access": "none",
            },
        ],
        "collaboration_protocol": default_collaboration_protocol(),
    }


def dynamic_default_role(critical_debate: bool) -> str:
    if critical_debate:
        return "Challenges assumptions, debates weak points, and identifies risks in other agents' findings."
    return "Works on the core task and shares concise findings for final synthesis."


def dynamic_default_description(critical_debate: bool) -> str:
    if critical_debate:
        return "A critical debate subagent that finds contradictions, missing cases, and overconfident claims."
    return "A dynamic task subagent created by the Orchestrator for this specific user request."


def dynamic_default_rules(critical_debate: bool) -> list[str]:
    if critical_debate:
        return [
            "Challenge assumptions and weak reasoning from other agents.",
            "Publish critique events when risks, contradictions, or missing cases are found.",
        ]
    return [
        "Focus on the assigned subtask.",
        "Share findings useful to other subagents and final synthesis.",
    ]


def dynamic_default_subtask(agent_name: str, critical_debate: bool) -> str:
    if critical_debate:
        return "Critically debate the emerging answer, challenge assumptions, and identify risks."
    return f"Contribute as {agent_name} to the user task and share useful findings."


def dynamic_default_expected_output(critical_debate: bool) -> str:
    if critical_debate:
        return "Critiques, risks, counterarguments, missing cases, and corrections."
    return "Concise task findings and recommendations for final synthesis."


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
            "role": agent.get("role", ""),
            "description": agent.get("description", ""),
            "rules": agent.get("rules", []),
            "critical_debate": agent.get("critical_debate", False),
            "workspace_access": agent.get("workspace_access", "none"),
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
