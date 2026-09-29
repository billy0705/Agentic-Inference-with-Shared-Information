import json
import re
from collections.abc import Mapping
from typing import Any, Literal, NotRequired, TypedDict

from muffin.prompts import render_prompt


Mode = Literal["multi_agent", "direct"]


class SelectedAgent(TypedDict, total=False):
    name: str
    subtask: str
    expected_output: str
    role: str
    description: str
    rules: list[str]
    critical_debate: bool


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
    direct_certainty: NotRequired[str]


DEFAULT_EVENT_TYPES = ["finding", "critique", "warning"]
MAX_SELECTED_AGENTS = 20
DEFAULT_MIN_DYNAMIC_SUBAGENTS = 3
DEFAULT_MAX_DYNAMIC_SUBAGENTS = 3

CODE_KEYWORDS = ("code", "programming", "bug", "error", "pytest", "function", "api", "implementation", "repository")
RESEARCH_KEYWORDS = ("research", "comparison", "compare", "literature", "recent work", "background", "evidence")
REASONING_KEYWORDS = ("math", "calculate", "calculation", "proof", "prove", "theorem", "logic", "philosophy", "reasoning", "argument", "theory")


def resolve_dynamic_subagent_limits(min_subagents: int, max_subagents: int) -> tuple[int, int]:
    min_count = max(1, int(min_subagents))
    max_count = max(1, min(int(max_subagents), MAX_SELECTED_AGENTS))
    if min_count > max_count:
        min_count = max_count
    return min_count, max_count


async def create_model_based_plan(
    task: str,
    llm: Any,
    min_dynamic_subagents: int = DEFAULT_MIN_DYNAMIC_SUBAGENTS,
    max_dynamic_subagents: int = DEFAULT_MAX_DYNAMIC_SUBAGENTS,
    think_mode: bool = True,
) -> OrchestratorPlan:
    min_dynamic_subagents, max_dynamic_subagents = resolve_dynamic_subagent_limits(
        min_dynamic_subagents,
        max_dynamic_subagents,
    )
    prompt = build_dynamic_orchestrator_prompt(
        task,
        min_dynamic_subagents=min_dynamic_subagents,
        max_dynamic_subagents=max_dynamic_subagents,
        think_mode=think_mode,
    )
    try:
        response = await llm.ainvoke(prompt)
        content = getattr(response, "content", str(response))
        raw_plan = extract_json_object(content)
    except Exception as exc:
        return create_dynamic_fallback_plan(
            task,
            reason=f"Model orchestrator failed or returned invalid JSON: {exc}",
            min_dynamic_subagents=min_dynamic_subagents,
            max_dynamic_subagents=max_dynamic_subagents,
        )

    return validate_orchestrator_plan(
        raw_plan,
        task,
        min_dynamic_subagents=min_dynamic_subagents,
        max_dynamic_subagents=max_dynamic_subagents,
    )


async def create_orchestrator_plan(
    task: str,
    llm: Any,
    min_dynamic_subagents: int = DEFAULT_MIN_DYNAMIC_SUBAGENTS,
    max_dynamic_subagents: int = DEFAULT_MAX_DYNAMIC_SUBAGENTS,
    think_mode: bool = True,
) -> OrchestratorPlan:
    return await create_model_based_plan(
        task,
        llm,
        min_dynamic_subagents=min_dynamic_subagents,
        max_dynamic_subagents=max_dynamic_subagents,
        think_mode=think_mode,
    )


def build_dynamic_orchestrator_prompt(
    task: str,
    min_dynamic_subagents: int = DEFAULT_MIN_DYNAMIC_SUBAGENTS,
    max_dynamic_subagents: int = DEFAULT_MAX_DYNAMIC_SUBAGENTS,
    think_mode: bool = True,
) -> str:
    min_dynamic_subagents, max_dynamic_subagents = resolve_dynamic_subagent_limits(
        min_dynamic_subagents,
        max_dynamic_subagents,
    )
    return render_prompt(
        "orchestrator/dynamic_model_plan.j2",
        think_mode=think_mode,
        task=task,
        min_dynamic_subagents=min_dynamic_subagents,
        max_dynamic_subagents=max_dynamic_subagents,
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


def validate_orchestrator_plan(
    raw_plan: Mapping[str, Any],
    task: str,
    min_dynamic_subagents: int = DEFAULT_MIN_DYNAMIC_SUBAGENTS,
    max_dynamic_subagents: int = DEFAULT_MAX_DYNAMIC_SUBAGENTS,
) -> OrchestratorPlan:
    min_dynamic_subagents, max_dynamic_subagents = resolve_dynamic_subagent_limits(
        min_dynamic_subagents,
        max_dynamic_subagents,
    )
    mode = raw_plan.get("mode")
    if mode == "direct":
        if not is_truthful_dynamic_direct_plan(raw_plan, task):
            return create_dynamic_fallback_plan(
                task,
                reason=(
                    "Dynamic orchestrator selected direct mode without being truthfully 100% certain; "
                    "dynamic direct is only allowed when the answer is certain enough that multi-agent verification would not help. "
                    "Tasks requiring state tracking, sequence reconstruction, validity checks, legality checks, or hidden context must use dynamic multi-agent review."
                ),
                min_dynamic_subagents=min_dynamic_subagents,
                max_dynamic_subagents=max_dynamic_subagents,
            )
        return create_direct_plan(raw_plan, task)
    if mode != "multi_agent":
        return create_dynamic_fallback_plan(
            task,
            reason="Model orchestrator returned an invalid mode.",
            min_dynamic_subagents=min_dynamic_subagents,
            max_dynamic_subagents=max_dynamic_subagents,
        )

    task_type = _clean_text(raw_plan.get("task_type")) or infer_fallback_task_type(task)
    if task_type.lower() == "unknown" and task.strip():
        task_type = infer_fallback_task_type(task)
    task_summary = _clean_text(raw_plan.get("task_summary")) or f"Handle the user task: {task}"
    reason = _clean_text(raw_plan.get("reason")) or "The model orchestrator selected this route."
    collaboration_protocol = normalize_collaboration_protocol(raw_plan.get("collaboration_protocol"))

    selected_agents = normalize_dynamic_selected_agents(
        raw_plan.get("selected_agents"), max_agents=max_dynamic_subagents
    )
    if not is_valid_dynamic_agent_pool(selected_agents, min_agents=min_dynamic_subagents):
        return create_dynamic_fallback_plan(
            task,
            reason="Model orchestrator did not select a valid dynamic multi-agent pool.",
            min_dynamic_subagents=min_dynamic_subagents,
            max_dynamic_subagents=max_dynamic_subagents,
        )
    return {
        "mode": "multi_agent",
        "task_type": task_type,
        "task_summary": task_summary,
        "reason": reason,
        "selected_agents": selected_agents[:max_dynamic_subagents],
        "collaboration_protocol": collaboration_protocol,
    }


def create_direct_plan(raw_plan: Mapping[str, Any], task: str) -> OrchestratorPlan:
    task_type = _clean_text(raw_plan.get("task_type")) or infer_fallback_task_type(task)
    task_summary = _clean_text(raw_plan.get("task_summary")) or f"Answer the user task directly: {task}"
    reason = _clean_text(raw_plan.get("reason")) or (
        "The orchestrator selected direct mode because it judged one direct response sufficient and no subagent "
        "specialization, independent verification, or debate was needed."
    )
    return {
        "mode": "direct",
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


def normalize_dynamic_selected_agents(
    value: Any,
    max_agents: int = DEFAULT_MAX_DYNAMIC_SUBAGENTS,
) -> list[SelectedAgent]:
    _, max_agents = resolve_dynamic_subagent_limits(1, max_agents)
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
            }
        )
        seen.add(name)
        if len(selected_agents) == max_agents:
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


def is_valid_dynamic_agent_pool(
    selected_agents: list[SelectedAgent],
    min_agents: int = DEFAULT_MIN_DYNAMIC_SUBAGENTS,
) -> bool:
    min_agents, _ = resolve_dynamic_subagent_limits(min_agents, DEFAULT_MAX_DYNAMIC_SUBAGENTS)
    return len(selected_agents) >= min_agents and any(agent.get("critical_debate") for agent in selected_agents)


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


def create_dynamic_fallback_plan(
    task: str,
    reason: str | None = None,
    min_dynamic_subagents: int = DEFAULT_MIN_DYNAMIC_SUBAGENTS,
    max_dynamic_subagents: int = DEFAULT_MAX_DYNAMIC_SUBAGENTS,
) -> OrchestratorPlan:
    min_dynamic_subagents, max_dynamic_subagents = resolve_dynamic_subagent_limits(
        min_dynamic_subagents,
        max_dynamic_subagents,
    )
    selected_agents: list[SelectedAgent] = [
        {
            "name": "TaskWorker",
            "role": dynamic_default_role(False),
            "description": dynamic_default_description(False),
            "rules": dynamic_default_rules(False),
            "subtask": dynamic_default_subtask("TaskWorker", False),
            "expected_output": dynamic_default_expected_output(False),
            "critical_debate": False,
        },
        {
            "name": "CriticalDebateAgent",
            "role": dynamic_default_role(True),
            "description": dynamic_default_description(True),
            "rules": dynamic_default_rules(True),
            "subtask": dynamic_default_subtask("CriticalDebateAgent", True),
            "expected_output": dynamic_default_expected_output(True),
            "critical_debate": True,
        },
    ]
    for name in ("EvidenceReviewAgent", "ConsistencyCheckAgent"):
        if len(selected_agents) >= min_dynamic_subagents:
            break
        selected_agents.append(
            {
                "name": name,
                "role": dynamic_default_role(False),
                "description": dynamic_default_description(False),
                "rules": dynamic_default_rules(False),
                "subtask": dynamic_default_subtask(name, False),
                "expected_output": dynamic_default_expected_output(False),
                "critical_debate": False,
            }
        )
    return {
        "mode": "multi_agent",
        "task_type": infer_fallback_task_type(task),
        "task_summary": f"Handle the user task with dynamic subagents: {task}",
        "reason": reason or "Using deterministic dynamic fallback because model orchestration was unavailable or invalid.",
        "selected_agents": selected_agents[:max_dynamic_subagents],
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
        }
        for agent in plan["selected_agents"]
    ]


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
