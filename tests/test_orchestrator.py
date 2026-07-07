from dataclasses import dataclass

import pytest

from multi_agent_sync.agents.registry import AGENT_REGISTRY
from multi_agent_sync.orchestrator.orchestrator import create_model_based_plan


@dataclass
class FakeResponse:
    content: str


class StaticLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    async def ainvoke(self, prompt: str) -> FakeResponse:
        return FakeResponse(self.content)


def agent_names(plan: dict) -> list[str]:
    return [agent["name"] for agent in plan["selected_agents"]]


@pytest.mark.asyncio
async def test_direct_mode_is_allowed_for_easy_factual_tasks():
    llm = StaticLLM(
        """
        {
          "mode": "direct",
          "task_type": "simple factual question",
          "task_summary": "The user asks what an API is.",
          "reason": "A concise factual answer is enough.",
          "selected_agents": [
            {
              "name": "ResearchAgent",
              "subtask": "Unused",
              "expected_output": "Unused"
            }
          ],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": true,
            "notes": "No collaboration needed."
          }
        }
        """
    )

    plan = await create_model_based_plan("What is an API?", llm, AGENT_REGISTRY)

    assert plan["mode"] == "direct"
    assert plan["subagent_mode"] == "fixed"
    assert plan["selected_agents"] == []


@pytest.mark.asyncio
async def test_direct_mode_for_non_easy_task_falls_back_to_multi_agent():
    llm = StaticLLM(
        """
        {
          "mode": "direct",
          "task_type": "software implementation task",
          "task_summary": "The user asks for code changes.",
          "reason": "The model incorrectly selected direct mode.",
          "selected_agents": [],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": true,
            "notes": "No collaboration needed."
          }
        }
        """
    )

    plan = await create_model_based_plan("Implement a benchmark runner and tests.", llm, AGENT_REGISTRY)

    assert plan["mode"] == "multi_agent"
    assert plan["subagent_mode"] == "fixed"
    assert len(plan["selected_agents"]) >= 2
    assert {"CriticAgent", "VerifierAgent"} & set(agent_names(plan))


@pytest.mark.asyncio
async def test_fixed_multi_agent_plan_adds_verifier_when_critic_or_verifier_is_missing():
    llm = StaticLLM(
        """
        {
          "mode": "multi_agent",
          "task_type": "reasoning task",
          "task_summary": "The user asks for reasoning.",
          "reason": "The task needs solver work.",
          "selected_agents": [
            {
              "name": "SolverAgent",
              "subtask": "Solve the task.",
              "expected_output": "A concise answer."
            }
          ],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": true,
            "notes": "Share findings."
          }
        }
        """
    )

    plan = await create_model_based_plan("Reason about this question.", llm, AGENT_REGISTRY)

    assert plan["mode"] == "multi_agent"
    assert agent_names(plan) == ["SolverAgent", "VerifierAgent"]


@pytest.mark.asyncio
async def test_coding_task_selects_coding_agent():
    llm = StaticLLM(
        """
        {
          "mode": "multi_agent",
          "task_type": "debugging implementation task",
          "task_summary": "The user wants help debugging a failing pytest error.",
          "reason": "The task needs implementation analysis, critique, and validation.",
          "selected_agents": [
            {
              "name": "CodingAgent",
              "subtask": "Inspect the likely implementation failure and propose a concrete fix path.",
              "expected_output": "A code-oriented debugging plan."
            },
            {
              "name": "CriticAgent",
              "subtask": "Find risks, missing cases, and weak assumptions in the fix path.",
              "expected_output": "A short critique of the debugging plan."
            },
            {
              "name": "VerifierAgent",
              "subtask": "Check that the proposed fix has suitable verification steps.",
              "expected_output": "Validation checks and test expectations."
            }
          ],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": true,
            "notes": "Agents should use findings and critiques from each other before final synthesis."
          }
        }
        """
    )

    plan = await create_model_based_plan("Debug this failing pytest error in my API module.", llm, AGENT_REGISTRY)

    assert plan["mode"] == "multi_agent"
    assert "CodingAgent" in agent_names(plan)


@pytest.mark.asyncio
async def test_math_task_selects_solver_and_verifier():
    llm = StaticLLM(
        """
        {
          "mode": "multi_agent",
          "task_type": "math reasoning problem",
          "task_summary": "The user asks for a force and acceleration calculation.",
          "reason": "The task needs reasoning and independent correctness checking.",
          "selected_agents": [
            {
              "name": "SolverAgent",
              "subtask": "Solve the calculation and state assumptions.",
              "expected_output": "A clear solution with intermediate values."
            },
            {
              "name": "VerifierAgent",
              "subtask": "Check the arithmetic, units, and final conclusion.",
              "expected_output": "A validation note with any corrections."
            }
          ],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": true,
            "notes": "Verifier should use solver findings and publish corrections."
          }
        }
        """
    )

    plan = await create_model_based_plan("Calculate the acceleration of a 2 kg object under a 10 N force.", llm, AGENT_REGISTRY)

    assert agent_names(plan) == ["SolverAgent", "VerifierAgent"]


@pytest.mark.asyncio
async def test_philosophy_task_selects_solver_and_critic():
    llm = StaticLLM(
        """
        {
          "mode": "multi_agent",
          "task_type": "philosophical argument analysis",
          "task_summary": "The user asks for analysis of a philosophical argument.",
          "reason": "The task benefits from reasoning and counterargument.",
          "selected_agents": [
            {
              "name": "SolverAgent",
              "subtask": "Analyze the argument structure and key concepts.",
              "expected_output": "A reasoned interpretation."
            },
            {
              "name": "CriticAgent",
              "subtask": "Identify counterarguments, ambiguities, and alternative interpretations.",
              "expected_output": "A concise critique."
            }
          ],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": true,
            "notes": "Critic should challenge the solver's interpretation."
          }
        }
        """
    )

    plan = await create_model_based_plan("Is moral realism compatible with evolutionary debunking arguments?", llm, AGENT_REGISTRY)

    assert agent_names(plan) == ["SolverAgent", "CriticAgent"]


@pytest.mark.asyncio
async def test_invalid_invented_agent_names_are_removed():
    llm = StaticLLM(
        """
        {
          "mode": "multi_agent",
          "task_type": "software design review",
          "task_summary": "The user asks for a software design review.",
          "reason": "The task needs code-aware planning and critique.",
          "selected_agents": [
            {
              "name": "ArchitectAgent",
              "subtask": "Design the architecture.",
              "expected_output": "An architecture plan."
            },
            {
              "name": "CodingAgent",
              "subtask": "Plan concrete modules and APIs.",
              "expected_output": "Implementation structure."
            },
            {
              "name": "CriticAgent",
              "subtask": "Review the proposed design for risks.",
              "expected_output": "Risks and edge cases."
            }
          ],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": true,
            "notes": "Share implementation findings and critiques."
          }
        }
        """
    )

    plan = await create_model_based_plan("Design a clean API module boundary for this repository.", llm, AGENT_REGISTRY)

    assert agent_names(plan) == ["CodingAgent", "CriticAgent"]
    assert "ArchitectAgent" not in agent_names(plan)


@pytest.mark.asyncio
async def test_invalid_json_falls_back_safely():
    plan = await create_model_based_plan("Prove this theorem and check each step.", StaticLLM("not json"), AGENT_REGISTRY)

    assert plan["mode"] == "multi_agent"
    assert agent_names(plan) == ["SolverAgent", "CriticAgent", "VerifierAgent"]
    assert plan["task_type"] != "unknown"


@pytest.mark.asyncio
async def test_selected_agents_are_always_from_registry_and_never_architect():
    llm = StaticLLM(
        """
        {
          "mode": "multi_agent",
          "task_type": "oversized plan",
          "task_summary": "The user asks for a complex answer.",
          "reason": "The task needs several views.",
          "selected_agents": [
            {"name": "ResearchAgent", "subtask": "Research context.", "expected_output": "Context."},
            {"name": "SolverAgent", "subtask": "Reason through the problem.", "expected_output": "Solution."},
            {"name": "CodingAgent", "subtask": "Plan implementation.", "expected_output": "Code plan."},
            {"name": "CriticAgent", "subtask": "Critique risks.", "expected_output": "Critique."},
            {"name": "VerifierAgent", "subtask": "Verify details.", "expected_output": "Checks."},
            {"name": "ArchitectAgent", "subtask": "Invented role.", "expected_output": "Invalid."}
          ],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": true,
            "notes": "Share useful events."
          }
        }
        """
    )

    plan = await create_model_based_plan("Research, design, implement, critique, and verify this API approach.", llm, AGENT_REGISTRY)

    assert len(plan["selected_agents"]) <= 4
    assert set(agent_names(plan)) <= set(AGENT_REGISTRY)
    assert "ArchitectAgent" not in agent_names(plan)


@pytest.mark.asyncio
async def test_dynamic_mode_accepts_orchestrator_named_agents_with_rules():
    llm = StaticLLM(
        """
        {
          "mode": "multi_agent",
          "task_type": "dynamic implementation review",
          "task_summary": "The user asks for a dynamic multi-agent design.",
          "reason": "The task needs a generated worker and critical debate.",
          "selected_agents": [
            {
              "name": "Implementation Planner",
              "role": "Plans concrete implementation work.",
              "description": "Focuses on files, tests, runtime flow, and compatibility.",
              "rules": ["Share actionable findings.", "Keep fixed mode compatible."],
              "subtask": "Design the runtime implementation.",
              "expected_output": "Implementation steps and risks.",
              "critical_debate": false
            },
            {
              "name": "Critical Debate Agent",
              "role": "Challenges the implementation plan.",
              "description": "Finds contradictions, missing cases, and weak assumptions.",
              "rules": ["Publish critique events.", "Challenge overconfident claims."],
              "subtask": "Debate and critique the plan.",
              "expected_output": "Critiques and corrections.",
              "critical_debate": true
            }
          ],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": true,
            "notes": "Generated agents should share findings and critiques."
          }
        }
        """
    )

    plan = await create_model_based_plan(
        "Implement dynamic subagents.",
        llm,
        AGENT_REGISTRY,
        subagent_mode="dynamic",
    )

    assert plan["mode"] == "multi_agent"
    assert plan["subagent_mode"] == "dynamic"
    assert agent_names(plan) == ["ImplementationPlanner", "CriticalDebateAgent"]
    assert plan["selected_agents"][0]["role"] == "Plans concrete implementation work."
    assert plan["selected_agents"][0]["description"] == "Focuses on files, tests, runtime flow, and compatibility."
    assert plan["selected_agents"][0]["rules"] == ["Share actionable findings.", "Keep fixed mode compatible."]
    assert plan["selected_agents"][1]["critical_debate"] is True


@pytest.mark.asyncio
async def test_dynamic_mode_invalid_multi_agent_plan_falls_back_to_worker_and_critical_debate():
    llm = StaticLLM(
        """
        {
          "mode": "multi_agent",
          "task_type": "invalid dynamic plan",
          "task_summary": "The user asks for dynamic agents.",
          "reason": "The model returned too few dynamic agents.",
          "selected_agents": [
            {
              "name": "",
              "role": "",
              "description": "",
              "rules": [],
              "subtask": "",
              "expected_output": "",
              "critical_debate": false
            }
          ],
          "collaboration_protocol": {
            "event_types_to_share": ["finding"],
            "reactive_steps": true,
            "notes": "Share findings."
          }
        }
        """
    )

    plan = await create_model_based_plan(
        "Implement dynamic subagents.",
        llm,
        AGENT_REGISTRY,
        subagent_mode="dynamic",
    )

    assert plan["mode"] == "multi_agent"
    assert plan["subagent_mode"] == "dynamic"
    assert agent_names(plan) == ["TaskWorker", "CriticalDebateAgent"]
    assert len(plan["selected_agents"]) >= 2
    assert any(agent["critical_debate"] for agent in plan["selected_agents"])
