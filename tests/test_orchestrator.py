from dataclasses import dataclass

import pytest

from multi_agent_sync.agents.registry import AGENT_REGISTRY
from multi_agent_sync.orchestrator.orchestrator import (
    create_model_based_plan,
    default_collaboration_protocol,
    selected_agents_to_assignments,
    validate_orchestrator_plan,
)


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
async def test_orchestrator_direct_mode_is_respected_for_non_easy_tasks():
    llm = StaticLLM(
        """
        {
          "mode": "direct",
          "task_type": "software implementation task",
          "task_summary": "The user asks for code changes.",
          "reason": "The orchestrator judged that a single concrete implementation answer is enough for this request.",
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

    assert plan["mode"] == "direct"
    assert plan["subagent_mode"] == "fixed"
    assert plan["selected_agents"] == []
    assert plan["reason"] == "The orchestrator judged that a single concrete implementation answer is enough for this request."


@pytest.mark.asyncio
async def test_dynamic_direct_without_truthful_100_percent_certainty_falls_back_to_multi_agent():
    llm = StaticLLM(
        """
        {
          "mode": "direct",
          "task_type": "benchmark question",
          "task_summary": "The user asks a benchmark question.",
          "reason": "I am confident enough to answer directly.",
          "selected_agents": [],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": false,
            "notes": "No collaboration needed."
          }
        }
        """
    )

    plan = await create_model_based_plan("Which option is best for this benchmark item?", llm, AGENT_REGISTRY, subagent_mode="dynamic")

    assert plan["mode"] == "multi_agent"
    assert plan["subagent_mode"] == "dynamic"
    assert "100% certain" in plan["reason"]
    assert agent_names(plan) == ["TaskWorker", "CriticalDebateAgent"]


@pytest.mark.asyncio
async def test_dynamic_direct_with_truthful_100_percent_certainty_is_allowed():
    llm = StaticLLM(
        """
        {
          "mode": "direct",
          "task_type": "deterministic lookup",
          "task_summary": "The user asks a deterministic lookup question.",
          "reason": "All required facts are explicitly provided in the prompt, the answer is deterministic, and no decomposition, verification, or debate would add useful checks.",
          "direct_certainty": "100_percent",
          "selected_agents": [],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": false,
            "notes": "No collaboration needed."
          }
        }
        """
    )

    plan = await create_model_based_plan("The prompt states A is correct. Which option is correct?", llm, AGENT_REGISTRY, subagent_mode="dynamic")

    assert plan["mode"] == "direct"
    assert plan["subagent_mode"] == "dynamic"
    assert plan["selected_agents"] == []
    assert plan["direct_certainty"] == "100_percent"


@pytest.mark.asyncio
async def test_dynamic_direct_for_state_validity_sequence_task_falls_back_to_multi_agent():
    llm = StaticLLM(
        """
        {
          "mode": "direct",
          "task_type": "state validity question",
          "task_summary": "The user asks for a valid destination square from a move sequence.",
          "reason": "The destination square is explicitly provided in the prompt, deterministic, and verification would not change the answer.",
          "direct_certainty": "100_percent",
          "selected_agents": [],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": false,
            "notes": "No collaboration needed."
          }
        }
        """
    )

    plan = await create_model_based_plan(
        (
            "Given the chess game, give one valid destination square for the chess piece at c8. "
            "State the destination square in the form (X), where X follows the regex [a-h][1-8]."
        ),
        llm,
        AGENT_REGISTRY,
        subagent_mode="dynamic",
    )

    assert plan["mode"] == "multi_agent"
    assert plan["subagent_mode"] == "dynamic"
    assert "state tracking" in plan["reason"]
    assert agent_names(plan) == ["TaskWorker", "CriticalDebateAgent"]


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


def test_dynamic_plan_preserves_workspace_access_in_assignments():
    plan = validate_orchestrator_plan(
        {
            "mode": "multi_agent",
            "task_type": "software repair task",
            "task_summary": "Fix code in a Docker workspace.",
            "reason": "The task needs file inspection and editing.",
            "selected_agents": [
                {
                    "name": "Patch Author",
                    "role": "Edits code.",
                    "description": "Applies the concrete fix.",
                    "rules": ["Use Docker workspace tools."],
                    "subtask": "Inspect and edit files.",
                    "expected_output": "Implemented change summary.",
                    "critical_debate": False,
                    "workspace_access": "write",
                },
                {
                    "name": "Critical Reviewer",
                    "role": "Reviews risks.",
                    "description": "Finds weak assumptions.",
                    "rules": ["Publish critiques."],
                    "subtask": "Review the proposed change.",
                    "expected_output": "Risk notes.",
                    "critical_debate": True,
                    "workspace_access": "none",
                },
            ],
            "collaboration_protocol": default_collaboration_protocol(),
        },
        task="Fix a repository bug.",
        available_agents={},
        subagent_mode="dynamic",
    )

    assignments = selected_agents_to_assignments(plan)

    assert assignments[0]["agent_name"] == "PatchAuthor"
    assert assignments[0]["workspace_access"] == "write"
    assert assignments[1]["agent_name"] == "CriticalReviewer"
    assert assignments[1]["workspace_access"] == "none"


def test_ordered_dynamic_plan_preserves_orchestrator_dependency_dag_in_assignments():
    plan = validate_orchestrator_plan(
        {
            "mode": "multi_agent",
            "task_type": "chess state tracking",
            "task_summary": "Reconstruct the board and validate moves.",
            "reason": "The task requires staged reconstruction and verification.",
            "selected_agents": [
                {
                    "name": "Board Agent",
                    "role": "Reconstructs the board.",
                    "subtask": "Reconstruct the board.",
                    "expected_output": "Board state.",
                    "critical_debate": False,
                    "depends_on": [],
                },
                {
                    "name": "Candidate Agent",
                    "role": "Generates moves.",
                    "subtask": "Generate candidate moves.",
                    "expected_output": "Candidate squares.",
                    "critical_debate": False,
                    "depends_on": ["Board Agent"],
                },
                {
                    "name": "Review Agent",
                    "role": "Checks move legality.",
                    "subtask": "Review candidates.",
                    "expected_output": "Validated square.",
                    "critical_debate": True,
                    "depends_on": ["Board Agent", "Candidate Agent"],
                },
            ],
            "collaboration_protocol": {
                "event_types_to_share": ["finding", "critique"],
                "reactive_steps": False,
                "notes": "Share upstream results.",
            },
        },
        "Complete the chess move.",
        AGENT_REGISTRY,
        subagent_mode="dynamic",
        ordered_step_one=True,
    )

    assert [agent["depends_on"] for agent in plan["selected_agents"]] == [
        [],
        ["BoardAgent"],
        ["BoardAgent", "CandidateAgent"],
    ]
    assert [assignment["depends_on"] for assignment in selected_agents_to_assignments(plan)] == [
        [],
        ["BoardAgent"],
        ["BoardAgent", "CandidateAgent"],
    ]


def test_ordered_dynamic_plan_rejects_cyclic_dependency_graph():
    plan = validate_orchestrator_plan(
        {
            "mode": "multi_agent",
            "task_type": "chess state tracking",
            "task_summary": "Reconstruct the board and validate moves.",
            "reason": "The task requires staged reconstruction and verification.",
            "selected_agents": [
                {
                    "name": "BoardAgent",
                    "subtask": "Reconstruct the board.",
                    "expected_output": "Board state.",
                    "critical_debate": False,
                    "depends_on": ["ReviewAgent"],
                },
                {
                    "name": "ReviewAgent",
                    "subtask": "Review candidates.",
                    "expected_output": "Validated square.",
                    "critical_debate": True,
                    "depends_on": ["BoardAgent"],
                },
            ],
            "collaboration_protocol": {
                "event_types_to_share": ["finding", "critique"],
                "reactive_steps": False,
                "notes": "Share upstream results.",
            },
        },
        "Complete the chess move.",
        AGENT_REGISTRY,
        subagent_mode="dynamic",
        ordered_step_one=True,
    )

    assert "valid acyclic dependency map" in plan["reason"]
    assert agent_names(plan) == ["TaskWorker", "CriticalDebateAgent"]


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
