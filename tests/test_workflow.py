import asyncio
import importlib.metadata
from dataclasses import dataclass

import pytest

from multi_agent_sync.graph.workflow import run_workflow
from multi_agent_sync.graph.nodes import apply_workspace_access_policy
from multi_agent_sync.workspace.docker import BashResult, DockerWorkspace


@dataclass
class FakeResponse:
    content: str


class FakeLLM:
    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "You are the model-based orchestrator" in prompt:
            return FakeResponse(orchestrator_response_for_prompt(prompt))
        if "Answer the user task directly" in prompt:
            return FakeResponse("A direct answer from one LLM call.")
        if "Synthesizer" in prompt or "final answer" in prompt.lower():
            return FakeResponse("A concise final plan that combines research, coding, and critique outputs.")
        return FakeResponse(
            "SUMMARY:\nUseful step summary.\n"
            "SHARE_FINDING:\nShare a concise implementation-relevant finding.\n"
            "CONFIDENCE:\n0.8\n"
            "LOCAL_NOTES:\nKeep the prototype local and event-driven."
        )


class DynamicFakeLLM:
    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "dynamic subagent orchestrator" in prompt:
            return FakeResponse(
                """
                {
                  "mode": "multi_agent",
                  "task_type": "dynamic software task",
                  "task_summary": "The user asks for dynamic workers.",
                  "reason": "The task needs specialized planning and critical debate.",
                  "selected_agents": [
                    {
                      "name": "Implementation Planner",
                      "role": "Plans the code changes.",
                      "description": "Identifies files, tests, and integration points.",
                      "rules": ["Share implementation findings."],
                      "subtask": "Plan the implementation details.",
                      "expected_output": "Implementation guidance.",
                      "critical_debate": false
                    },
                    {
                      "name": "Critical Debate Agent",
                      "role": "Challenges the plan.",
                      "description": "Finds missing assumptions and edge cases.",
                      "rules": ["Publish critique events."],
                      "subtask": "Critique the implementation plan.",
                      "expected_output": "Critiques and risks.",
                      "critical_debate": true
                    }
                  ],
                  "collaboration_protocol": {
                    "event_types_to_share": ["finding", "critique", "warning"],
                    "reactive_steps": true,
                    "notes": "Use each other's findings."
                  }
                }
                """
            )
        if "You are CriticalDebateAgent" in prompt:
            assert "Finds missing assumptions and edge cases." in prompt
            assert "Publish critique events." in prompt
            return FakeResponse(
                "SUMMARY:\nCritical debate found a risk.\n"
                "SHARE_FINDING:\nThe dynamic plan needs compatibility checks.\n"
                "CONFIDENCE:\n0.85\n"
                "LOCAL_NOTES:\nCritique complete."
            )
        if "You are ImplementationPlanner" in prompt:
            assert "Identifies files, tests, and integration points." in prompt
            assert "Share implementation findings." in prompt
            return FakeResponse(
                "SUMMARY:\nImplementation plan created.\n"
                "SHARE_FINDING:\nRuntime should instantiate DynamicAgent.\n"
                "CONFIDENCE:\n0.8\n"
                "LOCAL_NOTES:\nPlanner complete."
            )
        if "Synthesizer" in prompt:
            assert "ImplementationPlanner" in prompt
            assert "CriticalDebateAgent" in prompt
            return FakeResponse("Synthesized dynamic-agent answer.")
        return FakeResponse("Unexpected prompt")


class HangingSynthesizerLLM(FakeLLM):
    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "Synthesizer" in prompt:
            await asyncio.sleep(10)
        return await super().ainvoke(prompt)


class LateFindingLLM:
    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "You are the model-based orchestrator" in prompt:
            return FakeResponse(orchestrator_response_for_prompt(prompt))
        if "Synthesizer" in prompt:
            return FakeResponse("Synthesized answer from traced agent communication.")
        if "You are SolverAgent" in prompt:
            await asyncio.sleep(0.01)
            return FakeResponse(
                "SUMMARY:\nSolver computed the threshold.\n"
                "SHARE_FINDING:\nSolver late finding for verifier.\n"
                "CONFIDENCE:\n0.9\n"
                "LOCAL_NOTES:\nSolver notes."
            )
        if "You are VerifierAgent" in prompt and "reactive follow-up step" in prompt:
            assert "Solver late finding for verifier." in prompt
            return FakeResponse(
                "SUMMARY:\nReactive verifier used SolverAgent finding.\n"
                "SHARE_FINDING:\nVerified after receiving SolverAgent finding.\n"
                "CONFIDENCE:\n0.95\n"
                "LOCAL_NOTES:\nReactive verification completed."
            )
        if "You are VerifierAgent" in prompt:
            await asyncio.sleep(0.05)
            return FakeResponse(
                "SUMMARY:\nVerifier started before SolverAgent finding was available.\n"
                "SHARE_FINDING:\n\n"
                "CONFIDENCE:\n0.6\n"
                "LOCAL_NOTES:\nInitial verifier notes."
            )
        return FakeResponse("SUMMARY:\nNo-op.\nSHARE_FINDING:\n\nCONFIDENCE:\n0.5\nLOCAL_NOTES:\n")


class WorkflowToolLLM:
    def __init__(self) -> None:
        self.coding_prompts = 0

    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "You are the model-based orchestrator" in prompt:
            return FakeResponse(orchestrator_response_for_prompt(prompt))
        if "You are CodingAgent" in prompt and "Docker workspace" in prompt:
            self.coding_prompts += 1
            if self.coding_prompts == 1:
                return FakeResponse(
                    'ACTION:\n{"tool": "bash", "command": "printf workflow"}\n'
                    "SHARE_FINDING:\nRunning a Docker-contained workspace inspection.\n"
                    "CONFIDENCE:\n0.8\n"
                    "LOCAL_NOTES:\nNeed command output."
                )
            return FakeResponse(
                "FINAL:\nCodingAgent finished after using Docker bash.\n"
                "SHARE_FINDING:\nDocker bash output was available to the coding agent.\n"
                "CONFIDENCE:\n0.9\n"
                "LOCAL_NOTES:\nDone."
            )
        if "You are CodingAgent" in prompt:
            raise AssertionError("CodingAgent should receive the Docker workspace tool prompt")
        if "Synthesizer" in prompt:
            return FakeResponse("Synthesized workspace tool answer.")
        return FakeResponse(
            "SUMMARY:\nText-only agent step.\n"
            "SHARE_FINDING:\nText-only finding.\n"
            "CONFIDENCE:\n0.7\n"
            "LOCAL_NOTES:\nNo bash used."
        )


class WorkflowFakeDockerWorkspace(DockerWorkspace):
    def __init__(self) -> None:
        super().__init__(container_name="workflow-fake-container")
        self.commands: list[str] = []

    async def run_bash(self, command: str, *, timeout_seconds: float | None = None) -> BashResult:
        self.commands.append(command)
        return BashResult(
            command=command,
            exit_code=0,
            stdout="workflow output from docker\n",
            stderr="",
            timed_out=False,
            container_name=self.container_name,
        )


def orchestrator_response_for_prompt(prompt: str) -> str:
    if "What is an API?" in prompt or "Which one of the following" in prompt:
        return """
        {
          "mode": "direct",
          "task_type": "simple factual question",
          "task_summary": "The user asks for a concise direct answer.",
          "reason": "The task is simple enough for one LLM response.",
          "selected_agents": [],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": true,
            "notes": "No agent collaboration is needed."
          }
        }
        """
    if "Calculate" in prompt:
        return """
        {
          "mode": "multi_agent",
          "task_type": "math calculation",
          "task_summary": "The user asks for a calculation.",
          "reason": "The task benefits from solving and independent verification.",
          "selected_agents": [
            {
              "name": "SolverAgent",
              "subtask": "Solve the calculation step by step at summary level.",
              "expected_output": "A concise calculated result."
            },
            {
              "name": "VerifierAgent",
              "subtask": "Check the calculation, assumptions, and final result.",
              "expected_output": "Validation notes and corrections if needed."
            }
          ],
          "collaboration_protocol": {
            "event_types_to_share": ["finding", "critique", "warning"],
            "reactive_steps": true,
            "notes": "Verifier should use solver findings when available."
          }
        }
        """
    return """
    {
      "mode": "multi_agent",
      "task_type": "software prototype task",
      "task_summary": "The user asks for a software prototype plan.",
      "reason": "The task needs implementation planning, context, and critique.",
      "selected_agents": [
        {
          "name": "ResearchAgent",
          "subtask": "Identify assumptions and context for the software prototype.",
          "expected_output": "Concise context and constraints."
        },
        {
          "name": "CodingAgent",
          "subtask": "Plan the implementation modules, APIs, and tests.",
          "expected_output": "Concrete implementation plan."
        },
        {
          "name": "CriticAgent",
          "subtask": "Review the plan for missing cases, risks, and edge conditions.",
          "expected_output": "Concise critique and risks."
        }
      ],
      "collaboration_protocol": {
        "event_types_to_share": ["finding", "critique", "warning"],
        "reactive_steps": true,
        "notes": "Agents should use useful findings and critiques from each other."
      }
    }
    """


@pytest.mark.asyncio
async def test_langgraph_workflow_runs_from_start_to_end():
    state = await run_workflow(
        task="Build a prototype chess website",
        llm=FakeLLM(),
        max_steps_per_agent=1,
        total_runtime_timeout=10,
        stream_to_console=False,
    )

    assert state["event_log"]
    assert state["agent_outputs"]
    assert state["final_answer"]
    assert state["mode"] == "multi_agent"
    assert state["agent_traces"]
    assert any(event.event_type == "final_summary" and event.content == "Final answer generated." for event in state["event_log"])


@pytest.mark.asyncio
async def test_easy_direct_orchestrator_response_uses_direct_answer_node():
    state = await run_workflow(
        task="What is an API?",
        llm=FakeLLM(),
        max_steps_per_agent=1,
        stream_to_console=False,
    )

    assert state["mode"] == "direct"
    assert state["agent_outputs"] == {}
    assert state["agent_traces"] == {}
    assert state["final_answer"] == "A direct answer from one LLM call."
    assert any(
        event.event_type == "plan_created"
        and "mode=direct" in event.content
        and "selected_agents=none" in event.content
        for event in state["event_log"]
    )


@pytest.mark.asyncio
async def test_non_easy_direct_orchestrator_response_falls_back_to_multi_agent_runtime():
    state = await run_workflow(
        task="Which one of the following implementation strategies should we use?",
        llm=FakeLLM(),
        max_steps_per_agent=1,
        stream_to_console=False,
    )

    assert state["mode"] == "multi_agent"
    assert set(state["agent_outputs"]) == {"CodingAgent", "CriticAgent", "VerifierAgent"}
    assert state["agent_traces"]
    assert "A concise final plan that combines research, coding, and critique outputs." in state["final_answer"]


@pytest.mark.asyncio
async def test_calculation_runtime_constructs_only_solver_and_verifier_agents():
    state = await run_workflow(
        task="Calculate 2 + 2.",
        llm=FakeLLM(),
        max_steps_per_agent=1,
        total_runtime_timeout=5,
        stream_to_console=False,
    )

    assert state["mode"] == "multi_agent"
    assert state["task_type"] == "math calculation"
    assert set(state["agent_outputs"]) == {"SolverAgent", "VerifierAgent"}
    assert "CodingAgent" not in state["agent_outputs"]
    assert set(state["agent_traces"]) == {"SolverAgent", "VerifierAgent"}


@pytest.mark.asyncio
async def test_dynamic_workflow_constructs_free_named_agents_and_synthesizes_outputs():
    state = await run_workflow(
        task="Implement dynamic subagents.",
        llm=DynamicFakeLLM(),
        max_steps_per_agent=1,
        total_runtime_timeout=5,
        stream_to_console=False,
        subagent_mode="dynamic",
    )

    assert state["subagent_mode"] == "dynamic"
    assert set(state["agent_outputs"]) == {"ImplementationPlanner", "CriticalDebateAgent"}
    assert state["selected_agents"][0]["description"] == "Identifies files, tests, and integration points."
    assert state["selected_agents"][1]["critical_debate"] is True
    assert any(event.event_type == "critique" and event.source == "CriticalDebateAgent" for event in state["event_log"])
    assert state["final_answer"] == "Synthesized dynamic-agent answer."


@pytest.mark.asyncio
async def test_late_solver_finding_is_traced_and_used_by_reactive_verifier_step():
    state = await run_workflow(
        task="Calculate 2 + 2.",
        llm=LateFindingLLM(),
        max_steps_per_agent=1,
        total_runtime_timeout=5,
        stream_to_console=False,
    )

    verifier_trace = state["agent_traces"]["VerifierAgent"]
    received_solver_findings = [
        receipt
        for receipt in verifier_trace["event_receipts"]
        if receipt["source"] == "SolverAgent" and receipt["event_type"] == "finding"
    ]

    assert received_solver_findings
    solver_finding_receipt = next(receipt for receipt in received_solver_findings if receipt["used_in_step"] == 2)
    assert solver_finding_receipt["accepted"] is True
    assert solver_finding_receipt["ignored_reason"] is None

    reactive_steps = [step for step in verifier_trace["steps"] if step["is_reactive"]]
    assert len(reactive_steps) == 1
    reactive_step = reactive_steps[0]
    assert reactive_step["reactive_reason"] == "important_unused_events_received"
    assert solver_finding_receipt["event_id"] in reactive_step["used_event_ids"]
    assert "Solver late finding for verifier." in reactive_step["prompt"]


@pytest.mark.asyncio
async def test_agent_message_streaming_can_be_disabled():
    state = await run_workflow(
        task="Calculate 2 + 2.",
        llm=LateFindingLLM(),
        max_steps_per_agent=1,
        total_runtime_timeout=5,
        stream_to_console=False,
        enable_agent_message_streaming=False,
    )

    verifier_trace = state["agent_traces"]["VerifierAgent"]

    assert not [
        receipt
        for receipt in verifier_trace["event_receipts"]
        if receipt["source"] == "SolverAgent" and receipt["event_type"] == "finding"
    ]
    assert not [step for step in verifier_trace["steps"] if step["is_reactive"]]
    assert not any(event.event_type == "message_received" for event in state["event_log"])


@pytest.mark.asyncio
async def test_workflow_injects_docker_bash_tool_into_fixed_coding_agent():
    workspace = WorkflowFakeDockerWorkspace()

    state = await run_workflow(
        task="Fix a repository bug.",
        llm=WorkflowToolLLM(),
        max_steps_per_agent=2,
        total_runtime_timeout=5,
        stream_to_console=False,
        docker_workspace=workspace,
        enable_workspace_tools=True,
    )

    assert workspace.commands == ["printf workflow"]
    coding_trace = state["agent_traces"]["CodingAgent"]
    assert coding_trace["assignment"]["workspace_access"] == "write"
    assert coding_trace["steps"][0]["parsed_output"]["tool_result"]["stdout"] == "workflow output from docker\n"
    assert state["final_answer"] == "Synthesized workspace tool answer."


def test_dynamic_workspace_policy_does_not_let_critical_debate_consume_writer_slot():
    assignments = apply_workspace_access_policy(
        [
            {"agent_name": "CriticalReviewer", "critical_debate": True, "workspace_access": "write"},
            {"agent_name": "PatchAuthor", "critical_debate": False, "workspace_access": "write"},
        ],
        subagent_mode="dynamic",
        enable_workspace_tools=True,
    )

    assert assignments[0]["workspace_access"] == "read"
    assert assignments[1]["workspace_access"] == "write"


def test_fixed_workspace_policy_grants_writer_when_coding_agent_is_absent():
    assignments = apply_workspace_access_policy(
        [
            {"agent_name": "SolverAgent"},
            {"agent_name": "VerifierAgent"},
        ],
        subagent_mode="fixed",
        enable_workspace_tools=True,
    )

    assert assignments[0]["workspace_access"] == "write"
    assert assignments[1]["workspace_access"] == "none"


@pytest.mark.asyncio
async def test_agent_traces_include_prompt_response_parsed_output_and_published_events():
    state = await run_workflow(
        task="Calculate 2 + 2.",
        llm=FakeLLM(),
        max_steps_per_agent=1,
        total_runtime_timeout=5,
        stream_to_console=False,
    )

    solver_trace = state["agent_traces"]["SolverAgent"]
    step = solver_trace["steps"][0]

    assert solver_trace["assignment"]["agent_name"] == "SolverAgent"
    assert step["agent_name"] == "SolverAgent"
    assert step["step"] == 1
    assert "Overall user task" in step["prompt"]
    assert "SUMMARY:" in step["raw_response"]
    assert step["parsed_output"]["share_finding"] == "Share a concise implementation-relevant finding."
    assert step["published_events"]
    assert step["duration_seconds"] >= 0


@pytest.mark.asyncio
async def test_missing_options_guard_is_added_to_final_answer():
    state = await run_workflow(
        task="Which one of the following options is closest to the threshold value?",
        llm=FakeLLM(),
        stream_to_console=False,
    )

    assert "The options are missing, so I cannot choose one of them." in state["final_answer"]


@pytest.mark.asyncio
async def test_workflow_uses_fallback_when_synthesizer_times_out():
    state = await run_workflow(
        task="Build a prototype chess website",
        llm=HangingSynthesizerLLM(),
        max_steps_per_agent=1,
        total_runtime_timeout=10,
        synthesis_timeout=0.05,
        stream_to_console=False,
    )

    assert "Synthesis timed out" in state["final_answer"]
    assert any(event.event_type == "warning" and event.source == "Synthesizer" for event in state["event_log"])
    assert any(event.event_type == "final_summary" and event.content == "Final answer generated." for event in state["event_log"])


def test_autogen_dependency_is_not_used():
    installed = {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}
    assert "autogen" not in installed
    assert "pyautogen" not in installed
