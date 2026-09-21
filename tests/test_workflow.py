import asyncio
import importlib.metadata
from dataclasses import dataclass

import pytest

from multi_agent_sync.graph.workflow import run_workflow
from multi_agent_sync.graph import nodes
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
        if "Summarizer" in prompt or "final answer" in prompt.lower():
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
                    },
                    {
                      "name": "Verification Agent",
                      "role": "Verifies implementation coverage.",
                      "description": "Checks tests, edge cases, and consistency with the task.",
                      "rules": ["Publish verification findings."],
                      "subtask": "Verify the implementation plan.",
                      "expected_output": "Verification findings and remaining gaps.",
                      "critical_debate": false
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
        if "Agent name:\nCriticalDebateAgent" in prompt:
            assert "Finds missing assumptions and edge cases." in prompt
            assert "Publish critique events." in prompt
            return FakeResponse(
                "SUMMARY:\nCritical debate found a risk.\n"
                "SHARE_FINDING:\nThe dynamic plan needs compatibility checks.\n"
                "CONFIDENCE:\n0.85\n"
                "LOCAL_NOTES:\nCritique complete."
            )
        if "Agent name:\nImplementationPlanner" in prompt:
            assert "Identifies files, tests, and integration points." in prompt
            assert "Share implementation findings." in prompt
            return FakeResponse(
                "SUMMARY:\nImplementation plan created.\n"
                "SHARE_FINDING:\nRuntime should instantiate DynamicAgent.\n"
                "CONFIDENCE:\n0.8\n"
                "LOCAL_NOTES:\nPlanner complete."
            )
        if "Agent name:\nVerificationAgent" in prompt:
            assert "Checks tests, edge cases, and consistency with the task." in prompt
            assert "Publish verification findings." in prompt
            return FakeResponse(
                "SUMMARY:\nVerification plan created.\n"
                "SHARE_FINDING:\nRuntime should include verification coverage.\n"
                "CONFIDENCE:\n0.8\n"
                "LOCAL_NOTES:\nVerification complete."
            )
        if "Summarizer" in prompt:
            assert "ImplementationPlanner" in prompt
            assert "CriticalDebateAgent" in prompt
            assert "VerificationAgent" in prompt
            return FakeResponse("Summarized dynamic-agent answer.")
        return FakeResponse("Unexpected prompt")


class DynamicDirectFakeLLM:
    def __init__(self) -> None:
        self.direct_prompts: list[str] = []
        self.debate_prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "dynamic subagent orchestrator" in prompt:
            return FakeResponse(
                """
                {
                  "mode": "direct",
                  "task_type": "direct benchmark question",
                  "task_summary": "The user asks a question the orchestrator routes directly.",
                  "reason": "This is a deterministic toy test where the answer is known exactly, all required options are present, and no decomposition or verification would change the answer.",
                  "direct_certainty": "100_percent",
                  "selected_agents": [],
                  "collaboration_protocol": {
                    "event_types_to_share": ["finding", "critique", "warning"],
                    "reactive_steps": false,
                    "notes": "No agent collaboration is needed."
                  }
                }
                """
            )
        if "These are the solutions to the problem from other agents:" in prompt:
            self.debate_prompts.append(prompt)
            return FakeResponse("Revised dynamic direct answer. Final Answer: B")
        if "Answer the user task directly" in prompt:
            self.direct_prompts.append(prompt)
            return FakeResponse("Initial dynamic direct answer. Final Answer: A")
        return FakeResponse("Unexpected prompt")


class HangingSummarizerLLM(FakeLLM):
    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "Summarizer" in prompt:
            await asyncio.sleep(10)
        return await super().ainvoke(prompt)


class LateFindingLLM:
    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "You are the model-based orchestrator" in prompt:
            return FakeResponse(orchestrator_response_for_prompt(prompt))
        if "Summarizer" in prompt:
            return FakeResponse("Summarized answer from traced agent communication.")
        if "Agent name:\nSolverAgent" in prompt:
            await asyncio.sleep(0.01)
            return FakeResponse(
                "SUMMARY:\nSolver computed the threshold.\n"
                "SHARE_FINDING:\nSolver late finding for verifier.\n"
                "CONFIDENCE:\n0.9\n"
                "LOCAL_NOTES:\nSolver notes."
            )
        if "Agent name:\nVerifierAgent" in prompt and "reactive follow-up step" in prompt:
            assert "Solver late finding for verifier." in prompt
            return FakeResponse(
                "SUMMARY:\nReactive verifier used SolverAgent finding.\n"
                "SHARE_FINDING:\nVerified after receiving SolverAgent finding.\n"
                "CONFIDENCE:\n0.95\n"
                "LOCAL_NOTES:\nReactive verification completed."
            )
        if "Agent name:\nVerifierAgent" in prompt:
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
        if "Summarizer" in prompt:
            return FakeResponse("Summarized workspace tool answer.")
        return FakeResponse(
            "SUMMARY:\nText-only agent step.\n"
            "SHARE_FINDING:\nText-only finding.\n"
            "CONFIDENCE:\n0.7\n"
            "LOCAL_NOTES:\nNo bash used."
        )


class SummarizerTraceLLM(FakeLLM):
    def __init__(self) -> None:
        self.summarizer_prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "You are the Summarizer" in prompt:
            self.summarizer_prompts.append(prompt)
            return FakeResponse("Final Answer: summarized-agent-output")
        return await super().ainvoke(prompt)


class PromptCaptureLLM(SummarizerTraceLLM):
    def __init__(self) -> None:
        super().__init__()
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return await super().ainvoke(prompt)


class SummarizerLastSummaryLLM:
    def __init__(self) -> None:
        self.agent_prompt_counts: dict[str, int] = {}
        self.summarizer_prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "You are the model-based orchestrator" in prompt:
            return FakeResponse(orchestrator_response_for_prompt("Calculate 2 + 2."))
        if "You are the Summarizer" in prompt:
            self.summarizer_prompts.append(prompt)
            assert "Solver step 2 final summary." in prompt
            assert "Verifier step 2 final summary." in prompt
            assert "Solver step 1 stale summary." not in prompt
            assert "Verifier step 1 stale summary." not in prompt
            assert "SHARE_FINDING" not in prompt
            assert "LOCAL_NOTES" not in prompt
            assert "Runtime event log" not in prompt
            assert "[finding]" not in prompt
            return FakeResponse("Final Answer: last-summaries-only")
        if "Agent name:\nSolverAgent" in prompt:
            return self.agent_response("Solver")
        if "Agent name:\nVerifierAgent" in prompt:
            return self.agent_response("Verifier")
        return FakeResponse("Unexpected prompt")

    def agent_response(self, label: str) -> FakeResponse:
        count = self.agent_prompt_counts.get(label, 0) + 1
        self.agent_prompt_counts[label] = count
        if count == 1:
            return FakeResponse(
                f"SUMMARY:\n{label} step 1 stale summary.\n"
                f"SHARE_FINDING:\n{label} shared finding should stay out of summarizer input.\n"
                "LOCAL_NOTES:\nANSWER_CHOICE: stale\nANSWER_REASON: The stale candidate is only provisional."
            )
        return FakeResponse(
            f"SUMMARY:\n{label} step 2 final summary.\n"
            f"SHARE_FINDING:\n{label} final shared finding should stay out of summarizer input.\n"
            "LOCAL_NOTES:\nANSWER_CHOICE: final\nANSWER_REASON: The final candidate matches the last evidence."
        )


class FullTraceSummarizerLLM(SummarizerLastSummaryLLM):
    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "You are the Summarizer" in prompt:
            self.summarizer_prompts.append(prompt)
            return FakeResponse("Final Answer: full-trace-summary")
        return await super().ainvoke(prompt)


class RoleAwareAggregationLLM:
    def __init__(self) -> None:
        self.summarizer_prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "dynamic subagent orchestrator" in prompt:
            return FakeResponse(
                """
                {
                  "mode": "multi_agent",
                  "task_type": "multiple choice benchmark",
                  "task_summary": "Select the best option.",
                  "reason": "The task needs independent solving and critique.",
                  "selected_agents": [
                    {
                      "name": "Domain Solver",
                      "role": "Primary domain solver.",
                      "description": "Solves the question directly from domain knowledge.",
                      "rules": ["Give a final answer candidate."],
                      "subtask": "Solve the question and choose the best option.",
                      "expected_output": "A final option with explanation.",
                      "critical_debate": false
                    },
                    {
                      "name": "Option Verifier",
                      "role": "Checks option consistency.",
                      "description": "Verifies whether the chosen option matches the evidence.",
                      "rules": ["Prefer evidence over confidence."],
                      "subtask": "Validate the candidates against the prompt.",
                      "expected_output": "A verified option with concise justification.",
                      "critical_debate": false
                    },
                    {
                      "name": "Critical Reviewer",
                      "role": "Challenges weak assumptions.",
                      "description": "Looks for contradictions in the proposed options.",
                      "rules": ["Challenge unsupported claims."],
                      "subtask": "Critique the option choices.",
                      "expected_output": "A critique and final candidate.",
                      "critical_debate": true
                    }
                  ],
                  "collaboration_protocol": {
                    "event_types_to_share": ["finding", "critique", "warning"],
                    "reactive_steps": true,
                    "notes": "Use role-specific outputs."
                  }
                }
                """
            )
        if "Agent name:\nDomainSolver" in prompt:
            return FakeResponse("FINAL:\nDomain evidence supports option A. Final Answer: A\nCONFIDENCE:\n0.8")
        if "Agent name:\nOptionVerifier" in prompt:
            return FakeResponse("FINAL:\nThe verified candidate is option A. Final Answer: A\nCONFIDENCE:\n0.9")
        if "Agent name:\nCriticalReviewer" in prompt:
            return FakeResponse("FINAL:\nA possible objection points to option B. Final Answer: B\nCONFIDENCE:\n0.6")
        if "You are the Summarizer" in prompt:
            self.summarizer_prompts.append(prompt)
            return FakeResponse("Final Answer: A")
        return FakeResponse("Unexpected prompt")


class MajorityConflictSummarizerLLM:
    async def ainvoke(self, prompt: str) -> FakeResponse:
        assert "Candidate aggregation:" in prompt
        assert "selected_candidate: B" in prompt
        return FakeResponse("The option text contradicts the majority candidate.\n\nFinal Answer: C")


class DynamicOrchestrationFakeLLM:
    def __init__(self) -> None:
        self.controller_prompts: list[str] = []
        self.agent_prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "dynamic orchestration controller" in prompt:
            self.controller_prompts.append(prompt)
            if len(self.controller_prompts) == 1:
                return FakeResponse(
                    """
                    {
                      "action": "run_agents",
                      "reason": "Start with independent solving and a critical verification pass.",
                      "round_goal": "Get a candidate answer and check it.",
                      "selected_agents": [
                        {
                          "name": "Evidence Solver",
                          "role": "Solves the question from the prompt evidence.",
                          "description": "Owns the direct evidence-to-answer mapping.",
                          "rules": ["Share the answer and one short reason."],
                          "subtask": "Choose the best option from the prompt evidence.",
                          "expected_output": "A candidate answer with a short reason.",
                          "critical_debate": false,
                          "workspace_access": "none"
                        },
                        {
                          "name": "Critical Verifier",
                          "role": "Checks whether the candidate satisfies the prompt.",
                          "description": "Owns contradiction and option-fit checks.",
                          "rules": ["Reject unsupported answers."],
                          "subtask": "Verify the candidate against the question.",
                          "expected_output": "A verified candidate or correction.",
                          "critical_debate": true,
                          "workspace_access": "none"
                        }
                      ],
                      "collaboration_protocol": {
                        "event_types_to_share": ["finding", "critique", "warning"],
                        "reactive_steps": true,
                        "notes": "Agents should compare answer candidates and short reasons."
                      }
                    }
                    """
                )
            assert "Candidate aggregation:" in prompt
            assert "EvidenceSolver" in prompt
            assert "CriticalVerifier" in prompt
            return FakeResponse(
                """
                {
                  "action": "final",
                  "reason": "Both agents independently selected A and no unresolved disagreement remains.",
                  "final_answer": "The solver and verifier agree on the supported option.\\n\\nFinal Answer: A"
                }
                """
            )
        if "Agent name:\nEvidenceSolver" in prompt:
            self.agent_prompts.append(prompt)
            return FakeResponse(
                "FINAL:\nEvidence supports option A. Final Answer: A\n"
                "SHARE_FINDING:\nCandidate A because it matches the prompt evidence.\n"
                "LOCAL_NOTES:\nANSWER_CHOICE: A\nANSWER_REASON: A matches the prompt evidence."
            )
        if "Agent name:\nCriticalVerifier" in prompt:
            self.agent_prompts.append(prompt)
            return FakeResponse(
                "FINAL:\nVerifier confirms option A. Final Answer: A\n"
                "SHARE_FINDING:\nCandidate A passes the option-fit check.\n"
                "LOCAL_NOTES:\nANSWER_CHOICE: A\nANSWER_REASON: A is consistent with the question constraints."
            )
        return FakeResponse("Unexpected prompt")


class DynamicOrchestrationBareFinalLLM:
    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "dynamic orchestration controller" in prompt:
            return FakeResponse(
                """
                {
                  "action": "final",
                  "reason": "The prompt evidence is sufficient to select option B.",
                  "final_answer": "B"
                }
                """
            )
        return FakeResponse("Unexpected prompt")


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
    lifecycle_types = {"task_started", "plan_created", "agent_started", "final_summary"}
    assert not any(event.event_type in lifecycle_types for event in state["event_log"])


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
    assert state["orchestrator_plan"]["mode"] == "direct"
    assert not any(event.event_type in {"task_started", "plan_created", "final_summary"} for event in state["event_log"])


@pytest.mark.asyncio
async def test_workspace_tools_force_multi_agent_runtime_even_for_direct_orchestrator_response():
    workspace = WorkflowFakeDockerWorkspace()
    state = await run_workflow(
        task="What is an API? Inspect the repository.",
        llm=WorkflowToolLLM(),
        max_steps_per_agent=2,
        total_runtime_timeout=10,
        stream_to_console=False,
        enable_workspace_tools=True,
        docker_workspace=workspace,
    )

    assert state["mode"] == "multi_agent"
    assert "CodingAgent" in state["agent_outputs"]
    assert workspace.commands == ["printf workflow"]
    assert state["agent_traces"]


@pytest.mark.asyncio
async def test_non_easy_direct_orchestrator_response_uses_direct_answer_runtime():
    state = await run_workflow(
        task="Which one of the following implementation strategies should we use?",
        llm=FakeLLM(),
        max_steps_per_agent=1,
        stream_to_console=False,
    )

    assert state["mode"] == "direct"
    assert state["agent_outputs"] == {}
    assert state["agent_traces"] == {}
    assert state["reason"] == "The task is simple enough for one LLM response."
    assert "A direct answer from one LLM call." in state["final_answer"]


@pytest.mark.asyncio
async def test_dynamic_direct_route_runs_one_debate_revision_after_initial_answer():
    llm = DynamicDirectFakeLLM()

    state = await run_workflow(
        task="Which option is best?",
        llm=llm,
        max_steps_per_agent=1,
        stream_to_console=False,
        subagent_mode="dynamic",
        benchmark="mmlu_pro",
    )

    assert state["mode"] == "direct"
    assert state["subagent_mode"] == "dynamic"
    assert state["agent_outputs"] == {}
    assert state["agent_traces"] == {}
    assert state["final_answer"] == "Revised dynamic direct answer. Final Answer: B"
    assert len(llm.direct_prompts) == 1
    assert len(llm.debate_prompts) == 1
    assert "Initial dynamic direct answer. Final Answer: A" in llm.debate_prompts[0]
    assert "Using the reasoning from other agents as additional advice" in llm.debate_prompts[0]
    assert "Put your answer in the form (X) at the end of your response." in llm.debate_prompts[0]
    assert state["direct_trace"]["mode"] == "dynamic_direct_debate"
    assert [step["kind"] for step in state["direct_trace"]["steps"]] == ["direct_answer", "debate_revision"]


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
    assert set(state["agent_outputs"]) == {"ImplementationPlanner", "CriticalDebateAgent", "VerificationAgent"}
    assert state["selected_agents"][0]["description"] == "Identifies files, tests, and integration points."
    assert state["selected_agents"][1]["critical_debate"] is True
    assert any(event.event_type == "critique" and event.source == "CriticalDebateAgent" for event in state["event_log"])
    assert state["final_answer"] == "Summarized dynamic-agent answer."


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
    assert state["final_answer"] == "Summarized workspace tool answer."


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
    assert step["prompt"].startswith("<|think|>\n")
    assert "[GLOBAL STATIC PREFIX]\nCalculate 2 + 2." in step["prompt"]
    assert "Agent name:\nSolverAgent" in step["prompt"]
    assert "[STEP DYNAMIC SUFFIX]" in step["prompt"]
    assert "SUMMARY:" in step["raw_response"]
    assert step["parsed_output"]["share_finding"] == "Share a concise implementation-relevant finding."
    assert step["published_events"]
    assert step["duration_seconds"] >= 0


@pytest.mark.asyncio
async def test_summarize_outputs_summarizer_prompt_and_response_are_traced():
    llm = SummarizerTraceLLM()

    state = await run_workflow(
        task="Calculate 2 + 2.",
        llm=llm,
        max_steps_per_agent=1,
        total_runtime_timeout=5,
        stream_to_console=False,
        synthesizer_mode="summarize_outputs",
    )

    assert state["final_answer"] == "Final Answer: summarized-agent-output"
    assert len(llm.summarizer_prompts) == 1
    assert llm.summarizer_prompts[0].startswith("<|think|>\n")
    assert "You are the Summarizer" in llm.summarizer_prompts[0]
    assert "Do not solve the task again." in llm.summarizer_prompts[0]
    assert "choose and summarize the best-supported answer from the agents' last summaries" in llm.summarizer_prompts[0]
    assert "Use the candidate aggregation as evidence, not as an automatic authority." in llm.summarizer_prompts[0]
    assert "Agent last summaries:" in llm.summarizer_prompts[0]
    assert "Runtime event log" not in llm.summarizer_prompts[0]
    assert "[agent_done]" not in llm.summarizer_prompts[0]
    assert state["synthesizer_trace"]["prompt"] == llm.summarizer_prompts[0]
    assert state["synthesizer_trace"]["raw_response"] == "Final Answer: summarized-agent-output"
    assert state["synthesizer_trace"]["final_answer"] == "Final Answer: summarized-agent-output"
    assert state["synthesizer_trace"]["timed_out"] is False


@pytest.mark.asyncio
async def test_workflow_can_disable_think_mode_for_runtime_prompts():
    llm = PromptCaptureLLM()

    state = await run_workflow(
        task="Calculate 2 + 2.",
        llm=llm,
        max_steps_per_agent=1,
        total_runtime_timeout=5,
        stream_to_console=False,
        synthesizer_mode="summarize_outputs",
        think_mode=False,
    )

    solver_prompt = state["agent_traces"]["SolverAgent"]["steps"][0]["prompt"]
    summarizer_prompt = state["synthesizer_trace"]["prompt"]

    assert state["think_mode"] is False
    assert "You are the model-based orchestrator" in llm.prompts[0]
    assert not llm.prompts[0].startswith("<|think|>")
    assert not solver_prompt.startswith("<|think|>")
    assert not summarizer_prompt.startswith("<|think|>")
    assert not any(prompt.startswith("<|think|>") for prompt in llm.summarizer_prompts)


@pytest.mark.asyncio
async def test_summarizer_receives_only_each_agents_last_summary():
    llm = SummarizerLastSummaryLLM()

    state = await run_workflow(
        task="Calculate 2 + 2.",
        llm=llm,
        max_steps_per_agent=2,
        total_runtime_timeout=5,
        stream_to_console=False,
        synthesizer_mode="summarize_outputs",
        enable_agent_message_streaming=False,
    )

    assert state["final_answer"] == "Final Answer: last-summaries-only"
    assert len(llm.summarizer_prompts) == 1
    assert state["synthesizer_trace"]["prompt"] == llm.summarizer_prompts[0]
    assert state["synthesizer_trace"]["agent_last_summaries"] == {
        "SolverAgent": "Solver step 2 final summary.",
        "VerifierAgent": "Verifier step 2 final summary.",
    }


@pytest.mark.asyncio
async def test_full_trace_summarizer_receives_every_raw_agent_step():
    llm = FullTraceSummarizerLLM()

    state = await run_workflow(
        task="Calculate 2 + 2.",
        llm=llm,
        max_steps_per_agent=2,
        total_runtime_timeout=5,
        stream_to_console=False,
        synthesizer_mode="summarize_outputs",
        enable_agent_message_streaming=False,
        full_trace_sharing=True,
    )

    prompt = state["synthesizer_trace"]["prompt"]
    assert "Complete agent traces:" in prompt
    for agent_name, trace in state["agent_traces"].items():
        assert f"=== {agent_name} ===" in prompt
        for step in trace["steps"]:
            assert step["raw_response"] in prompt
    assert state["synthesizer_trace"]["full_trace_sharing"] is True


@pytest.mark.asyncio
async def test_dynamic_summarize_outputs_uses_candidate_majority_with_role_context():
    llm = RoleAwareAggregationLLM()

    state = await run_workflow(
        task="Which option is best? A. correct B. tempting C. wrong D. wrong",
        llm=llm,
        max_steps_per_agent=1,
        total_runtime_timeout=5,
        stream_to_console=False,
        subagent_mode="dynamic",
        synthesizer_mode="summarize_outputs",
        benchmark="gpqa",
    )

    assert state["final_answer"] == "Final Answer: A"
    assert len(llm.summarizer_prompts) == 1
    summarizer_prompt = llm.summarizer_prompts[0]
    assert "Candidate aggregation:" in summarizer_prompt
    assert "DomainSolver" in summarizer_prompt
    assert "Primary domain solver." in summarizer_prompt
    assert "Solve the question and choose the best option." in summarizer_prompt
    assert "candidate: A" in summarizer_prompt
    assert "candidate: B" in summarizer_prompt
    aggregation = state["synthesizer_trace"]["candidate_aggregation"]
    assert aggregation["selected_candidate"] == "A"
    assert aggregation["selection_rule"] == "weak_majority_requires_evidence_review"
    assert aggregation["consensus_strength"] == "weak_majority"
    assert aggregation["needs_review"] is True
    assert aggregation["counts"] == {"A": 2, "B": 1}


@pytest.mark.asyncio
async def test_dynamic_orchestration_runs_agent_round_then_orchestrator_final():
    from multi_agent_sync.graph.dynamic_orchestration import run_dynamic_orchestration_workflow

    llm = DynamicOrchestrationFakeLLM()

    state = await run_dynamic_orchestration_workflow(
        task="Which option is correct? A. supported B. unsupported C. unsupported D. unsupported",
        llm=llm,
        benchmark="gpqa",
        max_steps_per_agent=1,
        max_orchestrator_rounds=2,
        total_runtime_timeout=5,
        stream_to_console=False,
    )

    assert state["method"] == "dynamic_orchestration"
    assert state["final_answer"] == "The solver and verifier agree on the supported option.\n\nFinal Answer: A"
    assert len(llm.controller_prompts) == 2
    assert "Avoid backslashes in JSON strings" in llm.controller_prompts[0]
    assert len(llm.agent_prompts) == 2
    trace = state["dynamic_orchestration_trace"]
    assert trace["forced_final"] is False
    assert [round_trace["decision"]["action"] for round_trace in trace["rounds"]] == ["run_agents", "final"]
    first_round = trace["rounds"][0]
    assert [agent["name"] for agent in first_round["selected_agents"]] == ["EvidenceSolver", "CriticalVerifier"]
    assert first_round["candidate_aggregation"]["counts"] == {"A": 2}
    assert first_round["candidate_aggregation"]["consensus_strength"] == "unanimous"
    assert "EvidenceSolver" in first_round["agent_outputs"]
    assert "CriticalVerifier" in first_round["agent_traces"]
    assert "synthesizer_trace" not in state


@pytest.mark.asyncio
async def test_dynamic_orchestration_formats_bare_final_candidate_for_benchmark():
    from multi_agent_sync.evaluation.benchmarks.gpqa import extract_answer
    from multi_agent_sync.graph.dynamic_orchestration import run_dynamic_orchestration_workflow

    state = await run_dynamic_orchestration_workflow(
        task="Which option is correct? A. unsupported B. supported C. unsupported D. unsupported",
        llm=DynamicOrchestrationBareFinalLLM(),
        benchmark="gpqa",
        max_orchestrator_rounds=1,
        stream_to_console=False,
    )

    assert state["final_answer"] == "Final Answer: B"
    assert state["dynamic_orchestration_trace"]["final_answer"] == "Final Answer: B"
    assert extract_answer(state["final_answer"]) == "B"


def test_dynamic_orchestration_parses_controller_json_with_latex_backslash_escape():
    from multi_agent_sync.graph.dynamic_orchestration import parse_orchestration_decision

    decision = parse_orchestration_decision(
        r"""
        {
          "action": "final",
          "reason": "The value \sqrt{2} check supports this conclusion.",
          "final_answer": "Final Answer: B"
        }
        """,
        "Which option is correct?",
    )

    assert decision["action"] == "final"
    assert decision["final_answer"] == "Final Answer: B"
    assert r"\sqrt{2}" in decision["reason"]


def test_dynamic_orchestration_fallback_does_not_expose_prompt_answer_placeholder():
    from multi_agent_sync.evaluation.benchmarks.olymmath import extract_answer
    from multi_agent_sync.graph.dynamic_orchestration import fallback_final_answer
    from multi_agent_sync.graph.nodes import build_agent_last_summaries, build_candidate_aggregation

    task = "Solve the math problem.\n\nFinal Answer: <answer>"
    state = {
        "task": task,
        "benchmark": "olymmath",
        "assignments": [{"agent_name": "Solver", "task": "Solve.", "role": "Solver"}],
        "agent_outputs": {"Solver": "The recurrence gives the result. Final Answer: 144"},
        "agent_traces": {
            "Solver": {
                "steps": [
                    {
                        "parsed_output": {
                            "local_notes": "ANSWER_CHOICE: 144\nANSWER_REASON: The recurrence sum gives 144."
                        }
                    }
                ]
            }
        },
        "synthesizer_mode": "summarize_outputs",
    }
    aggregation = build_candidate_aggregation(state, build_agent_last_summaries(state))

    answer = fallback_final_answer(task, "olymmath", aggregation, state, "controller failed")

    assert "<answer>" not in answer
    assert answer == r"\boxed{144}"
    assert extract_answer(answer) == "144"


def test_dynamic_orchestration_fallback_without_candidate_omits_original_task_prompt():
    from multi_agent_sync.graph.dynamic_orchestration import fallback_final_answer

    task = "Solve the math problem.\n\nFinal Answer: <answer>"
    aggregation = {
        "selected_candidate": None,
        "selection_rule": "no_extractable_candidates",
        "needs_review": True,
    }

    answer = fallback_final_answer(
        task,
        "olymmath",
        aggregation,
        {
            "task": task,
            "benchmark": "olymmath",
            "agent_outputs": {},
            "synthesizer_mode": "summarize_outputs",
        },
        "controller failed",
    )

    assert "Final Answer: <answer>" not in answer
    assert "<answer>" not in answer
    assert "No agent outputs were available" in answer


@pytest.mark.asyncio
async def test_summarizer_valid_answer_is_not_overridden_by_candidate_majority():
    state = await nodes.synthesizer_node(
        {
            "task": "Choose one option.\n\nOptions:\nA. wrong\nB. tempting\nC. correct\nD. wrong",
            "benchmark": "gpqa",
            "run_id": "test-run",
            "llm": MajorityConflictSummarizerLLM(),
            "synthesizer_mode": "summarize_outputs",
            "assignments": [
                {"agent_name": "SolverOne", "role": "Solve", "task": "Choose an answer."},
                {"agent_name": "SolverTwo", "role": "Solve", "task": "Choose an answer."},
                {"agent_name": "Verifier", "role": "Check", "task": "Verify the answer."},
            ],
            "agent_traces": {
                "SolverOne": {
                    "steps": [
                        {
                            "parsed_output": {
                                "summary": "The first solver picked B.",
                                "local_notes": "ANSWER_CHOICE: B\nANSWER_REASON: B looked plausible.",
                            }
                        }
                    ]
                },
                "SolverTwo": {
                    "steps": [
                        {
                            "parsed_output": {
                                "summary": "The second solver picked B.",
                                "local_notes": "ANSWER_CHOICE: B\nANSWER_REASON: B looked plausible.",
                            }
                        }
                    ]
                },
                "Verifier": {
                    "steps": [
                        {
                            "parsed_output": {
                                "summary": "The verifier noticed C matches the option text.",
                                "local_notes": "ANSWER_CHOICE: C\nANSWER_REASON: C is option-consistent.",
                            }
                        }
                    ]
                },
            },
        }
    )

    assert state["final_answer"] == "The option text contradicts the majority candidate.\n\nFinal Answer: C"
    assert state["synthesizer_trace"]["final_answer"] == state["final_answer"]
    assert state["synthesizer_trace"]["candidate_aggregation"]["selected_candidate"] == "B"


def test_candidate_aggregation_prefers_answer_choice_from_agent_trace():
    state = {
        "benchmark": "gpqa",
        "assignments": [
            {"agent_name": "SolverAgent", "role": "Solve", "task": "Choose an answer."},
            {"agent_name": "ReviewerAgent", "role": "Review", "task": "Check the answer."},
        ],
        "agent_traces": {
            "SolverAgent": {
                "steps": [
                    {
                        "parsed_output": {
                            "summary": "Long reasoning mentions Final Answer: A before correcting course.",
                            "local_notes": "ANSWER_CHOICE: C\nANSWER_REASON: C is the corrected final candidate.",
                        }
                    }
                ]
            },
            "ReviewerAgent": {
                "steps": [
                    {
                        "parsed_output": {
                            "summary": "Long reasoning mentions Final Answer: A as a rejected candidate.",
                            "local_notes": "ANSWER_CHOICE: C\nANSWER_REASON: C is the corrected final candidate.",
                        }
                    }
                ]
            },
        },
    }

    aggregation = nodes.build_candidate_aggregation(
        state,
        {
            "SolverAgent": "Long reasoning mentions Final Answer: A before correcting course.",
            "ReviewerAgent": "Long reasoning mentions Final Answer: A as a rejected candidate.",
        },
    )

    assert aggregation["counts"] == {"C": 2}
    assert aggregation["selected_candidate"] == "C"
    assert aggregation["selection_rule"] == "majority_vote"
    assert [candidate["candidate_source"] for candidate in aggregation["candidates"]] == ["answer_choice", "answer_choice"]


@pytest.mark.asyncio
async def test_missing_options_guard_is_added_to_final_answer():
    state = await run_workflow(
        task="Which one of the following options is closest to the threshold value?",
        llm=FakeLLM(),
        stream_to_console=False,
    )

    assert "The options are missing, so I cannot choose one of them." in state["final_answer"]


@pytest.mark.asyncio
async def test_workflow_uses_fallback_when_summarizer_times_out():
    state = await run_workflow(
        task="Build a prototype chess website",
        llm=HangingSummarizerLLM(),
        max_steps_per_agent=1,
        total_runtime_timeout=10,
        synthesis_timeout=0.05,
        stream_to_console=False,
    )

    assert "Summarizer timed out" in state["final_answer"]
    assert any(event.event_type == "warning" and event.source == "Summarizer" for event in state["event_log"])
    assert not any(event.event_type == "final_summary" for event in state["event_log"])


def test_autogen_dependency_is_not_used():
    installed = {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}
    assert "autogen" not in installed
    assert "pyautogen" not in installed
