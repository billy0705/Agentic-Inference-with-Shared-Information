import asyncio
import importlib.metadata
from dataclasses import dataclass

import pytest

from multi_agent_sync.graph.workflow import run_workflow


@dataclass
class FakeResponse:
    content: str


class FakeLLM:
    async def ainvoke(self, prompt: str) -> FakeResponse:
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


class HangingSynthesizerLLM(FakeLLM):
    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "Synthesizer" in prompt:
            await asyncio.sleep(10)
        return await super().ainvoke(prompt)


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
async def test_direct_route_skips_agent_runtime():
    state = await run_workflow(
        task="What is an API?",
        llm=FakeLLM(),
        max_steps_per_agent=1,
        stream_to_console=False,
    )

    assert state["mode"] == "direct"
    assert state["assignments"] == []
    assert state["agent_outputs"] == {}
    assert state["agent_traces"] == {}
    assert state["final_answer"] == "A direct answer from one LLM call."
    assert any(
        event.event_type == "plan_created"
        and "mode=direct, task_type=simple_qa, selected_agents=none" in event.content
        for event in state["event_log"]
    )


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
    assert state["task_type"] == "calculation"
    assert set(state["agent_outputs"]) == {"SolverAgent", "VerifierAgent"}
    assert "CodingAgent" not in state["agent_outputs"]
    assert set(state["agent_traces"]) == {"SolverAgent", "VerifierAgent"}


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
