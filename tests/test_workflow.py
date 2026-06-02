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
    assert any(event.event_type == "final_summary" for event in state["event_log"])


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
    assert any(event.event_type == "final_summary" for event in state["event_log"])


def test_autogen_dependency_is_not_used():
    installed = {dist.metadata["Name"].lower() for dist in importlib.metadata.distributions()}
    assert "autogen" not in installed
    assert "pyautogen" not in installed
