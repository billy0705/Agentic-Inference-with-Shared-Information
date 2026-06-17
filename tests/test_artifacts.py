import json
from dataclasses import dataclass

import pytest

from multi_agent_sync.artifacts import save_run_artifacts
from multi_agent_sync.graph.workflow import run_workflow


@dataclass
class FakeResponse:
    content: str


class FakeLLM:
    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "Answer the user task directly" in prompt:
            return FakeResponse("Direct answer.")
        if "Synthesizer" in prompt:
            return FakeResponse("Synthesized answer.")
        return FakeResponse(
            "SUMMARY:\nSolved a step.\n"
            "SHARE_FINDING:\nUseful calculation finding.\n"
            "CONFIDENCE:\n0.8\n"
            "LOCAL_NOTES:\nChecked the reasoning."
        )


@pytest.mark.asyncio
async def test_save_run_artifacts_writes_expected_files(tmp_path):
    state = await run_workflow(
        task="Calculate 2 + 2.",
        llm=FakeLLM(),
        max_steps_per_agent=1,
        total_runtime_timeout=5,
        synthesis_timeout=5,
        stream_to_console=False,
    )

    run_dir = save_run_artifacts(state, root_dir=tmp_path)

    assert (run_dir / "task.txt").read_text() == "Calculate 2 + 2."
    assert (run_dir / "final_answer.md").read_text()
    assert (run_dir / "event_log.jsonl").read_text().strip()
    assert json.loads((run_dir / "agent_traces.json").read_text())
    sync_report = json.loads((run_dir / "sync_report.json").read_text())
    assert "messages_sent" in sync_report
    assert "agent_pairs" in sync_report
    orchestrator_plan = json.loads((run_dir / "orchestrator_plan.json").read_text())
    assert orchestrator_plan["task_type"] == "reasoning and verification task"
    assert orchestrator_plan["mode"] == "multi_agent"
    assert [agent["name"] for agent in orchestrator_plan["selected_agents"]] == [
        "SolverAgent",
        "CriticAgent",
        "VerifierAgent",
    ]
