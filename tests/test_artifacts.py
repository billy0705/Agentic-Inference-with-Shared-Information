import json
from dataclasses import dataclass

import pytest

from multi_agent_sync.artifacts import build_token_usage_by_step, format_token_usage_by_step, save_run_artifacts
from multi_agent_sync.graph.workflow import run_workflow


@dataclass
class FakeResponse:
    content: str


class FakeLLM:
    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "Answer the user task directly" in prompt:
            return FakeResponse("Direct answer.")
        if "Summarizer" in prompt:
            return FakeResponse("Summarized answer.")
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
    assert json.loads((run_dir / "token_usage_by_step.json").read_text())
    assert "Token usage by step" in (run_dir / "token_usage_by_step.txt").read_text()
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


def test_build_token_usage_by_step_extracts_each_agent_step():
    report = build_token_usage_by_step(
        {
            "ResearchAgent": {
                "steps": [
                    {
                        "step": 1,
                        "token_usage": {
                            "prompt_tokens": 12,
                            "completion_tokens": 5,
                            "total_tokens": 17,
                        },
                    }
                ]
            },
            "CodingAgent": {
                "steps": [
                    {
                        "step": 1,
                        "token_usage": {
                            "prompt_tokens": None,
                            "completion_tokens": None,
                            "total_tokens": None,
                        },
                    }
                ]
            },
        }
    )

    assert report == [
        {"agent": "ResearchAgent", "step": 1, "prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
        {"agent": "CodingAgent", "step": 1, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    ]


def test_format_token_usage_by_step_includes_every_step():
    report = [
        {"agent": "ResearchAgent", "step": 1, "prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17},
        {"agent": "CodingAgent", "step": 2, "prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
    ]

    assert format_token_usage_by_step(report) == (
        "Token usage by step\n"
        "- ResearchAgent step 1: prompt=12, completion=5, total=17\n"
        "- CodingAgent step 2: prompt=unknown, completion=unknown, total=unknown"
    )
