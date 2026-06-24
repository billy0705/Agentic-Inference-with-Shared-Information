import csv
import json
import types
from dataclasses import dataclass

import pytest

from multi_agent_sync.evaluation import main as evaluation
from multi_agent_sync.evaluation import gpqa
from multi_agent_sync.evaluation import runner
from multi_agent_sync.evaluation.types import BenchmarkSpec


@dataclass
class UsageResponse:
    content: str
    usage_metadata: dict[str, int] | None = None
    response_metadata: dict | None = None


class UsageLLM:
    def __init__(self, responses: list[UsageResponse]) -> None:
        self.responses = list(responses)

    async def ainvoke(self, prompt: str) -> UsageResponse:
        return self.responses.pop(0)


def test_parse_methods_accepts_comma_separated_methods():
    assert runner.parse_methods("multiagent_streaming,multiagent_no_streaming,plain_llm") == [
        "multiagent_streaming",
        "multiagent_no_streaming",
        "plain_llm",
    ]


def test_parse_methods_accepts_dynamic_multiagent_methods():
    assert runner.parse_methods("multiagent_dynamic_streaming,multiagent_dynamic_no_streaming,plain_llm") == [
        "multiagent_dynamic_streaming",
        "multiagent_dynamic_no_streaming",
        "plain_llm",
    ]


def test_parse_methods_keeps_multiagent_alias_for_streaming():
    assert runner.parse_methods("multiagent,plain_llm") == ["multiagent", "plain_llm"]


def test_parse_methods_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        runner.parse_methods("multiagent,architect")


def test_parser_accepts_simple_benchmark_command_shape():
    args = evaluation.build_parser().parse_args(
        [
            "--benchmark",
            "gpqa",
            "--methods",
            "multiagent_streaming,multiagent_no_streaming,plain_llm",
            "--limit",
            "10",
            "--data-file",
            "gpqa.csv",
        ]
    )

    assert args.benchmark == "gpqa"
    assert args.methods == "multiagent_streaming,multiagent_no_streaming,plain_llm"
    assert args.limit == 10
    assert args.data_file == "gpqa.csv"


def test_json_trace_saving_is_enabled_by_default():
    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa"])
    disabled_args = evaluation.build_parser().parse_args(["--benchmark", "gpqa", "--no-save-json-traces"])

    assert args.save_json_traces is True
    assert disabled_args.save_json_traces is False


@pytest.mark.asyncio
async def test_run_evaluation_sets_max_tokens_to_16384(monkeypatch, tmp_path):
    captured_llm_kwargs = {}

    def fake_get_llm(model=None, openai=True, max_tokens=None):
        captured_llm_kwargs["model"] = model
        captured_llm_kwargs["openai"] = openai
        captured_llm_kwargs["max_tokens"] = max_tokens
        return "fake-llm"

    async def fake_run_method(method, prompt, llm, args):
        return runner.RunResult(
            raw_output="Final Answer: A",
            returncode=0,
            elapsed_seconds=0.1,
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        )

    benchmark = BenchmarkSpec(
        name="fake",
        display_name="Fake",
        default_output_filename="fake.csv",
        load_items=lambda args: [
            {
                "Question": "Question?",
                "Correct Answer": "A correct answer",
                "Incorrect Answer 1": "Wrong answer 1",
                "Incorrect Answer 2": "Wrong answer 2",
                "Incorrect Answer 3": "Wrong answer 3",
                "Explanation": "Large explanation that should not be duplicated in traces.",
            }
        ],
        build_prompt=lambda row, rng: ("Question?", "A"),
        extract_answer=lambda text: "A",
    )

    monkeypatch.setattr(evaluation, "get_benchmarks", lambda: {"fake": benchmark})
    monkeypatch.setattr(evaluation, "get_llm", fake_get_llm)
    monkeypatch.setattr(runner, "run_method", fake_run_method)
    monkeypatch.setattr(runner, "progress", lambda items, desc: items)

    args = evaluation.build_parser().parse_args(
        ["--benchmark", "fake", "--methods", "plain_llm", "--output", str(tmp_path / "results.csv")]
    )

    await evaluation.run_evaluation(args)

    assert captured_llm_kwargs == {"model": None, "openai": True, "max_tokens": 16384}


@pytest.mark.asyncio
async def test_run_evaluation_writes_unique_csv_and_json_trace_by_default(monkeypatch, tmp_path):
    benchmark = BenchmarkSpec(
        name="fake",
        display_name="Fake",
        default_output_filename="fake.csv",
        load_items=lambda args: [
            {
                "Question": "Question?",
                "Correct Answer": "A correct answer",
                "Incorrect Answer 1": "Wrong answer 1",
                "Incorrect Answer 2": "Wrong answer 2",
                "Incorrect Answer 3": "Wrong answer 3",
                "Explanation": "Large explanation that should not be duplicated in traces.",
            }
        ],
        build_prompt=lambda row, rng: ("Question?", "A"),
        extract_answer=lambda text: "A",
    )

    async def fake_run_method(method, prompt, llm, args):
        return runner.RunResult(
            raw_output="Final Answer: A",
            returncode=0,
            elapsed_seconds=0.1,
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
            trace={
                "final_answer": "Final Answer: A",
                "orchestrator_plan": {"selected_agents": [{"name": "SolverAgent"}]},
                "event_log": [
                    {
                        "source": "SolverAgent",
                        "target": "broadcast",
                        "event_type": "finding",
                        "content": "Solver shared a useful finding.",
                        "confidence": 0.8,
                    }
                ],
                "agent_traces": {"SolverAgent": {"steps": [{"used_event_ids": ["event-1"]}]}},
            },
        )

    monkeypatch.setattr(evaluation, "get_benchmarks", lambda: {"fake": benchmark})
    monkeypatch.setattr(evaluation, "get_llm", lambda model=None, openai=True, max_tokens=None: "fake-llm")
    monkeypatch.setattr(runner, "run_method", fake_run_method)
    monkeypatch.setattr(runner, "progress", lambda items, desc: items)
    monkeypatch.setattr(runner, "create_run_id", lambda: "run-test", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    args = evaluation.build_parser().parse_args(["--benchmark", "fake", "--methods", "plain_llm", "--output-dir", str(tmp_path)])

    results = await evaluation.run_evaluation(args)

    output_path = tmp_path / "fake_run-test.csv"
    trace_path = tmp_path / "json_traces" / "run-test" / "examples" / "0000_plain_llm.json"
    run_config_path = tmp_path / "json_traces" / "run-test" / "run_config.json"
    summary_path = tmp_path / "json_traces" / "run-test" / "summary.json"

    assert output_path.exists()
    assert trace_path.exists()
    assert run_config_path.exists()
    assert summary_path.exists()
    assert results[0]["json_trace_path"] == str(trace_path)

    csv_rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))
    assert csv_rows[0]["json_trace_path"] == str(trace_path)

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["method"] == "plain_llm"
    assert trace["settings"]["save_json_traces"] is True
    assert trace["settings"]["resolved_model"] == "openai/gpt-oss-120b"
    assert "row" not in trace
    assert trace["question"] == "Question?"
    assert trace["options"] == {
        "correct": "A correct answer",
        "incorrect": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"],
    }
    assert trace["method_trace"]["orchestrator_plan"]["selected_agents"][0]["name"] == "SolverAgent"
    assert trace["multiagent_debug"]["subagents"] == [{"name": "SolverAgent"}]
    assert trace["multiagent_debug"]["messages"][0]["content"] == "Solver shared a useful finding."
    assert trace["multiagent_debug"]["workflow"] == [
        {"node": "orchestrator", "description": "Created plan and selected subagents."},
        {"node": "SolverAgent", "description": "Ran subagent and published/received messages."},
        {"node": "synthesizer", "description": "Combined subagent outputs and event log into final answer."},
    ]

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert run_config["settings"]["resolved_model"] == "openai/gpt-oss-120b"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["summary"]["plain_llm"]["avg_prompt_tokens"] == 1.0
    assert summary["summary"]["plain_llm"]["avg_completion_tokens"] == 2.0
    assert summary["summary"]["plain_llm"]["avg_total_tokens"] == 3.0
    assert summary["method_averages"]["plain_llm"] == {
        "correct_avg": 1.0,
        "time_avg_seconds": 0.1,
        "total_token_avg": 3.0,
    }


@pytest.mark.asyncio
async def test_run_evaluation_updates_results_summary_and_correctness_matrix_incrementally(monkeypatch, tmp_path):
    benchmark = BenchmarkSpec(
        name="fake",
        display_name="Fake",
        default_output_filename="fake.csv",
        load_items=lambda args: [
            {"Question": "Question 0?", "Correct Answer": "A", "Incorrect Answer 1": "B", "Incorrect Answer 2": "C", "Incorrect Answer 3": "D"},
            {"Question": "Question 1?", "Correct Answer": "A", "Incorrect Answer 1": "B", "Incorrect Answer 2": "C", "Incorrect Answer 3": "D"},
        ],
        build_prompt=lambda row, rng: (row["Question"], "A"),
        extract_answer=lambda text: text.removeprefix("Final Answer: ").strip(),
    )
    calls = []

    async def fake_run_method(method, prompt, llm, args):
        calls.append((method, prompt))
        output_path = tmp_path / "fake_run-test.csv"
        summary_path = tmp_path / "json_traces" / "run-test" / "summary.json"
        matrix_path = tmp_path / "json_traces" / "run-test" / "correctness.csv"
        if len(calls) == 2:
            assert output_path.exists()
            assert summary_path.exists()
            assert matrix_path.exists()
            csv_rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))
            matrix_rows = list(csv.DictReader(matrix_path.open(newline="", encoding="utf-8")))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            assert len(csv_rows) == 1
            assert matrix_rows == [{"task_id": "0", "plain_llm": "T", "multiagent_streaming": ""}]
            assert summary["method_averages"]["plain_llm"]["correct_avg"] == 1.0
        pred = "A" if method == "plain_llm" else "B"
        return runner.RunResult(
            raw_output=f"Final Answer: {pred}",
            returncode=0,
            elapsed_seconds=0.1,
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            trace={"prompt": prompt, "raw_output": f"Final Answer: {pred}"},
        )

    monkeypatch.setattr(evaluation, "get_benchmarks", lambda: {"fake": benchmark})
    monkeypatch.setattr(evaluation, "get_llm", lambda model=None, openai=True, max_tokens=None: "fake-llm")
    monkeypatch.setattr(runner, "run_method", fake_run_method)
    monkeypatch.setattr(runner, "progress", lambda items, desc: items)
    monkeypatch.setattr(runner, "create_run_id", lambda: "run-test", raising=False)

    args = evaluation.build_parser().parse_args(
        [
            "--benchmark",
            "fake",
            "--methods",
            "plain_llm,multiagent_streaming",
            "--output-dir",
            str(tmp_path),
        ]
    )

    await evaluation.run_evaluation(args)

    matrix_rows = list(csv.DictReader((tmp_path / "json_traces" / "run-test" / "correctness.csv").open(newline="", encoding="utf-8")))
    assert matrix_rows == [
        {"task_id": "0", "plain_llm": "T", "multiagent_streaming": "F"},
        {"task_id": "1", "plain_llm": "T", "multiagent_streaming": "F"},
    ]


def test_default_output_path_uses_run_id_to_avoid_overwriting():
    benchmark = gpqa.build_benchmark()
    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa"])

    assert runner.resolve_output_path(benchmark, args, run_id="run-123") == (
        runner.DEFAULT_OUTPUT_DIR / "gpqa_diamond_results_run-123.csv"
    )


def test_output_filename_is_written_inside_output_folder():
    benchmark = gpqa.build_benchmark()
    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa", "--output", "result.csv"])

    assert runner.resolve_output_path(benchmark, args) == runner.DEFAULT_OUTPUT_DIR / "result.csv"


def test_gpqa_owns_answer_extraction():
    benchmark = gpqa.build_benchmark()

    assert benchmark.extract_answer("Final Answer: C") == "C"


def test_summarize_results_groups_accuracy_by_method():
    summary = runner.summarize_results(
        [
            {
                "method": "multiagent",
                "correct": True,
                "pred": "A",
                "elapsed_seconds": 1.25,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            {
                "method": "multiagent",
                "correct": False,
                "pred": None,
                "elapsed_seconds": 2.75,
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
            {
                "method": "plain_llm",
                "correct": True,
                "pred": "B",
                "elapsed_seconds": 0.5,
                "prompt_tokens": 20,
                "completion_tokens": 3,
                "total_tokens": 23,
            },
        ]
    )

    assert summary["multiagent"]["total"] == 2
    assert summary["multiagent"]["accuracy"] == 0.5
    assert summary["multiagent"]["invalid_rate"] == 0.5
    assert summary["multiagent"]["elapsed_seconds"] == 4.0
    assert summary["multiagent"]["total_tokens"] == 33
    assert summary["multiagent"]["avg_prompt_tokens"] == 10.5
    assert summary["multiagent"]["avg_completion_tokens"] == 6.0
    assert summary["multiagent"]["avg_total_tokens"] == 16.5
    assert summary["plain_llm"]["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_run_method_records_elapsed_time_and_usage_metadata_tokens():
    llm = UsageLLM(
        [
            UsageResponse(
                content="Final Answer: A",
                usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
            )
        ]
    )
    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa"])

    result = await runner.run_method("plain_llm", "Question?", llm, args)

    assert result.raw_output == "Final Answer: A"
    assert result.returncode == 0
    assert result.elapsed_seconds >= 0
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 4
    assert result.total_tokens == 16


@pytest.mark.asyncio
async def test_run_method_records_response_metadata_token_usage():
    llm = UsageLLM(
        [
            UsageResponse(
                content="Final Answer: B",
                response_metadata={
                    "token_usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 6,
                        "total_tokens": 15,
                    }
                },
            )
        ]
    )
    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa"])

    result = await runner.run_method("plain_llm", "Question?", llm, args)

    assert result.prompt_tokens == 9
    assert result.completion_tokens == 6
    assert result.total_tokens == 15


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "expected_subagent_mode", "expected_streaming"),
    [
        ("multiagent_streaming", "fixed", True),
        ("multiagent_no_streaming", "fixed", False),
        ("multiagent_dynamic_streaming", "dynamic", True),
        ("multiagent_dynamic_no_streaming", "dynamic", False),
    ],
)
async def test_run_method_passes_subagent_mode_and_streaming_to_workflow(
    monkeypatch,
    method,
    expected_subagent_mode,
    expected_streaming,
):
    captured_kwargs = {}

    async def fake_run_workflow(**kwargs):
        captured_kwargs.update(kwargs)
        return {"final_answer": f"{method} answer"}

    monkeypatch.setattr(runner, "run_workflow", fake_run_workflow)
    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa"])

    result = await runner.run_method(method, "Question?", UsageLLM([]), args)

    assert result.raw_output == f"{method} answer"
    assert captured_kwargs["subagent_mode"] == expected_subagent_mode
    assert captured_kwargs["enable_agent_message_streaming"] is expected_streaming


def test_write_results_csv_includes_timing_and_token_columns(tmp_path):
    output_path = tmp_path / "results.csv"

    runner.write_results_csv(
        output_path,
        [
            {
                "benchmark": "gpqa",
                "method": "plain_llm",
                "index": 0,
                "gold": "A",
                "pred": "A",
                "correct": True,
                "returncode": 0,
                "error": "",
                "elapsed_seconds": 0.125,
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
                "raw_output": "Final Answer: A",
            }
        ],
    )

    rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))

    assert rows[0]["elapsed_seconds"] == "0.125"
    assert rows[0]["prompt_tokens"] == "12"
    assert rows[0]["completion_tokens"] == "4"
    assert rows[0]["total_tokens"] == "16"
    assert "json_trace_path" in rows[0]


def test_evaluation_main_module_is_importable():
    from multi_agent_sync.evaluation import main as module

    assert callable(module.main)
    assert callable(module.run_evaluation)


def test_load_local_gpqa_csv_rows(tmp_path):
    path = tmp_path / "gpqa.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Question",
                "Correct Answer",
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Question": "Question?",
                "Correct Answer": "Correct",
                "Incorrect Answer 1": "Wrong 1",
                "Incorrect Answer 2": "Wrong 2",
                "Incorrect Answer 3": "Wrong 3",
            }
        )

    rows = gpqa.load_local_gpqa_rows(path, limit=1)

    assert rows == [
        {
            "Question": "Question?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
        }
    ]


def test_load_gpqa_dataset_falls_back_to_default_local_file_when_hub_is_gated(monkeypatch, tmp_path):
    path = tmp_path / "gpqa_diamond.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Question",
                "Correct Answer",
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Question": "Local question?",
                "Correct Answer": "Correct",
                "Incorrect Answer 1": "Wrong 1",
                "Incorrect Answer 2": "Wrong 2",
                "Incorrect Answer 3": "Wrong 3",
            }
        )

    def gated_load_dataset(*args, **kwargs):
        raise RuntimeError("Dataset is gated. You must be authenticated to access it.")

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        types.SimpleNamespace(load_dataset=gated_load_dataset),
    )
    monkeypatch.setattr(gpqa, "DEFAULT_LOCAL_DATA_FILE", path, raising=False)

    rows = gpqa.load_gpqa_dataset(limit=1)

    assert rows == [
        {
            "Question": "Local question?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
        }
    ]


def test_gated_dataset_error_has_actionable_message():
    error = gpqa.build_dataset_access_error(RuntimeError("Dataset is gated. You must be authenticated to access it."))

    assert "GPQA is a gated Hugging Face dataset" in error
    assert "HF_TOKEN" in error
    assert "--data-file" in error
