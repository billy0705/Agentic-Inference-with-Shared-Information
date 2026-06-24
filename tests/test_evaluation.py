import csv
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
        load_items=lambda args: [{"Question": "Question?"}],
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


def test_default_output_path_is_in_output_folder():
    benchmark = gpqa.build_benchmark()
    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa"])

    assert runner.resolve_output_path(benchmark, args) == runner.DEFAULT_OUTPUT_DIR / "gpqa_diamond_results.csv"


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
