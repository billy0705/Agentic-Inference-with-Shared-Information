import csv

import pytest

from multi_agent_sync.evaluation import main as evaluation
from multi_agent_sync.evaluation import gpqa
from multi_agent_sync.evaluation import runner


def test_parse_methods_accepts_comma_separated_methods():
    assert runner.parse_methods("multiagent_streaming,multiagent_no_streaming,plain_llm") == [
        "multiagent_streaming",
        "multiagent_no_streaming",
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
            {"method": "multiagent", "correct": True, "pred": "A"},
            {"method": "multiagent", "correct": False, "pred": None},
            {"method": "plain_llm", "correct": True, "pred": "B"},
        ]
    )

    assert summary["multiagent"]["total"] == 2
    assert summary["multiagent"]["accuracy"] == 0.5
    assert summary["multiagent"]["invalid_rate"] == 0.5
    assert summary["plain_llm"]["accuracy"] == 1.0


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


def test_gated_dataset_error_has_actionable_message():
    error = gpqa.build_dataset_access_error(RuntimeError("Dataset is gated. You must be authenticated to access it."))

    assert "GPQA is a gated Hugging Face dataset" in error
    assert "HF_TOKEN" in error
    assert "--data-file" in error
