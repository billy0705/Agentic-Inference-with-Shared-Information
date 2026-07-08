from __future__ import annotations

import argparse
import json

from multi_agent_sync.evaluation import swebench_harness
from multi_agent_sync.evaluation.swebench_harness import (
    build_predictions,
    build_run_evaluation_command,
    write_predictions_jsonl,
)


def test_build_predictions_uses_model_patch_metadata_only_for_requested_method():
    results = [
        {
            "benchmark": "swe_bench_verified",
            "method": "multiagent_streaming",
            "index": 0,
            "score_metadata": {
                "instance_id": "repo__repo-1",
                "model_patch": "diff --git a/a.py b/a.py\n",
            },
        },
        {
            "benchmark": "swe_bench_verified",
            "method": "plain_llm",
            "index": 0,
            "score_metadata": {
                "instance_id": "repo__repo-1",
                "model_patch": "diff --git a/other.py b/other.py\n",
            },
        },
        {
            "benchmark": "gpqa",
            "method": "multiagent_streaming",
            "index": 0,
            "score_metadata": {},
        },
    ]

    predictions = build_predictions(results, method="multiagent_streaming", model_name_or_path="fake-model")

    assert predictions == [
        {
            "instance_id": "repo__repo-1",
            "model_name_or_path": "fake-model",
            "model_patch": "diff --git a/a.py b/a.py\n",
        }
    ]


def test_write_predictions_jsonl_writes_one_json_object_per_line(tmp_path):
    predictions = [
        {
            "instance_id": "repo__repo-1",
            "model_name_or_path": "fake-model",
            "model_patch": "diff --git a/a.py b/a.py\n",
        }
    ]
    path = tmp_path / "predictions.jsonl"

    write_predictions_jsonl(path, predictions)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == predictions[0]


def test_build_run_evaluation_command_targets_verified_dataset(tmp_path):
    args = argparse.Namespace(
        swebench_max_workers=2,
        swebench_run_id="run-123",
        swebench_namespace="",
        swebench_instance_ids="repo__repo-1,repo__repo-2",
    )
    predictions_path = tmp_path / "predictions.jsonl"

    command = build_run_evaluation_command(predictions_path, args)

    assert command[:3] == [swebench_harness.current_python_executable(), "-m", "swebench.harness.run_evaluation"]
    assert "--dataset_name" in command
    assert "SWE-bench/SWE-bench_Verified" in command
    assert "--predictions_path" in command
    assert str(predictions_path) in command
    assert "--max_workers" in command
    assert "2" in command
    assert "--run_id" in command
    assert "run-123" in command
    assert "--namespace" in command
    assert "" in command
    assert "--instance_ids" in command
    assert "repo__repo-1,repo__repo-2" in command
