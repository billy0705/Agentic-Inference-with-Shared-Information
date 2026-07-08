from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DATASET_NAME = "SWE-bench/SWE-bench_Verified"


def build_predictions(results: list[dict[str, Any]], *, method: str, model_name_or_path: str) -> list[dict[str, str]]:
    predictions: list[dict[str, str]] = []
    for result in results:
        if result.get("benchmark") != "swe_bench_verified" or result.get("method") != method:
            continue
        metadata = result.get("score_metadata")
        if not isinstance(metadata, dict):
            continue
        instance_id = str(metadata.get("instance_id") or "").strip()
        model_patch = str(metadata.get("model_patch") or "")
        if not instance_id or not model_patch.strip():
            continue
        predictions.append(
            {
                "instance_id": instance_id,
                "model_name_or_path": model_name_or_path,
                "model_patch": model_patch,
            }
        )
    return predictions


def write_predictions_jsonl(path: Path, predictions: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + "\n")


def build_run_evaluation_command(predictions_path: Path, args: Any) -> list[str]:
    command = [
        current_python_executable(),
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        DATASET_NAME,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(getattr(args, "swebench_max_workers", 1)),
        "--run_id",
        str(getattr(args, "swebench_run_id", "multi-agent-sync")),
    ]
    namespace = getattr(args, "swebench_namespace", None)
    if namespace is not None:
        command.extend(["--namespace", str(namespace)])
    instance_ids = str(getattr(args, "swebench_instance_ids", "") or "").strip()
    if instance_ids:
        command.extend(["--instance_ids", instance_ids])
    return command


def run_official_evaluation(command: list[str], artifact_path: Path) -> dict[str, Any]:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    payload = {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    artifact_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "Official SWE-bench harness evaluation failed with exit code "
            f"{completed.returncode}. See artifact: {artifact_path}"
        )
    return payload


def current_python_executable() -> str:
    return sys.executable
