from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multi_agent_sync.evaluation.dataset_files import benchmark_data_path, save_rows
from multi_agent_sync.evaluation.types import BenchmarkScore, BenchmarkSpec, BenchmarkWorkflowConfig
from multi_agent_sync.prompts import render_prompt


DATASET_NAME = "SWE-bench/SWE-bench_Verified"
SPLIT_NAME = "test"
DEFAULT_OUTPUT_CSV = "swe_bench_verified_results.csv"
DEFAULT_LOCAL_DATA_FILE = benchmark_data_path("swe_bench_verified/test.jsonl")
SUCCESS_GOLD = "patch_required"
PATCH_PRODUCED = "patch_produced"
NO_PATCH = "no_patch"


@dataclass
class SweBenchFinalGuard:
    workspace: Any
    repo: str

    name: str = "swebench_final_guard"

    async def run(self, *, final_response: str | None = None) -> dict[str, Any]:
        del final_response
        command = build_git_diff_command(
            self.repo,
            empty_diff_error="Docker workspace git diff is empty. Modify repository files inside /workspace before FINAL.",
            output_diff=False,
        )
        result = await self.workspace.run_bash(command)
        payload = result.as_dict() if hasattr(result, "as_dict") else dict(result)
        payload.update(
            {
                "tool": self.name,
                "passed": result.exit_code == 0,
            }
        )
        if result.exit_code != 0:
            payload["error"] = "final_guard_failed"
            payload["message"] = result.stderr or result.stdout or "Docker workspace has no non-empty git diff."
        return payload


def build_benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="swe_bench_verified",
        display_name="SWE-bench Verified",
        default_output_filename=DEFAULT_OUTPUT_CSV,
        load_items=load_items,
        build_prompt=build_prompt,
        extract_answer=extract_answer,
        score_response=score_response,
        build_workflow_config=build_workflow_config,
    )


def load_items(args: argparse.Namespace) -> list[dict[str, Any]] | Any:
    return load_swe_bench_verified_dataset(limit=args.limit, data_file=args.data_file)


def load_swe_bench_verified_dataset(limit: int | None, data_file: str | None = None) -> list[dict[str, Any]] | Any:
    if data_file:
        print(f"Using local SWE-bench Verified data file: {data_file}")
        return load_local_rows(Path(data_file), limit=limit)
    if DEFAULT_LOCAL_DATA_FILE.exists():
        print(f"Using local SWE-bench Verified data file: {DEFAULT_LOCAL_DATA_FILE}")
        return load_local_rows(DEFAULT_LOCAL_DATA_FILE, limit=limit)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing optional dependency 'datasets'. Install it with: uv add datasets") from exc

    rows = [dict(row) for row in load_dataset(DATASET_NAME, split=SPLIT_NAME)]
    save_rows(DEFAULT_LOCAL_DATA_FILE, rows)
    return rows[:limit] if limit is not None and limit > 0 else rows


def load_local_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Local SWE-bench Verified data file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    elif suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        raise RuntimeError("Local SWE-bench Verified data file must be .csv, .jsonl, or .ndjson.")

    validate_rows(rows)
    return rows[:limit] if limit is not None and limit > 0 else rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    required_fields = {"repo", "instance_id", "base_commit", "problem_statement"}
    for index, row in enumerate(rows):
        missing = sorted(field for field in required_fields if not row.get(field))
        if missing:
            raise RuntimeError(f"SWE-bench Verified row {index} is missing required field(s): {', '.join(missing)}")


def build_prompt(row: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    del rng
    prompt = render_prompt(
        "evaluation/swe_bench_verified_question.j2",
        repo=row["repo"],
        instance_id=row["instance_id"],
        base_commit=row["base_commit"],
        problem_statement=row["problem_statement"],
        hints_text=row.get("hints_text") or "",
        fail_to_pass=format_test_list(row.get("FAIL_TO_PASS")),
        pass_to_pass=format_test_list(row.get("PASS_TO_PASS")),
        difficulty=row.get("difficulty") or "",
    )
    return prompt, SUCCESS_GOLD


def build_workflow_config(row: dict[str, Any], args: argparse.Namespace) -> BenchmarkWorkflowConfig:
    del args

    repo = str(row["repo"])

    async def export_git_diff(workspace: Any) -> str:
        result = await workspace.run_bash(build_git_diff_command(repo, output_diff=True))
        if result.exit_code != 0:
            raise RuntimeError(f"Could not export git diff from SWE-bench workspace: {result.stderr or result.stdout}")
        return result.stdout

    return BenchmarkWorkflowConfig(
        final_candidate_path="__git_diff__",
        final_guard_factory=lambda workspace: SweBenchFinalGuard(workspace=workspace, repo=repo),
        final_candidate_exporter=export_git_diff,
    )


def build_git_diff_command(
    repo: str,
    *,
    output_diff: bool,
    empty_diff_error: str = "SWE-bench workspace git diff was empty; the agent did not modify repository files.",
) -> str:
    diff_command = "git -C \"$repo_dir\" diff -- ." if output_diff else "git -C \"$repo_dir\" diff --stat -- ."
    return (
        "set -euo pipefail\n"
        "repo_dir=''\n"
        "if git rev-parse --show-toplevel >/dev/null 2>&1; then\n"
        "  repo_dir=$(git rev-parse --show-toplevel)\n"
        "else\n"
        "  repo_git_dir=$(find . -mindepth 2 -maxdepth 2 -type d -name .git -print -quit)\n"
        "  if [ -n \"$repo_git_dir\" ]; then\n"
        "    repo_dir=${repo_git_dir%/.git}\n"
        "  fi\n"
        "fi\n"
        "if [ -z \"$repo_dir\" ]; then\n"
        f"  echo 'No Git repository found in /workspace. Clone https://github.com/{repo}.git in the Docker workspace before finishing.' >&2\n"
        "  exit 1\n"
        "fi\n"
        "git -C \"$repo_dir\" add -N .\n"
        "if git -C \"$repo_dir\" diff --quiet -- .; then\n"
        f"  echo {json.dumps(empty_diff_error)} >&2\n"
        "  exit 2\n"
        "fi\n"
        f"{diff_command}"
    )


def format_test_list(value: Any) -> str:
    parsed = parse_jsonish_list(value)
    if not parsed:
        return "- None provided."
    return "\n".join(f"- {item}" for item in parsed)


def parse_jsonish_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def extract_answer(text: str) -> str | None:
    return PATCH_PRODUCED if extract_prediction_patch(text) else None


def score_response(row: dict[str, Any], raw_output: str, args: argparse.Namespace) -> BenchmarkScore:
    del args
    patch = extract_prediction_patch(raw_output)
    metadata: dict[str, Any] = {
        "instance_id": row.get("instance_id"),
        "repo": row.get("repo"),
        "prediction_only": True,
        "official_evaluation": "not_run",
    }
    if patch:
        metadata["model_patch"] = patch
        metadata["model_patch_bytes"] = len(patch.encode("utf-8"))
        return BenchmarkScore(
            pred=PATCH_PRODUCED,
            correct=False,
            error="Prediction patch produced; official SWE-bench harness evaluation was not run.",
            metadata=metadata,
        )
    return BenchmarkScore(
        pred=NO_PATCH,
        correct=False,
        error="No diff patch was found in the model output.",
        metadata=metadata,
    )


def extract_prediction_patch(text: str) -> str | None:
    if not text:
        return None

    fenced = re.findall(r"```(?:diff|patch)?\s*(?P<patch>diff --git .*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[-1].strip()

    marker = "diff --git "
    index = text.find(marker)
    if index == -1:
        return None
    return text[index:].strip()
