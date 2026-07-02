from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from multi_agent_sync.evaluation.types import BenchmarkScore, BenchmarkSpec
from multi_agent_sync.prompts import render_prompt


TASK_PREFIX = "For each of the following (in-progress) chess games, please complete the notation for the last shown move by filling in the destination square:"
OUTPUT_REGEX = "[a-h][1-8]"
DEFAULT_OUTPUT_CSV = "chess_results.csv"
DEFAULT_LOCAL_DATA_FILE = Path("data/chess/synthetic_short_task.json")
DEFAULT_DOWNLOAD_URL = (
    "https://raw.githubusercontent.com/google/BIG-bench/main/"
    "bigbench/benchmark_tasks/chess_state_tracking/synthetic_short/task.json"
)
SQUARE_PATTERN = re.compile(r"\b(?P<square>[a-h][1-8])\b", flags=re.IGNORECASE)


def build_benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="chess",
        display_name="Chess State Tracking",
        default_output_filename=DEFAULT_OUTPUT_CSV,
        load_items=load_items,
        build_prompt=build_prompt,
        extract_answer=extract_answer,
        score_response=score_response,
    )


def load_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    return load_chess_dataset(limit=args.limit, data_file=args.data_file)


def build_prompt(row: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    del rng
    targets = normalized_targets(row)
    prompt = render_prompt(
        "evaluation/chess_question.j2",
        task_prefix=TASK_PREFIX,
        game_prefix=row["input"],
        output_regex=OUTPUT_REGEX,
    )
    return prompt, targets[0]


def extract_answer(text: str) -> str | None:
    if not text:
        return None

    strict_patterns = [
        rf"Final\s+Answer\s*:\s*(?P<square>{OUTPUT_REGEX})\b",
        rf"Final\s+answer\s*:\s*(?P<square>{OUTPUT_REGEX})\b",
        rf"final_answer\s*:\s*(?P<square>{OUTPUT_REGEX})\b",
        rf"Answer\s*:\s*(?P<square>{OUTPUT_REGEX})\b",
        rf"answer\s*:\s*(?P<square>{OUTPUT_REGEX})\b",
        rf"\bdestination\s+square\s+is\s*(?P<square>{OUTPUT_REGEX})\b",
    ]
    for pattern in strict_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group("square").lower()

    matches = [match.group("square").lower() for match in SQUARE_PATTERN.finditer(text)]
    if len(matches) == 1:
        return matches[0]
    return None


def score_response(row: dict[str, Any], raw_output: str, args: argparse.Namespace) -> BenchmarkScore:
    del args
    targets = normalized_targets(row)
    pred = extract_answer(raw_output)
    return BenchmarkScore(
        pred=pred,
        correct=pred in targets if pred is not None else False,
        metadata={
            "output_regex": OUTPUT_REGEX,
            "valid_targets": targets,
        },
    )


def load_chess_dataset(limit: int | None, data_file: str | None = None) -> list[dict[str, Any]]:
    path = Path(data_file) if data_file else DEFAULT_LOCAL_DATA_FILE
    if path.exists():
        print(f"Using local benchmark data file: {path}")
        return load_local_rows(path, limit=limit)

    if data_file:
        raise RuntimeError(f"Local chess benchmark data file does not exist: {path}")

    print(f"Local benchmark data file not found, downloading BIG-bench chess task: {path}")
    rows = download_chess_rows()
    save_bigbench_task(path, rows)
    print(f"Saved benchmark data file: {path}")
    return rows[:limit] if limit is not None and limit > 0 else rows


def download_chess_rows() -> list[dict[str, Any]]:
    with urlopen(DEFAULT_DOWNLOAD_URL, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return rows_from_payload(payload)


def save_bigbench_task(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "synthetic_short",
        "description": "BIG-bench chess_state_tracking synthetic_short task",
        "preferred_score": "exact_str_match",
        "metrics": ["exact_str_match"],
        "task_prefix": TASK_PREFIX + "\n",
        "output_regex": OUTPUT_REGEX,
        "examples": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_local_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = rows_from_payload(payload)
    elif suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        validate_rows(rows)
    else:
        raise RuntimeError("Local chess data file must be .json, .jsonl, or .ndjson.")

    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        rows = [dict(row) for row in payload["examples"]]
    elif isinstance(payload, list):
        rows = [dict(row) for row in payload]
    else:
        raise RuntimeError("Chess benchmark JSON must be a BIG-bench task object with examples or a list of rows.")
    validate_rows(rows)
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(rows):
        if not isinstance(row.get("input"), str) or not row["input"].strip():
            raise RuntimeError(f"Chess row {index} is missing required field: input")
        targets = row.get("target")
        if not isinstance(targets, list) or not targets:
            raise RuntimeError(f"Chess row {index} is missing required field: target")
        invalid_targets = [target for target in targets if not isinstance(target, str) or not re.fullmatch(OUTPUT_REGEX, target)]
        if invalid_targets:
            raise RuntimeError(f"Chess row {index} has invalid target square(s): {invalid_targets}")


def normalized_targets(row: dict[str, Any]) -> list[str]:
    targets = row.get("target")
    if not isinstance(targets, list) or not targets:
        raise RuntimeError("Chess row is missing target squares.")
    return [str(target).lower() for target in targets]
