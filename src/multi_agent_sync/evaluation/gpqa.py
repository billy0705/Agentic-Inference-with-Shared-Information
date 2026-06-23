from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from multi_agent_sync.evaluation.types import BenchmarkSpec
from multi_agent_sync.prompts import render_prompt


DATASET_NAME = "Idavidrein/gpqa"
SUBSET_NAME = "gpqa_diamond"
DEFAULT_OUTPUT_CSV = "gpqa_diamond_results.csv"


def build_benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="gpqa",
        display_name="GPQA-Diamond",
        default_output_filename=DEFAULT_OUTPUT_CSV,
        load_items=load_items,
        build_prompt=build_prompt,
        extract_answer=extract_answer,
    )


def load_items(args: argparse.Namespace) -> list[dict[str, Any]] | Any:
    return load_gpqa_dataset(limit=args.limit, data_file=args.data_file)


def build_prompt(row: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    question = row["Question"]
    options = [
        ("correct", row["Correct Answer"]),
        ("incorrect", row["Incorrect Answer 1"]),
        ("incorrect", row["Incorrect Answer 2"]),
        ("incorrect", row["Incorrect Answer 3"]),
    ]
    rng.shuffle(options)

    labels = ["A", "B", "C", "D"]
    correct_label = ""
    option_lines = []
    for label, (kind, answer_text) in zip(labels, options, strict=True):
        option_lines.append(f"{label}. {answer_text}")
        if kind == "correct":
            correct_label = label

    prompt = render_prompt(
        "evaluation/gpqa_question.j2",
        question=question,
        option_lines=option_lines,
    )
    return prompt, correct_label


def extract_answer(text: str) -> str | None:
    if not text:
        return None

    strict_patterns = [
        r"Final Answer\s*:\s*([ABCD])",
        r"Final answer\s*:\s*([ABCD])",
        r"final_answer\s*:\s*([ABCD])",
        r"Answer\s*:\s*([ABCD])",
        r"answer\s*:\s*([ABCD])",
        r"\bFinal\s*Answer\s*is\s*([ABCD])\b",
        r"\bThe\s*answer\s*is\s*([ABCD])\b",
    ]
    for pattern in strict_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()

    candidates = re.findall(r"\b([ABCD])\b", text.upper())
    return candidates[-1] if candidates else None


def load_gpqa_dataset(limit: int | None, data_file: str | None = None) -> list[dict[str, Any]] | Any:
    if data_file:
        return load_local_gpqa_rows(Path(data_file), limit=limit)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing optional dependency 'datasets'. Install it with: uv add datasets") from exc

    try:
        dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split="train")
    except Exception as exc:
        message = str(exc).lower()
        if "gated" in message or "authenticated" in message:
            raise RuntimeError(build_dataset_access_error(exc)) from exc
        raise

    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def load_local_gpqa_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Local GPQA data file does not exist: {path}")

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        raise RuntimeError("Local GPQA data file must be .csv, .jsonl, or .ndjson.")

    validate_gpqa_rows(rows)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def validate_gpqa_rows(rows: list[dict[str, Any]]) -> None:
    required_fields = {
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    }
    for index, row in enumerate(rows):
        missing = sorted(field for field in required_fields if field not in row or row[field] in (None, ""))
        if missing:
            raise RuntimeError(f"Local GPQA row {index} is missing required field(s): {', '.join(missing)}")


def build_dataset_access_error(exc: Exception) -> str:
    return (
        f"GPQA is a gated Hugging Face dataset and could not be loaded: {exc}\n\n"
        "Use one of these options:\n"
        "1. Authenticate with Hugging Face, make sure your account has access to the GPQA dataset, then rerun the benchmark.\n"
        "2. Set HF_TOKEN to a token from an account with dataset access.\n"
        "3. Run with --data-file /path/to/gpqa.csv or --data-file /path/to/gpqa.jsonl using rows with the fields "
        "Question, Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3."
    )
