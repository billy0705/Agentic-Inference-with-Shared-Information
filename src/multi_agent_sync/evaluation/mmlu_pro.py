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


DATASET_NAME = "TIGER-Lab/MMLU-Pro"
SPLIT_NAME = "test"
DEFAULT_OUTPUT_CSV = "mmlu_pro_results.csv"
DEFAULT_LOCAL_DATA_FILE = Path("data/mmlu_pro_test.jsonl")
LABELS = list("ABCDEFGHIJ")


def build_benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="mmlu_pro",
        display_name="MMLU-Pro",
        default_output_filename=DEFAULT_OUTPUT_CSV,
        load_items=load_items,
        build_prompt=build_prompt,
        extract_answer=extract_answer,
    )


def load_items(args: argparse.Namespace) -> list[dict[str, Any]] | Any:
    return load_mmlu_pro_dataset(limit=args.limit, data_file=args.data_file)


def build_prompt(row: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    del rng
    options = normalize_options(row["options"])
    labels = labels_for_options(options)
    answer = normalize_answer(row["answer"])
    validate_answer_for_options(answer, options)
    validate_answer_index(row, answer)
    option_lines = [f"{label}. {option}" for label, option in zip(labels, options, strict=True)]
    prompt = render_prompt(
        "evaluation/mmlu_pro_question.j2",
        question=row["question"],
        category=row.get("category"),
        option_lines=option_lines,
        label_choices=", ".join(labels),
        final_answer_format=f"<{'/'.join(labels)}>",
    )
    return prompt, answer


def extract_answer(text: str) -> str | None:
    if not text:
        return None

    label_pattern = rf"(?P<label>[{''.join(LABELS)}])"
    strict_patterns = [
        rf"Final\s+Answer\s*:\s*\(?{label_pattern}\)?",
        rf"Final\s+answer\s*:\s*\(?{label_pattern}\)?",
        rf"final_answer\s*:\s*\(?{label_pattern}\)?",
        rf"Answer\s*:\s*\(?{label_pattern}\)?",
        rf"answer\s*:\s*\(?{label_pattern}\)?",
        rf"\bThe\s+answer\s+is\s+(?:option\s+)?\(?{label_pattern}\)?",
        rf"\banswer\s+is\s+(?:option\s+)?\(?{label_pattern}\)?",
    ]
    for pattern in strict_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group("label").upper()

    stripped = text.strip()
    direct_match = re.fullmatch(rf"\(?\s*([{''.join(LABELS)}])\s*\)?\.?", stripped, flags=re.IGNORECASE)
    if direct_match:
        return direct_match.group(1).upper()
    return None


def load_mmlu_pro_dataset(limit: int | None, data_file: str | None = None) -> list[dict[str, Any]] | Any:
    if data_file:
        return load_local_mmlu_pro_rows(Path(data_file), limit=limit)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing optional dependency 'datasets'. Install it with: uv add datasets") from exc

    try:
        dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME)
    except Exception:
        if DEFAULT_LOCAL_DATA_FILE.exists():
            return load_local_mmlu_pro_rows(DEFAULT_LOCAL_DATA_FILE, limit=limit)
        raise

    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def load_local_mmlu_pro_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Local MMLU-Pro data file does not exist: {path}")

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = [parse_csv_row(row) for row in csv.DictReader(f)]
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    else:
        raise RuntimeError("Local MMLU-Pro data file must be .csv, .jsonl, or .ndjson.")

    validate_mmlu_pro_rows(rows)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def parse_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed = dict(row)
    if isinstance(parsed.get("options"), str):
        try:
            parsed["options"] = json.loads(parsed["options"])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Local MMLU-Pro CSV options field must be a JSON list.") from exc
    if isinstance(parsed.get("answer_index"), str) and parsed["answer_index"].strip():
        parsed["answer_index"] = int(parsed["answer_index"])
    return parsed


def validate_mmlu_pro_rows(rows: list[dict[str, Any]]) -> None:
    required_fields = {"question", "options", "answer"}
    for index, row in enumerate(rows):
        missing = sorted(field for field in required_fields if field not in row or row[field] in (None, ""))
        if missing:
            raise RuntimeError(f"Local MMLU-Pro row {index} is missing required field(s): {', '.join(missing)}")

        options = normalize_options(row["options"])
        validate_option_count(options, row_index=index)
        answer = normalize_answer(row["answer"])
        validate_answer_for_options(answer, options, row_index=index)
        validate_answer_index(row, answer, row_index=index)


def normalize_options(options: Any) -> list[str]:
    if not isinstance(options, list):
        raise RuntimeError("MMLU-Pro options must be a list.")
    return [str(option) for option in options]


def labels_for_options(options: list[str]) -> list[str]:
    validate_option_count(options)
    return LABELS[: len(options)]


def validate_option_count(options: list[str], row_index: int | None = None) -> None:
    if not options:
        prefix = f"Local MMLU-Pro row {row_index} " if row_index is not None else "MMLU-Pro row "
        raise RuntimeError(f"{prefix}must include at least one option.")
    if len(options) > len(LABELS):
        prefix = f"Local MMLU-Pro row {row_index} " if row_index is not None else "MMLU-Pro row "
        raise RuntimeError(f"{prefix}must include no more than 10 options.")


def normalize_answer(answer: Any) -> str:
    normalized = str(answer).strip().upper()
    if normalized not in LABELS:
        raise RuntimeError(f"MMLU-Pro answer must be one of {', '.join(LABELS)}.")
    return normalized


def validate_answer_for_options(answer: str, options: list[str], row_index: int | None = None) -> None:
    labels = labels_for_options(options)
    if answer not in labels:
        prefix = f"Local MMLU-Pro row {row_index} " if row_index is not None else "MMLU-Pro row "
        raise RuntimeError(f"{prefix}answer {answer} is outside the available options {', '.join(labels)}.")


def validate_answer_index(row: dict[str, Any], answer: str, row_index: int | None = None) -> None:
    if row.get("answer_index") in (None, ""):
        return
    try:
        answer_index = int(row["answer_index"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("MMLU-Pro answer_index must be an integer.") from exc
    if answer_index < 0 or answer_index >= len(LABELS):
        raise RuntimeError("MMLU-Pro answer_index must be between 0 and 9.")
    expected = LABELS[answer_index]
    if expected != answer:
        prefix = f"Local MMLU-Pro row {row_index} " if row_index is not None else "MMLU-Pro row "
        raise RuntimeError(f"{prefix}answer_index {answer_index} does not match answer {answer}.")
