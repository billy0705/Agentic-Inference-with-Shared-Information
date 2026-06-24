from __future__ import annotations

import argparse
import csv
import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from multi_agent_sync.evaluation.types import BenchmarkSpec
from multi_agent_sync.prompts import render_prompt


DATASET_NAME = "openai/gsm8k"
SUBSET_NAME = "main"
SPLIT_NAME = "test"
DEFAULT_OUTPUT_CSV = "gsm8k_results.csv"
DEFAULT_LOCAL_DATA_FILE = Path("data/gsm8k_test.jsonl")
NUMBER_PATTERN = r"(?:[-+]?\s*[$€£]?\s*(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+))"


def build_benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="gsm8k",
        display_name="GSM8K",
        default_output_filename=DEFAULT_OUTPUT_CSV,
        load_items=load_items,
        build_prompt=build_prompt,
        extract_answer=extract_answer,
    )


def load_items(args: argparse.Namespace) -> list[dict[str, Any]] | Any:
    return load_gsm8k_dataset(limit=args.limit, data_file=args.data_file)


def build_prompt(row: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    del rng
    prompt = render_prompt(
        "evaluation/gsm8k_question.j2",
        question=row["question"],
    )
    return prompt, extract_gold_answer(row["answer"])


def extract_gold_answer(answer: str) -> str:
    match = re.search(r"####\s*(?P<answer>[^\n]+)", answer)
    if not match:
        raise RuntimeError("GSM8K answer is missing the final '####' answer marker.")
    normalized = normalize_number(match.group("answer"))
    if normalized is None:
        raise RuntimeError(f"Could not parse GSM8K gold answer: {match.group('answer')}")
    return normalized


def extract_answer(text: str) -> str | None:
    if not text:
        return None

    strict_patterns = [
        rf"####\s*(?P<number>{NUMBER_PATTERN})",
        rf"Final\s+Answer\s*:\s*(?P<number>{NUMBER_PATTERN})",
        rf"Final\s+answer\s*:\s*(?P<number>{NUMBER_PATTERN})",
        rf"final_answer\s*:\s*(?P<number>{NUMBER_PATTERN})",
        rf"Answer\s*:\s*(?P<number>{NUMBER_PATTERN})",
        rf"answer\s*:\s*(?P<number>{NUMBER_PATTERN})",
        rf"\bThe\s+answer\s+is\s*(?P<number>{NUMBER_PATTERN})",
    ]
    for pattern in strict_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            normalized = normalize_number(match.group("number"))
            if normalized is not None:
                return normalized

    for candidate in reversed(re.findall(NUMBER_PATTERN, text)):
        normalized = normalize_number(candidate)
        if normalized is not None:
            return normalized
    return None


def normalize_number(value: str) -> str | None:
    cleaned = value.strip()
    cleaned = cleaned.rstrip(".,;:)")
    cleaned = re.sub(r"[\s,$€£]", "", cleaned)
    if not cleaned:
        return None

    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return None

    normalized = number.normalize()
    if normalized == normalized.to_integral_value():
        return format(normalized.quantize(Decimal(1)), "f")
    return format(normalized, "f")


def load_gsm8k_dataset(limit: int | None, data_file: str | None = None) -> list[dict[str, Any]] | Any:
    if data_file:
        return load_local_gsm8k_rows(Path(data_file), limit=limit)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing optional dependency 'datasets'. Install it with: uv add datasets") from exc

    try:
        dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split=SPLIT_NAME)
    except Exception:
        if DEFAULT_LOCAL_DATA_FILE.exists():
            return load_local_gsm8k_rows(DEFAULT_LOCAL_DATA_FILE, limit=limit)
        raise

    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def load_local_gsm8k_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Local GSM8K data file does not exist: {path}")

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
        raise RuntimeError("Local GSM8K data file must be .csv, .jsonl, or .ndjson.")

    validate_gsm8k_rows(rows)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def validate_gsm8k_rows(rows: list[dict[str, Any]]) -> None:
    required_fields = {"question", "answer"}
    for index, row in enumerate(rows):
        missing = sorted(field for field in required_fields if field not in row or row[field] in (None, ""))
        if missing:
            raise RuntimeError(f"Local GSM8K row {index} is missing required field(s): {', '.join(missing)}")
