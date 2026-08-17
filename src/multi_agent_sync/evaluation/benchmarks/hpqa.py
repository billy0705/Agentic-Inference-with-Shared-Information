from __future__ import annotations

import argparse
import csv
import json
import random
import re
import textwrap
from pathlib import Path
from typing import Any

from multi_agent_sync.evaluation.dataset_files import benchmark_data_path, load_or_download_rows
from multi_agent_sync.evaluation.types import BenchmarkScore, BenchmarkSpec
from multi_agent_sync.prompts import render_prompt


DATASET_NAME = "hotpotqa/hotpot_qa"
SUBSET_NAME = "distractor"
DEFAULT_OUTPUT_CSV = "hotpotqa_results.csv"
DEFAULT_LOCAL_DATA_FILE = benchmark_data_path("hotpotqa/hotpotqa_distractor.jsonl")
ANSWER_STRIP_RE = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^\w\s]")


def build_benchmark() -> BenchmarkSpec:
    """Return the HotpotQA benchmark specification."""

    return BenchmarkSpec(
        name="hotpotqa",
        display_name="HotpotQA",
        default_output_filename=DEFAULT_OUTPUT_CSV,
        load_items=load_items,
        build_prompt=build_prompt,
        extract_answer=extract_answer,
        score_response=score_response,
    )


def load_items(args: argparse.Namespace) -> list[dict[str, Any]] | Any:
    """Load HotpotQA rows from a local file or Hugging Face."""

    return load_hotpotqa_dataset(limit=args.limit, data_file=args.data_file)


def build_prompt(row: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    """Render the HotpotQA prompt and return the normalized gold answer."""

    del rng
    question = str(get_row_value(row, "question", "Question"))
    context = format_context(get_row_value(row, "context", "Context"))
    prompt = render_prompt(
        "evaluation/hotpotqa_question.j2",
        question=question,
        context=context,
    )
    return prompt, normalize_answer(get_row_value(row, "answer", "Answer"))


def extract_answer(text: str) -> str | None:
    """Extract a normalized HotpotQA answer from model output."""

    if not text:
        return None

    strict_patterns = [
        r"Final Answer\s*:\s*(?P<answer>.+)",
        r"Final answer\s*:\s*(?P<answer>.+)",
        r"final_answer\s*:\s*(?P<answer>.+)",
        r"Answer\s*:\s*(?P<answer>.+)",
        r"answer\s*:\s*(?P<answer>.+)",
        r"\bThe\s+answer\s+is\s*(?P<answer>.+)",
    ]
    for pattern in strict_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            normalized = normalize_answer(match.group("answer"))
            if normalized:
                return normalized

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        normalized = normalize_answer(lines[-1])
        if normalized:
            return normalized

    normalized = normalize_answer(text)
    return normalized or None


def score_response(row: dict[str, Any], raw_output: str, args: argparse.Namespace) -> BenchmarkScore:
    """Score a HotpotQA response with normalized exact match."""

    del args
    gold = normalize_answer(get_row_value(row, "answer", "Answer"))
    pred = extract_answer(raw_output)
    return BenchmarkScore(
        pred=pred,
        correct=pred == gold,
        metadata={
            "gold_normalized": gold,
            "pred_normalized": pred,
        },
    )


def load_hotpotqa_dataset(limit: int | None, data_file: str | None = None) -> list[dict[str, Any]] | Any:
    """Load HotpotQA data, preferring a local cached file when available."""

    if data_file:
        print(f"Using local benchmark data file: {data_file}")
        return load_local_hotpotqa_rows(Path(data_file), limit=limit)

    def download_rows() -> Any:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing optional dependency 'datasets'. Install it with: uv add datasets") from exc
        return load_dataset(DATASET_NAME, SUBSET_NAME, split="train")

    try:
        return load_or_download_rows(
            DEFAULT_LOCAL_DATA_FILE,
            load_local_rows=load_local_hotpotqa_rows,
            download_rows=download_rows,
            limit=limit,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "gated" in message or "authenticated" in message:
            raise RuntimeError(build_dataset_access_error(exc)) from exc
        raise


def load_local_hotpotqa_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load local HotpotQA rows from CSV, JSONL, or NDJSON."""

    if not path.exists():
        raise RuntimeError(f"Local HotpotQA data file does not exist: {path}")

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = [coerce_hotpotqa_row(row) for row in csv.DictReader(f)]
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(coerce_hotpotqa_row(json.loads(line)))
    else:
        raise RuntimeError("Local HotpotQA data file must be .csv, .jsonl, or .ndjson.")

    validate_hotpotqa_rows(rows)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def validate_hotpotqa_rows(rows: list[dict[str, Any]]) -> None:
    """Validate the minimum HotpotQA row shape required by the benchmark."""

    required_fields = {"question", "answer", "context"}
    for index, row in enumerate(rows):
        missing = sorted(field for field in required_fields if field not in row or row[field] in (None, ""))
        if missing:
            raise RuntimeError(f"Local HotpotQA row {index} is missing required field(s): {', '.join(missing)}")


def build_dataset_access_error(exc: Exception) -> str:
    """Build a helpful error message for gated or authenticated dataset access."""

    return (
        f"HotpotQA is a gated Hugging Face dataset and could not be loaded: {exc}\n\n"
        "Use one of these options:\n"
        "1. Authenticate with Hugging Face, make sure your account has access to the HotpotQA dataset, then rerun the benchmark.\n"
        "2. Set HF_TOKEN to a token from an account with dataset access.\n"
        "3. Run with --data-file /path/to/hotpotqa.jsonl, /path/to/hotpotqa.ndjson, or /path/to/hotpotqa.csv using rows with the fields "
        "question, answer, and context."
    )


def coerce_hotpotqa_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert JSON string fields from CSV back into structured HotpotQA values."""

    coerced = dict(row)
    for field_name in ("context", "supporting_facts"):
        if field_name in coerced:
            coerced[field_name] = parse_jsonish_value(coerced[field_name])
    return coerced


def format_context(context: Any) -> str:
    """Format the HotpotQA context paragraphs into readable prompt text."""

    parsed_context = parse_jsonish_value(context)
    titles, sentences = unpack_context(parsed_context)
    lines: list[str] = []
    for title, paragraph in zip(titles, sentences, strict=True):
        lines.append(f"Title: {title}")
        for sentence in paragraph:
            lines.append(f"  {sentence}")
        lines.append("")
    return "\n".join(lines).strip()


def unpack_context(context: Any) -> tuple[list[Any], list[list[Any]]]:
    """Extract the title and sentence lists from a HotpotQA context object."""

    if isinstance(context, dict):
        titles = context.get("title")
        sentences = context.get("sentences")
    elif isinstance(context, list) and len(context) == 2:
        titles, sentences = context
    else:
        raise RuntimeError("HotpotQA context must be a dict with title/sentences or a two-item list.")

    if not isinstance(titles, list) or not isinstance(sentences, list):
        raise RuntimeError("HotpotQA context is missing title or sentence lists.")

    return titles, sentences


def parse_jsonish_value(value: Any) -> Any:
    """Parse JSON-encoded strings while leaving native Python values untouched."""

    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def normalize_answer(value: Any) -> str:
    """Normalize a HotpotQA answer for exact-match comparison."""

    text = str(value).strip().lower()
    text = NON_WORD_RE.sub(" ", text)
    text = ANSWER_STRIP_RE.sub(" ", text)
    text = " ".join(text.split())
    return text


def get_row_value(row: dict[str, Any], *field_names: str) -> Any:
    """Return the first populated value from a row across candidate field names."""

    for field_name in field_names:
        value = row.get(field_name)
        if value not in (None, ""):
            return value
    joined_names = ", ".join(field_names)
    raise RuntimeError(f"HotpotQA row is missing required field(s): {joined_names}")


def manual_check_rows(
    limit: int = 3,
    data_file: str | None = None,
    indices: list[int] | None = None,
    seed: int = 0,
    prompt_preview_chars: int = 1200,
) -> int:
    """Print a compact sanity-check report for a few HotpotQA rows."""

    rows = load_hotpotqa_dataset(limit=None if indices else limit, data_file=data_file)
    if not rows:
        print("No HotpotQA rows were loaded.")
        return 1

    selected_rows: list[tuple[int, dict[str, Any]]]
    if indices:
        selected_rows = []
        for index in indices:
            if index < 0 or index >= len(rows):
                raise IndexError(f"Row index {index} is out of range for {len(rows)} loaded row(s).")
            selected_rows.append((index, rows[index]))
    else:
        selected_rows = list(enumerate(rows[:limit]))

    rng = random.Random(seed)
    for index, row in selected_rows:
        prompt, gold = build_prompt(row, rng)
        sample_output = f"Final Answer: {gold}"
        extracted = extract_answer(sample_output)
        score = score_response(row, sample_output, argparse.Namespace())

        print(f"\n=== Row {index} ===")
        print(f"Question: {get_row_value(row, 'question', 'Question')}")
        print(f"Gold answer: {gold}")
        print(f"Extracted self-check answer: {extracted}")
        print(f"Exact-match self-check: {score.correct}")
        print("Context preview:")
        print(textwrap.shorten(format_context(get_row_value(row, 'context', 'Context')).replace("\n", " | "), width=prompt_preview_chars, placeholder=" ..."))
        print("Prompt preview:")
        print(textwrap.shorten(prompt.replace("\n", " | "), width=prompt_preview_chars, placeholder=" ..."))

    return 0


def main(argv: list[str] | None = None) -> int:
    """Run a small standalone HotpotQA sanity check from the command line."""

    parser = argparse.ArgumentParser(description="Run quick manual checks for the HotpotQA benchmark module.")
    parser.add_argument("--limit", type=int, default=3, help="Number of rows to preview when no explicit indices are given.")
    parser.add_argument("--data-file", default=None, help="Optional local HotpotQA CSV, JSONL, or NDJSON file.")
    parser.add_argument(
        "--indices",
        default="",
        help="Optional comma-separated row indices to inspect instead of the first --limit rows.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used when building prompts for preview.")
    parser.add_argument(
        "--prompt-preview-chars",
        type=int,
        default=1200,
        help="Maximum characters to print for each prompt and context preview.",
    )
    args = parser.parse_args(argv)

    indices = [int(value.strip()) for value in args.indices.split(",") if value.strip()] if args.indices else None
    return manual_check_rows(
        limit=args.limit,
        data_file=args.data_file,
        indices=indices,
        seed=args.seed,
        prompt_preview_chars=args.prompt_preview_chars,
    )


if __name__ == "__main__":
    raise SystemExit(main())
