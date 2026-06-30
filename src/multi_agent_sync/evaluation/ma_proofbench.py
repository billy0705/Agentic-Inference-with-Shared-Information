from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from multi_agent_sync.evaluation.types import BenchmarkScore, BenchmarkSpec
from multi_agent_sync.prompts import render_prompt


DATASET_NAME = "openbmb/MA-ProofBench"
SPLIT_NAME = "test"
DEFAULT_OUTPUT_CSV = "ma_proofbench_results.csv"
DEFAULT_LOCAL_DATA_FILE = Path("data/ma_proofbench_test.jsonl")
SUCCESS_GOLD = "lean_verifies"
SUCCESS_PRED = "verified"
FAILED_PRED = "failed"
LABELS = {"level1", "level2"}


@dataclass(frozen=True)
class LeanVerificationResult:
    passed: bool
    verifier_output: str
    returncode: int = 0
    backend: str = "kimina-server"


def build_benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="ma_proofbench",
        display_name="MA-ProofBench",
        default_output_filename=DEFAULT_OUTPUT_CSV,
        load_items=load_items,
        build_prompt=build_prompt,
        extract_answer=extract_answer,
        score_response=score_response,
    )


def load_items(args: argparse.Namespace) -> list[dict[str, Any]] | Any:
    return load_ma_proofbench_dataset(
        limit=args.limit,
        data_file=args.data_file,
        level=getattr(args, "ma_proofbench_level", "all"),
    )


def build_prompt(row: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    del rng
    prompt = render_prompt(
        "evaluation/ma_proofbench_question.j2",
        informal_statement=row["informal_statement"],
        formal_statement=row["formal_statement"],
        topic=row.get("topic"),
        tag=row.get("tag"),
        level=row.get("split"),
        mathlib_version=row.get("version"),
    )
    return prompt, SUCCESS_GOLD


def extract_answer(text: str) -> str | None:
    return SUCCESS_PRED if extract_lean_code(text) else None


def score_response(row: dict[str, Any], raw_output: str, args: argparse.Namespace) -> BenchmarkScore:
    code = extract_lean_code(raw_output)
    metadata: dict[str, Any] = {
        "problem_id": row.get("id"),
        "level": row.get("split"),
        "topic": row.get("topic"),
        "tag": row.get("tag"),
        "mathlib_version": row.get("version"),
    }
    if not code:
        return BenchmarkScore(pred=None, correct=False, error="No Lean code block found.", metadata=metadata)

    code = normalize_lean_candidate(code, str(row.get("header") or ""))
    metadata["lean_code"] = code
    if contains_sorry(code):
        metadata["verification_passed"] = False
        return BenchmarkScore(
            pred="contains_sorry",
            correct=False,
            error="Generated Lean code contains a sorry placeholder.",
            metadata=metadata,
        )

    if not theorem_statement_is_unchanged(row["formal_statement"], code):
        metadata["verification_passed"] = False
        return BenchmarkScore(
            pred="altered_statement",
            correct=False,
            error="Generated Lean code changes the original theorem statement.",
            metadata=metadata,
        )

    verification = run_lean_verifier(code, args)
    metadata["verification_passed"] = verification.passed
    metadata["verifier_backend"] = verification.backend
    metadata["verifier_returncode"] = verification.returncode
    metadata["verifier_output"] = verification.verifier_output

    return BenchmarkScore(
        pred=SUCCESS_PRED if verification.passed else FAILED_PRED,
        correct=verification.passed,
        error="" if verification.passed else verification.verifier_output,
        metadata=metadata,
    )


def extract_lean_code(text: str) -> str | None:
    if not text:
        return None

    fences = re.findall(r"```(?:lean4|lean)\s*(?P<code>.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fences:
        return fences[-1].strip()

    stripped = text.strip()
    if "theorem " in stripped and ("import " in stripped or ":= by" in stripped):
        return stripped
    return None


def contains_sorry(code: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z0-9_'])sorry(?![A-Za-z0-9_'])", code))


def normalize_lean_candidate(code: str, header: str) -> str:
    if not header.strip():
        return code.strip()

    candidate_header, body = split_lean_header_and_body(code.strip())
    merged_header = merge_lean_headers(header, candidate_header)
    return f"{merged_header}\n\n{body.strip()}" if body.strip() else merged_header


def split_lean_header_and_body(code: str) -> tuple[str, str]:
    header_lines: list[str] = []
    body_lines = code.splitlines()
    for index, line in enumerate(code.splitlines()):
        stripped = line.strip()
        if stripped.startswith(("import ", "open ", "set_option ")):
            header_lines.append(line.rstrip())
            body_lines = code.splitlines()[index + 1 :]
        elif stripped == "" or stripped.startswith("--"):
            continue
        else:
            body_lines = code.splitlines()[index:]
            break
    return "\n".join(header_lines).strip(), "\n".join(body_lines).lstrip("\n")


def merge_lean_headers(dataset_header: str, candidate_header: str) -> str:
    imports: list[str] = []
    opens: list[str] = []
    set_options: list[str] = []
    other_lines: list[str] = []
    seen: set[str] = set()
    for header in (dataset_header, candidate_header):
        for line in header.splitlines():
            stripped = line.strip()
            if not stripped or stripped in seen:
                continue
            seen.add(stripped)
            if stripped.startswith("import "):
                imports.append(stripped)
            elif stripped.startswith("open "):
                opens.append(stripped)
            elif stripped.startswith("set_option "):
                set_options.append(stripped)
            else:
                other_lines.append(stripped)

    parts = ["\n".join(lines) for lines in (imports, opens, set_options, other_lines) if lines]
    return "\n\n".join(parts)


def theorem_statement_is_unchanged(formal_statement: str, candidate_code: str) -> bool:
    theorem_name = extract_theorem_name(formal_statement)
    if theorem_name is None:
        return False

    expected = extract_theorem_signature(formal_statement, theorem_name)
    candidate = extract_theorem_signature(candidate_code, theorem_name)
    return expected is not None and normalize_lean_whitespace(expected) == normalize_lean_whitespace(candidate or "")


def extract_theorem_name(code: str) -> str | None:
    match = re.search(r"\btheorem\s+([^\s:{]+)", code)
    return match.group(1) if match else None


def extract_theorem_signature(code: str, theorem_name: str) -> str | None:
    start_match = re.search(rf"\btheorem\s+{re.escape(theorem_name)}\b", code)
    if not start_match:
        return None
    proof_match = re.search(r":=\s*by\b", code[start_match.start() :], flags=re.DOTALL)
    if not proof_match:
        return None
    end = start_match.start() + proof_match.start()
    return code[start_match.start() : end]


def normalize_lean_whitespace(code: str) -> str:
    return re.sub(r"\s+", " ", code).strip()


def run_lean_verifier(code: str, args: argparse.Namespace) -> LeanVerificationResult:
    return run_kimina_server_verifier(code, args)


def run_kimina_server_verifier(code: str, args: argparse.Namespace) -> LeanVerificationResult:
    try:
        async_client_cls, snippet_cls = load_kimina_client_classes()
    except RuntimeError as exc:
        return LeanVerificationResult(
            passed=False,
            verifier_output=str(exc),
            returncode=127,
            backend="kimina-server",
        )

    async def query_server() -> LeanVerificationResult:
        host = getattr(args, "kimina_host", "127.0.0.1")
        port = int(getattr(args, "kimina_port", 8000))
        timeout = int(getattr(args, "lean_timeout", 180) or 180)
        max_workers = int(getattr(args, "kimina_max_workers", 1) or 1)
        client = async_client_cls(api_url=f"http://{host}:{port}")
        snippet = snippet_cls(id="0", code=code)
        started_at = time.perf_counter()
        check_response = await client.check(
            snips=[snippet],
            timeout=timeout,
            reuse=True,
            batch_size=1,
            max_workers=max_workers,
            show_progress=False,
        )
        result = sorted(check_response.results, key=lambda item: int(item.id))[0]
        elapsed_seconds = time.perf_counter() - started_at
        if result.error is not None:
            return LeanVerificationResult(
                passed=False,
                verifier_output=str(result.error),
                returncode=1,
                backend="kimina-server",
            )
        response = result.response
        if response is None or (isinstance(response, dict) and "message" in response):
            message = response.get("message", "No response") if isinstance(response, dict) else "No response"
            return LeanVerificationResult(
                passed=False,
                verifier_output=str(message),
                returncode=1,
                backend="kimina-server",
            )
        return collect_kimina_result(code=code, response=response, elapsed_seconds=elapsed_seconds)

    return run_coroutine_sync(query_server)


def load_kimina_client_classes() -> tuple[Any, Any]:
    kimina_client_path = os.environ.get("KIMINA_CLIENT_PATH")
    if kimina_client_path:
        path = Path(kimina_client_path).expanduser()
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from kimina_client import AsyncKiminaClient, Snippet
    except ImportError as exc:
        raise RuntimeError(
            "kimina_client is required for Kimina Lean Server verification. "
            "Install the Kimina Lean Server client package or set KIMINA_CLIENT_PATH to its client directory."
        ) from exc
    return AsyncKiminaClient, Snippet


def run_coroutine_sync(coro_factory: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro_factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: asyncio.run(coro_factory()))
        return future.result()


def collect_kimina_result(code: str, response: dict[str, Any], elapsed_seconds: float) -> LeanVerificationResult:
    errors = [message for message in response.get("messages", []) if message.get("severity") == "error"]
    warnings = [message for message in response.get("messages", []) if message.get("severity") == "warning"]
    infos = [message for message in response.get("messages", []) if message.get("severity") == "info"]
    sorries = response.get("sorries", [])
    compile_pass = not errors
    complete = compile_pass and not sorries and not any(
        "declaration uses `sorry`" in str(warning.get("data", ""))
        or "declaration uses 'sorry'" in str(warning.get("data", ""))
        or "failed" in str(warning.get("data", "")).lower()
        for warning in warnings
    )
    payload = {
        "pass": compile_pass,
        "complete": complete,
        "system_errors": None,
        "time": elapsed_seconds,
        "verified_code": code,
        "errors": errors,
        "warnings": warnings,
        "infos": infos,
        "sorries": sorries,
        "tactics": response.get("tactics", []),
    }
    return LeanVerificationResult(
        passed=complete,
        verifier_output=json.dumps(payload, ensure_ascii=False, sort_keys=True),
        returncode=0 if complete else 1,
        backend="kimina-server",
    )


def load_ma_proofbench_dataset(
    limit: int | None,
    data_file: str | None = None,
    level: str = "all",
) -> list[dict[str, Any]] | Any:
    validate_level(level)
    if data_file:
        return load_local_ma_proofbench_rows(Path(data_file), limit=limit, level=level)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing optional dependency 'datasets'. Install it with: uv add datasets") from exc

    try:
        dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME)
    except Exception:
        if DEFAULT_LOCAL_DATA_FILE.exists():
            return load_local_ma_proofbench_rows(DEFAULT_LOCAL_DATA_FILE, limit=limit, level=level)
        raise

    if level != "all":
        dataset = dataset.filter(lambda row: row["split"] == level)
    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def load_local_ma_proofbench_rows(path: Path, limit: int | None = None, level: str = "all") -> list[dict[str, Any]]:
    validate_level(level)
    if not path.exists():
        raise RuntimeError(f"Local MA-ProofBench data file does not exist: {path}")

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
        raise RuntimeError("Local MA-ProofBench data file must be .csv, .jsonl, or .ndjson.")

    validate_ma_proofbench_rows(rows)
    if level != "all":
        rows = [row for row in rows if row["split"] == level]
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def validate_level(level: str) -> None:
    if level != "all" and level not in LABELS:
        raise RuntimeError("MA-ProofBench level must be one of: all, level1, level2.")


def validate_ma_proofbench_rows(rows: list[dict[str, Any]]) -> None:
    required_fields = {
        "id",
        "split",
        "informal_statement",
        "formal_statement",
        "header",
        "topic",
        "tag",
        "version",
    }
    for index, row in enumerate(rows):
        missing = sorted(field for field in required_fields if field not in row or row[field] in (None, ""))
        if missing:
            raise RuntimeError(f"Local MA-ProofBench row {index} is missing required field(s): {', '.join(missing)}")
        if row["split"] not in LABELS:
            raise RuntimeError(f"Local MA-ProofBench row {index} split must be level1 or level2.")
