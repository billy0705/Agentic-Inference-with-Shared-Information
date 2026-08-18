from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from multi_agent_sync.evaluation.benchmarks import ma_proofbench
from multi_agent_sync.evaluation.dataset_files import benchmark_data_path, save_rows
from multi_agent_sync.evaluation.lean_feedback import LeanVerifierTool
from multi_agent_sync.evaluation.types import BenchmarkScore, BenchmarkSpec, BenchmarkWorkflowConfig
from multi_agent_sync.prompts import render_prompt


DATASET_NAME = "RUC-AIBOX/OlymMATH"
SPLIT_NAME = "test"
DEFAULT_OUTPUT_CSV = "olymmath_results.csv"
DEFAULT_LEAN_OUTPUT_CSV = "olymmath_lean_results.csv"
DEFAULT_SUBSET = "en-easy"
NATURAL_SUBSETS = {"en-easy", "en-hard", "zh-easy", "zh-hard"}
HF_SUBSET_NAMES = {
    "en-easy": "en-easy",
    "en-hard": "en-hard",
    "zh-easy": "zh-easy",
    "zh-hard": "zh-hard",
    "lean": "lean",
}
RAW_FILENAMES = {
    "en-easy": "OlymMATH-EN-EASY.jsonl",
    "en-hard": "OlymMATH-EN-HARD.jsonl",
    "zh-easy": "OlymMATH-ZH-EASY.jsonl",
    "zh-hard": "OlymMATH-ZH-HARD.jsonl",
    "lean": "OlymMATH-LEAN.jsonl",
}
DEFAULT_LOCAL_DATA_DIR = benchmark_data_path("OlymMATH")
SUCCESS_GOLD = "lean_verifies"
SUCCESS_PRED = "verified"
FAILED_PRED = "failed"


def build_benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="olymmath",
        display_name="OlymMATH",
        default_output_filename=DEFAULT_OUTPUT_CSV,
        load_items=load_items,
        build_prompt=build_prompt,
        extract_answer=extract_answer,
        score_response=score_response,
    )


def build_lean_benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="olymmath_lean",
        display_name="OlymMATH-LEAN",
        default_output_filename=DEFAULT_LEAN_OUTPUT_CSV,
        load_items=load_lean_items,
        build_prompt=build_lean_prompt,
        extract_answer=extract_lean_answer,
        score_response=score_lean_response,
        build_workflow_config=build_lean_workflow_config,
    )


def load_items(args: argparse.Namespace) -> list[dict[str, Any]] | Any:
    return load_olymmath_dataset(
        subset=getattr(args, "olymmath_subset", DEFAULT_SUBSET),
        limit=args.limit,
        data_file=args.data_file,
    )


def load_lean_items(args: argparse.Namespace) -> list[dict[str, Any]] | Any:
    return load_olymmath_dataset(subset="lean", limit=args.limit, data_file=args.data_file)


def build_prompt(row: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    del rng
    problem = str(row["problem"])
    answer = normalize_answer_text(str(row["answer"]))
    prompt = render_prompt(
        "evaluation/olymmath_question.j2",
        problem=problem,
        subject=row.get("subject"),
    )
    return prompt, answer


def build_lean_prompt(row: dict[str, Any], rng: random.Random) -> tuple[str, str]:
    del rng
    formal_statement = str(row.get("formal_statement_raw") or row["formal_statement"])
    prompt_statement = build_lean_prompt_statement(formal_statement)
    prompt = render_prompt(
        "evaluation/olymmath_lean_question.j2",
        informal_statement=row.get("en_informal") or row.get("zh_informal"),
        formal_statement=prompt_statement,
    )
    return prompt, SUCCESS_GOLD


def build_lean_workflow_config(row: dict[str, Any], args: argparse.Namespace) -> BenchmarkWorkflowConfig:
    formal_statement = str(row.get("formal_statement_raw") or row["formal_statement"])
    header, _ = ma_proofbench.split_lean_header_and_body(formal_statement)
    initial_code = build_lean_prompt_statement(formal_statement)

    def verify_candidate(code: str) -> ma_proofbench.LeanVerificationResult:
        normalized_code = ma_proofbench.normalize_lean_candidate(code, header)
        return ma_proofbench.run_lean_verifier(normalized_code, args)

    return BenchmarkWorkflowConfig(
        seed_files={ma_proofbench.LEAN_CANDIDATE_PATH: initial_code},
        final_candidate_path=ma_proofbench.LEAN_CANDIDATE_PATH,
        feedback_tool_factory=lambda workspace: LeanVerifierTool(
            workspace=workspace,
            verify_code=verify_candidate,
            default_path=ma_proofbench.LEAN_CANDIDATE_PATH,
        ),
    )


def build_lean_prompt_statement(formal_statement: str) -> str:
    theorem_name = ma_proofbench.extract_theorem_name(formal_statement)
    if theorem_name is not None:
        signature = ma_proofbench.extract_theorem_signature(formal_statement, theorem_name)
        if signature is not None:
            header, _ = ma_proofbench.split_lean_header_and_body(formal_statement)
            parts = [part for part in (header, f"{signature.strip()} := by") if part.strip()]
            return "\n\n".join(parts)

    lines = [
        line
        for line in formal_statement.splitlines()
        if not ma_proofbench.contains_sorry(line)
    ]
    return "\n".join(lines).strip()


def extract_answer(text: str) -> str | None:
    if not text:
        return None

    boxed = extract_last_boxed(text)
    if boxed:
        normalized_boxed = normalize_answer_text(boxed)
        if not is_placeholder_answer(boxed, normalized_boxed):
            return normalized_boxed

    found_json_answer, json_answer = extract_json_like_final_answer(text)
    if found_json_answer:
        return normalize_answer_text(json_answer) if json_answer is not None else None

    strict_patterns = [
        r"Final\s+Answer\s*:\s*(?P<answer>.+)",
        r"Final\s+answer\s*:\s*(?P<answer>.+)",
        r"final_answer\s*:\s*(?P<answer>.+)",
        r"Answer\s*:\s*(?P<answer>.+)",
        r"answer\s*:\s*(?P<answer>.+)",
        r"\bThe\s+answer\s+is\s*(?P<answer>.+)",
    ]
    for pattern in strict_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw_answer = first_answer_line(match.group("answer"))
            normalized_answer = normalize_answer_text(raw_answer)
            if is_placeholder_answer(raw_answer, normalized_answer):
                continue
            return normalized_answer

    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        if len(line) <= 200:
            normalized_line = normalize_answer_text(line)
            if not is_placeholder_answer(line, normalized_line):
                return normalized_line
    return None


def extract_json_like_final_answer(text: str) -> tuple[bool, str | None]:
    final_answer_match = re.search(
        r"[\"']final_answer[\"']\s*:\s*[\"'](?P<answer>[^\"']*)[\"']",
        text,
        flags=re.IGNORECASE,
    )
    if not final_answer_match:
        return False, None

    status_match = re.search(
        r"[\"']status[\"']\s*:\s*[\"'](?P<status>[^\"']*)[\"']",
        text,
        flags=re.IGNORECASE,
    )
    if status_match and status_match.group("status").strip().lower() == "continue":
        return True, None

    answer = final_answer_match.group("answer").strip()
    normalized_answer = normalize_answer_text(answer)
    if not answer or answer.lower() in {"n/a", "na", "none", "null"} or is_placeholder_answer(answer, normalized_answer):
        return True, None
    return True, answer


def is_placeholder_answer(raw_answer: str, normalized_answer: str | None = None) -> bool:
    raw = str(raw_answer or "").strip().lower()
    normalized = str(normalized_answer if normalized_answer is not None else normalize_answer_text(raw_answer)).strip().lower()
    placeholders = {"<answer>", "answer", "<finalanswer>", "finalanswer"}
    return raw in placeholders or normalized in placeholders or "<answer>" in raw


def score_response(row: dict[str, Any], raw_output: str, args: argparse.Namespace) -> BenchmarkScore:
    del args
    pred = extract_answer(raw_output)
    gold = normalize_answer_text(str(row["answer"]))
    metadata = {
        "unique_id": row.get("unique_id"),
        "subject": row.get("subject"),
        "gold_normalized": gold,
    }
    if pred is not None:
        metadata["pred_normalized"] = pred
    return BenchmarkScore(pred=pred, correct=answers_equivalent(pred, gold), metadata=metadata)


def extract_lean_answer(text: str) -> str | None:
    return SUCCESS_PRED if ma_proofbench.extract_lean_code(text) else None


def score_lean_response(row: dict[str, Any], raw_output: str, args: argparse.Namespace) -> BenchmarkScore:
    code = ma_proofbench.extract_lean_code(raw_output)
    formal_statement = str(row.get("formal_statement_raw") or row["formal_statement"])
    metadata: dict[str, Any] = {
        "unique_id": row.get("unique_id"),
        "subject": row.get("subject"),
    }
    if row.get("en_informal"):
        metadata["en_informal"] = row.get("en_informal")
    if not code:
        return BenchmarkScore(pred=None, correct=False, error="No Lean code block found.", metadata=metadata)

    header, _ = ma_proofbench.split_lean_header_and_body(formal_statement)
    code = ma_proofbench.normalize_lean_candidate(code, header)
    metadata["lean_code"] = code
    if ma_proofbench.contains_sorry(code):
        metadata["verification_passed"] = False
        return BenchmarkScore(
            pred="contains_sorry",
            correct=False,
            error="Generated Lean code contains a sorry placeholder.",
            metadata=metadata,
        )

    if not ma_proofbench.theorem_statement_is_unchanged(formal_statement, code):
        metadata["verification_passed"] = False
        return BenchmarkScore(
            pred="altered_statement",
            correct=False,
            error="Generated Lean code changes the original theorem statement.",
            metadata=metadata,
        )

    verification = ma_proofbench.run_lean_verifier(code, args)
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


def load_olymmath_dataset(subset: str, limit: int | None, data_file: str | None = None) -> list[dict[str, Any]] | Any:
    validate_subset(subset)
    if data_file:
        print(f"Using local benchmark data file: {data_file}")
        return load_local_rows(Path(data_file), subset=subset, limit=limit)
    fallback = DEFAULT_LOCAL_DATA_DIR / RAW_FILENAMES[subset]
    if all_default_data_files_exist():
        print(f"Using local OlymMATH data files from: {DEFAULT_LOCAL_DATA_DIR}")
        return load_local_rows(fallback, subset=subset, limit=limit)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing optional dependency 'datasets'. Install it with: uv add datasets") from exc

    download_missing_default_data_files(load_dataset)
    return load_local_rows(fallback, subset=subset, limit=limit)


def all_default_data_files_exist() -> bool:
    return all((DEFAULT_LOCAL_DATA_DIR / filename).exists() for filename in RAW_FILENAMES.values())


def download_missing_default_data_files(load_dataset: Any) -> None:
    missing = [filename for filename in RAW_FILENAMES.values() if not (DEFAULT_LOCAL_DATA_DIR / filename).exists()]
    if missing:
        print(f"OlymMATH local data incomplete, downloading {len(missing)} missing file(s) into: {DEFAULT_LOCAL_DATA_DIR}")
    for subset, filename in RAW_FILENAMES.items():
        path = DEFAULT_LOCAL_DATA_DIR / filename
        if path.exists():
            continue
        print(f"Downloading OlymMATH data file from Hugging Face: {path}")
        rows = [dict(row) for row in load_dataset(DATASET_NAME, HF_SUBSET_NAMES[subset], split=SPLIT_NAME)]
        save_rows(path, rows)
        print(f"Saved benchmark data file: {path}")


def load_local_rows(path: Path, subset: str, limit: int | None = None) -> list[dict[str, Any]]:
    validate_subset(subset)
    if not path.exists():
        raise RuntimeError(f"Local OlymMATH data file does not exist: {path}")

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
        raise RuntimeError("Local OlymMATH data file must be .csv, .jsonl, or .ndjson.")

    validate_rows(rows, subset=subset)
    if limit is not None and limit > 0:
        rows = rows[:limit]
    return rows


def validate_subset(subset: str) -> None:
    if subset not in HF_SUBSET_NAMES:
        supported = ", ".join(sorted(HF_SUBSET_NAMES))
        raise RuntimeError(f"Unknown OlymMATH subset {subset!r}. Supported subsets: {supported}.")


def validate_rows(rows: list[dict[str, Any]], subset: str) -> None:
    required_fields = {"unique_id", "subject", "formal_statement"} if subset == "lean" else {"problem", "answer", "subject", "unique_id"}
    for index, row in enumerate(rows):
        missing = sorted(field for field in required_fields if field not in row or row[field] in (None, ""))
        if missing:
            raise RuntimeError(f"Local OlymMATH row {index} is missing required field(s): {', '.join(missing)}")


def extract_last_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    index = start + len(marker)
    depth = 1
    chars: list[str] = []
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        index += 1
    return None


def first_answer_line(value: str) -> str:
    line = value.strip().splitlines()[0].strip()
    return line.rstrip("。.,;")


def normalize_answer_text(value: str) -> str:
    normalized = strip_wrappers(value)
    normalized = normalized.replace("\u2212", "-")
    normalized = normalized.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\,", "").replace("\\;", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace("\\cdot", "*").replace("\\times", "*")
    normalized = normalize_latex_fractions(normalized)
    normalized = normalize_latex_roots(normalized)
    normalized = normalize_latex_powers(normalized)
    normalized = normalized.replace("\\pi", "pi")
    normalized = normalized.replace("\\arccos", "arccos")
    normalized = normalized.replace("\\arcsin", "arcsin")
    normalized = normalized.replace("\\arctan", "arctan")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"(?<=\d)(?=sqrt\()", "*", normalized)
    normalized = re.sub(r"(?<=\))(?=sqrt\()", "*", normalized)
    decimal = normalize_decimal(normalized)
    return decimal if decimal is not None else normalized


def strip_wrappers(value: str) -> str:
    stripped = value.strip().strip("$").strip()
    stripped = stripped.removeprefix("答案：").removeprefix("答案:")
    stripped = re.sub(r"^\\boxed\s*", r"\\boxed", stripped)
    boxed = extract_last_boxed(stripped)
    if boxed is not None and stripped.startswith("\\boxed"):
        return boxed
    return stripped.rstrip("。.,;")


def normalize_latex_fractions(value: str) -> str:
    pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    previous = None
    current = value
    while previous != current:
        previous = current
        current = pattern.sub(replace_latex_fraction, current)
    return current


def replace_latex_fraction(match: re.Match[str]) -> str:
    numerator = match.group(1)
    denominator = match.group(2)
    if re.search(r"[+\-*/^]", numerator):
        numerator = f"({numerator})"
    if re.search(r"[+\-*/^]", denominator):
        denominator = f"({denominator})"
    return f"{numerator}/{denominator}"


def normalize_latex_roots(value: str) -> str:
    return re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", value)


def normalize_latex_powers(value: str) -> str:
    return re.sub(r"\^\s*\{([^{}]+)\}", r"^(\1)", value)


def normalize_decimal(value: str) -> str | None:
    candidate = value.replace(",", "")
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", candidate):
        return None
    try:
        number = Decimal(candidate)
    except InvalidOperation:
        return None
    normalized = number.normalize()
    if normalized == normalized.to_integral_value():
        return format(normalized.quantize(Decimal(1)), "f")
    return format(normalized, "f")


def answers_equivalent(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    pred_norm = normalize_answer_text(pred)
    gold_norm = normalize_answer_text(gold)
    if pred_norm == gold_norm:
        return True
    if numeric_equivalent(pred_norm, gold_norm):
        return True
    return sympy_equivalent(pred_norm, gold_norm)


def numeric_equivalent(left: str, right: str) -> bool:
    left_value = safe_eval_numeric_expression(left)
    right_value = safe_eval_numeric_expression(right)
    if left_value is None or right_value is None:
        return False
    return math.isclose(left_value, right_value, rel_tol=1e-9, abs_tol=1e-9)


def safe_eval_numeric_expression(expression: str) -> float | None:
    if len(expression) > 200 or any(token in expression for token in ("[", "]", "{", "}", "\\", "_", "=")):
        return None
    candidate = expression.replace("^", "**")
    try:
        tree = ast.parse(candidate, mode="eval")
    except SyntaxError:
        return None
    if not numeric_ast_is_safe(tree):
        return None
    names = {
        "sqrt": math.sqrt,
        "pi": math.pi,
        "arccos": math.acos,
        "arcsin": math.asin,
        "arctan": math.atan,
    }
    try:
        value = eval(compile(tree, "<olymmath-answer>", "eval"), {"__builtins__": {}}, names)
    except Exception:
        return None
    if not isinstance(value, int | float):
        return None
    try:
        numeric_value = float(value)
    except OverflowError:
        return None
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def numeric_ast_is_safe(tree: ast.AST) -> bool:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Call,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )
    allowed_names = {"sqrt", "pi", "arccos", "arcsin", "arctan"}
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return False
        if isinstance(node, ast.Constant) and not isinstance(node.value, int | float):
            return False
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            return False
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_names:
                return False
            if len(node.args) != 1 or node.keywords:
                return False
    return True


def sympy_equivalent(left: str, right: str) -> bool:
    if any(token in left + right for token in ("[", "]", "+∞", "\\infty", "infty")):
        return False
    try:
        import sympy as sp
    except ModuleNotFoundError:
        return False

    try:
        left_expr = sp.sympify(left.replace("^", "**"))
        right_expr = sp.sympify(right.replace("^", "**"))
    except Exception:
        return False

    try:
        return bool(sp.simplify(left_expr - right_expr) == 0)
    except Exception:
        return False
