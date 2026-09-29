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

from muffin.evaluation.dataset_files import benchmark_data_path, save_rows
from muffin.evaluation.types import BenchmarkScore, BenchmarkSpec
from muffin.prompts import render_prompt


DATASET_NAME = "RUC-AIBOX/OlymMATH"
SPLIT_NAME = "test"
DEFAULT_OUTPUT_CSV = "olymmath_results.csv"
HF_SUBSET_NAMES = {
    "en-easy": "en-easy",
    "en-hard": "en-hard",
}
RAW_FILENAMES = {
    "en-easy": "OlymMATH-EN-EASY.jsonl",
    "en-hard": "OlymMATH-EN-HARD.jsonl",
}
DEFAULT_LOCAL_DATA_DIR = benchmark_data_path("OlymMATH")


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


def load_items(args: argparse.Namespace) -> list[dict[str, Any]] | Any:
    return load_olymmath_dataset(
        subset=args.olymmath_subset,
        limit=args.limit,
        data_file=args.data_file,
    )

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

def extract_answer(text: str) -> str | None:
    if not text:
        return None

    # Some providers include their private reasoning before a closing </think>
    # tag.  Only the visible response after that tag is the submitted answer.
    answer_text = text
    think_end = text.rfind("</think>")
    visible_answer_only = think_end >= 0 and bool(text[think_end + len("</think>") :].strip())
    if visible_answer_only:
        answer_text = text[think_end + len("</think>") :]

    # Preserve the legacy priority for outputs without a completed thinking
    # block.  These can be truncated mid-generation, so their last line is not
    # necessarily a submitted answer.
    if not visible_answer_only:
        boxed = extract_last_boxed(answer_text)
        if boxed:
            normalized_boxed = normalize_answer_text(boxed)
            if not is_placeholder_answer(boxed, normalized_boxed):
                return normalized_boxed

    found_json_answer, json_answer = extract_json_like_final_answer(answer_text)
    if found_json_answer:
        return normalize_answer_text(json_answer) if json_answer is not None else None

    strict_patterns = [
        r"Final\s+Answer\s*:\s*(?P<answer>.+)",
        r"final_answer\s*:\s*(?P<answer>.+)",
        r"Answer\s*:\s*(?P<answer>.+)",
    ]
    marked_answers: list[tuple[int, str]] = []
    for pattern in strict_patterns:
        for match in re.finditer(pattern, answer_text, flags=re.IGNORECASE):
            raw_answer = first_answer_line(match.group("answer"))
            normalized_answer = normalize_answer_text(raw_answer)
            if is_placeholder_answer(raw_answer, normalized_answer):
                continue
            if not visible_answer_only:
                return normalized_answer
            marked_answers.append((match.start(), normalized_answer))
    if marked_answers:
        return max(marked_answers, key=lambda item: item[0])[1]

    if visible_answer_only:
        boxed = extract_last_boxed(answer_text)
        if boxed:
            normalized_boxed = normalize_answer_text(boxed)
            if not is_placeholder_answer(boxed, normalized_boxed):
                return normalized_boxed

    lines = [line.strip() for line in answer_text.splitlines() if line.strip()]
    if not visible_answer_only and lines and len(lines[-1]) <= 200:
        normalized_terminal = normalize_answer_text(lines[-1])
        math_tokens_removed = re.sub(
            r"sqrt|frac|pi|arccos|arcsin|arctan|circ",
            "",
            normalized_terminal,
            flags=re.IGNORECASE,
        )
        if not re.search(r"[A-Za-z]", math_tokens_removed) and not is_placeholder_answer(
            lines[-1], normalized_terminal
        ):
            return normalized_terminal

    if not visible_answer_only:
        for match in re.finditer(r"\bThe\s+answer\s+is\s*(?P<answer>.+)", answer_text, flags=re.IGNORECASE):
            raw_answer = first_answer_line(match.group("answer"))
            normalized_answer = normalize_answer_text(raw_answer)
            if not is_placeholder_answer(raw_answer, normalized_answer):
                return normalized_answer

    for line in reversed(lines):
        if len(line) <= 200:
            prose_answer = re.fullmatch(r"The\s+answer\s+is\s*(?P<answer>.+)", line, flags=re.IGNORECASE)
            if prose_answer:
                line = first_answer_line(prose_answer.group("answer"))
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
    required_fields = {"problem", "answer", "subject", "unique_id"}
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
    if stripped.startswith(r"\(") and stripped.endswith(r"\)"):
        stripped = stripped[2:-2].strip()
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
    normalized = re.sub(r"\^\s*\{([^{}]+)\}", r"^(\1)", value)
    return re.sub(r"\^\((-?\d+)\)", r"^\1", normalized)


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
