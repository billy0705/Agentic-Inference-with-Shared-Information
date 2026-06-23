#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print compact benchmark summaries from a results CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the results CSV file.")
    parser.add_argument(
        "--method",
        nargs="+",
        help="One or more method names to summarize. Defaults to all methods.",
    )
    return parser.parse_args()


def is_correct(value: str) -> bool:
    return value.strip().lower() == "true"


def print_summary(method: str, rows: list[dict[str, str]]) -> None:
    total = len(rows)
    correct = sum(is_correct(row.get("correct", "")) for row in rows)
    invalid_answers = sum(not row.get("pred", "").strip() for row in rows)
    accuracy = correct / total if total else 0.0
    invalid_rate = invalid_answers / total if total else 0.0

    print(f"Method: {method}")
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Invalid answers: {invalid_answers}")
    print(f"Invalid rate: {invalid_rate:.4f}")


def main() -> int:
    args = parse_args()

    with args.csv_path.open(newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    rows_by_method: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        method = row.get("method", "").strip()
        if method:
            rows_by_method.setdefault(method, []).append(row)

    methods = args.method or list(rows_by_method)
    for index, method in enumerate(methods):
        if index:
            print()
        print_summary(method, rows_by_method.get(method, []))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
