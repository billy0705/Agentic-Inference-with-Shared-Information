from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any


def load_or_download_rows(
    path: Path,
    *,
    load_local_rows: Callable[[Path, int | None], list[dict[str, Any]]],
    download_rows: Callable[[], Iterable[dict[str, Any]]],
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if path.exists():
        print(f"Using local benchmark data file: {path}")
        return load_local_rows(path, limit)

    print(f"Local benchmark data file not found, downloading from Hugging Face: {path}")
    rows = [dict(row) for row in download_rows()]
    save_rows(path, rows)
    print(f"Saved benchmark data file: {path}")
    return load_local_rows(path, limit)


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        save_csv_rows(path, rows)
        return
    if suffix in {".jsonl", ".ndjson"}:
        save_jsonl_rows(path, rows)
        return
    raise RuntimeError(f"Local benchmark data file must be .csv, .jsonl, or .ndjson: {path}")


def save_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ordered_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: csv_value(value) for key, value in row.items()} for row in rows)


def save_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def csv_value(value: Any) -> Any:
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for fieldname in row:
            if fieldname not in seen:
                seen.add(fieldname)
                fieldnames.append(fieldname)
    return fieldnames
