from __future__ import annotations

import argparse
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    display_name: str
    default_output_filename: str
    load_items: Callable[[argparse.Namespace], Iterable[dict[str, Any]]]
    build_prompt: Callable[[dict[str, Any], random.Random], tuple[str, str]]
    extract_answer: Callable[[str], str | None]
