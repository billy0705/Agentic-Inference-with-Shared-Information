from __future__ import annotations

import argparse
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any


def identity_answer_formatter(answer: str) -> str:
    return answer


@dataclass(frozen=True)
class BenchmarkScore:
    pred: str | None
    correct: bool
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkWorkflowConfig:
    seed_files: dict[str, str] = field(default_factory=dict)
    final_candidate_path: str | None = None
    feedback_tool_factory: Callable[[Any], Any] | None = None
    final_guard_factory: Callable[[Any], Any] | None = None
    final_candidate_exporter: Callable[[Any], Any] | None = None


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    display_name: str
    default_output_filename: str
    load_items: Callable[[argparse.Namespace], Iterable[dict[str, Any]]]
    build_prompt: Callable[[dict[str, Any], random.Random], tuple[str, str]]
    extract_answer: Callable[[str], str | None]
    format_answer: Callable[[str], str] = identity_answer_formatter
    score_response: Callable[[dict[str, Any], str, argparse.Namespace], BenchmarkScore] | None = None
    build_workflow_config: Callable[[dict[str, Any], argparse.Namespace], BenchmarkWorkflowConfig] | None = None
