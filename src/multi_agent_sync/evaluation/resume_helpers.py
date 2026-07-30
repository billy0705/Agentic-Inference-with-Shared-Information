import argparse
import argparse
import json
from unittest import runner
from pathlib import Path
from typing import Any
from rich import json
from multi_agent_sync.evaluation import runner

from multi_agent_sync.evaluation.types import BenchmarkSpec


def resolve_resume_output_path(resume_root: Path | None, benchmark: BenchmarkSpec, args: argparse.Namespace, run_id: str) -> Path:
    if resume_root is None:
        return runner.resolve_output_path(benchmark, args, run_id=run_id)
    config_path = resume_root / "run_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        output_path = config.get("output_path")
        if output_path:
            return Path(output_path)
    return resume_root / Path(benchmark.default_output_filename).name


def load_resume_results(resume_root: Path | None, benchmark_name: str, methods: list[str]) -> list[dict[str, Any]]:
    if resume_root is None:
        return []
    results = []
    for trace_path in sorted((resume_root / "examples").glob("*.json")):
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        method = str(trace.get("method", ""))
        if trace.get("benchmark") != benchmark_name or method not in methods:
            continue
        token_usage = trace.get("token_usage") if isinstance(trace.get("token_usage"), dict) else {}
        results.append(
            {
                "benchmark": trace.get("benchmark", benchmark_name),
                "method": method,
                "index": trace.get("index"),
                "gold": trace.get("gold"),
                "pred": trace.get("pred"),
                "correct": trace.get("correct"),
                "returncode": trace.get("returncode"),
                "error": trace.get("error", ""),
                "elapsed_seconds": trace.get("elapsed_seconds", 0.0),
                "prompt_tokens": token_usage.get("prompt_tokens"),
                "completion_tokens": token_usage.get("completion_tokens"),
                "total_tokens": token_usage.get("total_tokens"),
                "raw_output": trace.get("raw_output", ""),
                "score_metadata": trace.get("score_metadata", {}),
                "json_trace_path": str(trace_path),
            }
        )
    return results