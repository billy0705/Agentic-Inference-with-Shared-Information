"""LTS baselines (Learning to Share, Fioresi et al., ICML 2026).

The authors' scenario scripts from https://github.com/UCF-CRCV/autogen-lts
(git submodule at ``third_party/autogen-lts``) are executed unchanged apart from
the small, explicit edits in ``prepare_scenario_source``. Their AutoGen fork is
installed into the project environment through the ``lts`` extra
(``uv sync --extra lts``). Each scenario runs as a subprocess, because it is a
``__main__`` script that configures global logging, mutates ``os.environ`` and
writes files into its working directory; this package never imports AutoGen.

Variants:
  lts         LLMVerifiedMemoryParallelAgents with the learned decision transformer
  lts_llm     LLMVerifiedMemoryParallelAgents with the LLM memory verifier
  lts_addall  MemoryParallelAgents, every step is added to shared memory
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import shutil
import signal
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from multi_agent_sync.token_usage import TokenUsage


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LTS_REPO_DIR = REPO_ROOT / "third_party" / "autogen-lts"
DEFAULT_LTS_WEIGHTS = REPO_ROOT / "third_party" / "lts-weights" / "checkpoint.pt"
ASSISTANTBENCH_SUBDIR = Path("python/packages/agbench/benchmarks/AssistantBench")

LTS_TEAM_COUNT = 3  # num_teams / num_answers hard-coded in the authors' scenario scripts.


@dataclass(frozen=True)
class LTSVariant:
    method: str
    template: str
    use_decision_transformer: bool | None  # None: template has no memory verifier.


LTS_VARIANTS: dict[str, LTSVariant] = {
    "lts": LTSVariant("lts", "LLMVerifiedMemoryParallelAgents", True),
    "lts_llm": LTSVariant("lts_llm", "LLMVerifiedMemoryParallelAgents", False),
    "lts_addall": LTSVariant("lts_addall", "MemoryParallelAgents", None),
}

# Every edit is an exact string replacement that must match exactly once, so a
# change in the upstream scripts fails loudly instead of silently altering them.
_FILE_NAME_PATCH = ('filename = "__FILE_NAME__".strip()', 'filename = "".strip()')  # agbench templating: no attached file.
_NO_WEB_SURFER_PATCH = ("[coder, executor, file_surfer, web_surfer],", "[coder, executor, file_surfer],")
_DT_FLAG = "\n    use_decision_transformer = False\n"

_AGGREGATED_ANSWER_RE = re.compile(r"FINAL AGGREGATED ANSWER:\s*(.*)")
_TEAM_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.*)")
_KEEP_COUNTS_RE = re.compile(r"SharedMemory: FINAL counts: kept=(\d+) rejected=(\d+)")


def lts_repo_dir() -> Path:
    return Path(os.getenv("LTS_REPO_DIR", DEFAULT_LTS_REPO_DIR))


def lts_python() -> Path:
    return Path(os.getenv("LTS_PYTHON") or sys.executable)


def lts_weights_path() -> Path:
    return Path(os.getenv("DECISION_TRANSFORMER_WEIGHTS", DEFAULT_LTS_WEIGHTS))


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"LTS scenario patch '{label}' expected 1 match, found {count}.")
    return source.replace(old, new)


def prepare_scenario_source(source: str, variant: LTSVariant) -> str:
    """Apply the only edits made to the authors' scenario script.

    1. Fill agbench's ``__FILE_NAME__`` placeholder with "" (our tasks have no attached file).
    2. Drop WebSurfer from each team: our benchmarks are closed-book.
    3. Select the memory verifier (decision transformer vs. LLM) for LLMVerifiedMemoryParallelAgents.
    """
    source = _replace_once(source, *_FILE_NAME_PATCH, label="file_name")
    source = _replace_once(source, *_NO_WEB_SURFER_PATCH, label="no_web_surfer")
    if variant.use_decision_transformer is not None:
        flag = f"\n    use_decision_transformer = {variant.use_decision_transformer}\n"
        source = _replace_once(source, _DT_FLAG, flag, label="decision_transformer")
    return source


def build_model_config(model: str, base_url: str, max_tokens: int | None) -> dict[str, Any]:
    client_config: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "api_key": os.getenv("OPENAI_API_KEY", "EMPTY"),
        "temperature": 0,
        "model_info": {
            "vision": False,
            # Our benchmarks need no tools; only FileSurfer would use them, and it
            # catches the resulting error itself.
            "function_calling": False,
            "json_output": True,
            "structured_output": True,
            "family": "unknown",
            "multiple_system_messages": True,
        },
    }
    if max_tokens:
        client_config["max_tokens"] = max_tokens
    provider = "autogen_ext.models.openai.OpenAIChatCompletionClient"
    client = {"provider": provider, "config": client_config}
    # The scenario still constructs MultimodalWebSurfer (it is only left out of the
    # team), and its constructor insists on function calling. It never calls the model.
    web_surfer_config = {**client_config, "model_info": {**client_config["model_info"], "function_calling": True}}
    return {
        "model_config": client,
        "orchestrator_client": client,
        "coder_client": client,
        "web_surfer_client": {"provider": provider, "config": web_surfer_config},
        "file_surfer_client": client,
    }


def prepare_task_dir(task_dir: Path, prompt: str, variant: LTSVariant, model_config: dict[str, Any], repo_dir: Path) -> None:
    bench_dir = repo_dir / ASSISTANTBENCH_SUBDIR
    template_path = bench_dir / "Templates" / variant.template / "scenario.py"
    if not template_path.is_file():
        raise FileNotFoundError(
            f"LTS template not found at {template_path}. Run: git submodule update --init third_party/autogen-lts"
        )
    task_dir.mkdir(parents=True, exist_ok=True)
    scenario = prepare_scenario_source(template_path.read_text(encoding="utf-8"), variant)
    (task_dir / "scenario.py").write_text(scenario, encoding="utf-8")
    if variant.use_decision_transformer:
        shutil.copy2(bench_dir / "decision_transformer.py", task_dir / "decision_transformer.py")
    (task_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    # JSON is valid YAML; the scenario reads it with yaml.safe_load.
    (task_dir / "config.yaml").write_text(json.dumps(model_config, indent=2), encoding="utf-8")


def parse_final_answer(stdout: str) -> str | None:
    matches = _AGGREGATED_ANSWER_RE.findall(stdout) or _TEAM_ANSWER_RE.findall(stdout)
    if not matches:
        return None
    # Kept verbatim: brackets can be part of the answer (e.g. the interval [-1/8, 1/8]).
    answer = matches[-1].strip()
    return answer or None


def parse_token_usage(log_path: Path) -> TokenUsage:
    usage = TokenUsage()
    if not log_path.is_file():
        return usage
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("type") != "LLMCallEvent":
            continue
        prompt_tokens = int(record.get("prompt_tokens") or 0)
        completion_tokens = int(record.get("completion_tokens") or 0)
        usage.add(TokenUsage(prompt_tokens, completion_tokens, prompt_tokens + completion_tokens))
    return usage


def parse_memory_stats(stdout: str, task_dir: Path) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    counts = _KEEP_COUNTS_RE.findall(stdout)
    if counts:
        kept, rejected = (int(value) for value in counts[-1])
        stats.update({"kept": kept, "rejected": rejected, "verified": kept + rejected})
    memory_file = task_dir / ".shared_memory.json"
    if memory_file.is_file():
        try:
            stats["memory_entries"] = len(json.loads(memory_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError):
            pass
    return stats


def read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


async def run_scenario(python: Path, task_dir: Path, env: dict[str, str], timeout: float | None) -> tuple[int, str, bool]:
    process = await asyncio.create_subprocess_exec(
        str(python),
        "scenario.py",
        cwd=str(task_dir),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout_bytes, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except (TimeoutError, asyncio.CancelledError) as exc:
        timed_out = isinstance(exc, TimeoutError)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout_bytes, _ = await process.communicate()
        if not timed_out:
            raise
    stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
    (task_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    returncode = 124 if timed_out else (process.returncode or 0)
    return returncode, stdout, timed_out


def build_subprocess_env(python: Path, variant: LTSVariant) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    env["PYTHONUNBUFFERED"] = "1"
    # Code written by the Coder agent runs with `python` from PATH: use the same interpreter.
    env["PATH"] = f"{python.parent}{os.pathsep}{env.get('PATH', '')}"
    if variant.use_decision_transformer:
        weights = lts_weights_path()
        if not weights.is_file():
            raise FileNotFoundError(f"Decision transformer weights not found at {weights}. Run scripts/setup_lts.sh.")
        env["DECISION_TRANSFORMER_WEIGHTS"] = str(weights.resolve())
    return env


def resolve_task_dir(args: argparse.Namespace, method: str) -> Path:
    root = Path(getattr(args, "lts_work_dir", None) or Path(getattr(args, "output_dir", "output")) / "lts_workdirs")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return root / f"{stamp}-{method}-{uuid4().hex[:8]}"


async def run_lts(
    prompt: str,
    args: argparse.Namespace,
    *,
    method: str,
    model: str,
    base_url: str,
) -> tuple[str, int, dict[str, Any]]:
    variant = LTS_VARIANTS[method]
    python = lts_python()
    if not os.getenv("LTS_PYTHON") and importlib.util.find_spec("autogen_agentchat") is None:
        raise ModuleNotFoundError("LTS baselines need the AutoGen fork: run `uv sync --extra lts` (see scripts/setup_lts.sh).")
    if not python.is_file():
        raise FileNotFoundError(f"LTS Python interpreter not found at {python}.")

    task_dir = resolve_task_dir(args, method)
    model_config = build_model_config(model, base_url, getattr(args, "max_tokens", None))
    prepare_task_dir(task_dir, prompt, variant, model_config, lts_repo_dir())
    env = build_subprocess_env(python, variant)

    returncode, stdout, timed_out = await run_scenario(python, task_dir, env, getattr(args, "total_runtime_timeout", None))

    final_answer = parse_final_answer(stdout)
    # Without a final answer the run failed: leave the output empty so it is scored
    # as invalid, rather than letting the extractor read an error message as a prediction.
    raw_output = f"Final Answer: {final_answer}" if final_answer is not None else ""
    usage = parse_token_usage(task_dir / "log.jsonl")
    lts_metrics = read_json(task_dir / "first_vs_aggregated_metrics.json")
    return raw_output, returncode, {
        "method": method,
        "prompt": prompt,
        "lts_template": variant.template,
        "use_decision_transformer": variant.use_decision_transformer,
        "teams": LTS_TEAM_COUNT,
        "task_dir": str(task_dir),
        "returncode": returncode,
        "timed_out": timed_out,
        "final_answer": final_answer,
        "lts_metrics": lts_metrics,
        "lts_runtime_seconds": (lts_metrics or {}).get("aggregated", {}).get("runtime_seconds"),
        "memory": parse_memory_stats(stdout, task_dir),
        "token_usage": usage.as_dict(),
        "stdout_tail": stdout[-4000:],
        "raw_output": raw_output,
    }
