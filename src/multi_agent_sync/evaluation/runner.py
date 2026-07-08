from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from urllib.request import Request, urlopen
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from multi_agent_sync.llm import get_openai_base_url
from multi_agent_sync.evaluation.types import BenchmarkScore, BenchmarkSpec, BenchmarkWorkflowConfig
from multi_agent_sync.graph.workflow import run_workflow
from multi_agent_sync.prompts import render_prompt
from multi_agent_sync.workspace.docker import DockerWorkspace


DEFAULT_OUTPUT_DIR = Path("output")
VALID_METHODS = {
    "multiagent",
    "multiagent_streaming",
    "multiagent_no_streaming",
    "multiagent_dynamic_streaming",
    "multiagent_dynamic_no_streaming",
    "plain_llm",
    "single_agent",
    }


@dataclass
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens = add_optional_ints(self.prompt_tokens, other.prompt_tokens)
        self.completion_tokens = add_optional_ints(self.completion_tokens, other.completion_tokens)
        self.total_tokens = add_optional_ints(self.total_tokens, other.total_tokens)


@dataclass
class RunResult:
    raw_output: str
    returncode: int
    elapsed_seconds: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    trace: dict[str, Any] | None = None


class MeteredLLM:
    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self.usage = TokenUsage()

    async def ainvoke(self, prompt: str) -> Any:
        response = await self._llm.ainvoke(prompt)
        self.usage.add(extract_token_usage(response))
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def add_optional_ints(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_token_usage(response: Any) -> TokenUsage:
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}

    prompt_tokens = coerce_int(
        usage_metadata.get("input_tokens")
        or usage_metadata.get("prompt_tokens")
        or token_usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
    )
    completion_tokens = coerce_int(
        usage_metadata.get("output_tokens")
        or usage_metadata.get("completion_tokens")
        or token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
    )
    total_tokens = coerce_int(usage_metadata.get("total_tokens") or token_usage.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def parse_methods(methods: str) -> list[str]:
    parsed = [method.strip() for method in methods.split(",") if method.strip()]
    if not parsed:
        raise ValueError("At least one method is required.")

    unknown = [method for method in parsed if method not in VALID_METHODS]
    if unknown:
        supported = ", ".join(sorted(VALID_METHODS))
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}. Supported methods: {supported}.")

    deduped: list[str] = []
    for method in parsed:
        if method not in deduped:
            deduped.append(method)
    return deduped


def create_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{uuid4().hex[:8]}"


def resolve_output_path(benchmark: BenchmarkSpec, args: argparse.Namespace, run_id: str | None = None) -> Path:
    if args.output is not None:
        output_path = Path(args.output)
        if output_path.is_absolute() or output_path.parent != Path("."):
            return output_path

    resolved_run_id = run_id or create_run_id()
    output_root = resolve_run_output_root(benchmark.name, args, resolved_run_id)
    if args.output is None:
        default_output = Path(benchmark.default_output_filename)
        return output_root / default_output.name

    return output_root / Path(args.output)


def resolve_run_output_root(benchmark_name: str, args: argparse.Namespace, run_id: str) -> Path:
    return Path(args.output_dir) / safe_filename(benchmark_name) / resolve_model_output_name(args) / run_id


def resolve_correctness_matrix_path(trace_root: Path) -> Path:
    return trace_root / "correctness.csv"


def resolve_json_trace_root(args: argparse.Namespace, run_id: str) -> Path:
    return resolve_run_output_root(args.benchmark, args, run_id)


def resolve_example_trace_path(args: argparse.Namespace, run_id: str, index: int, method: str) -> Path:
    return resolve_json_trace_root(args, run_id) / "examples" / f"{index:04d}_{safe_filename(method)}.json"


def safe_filename(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return normalized.strip("._") or "trace"


def resolve_model_output_name(args: argparse.Namespace) -> str:
    resolved_model = resolve_model_name(args)
    model_name = resolved_model.rstrip("/").rsplit("/", 1)[-1]
    return safe_filename(model_name)


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [to_jsonable(item) for item in value]
    return str(value)


def build_run_config(
    benchmark: BenchmarkSpec,
    args: argparse.Namespace,
    methods: list[str],
    *,
    run_id: str,
    output_path: Path,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "benchmark": benchmark.name,
        "benchmark_display_name": benchmark.display_name,
        "methods": methods,
        "output_path": str(output_path),
        "settings": {
            "model": args.model,
            "resolved_model": resolve_model_name(args),
            "local_model": args.local_model,
            "limit": args.limit,
            "attempts": getattr(args, "attempts", 1),
            "ma_proofbench_level": getattr(args, "ma_proofbench_level", None),
            "olymmath_subset": getattr(args, "olymmath_subset", None),
            "lean_timeout": getattr(args, "lean_timeout", None),
            "kimina_host": getattr(args, "kimina_host", None),
            "kimina_port": getattr(args, "kimina_port", None),
            "kimina_max_workers": getattr(args, "kimina_max_workers", None),
            "lean_agent_workspace": getattr(args, "lean_agent_workspace", None),
            "workspace_image": getattr(args, "workspace_image", None),
            "max_steps": args.max_steps,
            "total_runtime_timeout": args.total_runtime_timeout,
            "synthesis_timeout": args.synthesis_timeout,
            "seed": args.seed,
            "data_file": args.data_file,
            "save_json_traces": args.save_json_traces,
        },
    }


def build_method_settings(method: str, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": method,
        "model": args.model,
        "resolved_model": resolve_model_name(args),
        "local_model": args.local_model,
        "subagent_mode": method_subagent_mode(method),
        "message_streaming": method_message_streaming(method),
        "max_steps": args.max_steps,
        "attempts": getattr(args, "attempts", 1),
        "olymmath_subset": getattr(args, "olymmath_subset", None),
        "kimina_host": getattr(args, "kimina_host", None),
        "kimina_port": getattr(args, "kimina_port", None),
        "kimina_max_workers": getattr(args, "kimina_max_workers", None),
        "lean_agent_workspace": getattr(args, "lean_agent_workspace", None),
        "workspace_image": getattr(args, "workspace_image", None),
        "total_runtime_timeout": args.total_runtime_timeout,
        "synthesis_timeout": args.synthesis_timeout,
        "save_json_traces": args.save_json_traces,
}


def resolve_model_name(args: argparse.Namespace) -> str:
    cached = getattr(args, "resolved_model", None)
    if cached:
        return str(cached)
    if args.model and args.model != "auto":
        return args.model
    if args.local_model:
        return resolve_local_model_name()
    detected_model = resolve_auto_openai_model_name()
    if detected_model is not None:
        return detected_model
    setattr(args, "local_model", True)
    return resolve_local_model_name()


def resolve_local_model_name() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen3:4b")


def resolve_auto_openai_model_name() -> str | None:
    local_fallback = resolve_local_model_name()
    base_url = get_openai_base_url()
    timeout = float(os.getenv("OPENAI_MODEL_LOOKUP_TIMEOUT", "2"))
    try:
        request = Request(f"{base_url}/models", headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"Could not auto-detect model from {base_url}/models ({exc}); using local model: {local_fallback}")
        return None

    model_id = first_model_id(payload)
    if model_id is None:
        print(f"No model id found in {base_url}/models response; using local model: {local_fallback}")
        return None

    print(f"Auto-detected model from API: {model_id}")
    return model_id


def first_model_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, list):
        return None
    for item in data:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"].strip():
            return item["id"].strip()
    return None


def method_subagent_mode(method: str) -> str:
    if method in {"plain_llm", "single_agent"}:
        return "none"
    return "dynamic" if "dynamic" in method else "fixed"


def method_message_streaming(method: str) -> bool | None:
    if method in {"plain_llm", "single_agent"}:
        return None
    return "no_streaming" not in method


def build_question_context(row: dict[str, Any]) -> dict[str, Any]:
    question = row.get("Question") or row.get("question") or row.get("problem") or row.get("informal_statement") or row.get("input", "")
    incorrect_answers = [
        row[field]
        for field in ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3")
        if row.get(field)
    ]
    options: dict[str, Any] = {}
    if row.get("Correct Answer") is not None:
        options["correct"] = row.get("Correct Answer")
    if incorrect_answers:
        options["incorrect"] = incorrect_answers
    if isinstance(row.get("options"), list):
        labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        options = {label: option for label, option in zip(labels, row["options"], strict=False)}
    context: dict[str, Any] = {
        "question": question,
        "options": options,
    }
    if row.get("answer"):
        match = re.search(r"####\s*(?P<answer>[^\n]+)", str(row["answer"]))
        if match:
            context["gold_answer"] = match.group("answer").strip()
        elif len(str(row["answer"]).strip()) == 1:
            context["gold_answer"] = str(row["answer"]).strip().upper()
    if isinstance(row.get("target"), list):
        context["gold_answer"] = row.get("target")
    if row.get("formal_statement"):
        context["formal_statement"] = row.get("formal_statement")
    if row.get("unique_id"):
        context["unique_id"] = row.get("unique_id")
    if row.get("subject"):
        context["subject"] = row.get("subject")
    if row.get("split"):
        context["split"] = row.get("split")
    if row.get("topic"):
        context["topic"] = row.get("topic")
    if row.get("tag"):
        context["tag"] = row.get("tag")
    return context


def score_benchmark_response(
    benchmark: BenchmarkSpec,
    row: dict[str, Any],
    raw_output: str,
    gold: str,
    args: argparse.Namespace,
) -> BenchmarkScore:
    if benchmark.score_response is not None:
        return benchmark.score_response(row, raw_output, args)

    pred = benchmark.extract_answer(raw_output)
    return BenchmarkScore(pred=pred, correct=pred == gold)


def build_multiagent_debug(method_trace: dict[str, Any]) -> dict[str, Any]:
    selected_agents = extract_selected_subagents(method_trace)
    return {
        "subagents": selected_agents,
        "messages": extract_agent_messages(method_trace),
        "workflow": build_debug_workflow(selected_agents, method_trace),
    }


def extract_selected_subagents(method_trace: dict[str, Any]) -> list[dict[str, Any]]:
    orchestrator_plan = method_trace.get("orchestrator_plan")
    if not isinstance(orchestrator_plan, dict):
        return []
    selected_agents = orchestrator_plan.get("selected_agents")
    if not isinstance(selected_agents, list):
        return []

    subagents: list[dict[str, Any]] = []
    for agent in selected_agents:
        if not isinstance(agent, dict) or not agent.get("name"):
            continue
        compact_agent = {
            key: agent[key]
            for key in ("name", "role", "description", "subtask", "expected_output", "critical_debate")
            if key in agent
        }
        subagents.append(compact_agent)
    return subagents


def extract_agent_messages(method_trace: dict[str, Any]) -> list[dict[str, Any]]:
    event_log = method_trace.get("event_log")
    if not isinstance(event_log, list):
        return []

    messages: list[dict[str, Any]] = []
    for event in event_log:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        if event_type not in {"finding", "critique", "warning", "question", "message_received", "agent_done"}:
            continue
        messages.append(
            {
                "source": event.get("source"),
                "target": event.get("target") or "broadcast",
                "event_type": event_type,
                "content": event.get("content", ""),
                "confidence": event.get("confidence"),
            }
        )
    return messages


def build_debug_workflow(selected_agents: list[dict[str, Any]], method_trace: dict[str, Any]) -> list[dict[str, str]]:
    if not selected_agents:
        if method_trace.get("method") == "single_agent":
            return [
                {"node": "single_agent", "description": "Iteratively answered and self-reviewed until final parseable output or max steps."},
            ]
        if method_trace.get("prompt") is not None:
            return [
                {"node": "plain_llm", "description": "Answered the benchmark prompt directly."},
            ]
        return []

    workflow = [{"node": "orchestrator", "description": "Created plan and selected subagents."}]
    workflow.extend(
        {
            "node": str(agent["name"]),
            "description": "Ran subagent and published/received messages.",
        }
        for agent in selected_agents
    )
    workflow.append({"node": "synthesizer", "description": "Combined subagent outputs and event log into final answer."})
    return workflow


async def run_plain_llm(prompt: str, llm: Any) -> tuple[str, int]:
    response = await llm.ainvoke(prompt)
    return getattr(response, "content", str(response)), 0


async def run_single_agent(prompt: str, llm: Any, args: argparse.Namespace) -> tuple[str, int, dict[str, Any]]:
    max_steps = max(1, int(getattr(args, "max_steps", 3)))
    answer_extractor = getattr(args, "answer_extractor", None)
    if not callable(answer_extractor):
        answer_extractor = lambda text: text.strip() if text.strip() else None

    steps: list[dict[str, Any]] = []
    raw_output = ""
    stopped_reason = "max_steps"

    for step_index in range(1, max_steps + 1):
        step_prompt = render_prompt(
            "evaluation/single_agent_step.j2",
            benchmark_prompt=prompt,
            previous_steps=steps,
        )
        response = await llm.ainvoke(step_prompt)
        content = getattr(response, "content", str(response)).strip()
        parsed_payload = parse_single_agent_payload(content)
        status = parsed_payload.get("status")
        final_answer = str(parsed_payload.get("final_answer") or "").strip()
        notes = str(parsed_payload.get("notes") or "").strip()
        candidate_output = final_answer or content
        parsed_answer = answer_extractor(candidate_output)
        raw_output = candidate_output

        step_trace = {
            "step": step_index,
            "status": status,
            "output": candidate_output,
            "notes": notes,
            "parsed_answer": parsed_answer,
            "raw_response": content,
        }
        steps.append(step_trace)

        if status == "final" and parsed_answer is not None:
            stopped_reason = "final_answer_parseable"
            break

    return raw_output, 0, {
        "method": "single_agent",
        "prompt": prompt,
        "raw_output": raw_output,
        "steps": steps,
        "stopped_reason": stopped_reason,
    }


def parse_single_agent_payload(content: str) -> dict[str, Any]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", stripped):
            try:
                payload, _ = decoder.raw_decode(stripped[match.start() :])
                break
            except json.JSONDecodeError:
                continue
        else:
            return {"status": "continue", "final_answer": "", "notes": "Model did not return JSON."}
    if not isinstance(payload, dict):
        return {"status": "continue", "final_answer": "", "notes": "Model did not return a JSON object."}

    status = str(payload.get("status") or "continue").strip().lower()
    if status not in {"continue", "final"}:
        status = "continue"
    return {
        "status": status,
        "final_answer": payload.get("final_answer") or "",
        "notes": payload.get("notes") or "",
    }


async def run_multiagent(
    prompt: str,
    llm: Any,
    args: argparse.Namespace,
    *,
    enable_agent_message_streaming: bool = True,
    subagent_mode: str = "fixed",
    workflow_config: BenchmarkWorkflowConfig | None = None,
) -> tuple[str, int, dict[str, Any]]:
    docker_workspace = None
    try:
        feedback_tool = None
        if workflow_config is not None:
            docker_workspace = await DockerWorkspace.create(
                image=getattr(args, "workspace_image", "python:3.12"),
                source_path=None,
                default_timeout_seconds=getattr(args, "workspace_command_timeout", 60.0),
                output_char_limit=getattr(args, "workspace_output_limit", 12000),
            )
            for path, content in workflow_config.seed_files.items():
                await docker_workspace.write_text(path, content)
            if workflow_config.feedback_tool_factory is not None:
                feedback_tool = workflow_config.feedback_tool_factory(docker_workspace)

        state = await run_workflow(
            task=prompt,
            llm=llm,
            subagent_mode=subagent_mode,
            max_steps_per_agent=args.max_steps,
            total_runtime_timeout=args.total_runtime_timeout,
            synthesis_timeout=args.synthesis_timeout,
            enable_agent_message_streaming=enable_agent_message_streaming,
            stream_to_console=False,
            no_color=True,
            enable_workspace_tools=workflow_config is not None,
            docker_workspace=docker_workspace,
            feedback_tool=feedback_tool,
        )
        raw_output = state["final_answer"]
        trace = extract_workflow_trace(state)
        if workflow_config is not None and docker_workspace is not None:
            trace["workspace"] = {
                "final_candidate_path": workflow_config.final_candidate_path,
                "seed_files": sorted(workflow_config.seed_files),
            }
            if workflow_config.final_candidate_path:
                final_candidate = await docker_workspace.read_text(workflow_config.final_candidate_path)
                trace["workspace"]["final_candidate"] = final_candidate
                raw_output = f"{raw_output}\n\n```lean4\n{final_candidate.strip()}\n```"
        return raw_output, 0, trace
    finally:
        if docker_workspace is not None:
            await docker_workspace.cleanup()


def extract_workflow_trace(state: dict[str, Any]) -> dict[str, Any]:
    trace_keys = [
        "run_id",
        "mode",
        "subagent_mode",
        "task_type",
        "reason",
        "plan",
        "selected_agents",
        "assignments",
        "orchestrator_plan",
        "event_log",
        "agent_outputs",
        "agent_traces",
        "final_answer",
    ]
    return {key: state.get(key) for key in trace_keys if key in state}


async def run_method(
    method: str,
    prompt: str,
    llm: Any,
    args: argparse.Namespace,
    *,
    workflow_config: BenchmarkWorkflowConfig | None = None,
) -> RunResult:
    metered_llm = MeteredLLM(llm)
    started_at = time.perf_counter()
    trace: dict[str, Any] | None = None
    if method == "plain_llm":
        raw_output, returncode = await run_plain_llm(prompt, metered_llm)
        trace = {"prompt": prompt, "raw_output": raw_output}
    elif method == "single_agent":
        raw_output, returncode, trace = await run_single_agent(prompt, metered_llm, args)
    elif method in {"multiagent", "multiagent_streaming"}:
        raw_output, returncode, trace = await run_multiagent(
            prompt,
            metered_llm,
            args,
            enable_agent_message_streaming=True,
            subagent_mode="fixed",
            workflow_config=workflow_config,
        )
    elif method == "multiagent_no_streaming":
        raw_output, returncode, trace = await run_multiagent(
            prompt,
            metered_llm,
            args,
            enable_agent_message_streaming=False,
            subagent_mode="fixed",
            workflow_config=workflow_config,
        )
    elif method == "multiagent_dynamic_streaming":
        raw_output, returncode, trace = await run_multiagent(
            prompt,
            metered_llm,
            args,
            enable_agent_message_streaming=True,
            subagent_mode="dynamic",
            workflow_config=workflow_config,
        )
    elif method == "multiagent_dynamic_no_streaming":
        raw_output, returncode, trace = await run_multiagent(
            prompt,
            metered_llm,
            args,
            enable_agent_message_streaming=False,
            subagent_mode="dynamic",
            workflow_config=workflow_config,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed_seconds = time.perf_counter() - started_at
    return RunResult(
        raw_output=raw_output,
        returncode=returncode,
        elapsed_seconds=elapsed_seconds,
        prompt_tokens=metered_llm.usage.prompt_tokens,
        completion_tokens=metered_llm.usage.completion_tokens,
        total_tokens=metered_llm.usage.total_tokens,
        trace=trace,
    )


def progress(items: Any, desc: str) -> Any:
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        return items
    return tqdm(items, desc=desc)


def write_results_csv(output_path: Path, results: list[dict[str, Any]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark",
                "method",
                "index",
                "gold",
                "pred",
                "correct",
                "returncode",
                "error",
                "elapsed_seconds",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "raw_output",
                "score_metadata",
                "json_trace_path",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def write_correctness_matrix_csv(trace_root: Path, results: list[dict[str, Any]], methods: list[str]) -> None:
    matrix_path = resolve_correctness_matrix_path(trace_root)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[int, dict[str, str]] = defaultdict(dict)
    for result in results:
        grouped[int(result["index"])][str(result["method"])] = "T" if result.get("correct") else "F"

    with matrix_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task_id", *methods])
        writer.writeheader()
        for task_id in sorted(grouped):
            row = {"task_id": str(task_id)}
            row.update({method: grouped[task_id].get(method, "") for method in methods})
            writer.writerow(row)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[result["method"]].append(result)

    summary: dict[str, dict[str, float | int]] = {}
    for method, method_results in grouped.items():
        total = len(method_results)
        correct_count = sum(1 for result in method_results if result["correct"])
        invalid_count = sum(1 for result in method_results if result["pred"] is None)
        elapsed_seconds = sum(float(result.get("elapsed_seconds") or 0.0) for result in method_results)
        prompt_tokens = sum(int(result.get("prompt_tokens") or 0) for result in method_results)
        completion_tokens = sum(int(result.get("completion_tokens") or 0) for result in method_results)
        total_tokens = sum(int(result.get("total_tokens") or 0) for result in method_results)
        summary[method] = {
            "total": total,
            "correct": correct_count,
            "accuracy": correct_count / total if total else 0.0,
            "invalid": invalid_count,
            "invalid_rate": invalid_count / total if total else 0.0,
            "elapsed_seconds": elapsed_seconds,
            "avg_elapsed_seconds": elapsed_seconds / total if total else 0.0,
            "prompt_tokens": prompt_tokens,
            "avg_prompt_tokens": prompt_tokens / total if total else 0.0,
            "completion_tokens": completion_tokens,
            "avg_completion_tokens": completion_tokens / total if total else 0.0,
            "total_tokens": total_tokens,
            "avg_total_tokens": total_tokens / total if total else 0.0,
        }
    return summary


def build_method_averages(summary: dict[str, dict[str, float | int]]) -> dict[str, dict[str, float]]:
    return {
        method: {
            "correct_avg": float(stats["accuracy"]),
            "invalid_avg": float(stats["invalid_rate"]),
            "time_avg_seconds": float(stats["avg_elapsed_seconds"]),
            "total_token_avg": float(stats["avg_total_tokens"]),
        }
        for method, stats in summary.items()
    }


def print_result(result: dict[str, Any]) -> None:
    print()
    print("=" * 80)
    print(f"Benchmark: {result['benchmark']}")
    print(f"Method: {result['method']}")
    print(f"Question: {result['index']}")
    print(f"Gold: {result['gold']}")
    print(f"Pred: {result['pred']}")
    print(f"Correct: {result['correct']}")
    print(f"Elapsed seconds: {result['elapsed_seconds']:.4f}")
    if result.get("total_tokens") is not None:
        print(
            f"Tokens: prompt={result.get('prompt_tokens')}, "
            f"completion={result.get('completion_tokens')}, total={result.get('total_tokens')}"
        )
    if result["error"]:
        print(f"Error: {result['error']}")
    print("=" * 80)


def print_summary(benchmark: BenchmarkSpec, summary: dict[str, dict[str, float | int]], output_path: Path) -> None:
    print()
    print("Benchmark finished.")
    print(f"Benchmark: {benchmark.display_name}")
    for method, stats in summary.items():
        print()
        print(f"Method: {method}")
        print(f"Total: {stats['total']}")
        print(f"Correct: {stats['correct']}")
        print(f"Accuracy: {stats['accuracy']:.4f}")
        print(f"Invalid answers: {stats['invalid']}")
        print(f"Invalid rate: {stats['invalid_rate']:.4f}")
        print(f"Total time: {stats['elapsed_seconds']:.4f}s")
        print(f"Avg time: {stats['avg_elapsed_seconds']:.4f}s")
        print(f"Prompt tokens: {stats['prompt_tokens']}")
        print(f"Avg prompt tokens: {stats['avg_prompt_tokens']:.2f}")
        print(f"Completion tokens: {stats['completion_tokens']}")
        print(f"Avg completion tokens: {stats['avg_completion_tokens']:.2f}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"Avg total tokens: {stats['avg_total_tokens']:.2f}")
    print()
    print(f"Saved results to: {output_path}")
