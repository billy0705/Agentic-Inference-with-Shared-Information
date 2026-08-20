from __future__ import annotations

import argparse
from typing import Any

from multi_agent_sync.agents.base import DEFAULT_AGENT_RUNTIME_TIMEOUT_SECONDS
from multi_agent_sync.evaluation.types import BenchmarkWorkflowConfig
from multi_agent_sync.graph.dynamic_orchestration import run_dynamic_orchestration_workflow
from multi_agent_sync.graph.workflow import run_workflow
from multi_agent_sync.workspace.docker import DockerWorkspace


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
        final_guard_tool = None
        returncode = 0
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
            if workflow_config.final_guard_factory is not None:
                final_guard_tool = workflow_config.final_guard_factory(docker_workspace)

        state = await run_workflow(
            task=prompt,
            llm=llm,
            subagent_mode=subagent_mode,
            max_steps_per_agent=args.max_steps,
            min_dynamic_subagents=getattr(args, "min_dynamic_subagents", 3),
            max_dynamic_subagents=getattr(args, "max_dynamic_subagents", 3),
            total_runtime_timeout=args.total_runtime_timeout,
            agent_runtime_timeout=getattr(args, "agent_runtime_timeout", DEFAULT_AGENT_RUNTIME_TIMEOUT_SECONDS),
            synthesis_timeout=args.synthesis_timeout,
            allow_agent_early_stop=getattr(args, "allow_agent_early_stop", False),
            think_mode=getattr(args, "think_mode", True),
            enable_agent_message_streaming=enable_agent_message_streaming,
            stream_to_console=False,
            no_color=True,
            enable_workspace_tools=workflow_config is not None,
            docker_workspace=docker_workspace,
            feedback_tool=feedback_tool,
            final_guard_tool=final_guard_tool,
            benchmark=str(getattr(args, "benchmark", "") or ""),
            synthesizer_mode=(
                "summarize_outputs"
                if subagent_mode == "dynamic" and enable_agent_message_streaming
                else "generic"
            ),
        )
        raw_output = state["final_answer"]
        trace = extract_workflow_trace(state)
        if workflow_config is not None and docker_workspace is not None:
            trace["workspace"] = {
                "final_candidate_path": workflow_config.final_candidate_path,
                "seed_files": sorted(workflow_config.seed_files),
            }
            try:
                if workflow_config.final_candidate_exporter is not None:
                    final_candidate = await workflow_config.final_candidate_exporter(docker_workspace)
                    trace["workspace"]["final_candidate"] = final_candidate
                    if final_candidate.strip():
                        raw_output = f"{raw_output}\n\n```diff\n{final_candidate.strip()}\n```"
                elif workflow_config.final_candidate_path:
                    final_candidate = await docker_workspace.read_text(workflow_config.final_candidate_path)
                    trace["workspace"]["final_candidate"] = final_candidate
                    raw_output = f"{raw_output}\n\n```lean4\n{final_candidate.strip()}\n```"
            except Exception as exc:
                returncode = 1
                trace["workspace"]["export_error"] = str(exc)
                raw_output = f"{raw_output}\n\nWorkspace export failed: {exc}".strip()
        return raw_output, returncode, trace
    finally:
        if docker_workspace is not None:
            await docker_workspace.cleanup()


async def run_dynamic_orchestration(
    prompt: str,
    llm: Any,
    args: argparse.Namespace,
    *,
    workflow_config: BenchmarkWorkflowConfig | None = None,
) -> tuple[str, int, dict[str, Any]]:
    docker_workspace = None
    try:
        feedback_tool = None
        final_guard_tool = None
        returncode = 0
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
            if workflow_config.final_guard_factory is not None:
                final_guard_tool = workflow_config.final_guard_factory(docker_workspace)

        state = await run_dynamic_orchestration_workflow(
            task=prompt,
            llm=llm,
            benchmark=str(getattr(args, "benchmark", "") or ""),
            max_steps_per_agent=args.max_steps,
            max_orchestrator_rounds=getattr(args, "max_orchestrator_rounds", 3),
            total_runtime_timeout=args.total_runtime_timeout,
            synthesis_timeout=args.synthesis_timeout,
            enable_agent_message_streaming=True,
            stream_to_console=False,
            no_color=True,
            enable_workspace_tools=workflow_config is not None,
            docker_workspace=docker_workspace,
            feedback_tool=feedback_tool,
            final_guard_tool=final_guard_tool,
            think_mode=getattr(args, "think_mode", True),
        )
        raw_output = state["final_answer"]
        trace = extract_workflow_trace(state)
        trace["method"] = "dynamic_orchestration"
        if workflow_config is not None and docker_workspace is not None:
            trace["workspace"] = {
                "final_candidate_path": workflow_config.final_candidate_path,
                "seed_files": sorted(workflow_config.seed_files),
            }
            try:
                if workflow_config.final_candidate_exporter is not None:
                    final_candidate = await workflow_config.final_candidate_exporter(docker_workspace)
                    trace["workspace"]["final_candidate"] = final_candidate
                    if final_candidate.strip():
                        raw_output = f"{raw_output}\n\n```diff\n{final_candidate.strip()}\n```"
                elif workflow_config.final_candidate_path:
                    final_candidate = await docker_workspace.read_text(workflow_config.final_candidate_path)
                    trace["workspace"]["final_candidate"] = final_candidate
                    raw_output = f"{raw_output}\n\n```lean4\n{final_candidate.strip()}\n```"
            except Exception as exc:
                returncode = 1
                trace["workspace"]["export_error"] = str(exc)
                raw_output = f"{raw_output}\n\nWorkspace export failed: {exc}".strip()
        return raw_output, returncode, trace
    finally:
        if docker_workspace is not None:
            await docker_workspace.cleanup()


def extract_workflow_trace(state: dict[str, Any]) -> dict[str, Any]:
    trace_keys = [
        "method",
        "run_id",
        "mode",
        "subagent_mode",
        "task_type",
        "reason",
        "plan",
        "selected_agents",
        "assignments",
        "allow_agent_early_stop",
        "think_mode",
        "agent_runtime_timeout",
        "orchestrator_plan",
        "event_log",
        "agent_outputs",
        "agent_traces",
        "synthesizer_mode",
        "synthesizer_trace",
        "direct_trace",
        "orchestrator_rounds",
        "dynamic_orchestration_trace",
        "final_answer",
    ]
    return {key: state.get(key) for key in trace_keys if key in state}
