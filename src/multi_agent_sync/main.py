from __future__ import annotations

import argparse
import asyncio
import os

from rich.console import Console

from multi_agent_sync.artifacts import build_token_usage_by_step, format_token_usage_by_step, save_run_artifacts
from multi_agent_sync.graph.workflow import run_workflow
from multi_agent_sync.llm import get_llm
from multi_agent_sync.workspace.docker import DockerWorkspace


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LangGraph multi-agent synchronization prototype.")
    parser.add_argument("task", nargs="+", help="Task to assign to the multi-agent runtime.")
    parser.add_argument("--model", default=None, help="Model name to use for the selected provider.")
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum inference steps per agent.")
    parser.add_argument(
        "--subagent-mode",
        choices=("fixed", "dynamic"),
        default="fixed",
        help="Use fixed registered agents or dynamic Orchestrator-defined subagents.",
    )
    parser.add_argument(
        "--total-runtime-timeout",
        type=float,
        default=600.0,
        help="Maximum total runtime for all agents, in seconds. Defaults to 600.",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored terminal output.")
    parser.add_argument("--runs-dir", default="runs", help="Directory where run artifacts are saved.")
    parser.add_argument(
        "--docker-workspace",
        action="store_true",
        help="Enable Docker-only bash workspace tools for eligible agents.",
    )
    parser.add_argument(
        "--workspace-image",
        default="python:3.12",
        help="Docker image used when --docker-workspace is enabled.",
    )
    parser.add_argument(
        "--workspace-source",
        default=".",
        help="Local directory copied into /workspace when --docker-workspace is enabled.",
    )
    parser.add_argument(
        "--workspace-command-timeout",
        type=float,
        default=60.0,
        help="Per-command timeout in seconds for Docker workspace bash.",
    )
    parser.add_argument(
        "--workspace-output-limit",
        type=int,
        default=12000,
        help="Maximum stdout/stderr characters retained per Docker bash command.",
    )
    return parser


async def async_main(args: argparse.Namespace) -> None:
    if args.model:
        os.environ["OPENAI_MODEL"] = args.model

    task = " ".join(args.task)
    llm = get_llm(args.model, openai=True)
    docker_workspace = None
    try:
        if args.docker_workspace:
            docker_workspace = await DockerWorkspace.create(
                image=args.workspace_image,
                source_path=args.workspace_source,
                default_timeout_seconds=args.workspace_command_timeout,
                output_char_limit=args.workspace_output_limit,
            )

        state = await run_workflow(
            task=task,
            llm=llm,
            subagent_mode=args.subagent_mode,
            max_steps_per_agent=args.max_steps,
            total_runtime_timeout=args.total_runtime_timeout,
            stream_to_console=True,
            no_color=args.no_color,
            enable_workspace_tools=args.docker_workspace,
            docker_workspace=docker_workspace,
        )

        console = Console(no_color=args.no_color)
        console.print("\n[bold]Final answer[/bold]")
        console.print(state["final_answer"])
        console.print(f"\n{format_token_usage_by_step(build_token_usage_by_step(state.get('agent_traces', {})))}")
        run_dir = save_run_artifacts(state, root_dir=args.runs_dir)
        console.print(f"\nRun artifacts: {run_dir}")
    finally:
        if docker_workspace is not None:
            await docker_workspace.cleanup()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(async_main(args))
