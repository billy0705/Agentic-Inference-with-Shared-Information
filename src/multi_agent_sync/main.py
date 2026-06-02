from __future__ import annotations

import argparse
import asyncio
import os

from rich.console import Console

from multi_agent_sync.artifacts import save_run_artifacts
from multi_agent_sync.graph.workflow import run_workflow
from multi_agent_sync.llm import get_llm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LangGraph multi-agent synchronization prototype.")
    parser.add_argument("task", nargs="+", help="Task to assign to the multi-agent runtime.")
    parser.add_argument("--model", default=None, help="Ollama model to use. Defaults to OLLAMA_MODEL or qwen3:4b.")
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum inference steps per agent.")
    parser.add_argument(
        "--total-runtime-timeout",
        type=float,
        default=600.0,
        help="Maximum total runtime for all agents, in seconds. Defaults to 600.",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored terminal output.")
    parser.add_argument("--runs-dir", default="runs", help="Directory where run artifacts are saved.")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    if args.model:
        os.environ["OLLAMA_MODEL"] = args.model

    task = " ".join(args.task)
    llm = get_llm(args.model)
    state = await run_workflow(
        task=task,
        llm=llm,
        max_steps_per_agent=args.max_steps,
        total_runtime_timeout=args.total_runtime_timeout,
        stream_to_console=True,
        no_color=args.no_color,
    )

    console = Console(no_color=args.no_color)
    console.print("\n[bold]Final answer[/bold]")
    console.print(state["final_answer"])
    run_dir = save_run_artifacts(state, root_dir=args.runs_dir)
    console.print(f"\nRun artifacts: {run_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(async_main(args))
