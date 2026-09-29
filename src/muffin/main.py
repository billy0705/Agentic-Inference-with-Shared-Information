import argparse
import asyncio

from rich.console import Console

from muffin.agents.base import DEFAULT_AGENT_RUNTIME_TIMEOUT_SECONDS
from muffin.graph.workflow import run_workflow
from muffin.llm import get_llm
from muffin.tracing.artifacts import build_token_usage_by_step, format_token_usage_by_step, save_run_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LangGraph multi-agent synchronization prototype.")
    parser.add_argument("task", nargs="+", help="Task to assign to the multi-agent runtime.")
    parser.add_argument("--model", default=None, help="Model name to use for the selected provider.")
    parser.add_argument(
        "--openai-base-url",
        default=None,
        help="OpenAI-compatible API URL. If omitted, use OPENAI_BASE_URL or http://localhost:8000/v1.",
    )
    parser.add_argument("--max-steps", type=int, default=3, help="Maximum inference steps per agent.")
    parser.add_argument(
        "--allow-agent-early-stop",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow agents to stop before --max-steps when they return FINAL. Disabled by default.",
    )
    parser.add_argument(
        "--think-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefix orchestrator, agent step, and summarizer prompts with <|think|>. Enabled by default.",
    )
    parser.add_argument(
        "--min-dynamic-subagents",
        type=int,
        default=3,
        help="Minimum dynamic subagents to create. Defaults to 3.",
    )
    parser.add_argument(
        "--max-dynamic-subagents",
        type=int,
        default=3,
        help="Maximum dynamic subagents to create. Defaults to 3.",
    )
    parser.add_argument(
        "--total-runtime-timeout",
        type=float,
        default=1800.0,
        help="Maximum total runtime for all agents, in seconds. Defaults to 1800.",
    )
    parser.add_argument(
        "--agent-runtime-timeout",
        type=float,
        default=DEFAULT_AGENT_RUNTIME_TIMEOUT_SECONDS,
        help=f"Maximum runtime per agent, in seconds. Defaults to {DEFAULT_AGENT_RUNTIME_TIMEOUT_SECONDS:g}.",
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colored terminal output.")
    parser.add_argument("--runs-dir", default="runs", help="Directory where run artifacts are saved.")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    task = " ".join(args.task)
    llm = get_llm(args.model, base_url=args.openai_base_url)
    state = await run_workflow(
        task=task,
        llm=llm,
        max_steps_per_agent=args.max_steps,
        min_dynamic_subagents=args.min_dynamic_subagents,
        max_dynamic_subagents=args.max_dynamic_subagents,
        total_runtime_timeout=args.total_runtime_timeout,
        agent_runtime_timeout=args.agent_runtime_timeout,
        allow_agent_early_stop=args.allow_agent_early_stop,
        think_mode=args.think_mode,
        stream_to_console=True,
        no_color=args.no_color,
    )

    console = Console(no_color=args.no_color)
    console.print("\n[bold]Final answer[/bold]")
    console.print(state["final_answer"])
    console.print(f"\n{format_token_usage_by_step(build_token_usage_by_step(state.get('agent_traces', {})))}")
    run_dir = save_run_artifacts(state, root_dir=args.runs_dir)
    console.print(f"\nRun artifacts: {run_dir}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(async_main(args))
