from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.graph.state import GraphState
from multi_agent_sync.sync_report import build_sync_report


def save_run_artifacts(state: GraphState, root_dir: str | Path = "runs") -> Path:
    run_dir = Path(root_dir) / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "task.txt").write_text(state["task"])
    (run_dir / "final_answer.md").write_text(state["final_answer"])
    _write_event_log(run_dir / "event_log.jsonl", state.get("event_log", []))
    (run_dir / "agent_traces.json").write_text(json.dumps(state.get("agent_traces", {}), indent=2))
    (run_dir / "orchestrator_plan.json").write_text(json.dumps(state.get("orchestrator_plan", {}), indent=2))
    sync_report = build_sync_report(state.get("event_log", []), state.get("agent_traces", {}))
    (run_dir / "sync_report.json").write_text(json.dumps(sync_report, indent=2))

    return run_dir


def _write_event_log(path: Path, event_log: list[AgentEvent]) -> None:
    lines = [json.dumps(event.model_dump(mode="json")) for event in event_log]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
