import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_eval_traces.py"


def load_compare_module():
    assert SCRIPT_PATH.exists()
    spec = importlib.util.spec_from_file_location("compare_eval_traces", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_trace(path: Path, *, method: str, pred: str, correct: bool, agents: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "index": 0,
                "gold": "A",
                "method": method,
                "pred": pred,
                "correct": correct,
                "method_trace": {
                    "orchestrator_plan": {
                        "selected_agents": [{"name": agent} for agent in agents],
                    },
                    "event_log": [{"event_type": "finding"}, {"event_type": "critique"}],
                    "agent_traces": {
                        agent: {"steps": [{"used_event_ids": ["event-1"]}], "event_receipts": [{"accepted": True}]}
                        for agent in agents
                    },
                },
            }
        ),
        encoding="utf-8",
    )


def test_compare_eval_traces_reports_method_differences(tmp_path):
    module = load_compare_module()
    trace_dir = tmp_path / "examples"
    trace_dir.mkdir()
    write_trace(trace_dir / "0000_multiagent_streaming.json", method="multiagent_streaming", pred="A", correct=True, agents=["SolverAgent"])
    write_trace(
        trace_dir / "0000_multiagent_dynamic_streaming.json",
        method="multiagent_dynamic_streaming",
        pred="C",
        correct=False,
        agents=["TaskWorker", "CriticalDebateAgent"],
    )

    traces = module.load_trace_files(trace_dir)
    differences = module.find_method_differences(traces)
    rendered = module.format_differences(differences)

    assert len(differences) == 1
    assert differences[0]["index"] == 0
    assert "multiagent_dynamic_streaming" in rendered
    assert "CriticalDebateAgent" in rendered
