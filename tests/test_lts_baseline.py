from __future__ import annotations

import argparse
import json
import stat
from pathlib import Path

import pytest

from multi_agent_sync.evaluation import runner
from multi_agent_sync.evaluation.baselines import lts


TEMPLATES = lts.DEFAULT_LTS_REPO_DIR / lts.ASSISTANTBENCH_SUBDIR / "Templates"
requires_submodule = pytest.mark.skipif(not TEMPLATES.is_dir(), reason="third_party/autogen-lts submodule not checked out")


@requires_submodule
@pytest.mark.parametrize("method", sorted(lts.LTS_VARIANTS))
def test_patches_apply_to_upstream_templates(method):
    variant = lts.LTS_VARIANTS[method]
    source = (TEMPLATES / variant.template / "scenario.py").read_text(encoding="utf-8")

    patched = lts.prepare_scenario_source(source, variant)

    assert "__FILE_NAME__" not in patched
    assert "[coder, executor, file_surfer]," in patched
    assert "web_surfer]," not in patched
    if variant.use_decision_transformer is not None:
        assert f"\n    use_decision_transformer = {variant.use_decision_transformer}\n" in patched
    compile(patched, "scenario.py", "exec")


def test_patch_fails_loudly_when_upstream_changes():
    with pytest.raises(RuntimeError, match="file_name"):
        lts.prepare_scenario_source("print('changed upstream')", lts.LTS_VARIANTS["lts_addall"])


def test_parse_final_answer_prefers_aggregated_answer():
    stdout = "FINAL ANSWER: B\nFINAL ANSWER: C\nFINAL AGGREGATED ANSWER: C\nSharedMemory: FINAL keep rate: 3/5 (60.0%)\n"

    assert lts.parse_final_answer(stdout) == "C"
    assert lts.parse_final_answer("FINAL AGGREGATED ANSWER: [-1/8, 1/8]") == "[-1/8, 1/8]"
    assert lts.parse_final_answer("Team said FINAL ANSWER: 42") == "42"
    assert lts.parse_final_answer("no answer here") is None


def test_parse_token_usage_sums_llm_calls(tmp_path):
    log = tmp_path / "log.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"type": "LLMCallEvent", "prompt_tokens": 10, "completion_tokens": 3}),
                json.dumps({"type": "TextMessage", "message": "hi"}),
                "not json",
                json.dumps({"type": "LLMCallEvent", "prompt_tokens": 5, "completion_tokens": 2}),
            ]
        ),
        encoding="utf-8",
    )

    assert lts.parse_token_usage(log).as_dict() == {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20}


FAKE_SCENARIO = """#!/bin/sh
# Stands in for `.venv-lts/bin/python scenario.py`, run inside the task dir.
test -f prompt.txt && test -f config.yaml && test -f scenario.py || exit 3
{extra}
printf '%s\\n' '{{"type": "LLMCallEvent", "prompt_tokens": 100, "completion_tokens": 20}}' > log.jsonl
printf '%s' '{{"aggregated": {{"runtime_seconds": 1.5, "final_answer": "B"}}}}' > first_vs_aggregated_metrics.json
printf '%s' '{{"team0-step1 - x": "y"}}' > .shared_memory.json
echo "FINAL ANSWER: A"
echo "FINAL AGGREGATED ANSWER: B"
echo "SharedMemory: FINAL counts: kept=4 rejected=6"
"""


def make_fake_python(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "fake-python"
    path.write_text(FAKE_SCENARIO.format(extra=extra), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def lts_args(tmp_path: Path, timeout: float = 30.0) -> argparse.Namespace:
    return argparse.Namespace(model="test-model", output_dir=str(tmp_path / "out"), max_tokens=512, total_runtime_timeout=timeout)


@requires_submodule
@pytest.mark.asyncio
async def test_run_method_runs_lts_subprocess_and_collects_metrics(tmp_path, monkeypatch):
    weights = tmp_path / "checkpoint.pt"
    weights.write_bytes(b"weights")
    monkeypatch.setenv("LTS_PYTHON", str(make_fake_python(tmp_path)))
    monkeypatch.setenv("DECISION_TRANSFORMER_WEIGHTS", str(weights))
    monkeypatch.setenv("OPENAI_BASE_URL", "http://vllm.example:8000/v1")

    result = await runner.run_method("lts", "Question?\nFinal Answer: <A/B/C/D>", llm=object(), args=lts_args(tmp_path))

    assert result.raw_output == "Final Answer: B"
    assert result.returncode == 0
    assert (result.prompt_tokens, result.completion_tokens, result.total_tokens) == (100, 20, 120)
    trace = result.trace
    assert trace["method"] == "lts"
    assert trace["use_decision_transformer"] is True
    assert trace["lts_runtime_seconds"] == 1.5
    assert trace["memory"] == {"kept": 4, "rejected": 6, "verified": 10, "memory_entries": 1}

    task_dir = Path(trace["task_dir"])
    config = json.loads((task_dir / "config.yaml").read_text(encoding="utf-8"))
    client = config["orchestrator_client"]["config"]
    assert client["model"] == "test-model"
    assert client["base_url"] == "http://vllm.example:8000/v1"
    assert client["max_tokens"] == 512
    assert client["model_info"]["function_calling"] is False
    assert config["web_surfer_client"]["config"]["model_info"]["function_calling"] is True
    assert (task_dir / "decision_transformer.py").is_file()
    assert "use_decision_transformer = True" in (task_dir / "scenario.py").read_text(encoding="utf-8")


@requires_submodule
@pytest.mark.asyncio
async def test_lts_timeout_kills_subprocess(tmp_path, monkeypatch):
    monkeypatch.setenv("LTS_PYTHON", str(make_fake_python(tmp_path, extra="sleep 30")))

    result = await runner.run_method("lts_addall", "Question?", llm=object(), args=lts_args(tmp_path, timeout=0.5))

    assert result.returncode == 124
    assert result.trace["timed_out"] is True
    assert result.trace["final_answer"] is None
    assert result.raw_output == ""


def test_lts_methods_are_registered():
    assert runner.parse_methods("multiagent_streaming,lts,lts_llm,lts_addall") == [
        "multiagent_streaming",
        "lts",
        "lts_llm",
        "lts_addall",
    ]
    settings = runner.build_method_settings("lts_llm", argparse.Namespace(model="m", max_steps=3, total_runtime_timeout=1.0, synthesis_timeout=1.0, save_json_traces=False))
    assert settings["lts_use_decision_transformer"] is False
    assert settings["subagent_mode"] == "lts"
