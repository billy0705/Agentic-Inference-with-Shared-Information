import argparse
import csv
import json
import random
import types
from dataclasses import dataclass

import pytest

from multi_agent_sync.evaluation import main as evaluation
from multi_agent_sync.evaluation import chess, gpqa, gsm8k, ma_proofbench, mmlu_pro, olymmath
from multi_agent_sync.evaluation.benchmarks import swe_bench_verified
from multi_agent_sync.evaluation.baselines import multiagent_sync
from multi_agent_sync.evaluation import runner
from multi_agent_sync.evaluation.types import BenchmarkSpec, BenchmarkWorkflowConfig


def test_baselines_package_exports_evaluation_methods():
    from multi_agent_sync.evaluation import baselines

    assert baselines.run_plain_llm is not None
    assert baselines.run_single_agent is not None
    assert baselines.run_majority_vote is not None
    assert baselines.run_multiagent_debate is not None
    assert baselines.run_multiagent is not None


def test_benchmarks_package_exports_benchmark_modules():
    from multi_agent_sync.evaluation import benchmarks

    assert benchmarks.chess.build_benchmark is not None
    assert benchmarks.gpqa.build_benchmark is not None
    assert benchmarks.gsm8k.build_benchmark is not None
    assert benchmarks.ma_proofbench.build_benchmark is not None
    assert benchmarks.mmlu_pro.build_benchmark is not None
    assert benchmarks.olymmath.build_benchmark is not None


@dataclass
class UsageResponse:
    content: str
    usage_metadata: dict[str, int] | None = None
    response_metadata: dict | None = None


class UsageLLM:
    def __init__(self, responses: list[UsageResponse]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> UsageResponse:
        self.prompts.append(prompt)
        return self.responses.pop(0)


class RunnerFakeDockerWorkspace:
    created: list["RunnerFakeDockerWorkspace"] = []

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.cleaned = False
        RunnerFakeDockerWorkspace.created.append(self)

    @classmethod
    async def create(cls, **kwargs):
        return cls()

    async def write_text(self, path: str, content: str) -> None:
        self.files[path] = content

    async def read_text(self, path: str) -> str:
        return self.files[path]

    async def run_bash(self, command: str, *, timeout_seconds=None):
        return types.SimpleNamespace(
            command=command,
            exit_code=0,
            stdout="",
            stderr="",
            timed_out=False,
            container_name="fake",
        )

    async def cleanup(self) -> None:
        self.cleaned = True


def test_parse_methods_accepts_comma_separated_methods():
    assert runner.parse_methods("multiagent_streaming,multiagent_no_streaming,plain_llm") == [
        "multiagent_streaming",
        "multiagent_no_streaming",
        "plain_llm",
    ]


def test_parse_methods_accepts_dynamic_multiagent_methods():
    assert runner.parse_methods("multiagent_dynamic_streaming,multiagent_dynamic_no_streaming,plain_llm") == [
        "multiagent_dynamic_streaming",
        "multiagent_dynamic_no_streaming",
        "plain_llm",
    ]


def test_parse_methods_accepts_single_agent_method():
    assert runner.parse_methods("single_agent,plain_llm") == ["single_agent", "plain_llm"]


def test_parse_methods_accepts_multiagent_debate_method():
    assert runner.parse_methods("multiagent_debate,plain_llm") == ["multiagent_debate", "plain_llm"]


def test_parse_methods_accepts_majority_vote_method():
    assert runner.parse_methods("majority_vote,single_agent,plain_llm") == ["majority_vote", "single_agent", "plain_llm"]


def test_parse_methods_keeps_multiagent_alias_for_streaming():
    assert runner.parse_methods("multiagent,plain_llm") == ["multiagent", "plain_llm"]


def test_parse_methods_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        runner.parse_methods("multiagent,architect")


def test_parser_accepts_simple_benchmark_command_shape():
    args = evaluation.build_parser().parse_args(
        [
            "--benchmark",
            "gpqa",
            "--methods",
            "multiagent_streaming,multiagent_no_streaming,plain_llm",
            "--limit",
            "10",
            "--data-file",
            "gpqa.csv",
        ]
    )

    assert args.benchmark == "gpqa"
    assert args.methods == "multiagent_streaming,multiagent_no_streaming,plain_llm"
    assert args.limit == 10
    assert args.data_file == "gpqa.csv"


def test_parser_accepts_gsm8k_benchmark():
    args = evaluation.build_parser().parse_args(
        [
            "--benchmark",
            "gsm8k",
            "--methods",
            "plain_llm",
            "--limit",
            "10",
        ]
    )

    assert args.benchmark == "gsm8k"
    assert "gsm8k" in evaluation.get_benchmarks()


def test_parser_accepts_chess_benchmark():
    args = evaluation.build_parser().parse_args(["--benchmark", "chess", "--methods", "plain_llm", "--limit", "10"])

    assert args.benchmark == "chess"
    assert "chess" in evaluation.get_benchmarks()


def test_parser_accepts_mmlu_pro_benchmark():
    args = evaluation.build_parser().parse_args(
        [
            "--benchmark",
            "mmlu_pro",
            "--methods",
            "plain_llm",
            "--limit",
            "10",
        ]
    )

    assert args.benchmark == "mmlu_pro"
    assert "mmlu_pro" in evaluation.get_benchmarks()


def test_parser_accepts_ma_proofbench_with_all_levels_and_one_attempt_by_default():
    args = evaluation.build_parser().parse_args(
        [
            "--benchmark",
            "ma_proofbench",
            "--methods",
            "plain_llm",
        ]
    )

    assert args.benchmark == "ma_proofbench"
    assert args.ma_proofbench_level == "all"
    assert args.attempts == 1
    assert args.kimina_host == "127.0.0.1"
    assert args.kimina_port == 8001
    assert "ma_proofbench" in evaluation.get_benchmarks()


def test_parser_accepts_olymmath_benchmarks():
    args = evaluation.build_parser().parse_args(
        [
            "--benchmark",
            "olymmath",
            "--olymmath-subset",
            "en-hard",
            "--methods",
            "plain_llm",
        ]
    )
    lean_args = evaluation.build_parser().parse_args(["--benchmark", "olymmath_lean", "--methods", "plain_llm"])

    assert args.benchmark == "olymmath"
    assert args.olymmath_subset == "en-hard"
    assert lean_args.benchmark == "olymmath_lean"
    assert "olymmath" in evaluation.get_benchmarks()
    assert "olymmath_lean" in evaluation.get_benchmarks()


def test_parser_accepts_swe_bench_verified_benchmark():
    args = evaluation.build_parser().parse_args(
        [
            "--benchmark",
            "swe_bench_verified",
            "--methods",
            "multiagent_streaming",
            "--limit",
            "1",
            "--swebench-agent-workspace",
            "--swebench-run-harness",
            "--swebench-max-workers",
            "2",
        ]
    )

    assert args.benchmark == "swe_bench_verified"
    assert args.swebench_agent_workspace is True
    assert args.swebench_run_harness is True
    assert args.swebench_max_workers == 2
    assert "swe_bench_verified" in evaluation.get_benchmarks()


def test_swe_bench_verified_prompt_omits_gold_patches():
    row = {
        "repo": "astropy/astropy",
        "instance_id": "astropy__astropy-12907",
        "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",
        "problem_statement": "Fix separability_matrix for nested CompoundModels.",
        "hints_text": "Look at separable.py.",
        "patch": "GOLD_PATCH_SHOULD_NOT_APPEAR",
        "test_patch": "GOLD_TEST_PATCH_SHOULD_NOT_APPEAR",
        "FAIL_TO_PASS": '["test_new"]',
        "PASS_TO_PASS": '["test_existing"]',
        "difficulty": "15 min - 1 hour",
    }

    prompt, gold = swe_bench_verified.build_prompt(row, random.Random(0))

    assert gold == "patch_required"
    assert "Fix separability_matrix" in prompt
    assert "astropy/astropy" in prompt
    assert "d16bfe05a744909de4b27f5875fe0d4ed41ce607" in prompt
    assert "Required Docker workflow:" in prompt
    assert "Your first bash action must clone the repository" in prompt
    assert "git clone https://github.com/astropy/astropy.git ." in prompt
    assert "git checkout d16bfe05a744909de4b27f5875fe0d4ed41ce607" in prompt
    assert "modify the actual repository files inside the Docker workspace" in prompt
    assert "Do not write a patch only in your FINAL response" in prompt
    assert "git diff would show a non-empty patch" in prompt
    assert "The system will export the final patch from the Docker repository with git diff" in prompt
    assert "Do not return FINAL until the repository exists in /workspace" in prompt
    assert "GOLD_PATCH_SHOULD_NOT_APPEAR" not in prompt
    assert "GOLD_TEST_PATCH_SHOULD_NOT_APPEAR" not in prompt
    assert "diff --git" in prompt


def test_swe_bench_verified_extracts_prediction_patch_and_scores_prediction_only():
    raw_output = (
        "I changed the implementation.\n\n"
        "```diff\n"
        "diff --git a/pkg/mod.py b/pkg/mod.py\n"
        "--- a/pkg/mod.py\n"
        "+++ b/pkg/mod.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "```\n"
    )
    row = {"instance_id": "repo__repo-1"}

    score = swe_bench_verified.score_response(row, raw_output, argparse.Namespace())

    assert score.pred == "patch_produced"
    assert score.correct is False
    assert score.metadata["instance_id"] == "repo__repo-1"
    assert score.metadata["prediction_only"] is True
    assert score.metadata["model_patch"].startswith("diff --git")


@pytest.mark.asyncio
async def test_swe_bench_verified_multiagent_workspace_exports_git_diff(monkeypatch):
    RunnerFakeDockerWorkspace.created = []
    captured_workflow_kwargs = {}
    bash_commands = []

    async def fake_run_workflow(**kwargs):
        captured_workflow_kwargs.update(kwargs)
        workspace = kwargs["docker_workspace"]
        workspace.files["__git_diff__"] = (
            "diff --git a/pkg/mod.py b/pkg/mod.py\n"
            "--- a/pkg/mod.py\n"
            "+++ b/pkg/mod.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        return {
            "final_answer": "Agent finished.",
            "run_id": "run",
            "mode": "multi_agent",
            "subagent_mode": "fixed",
            "agent_traces": {},
        }

    async def fake_run_bash(self, command, *, timeout_seconds=None):
        bash_commands.append(command)
        if "git -C \"$repo_dir\" diff -- ." in command:
            return types.SimpleNamespace(
                command=command,
                exit_code=0,
                stdout=self.files["__git_diff__"],
                stderr="",
                timed_out=False,
                container_name="fake",
            )
        return types.SimpleNamespace(
            command=command,
            exit_code=0,
            stdout="",
            stderr="",
            timed_out=False,
            container_name="fake",
        )

    monkeypatch.setattr(multiagent_sync, "DockerWorkspace", RunnerFakeDockerWorkspace)
    monkeypatch.setattr(multiagent_sync, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(RunnerFakeDockerWorkspace, "run_bash", fake_run_bash)

    row = {
        "repo": "owner/repo",
        "instance_id": "owner__repo-1",
        "base_commit": "a" * 40,
        "problem_statement": "Fix the bug.",
        "hints_text": "",
        "FAIL_TO_PASS": "[]",
        "PASS_TO_PASS": "[]",
    }
    workflow_config = swe_bench_verified.build_workflow_config(row, argparse.Namespace())
    args = argparse.Namespace(
        max_steps=2,
        total_runtime_timeout=5,
        synthesis_timeout=1,
        workspace_image="python:3.12",
        workspace_command_timeout=30,
        workspace_output_limit=12000,
    )

    raw_output, returncode, trace = await runner.run_multiagent(
        "Fix the bug.",
        UsageLLM([]),
        args,
        workflow_config=workflow_config,
    )

    assert returncode == 0
    assert captured_workflow_kwargs["enable_workspace_tools"] is True
    assert captured_workflow_kwargs["final_guard_tool"].name == "swebench_final_guard"
    assert "diff --git a/pkg/mod.py b/pkg/mod.py" in raw_output
    assert trace["workspace"]["final_candidate_path"] == "__git_diff__"
    assert trace["workspace"]["final_candidate"].startswith("diff --git")
    assert any("No Git repository found in /workspace" in command for command in bash_commands)
    assert all("git clone https://github.com/owner/repo.git" not in command for command in bash_commands)


@pytest.mark.asyncio
async def test_multiagent_workspace_export_error_preserves_trace(monkeypatch):
    RunnerFakeDockerWorkspace.created = []

    async def fake_run_workflow(**kwargs):
        return {
            "final_answer": "Agent finished without cloning.",
            "run_id": "run",
            "mode": "multi_agent",
            "subagent_mode": "fixed",
            "orchestrator_plan": {
                "selected_agents": [{"name": "CodingAgent", "subtask": "Use Docker bash."}],
            },
            "event_log": [],
            "agent_outputs": {"CodingAgent": "No repo created."},
            "agent_traces": {"CodingAgent": {"steps": []}},
        }

    async def failing_exporter(workspace):
        raise RuntimeError("No Git repository found in /workspace.")

    monkeypatch.setattr(multiagent_sync, "DockerWorkspace", RunnerFakeDockerWorkspace)
    monkeypatch.setattr(multiagent_sync, "run_workflow", fake_run_workflow)

    workflow_config = BenchmarkWorkflowConfig(
        final_candidate_path="__git_diff__",
        final_candidate_exporter=failing_exporter,
    )
    args = argparse.Namespace(
        max_steps=2,
        total_runtime_timeout=5,
        synthesis_timeout=1,
        workspace_image="python:3.12",
        workspace_command_timeout=30,
        workspace_output_limit=12000,
    )

    raw_output, returncode, trace = await runner.run_multiagent(
        "Fix the bug.",
        UsageLLM([]),
        args,
        workflow_config=workflow_config,
    )

    assert returncode == 1
    assert "Workspace export failed: No Git repository found in /workspace." in raw_output
    assert trace["agent_outputs"] == {"CodingAgent": "No repo created."}
    assert trace["workspace"]["export_error"] == "No Git repository found in /workspace."


def test_parser_rejects_removed_local_lean_verifier_flags():
    with pytest.raises(SystemExit):
        evaluation.build_parser().parse_args(["--benchmark", "ma_proofbench", "--lean-verifier", "local"])
    with pytest.raises(SystemExit):
        evaluation.build_parser().parse_args(["--benchmark", "ma_proofbench", "--lean-command", "lake env lean"])
    with pytest.raises(SystemExit):
        evaluation.build_parser().parse_args(["--benchmark", "ma_proofbench", "--lean-workdir", "lean_ma_proofbench"])


def test_json_trace_saving_is_enabled_by_default():
    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa"])
    disabled_args = evaluation.build_parser().parse_args(["--benchmark", "gpqa", "--no-save-json-traces"])

    assert args.save_json_traces is True
    assert disabled_args.save_json_traces is False


@pytest.mark.asyncio
async def test_run_evaluation_sets_max_tokens_to_16384(monkeypatch, tmp_path):
    captured_llm_kwargs = {}

    def fake_get_llm(model=None, openai=True, max_tokens=None):
        captured_llm_kwargs["model"] = model
        captured_llm_kwargs["openai"] = openai
        captured_llm_kwargs["max_tokens"] = max_tokens
        return "fake-llm"

    async def fake_run_method(method, prompt, llm, args):
        return runner.RunResult(
            raw_output="Final Answer: A",
            returncode=0,
            elapsed_seconds=0.1,
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        )

    benchmark = BenchmarkSpec(
        name="fake",
        display_name="Fake",
        default_output_filename="fake.csv",
        load_items=lambda args: [
            {
                "Question": "Question?",
                "Correct Answer": "A correct answer",
                "Incorrect Answer 1": "Wrong answer 1",
                "Incorrect Answer 2": "Wrong answer 2",
                "Incorrect Answer 3": "Wrong answer 3",
                "Explanation": "Large explanation that should not be duplicated in traces.",
            }
        ],
        build_prompt=lambda row, rng: ("Question?", "A"),
        extract_answer=lambda text: "A",
    )

    monkeypatch.setattr(evaluation, "get_benchmarks", lambda: {"fake": benchmark})
    monkeypatch.setattr(evaluation, "get_llm", fake_get_llm)
    monkeypatch.setattr(runner, "run_method", fake_run_method)
    monkeypatch.setattr(runner, "progress", lambda items, desc: items)
    monkeypatch.setattr(runner, "resolve_auto_openai_model_name", lambda: "openai/gpt-oss-120b")

    args = evaluation.build_parser().parse_args(
        ["--benchmark", "fake", "--methods", "plain_llm", "--output", str(tmp_path / "results.csv")]
    )

    await evaluation.run_evaluation(args)

    assert captured_llm_kwargs == {"model": "openai/gpt-oss-120b", "openai": True, "max_tokens": 16384}


@pytest.mark.asyncio
async def test_run_evaluation_writes_swebench_predictions_jsonl(monkeypatch, tmp_path):
    benchmark = BenchmarkSpec(
        name="swe_bench_verified",
        display_name="SWE-bench Verified",
        default_output_filename="swe_bench_verified_results.csv",
        load_items=lambda args: [
            {
                "repo": "owner/repo",
                "instance_id": "owner__repo-1",
                "base_commit": "a" * 40,
                "problem_statement": "Fix the bug.",
            }
        ],
        build_prompt=lambda row, rng: ("Fix the bug.", "patch_required"),
        extract_answer=swe_bench_verified.extract_answer,
        score_response=swe_bench_verified.score_response,
    )

    async def fake_run_method(method, prompt, llm, args, **kwargs):
        return runner.RunResult(
            raw_output="```diff\ndiff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n```",
            returncode=0,
            elapsed_seconds=0.1,
            trace={},
        )

    monkeypatch.setattr(evaluation, "get_benchmarks", lambda: {"swe_bench_verified": benchmark})
    monkeypatch.setattr(evaluation, "get_llm", lambda model=None, openai=True, max_tokens=None: "fake-llm")
    monkeypatch.setattr(runner, "run_method", fake_run_method)
    monkeypatch.setattr(runner, "progress", lambda items, desc: items)
    monkeypatch.setattr(runner, "resolve_auto_openai_model_name", lambda: "fake-model")

    args = evaluation.build_parser().parse_args(
        [
            "--benchmark",
            "swe_bench_verified",
            "--methods",
            "multiagent_streaming",
            "--output-dir",
            str(tmp_path),
        ]
    )

    await evaluation.run_evaluation(args)

    prediction_paths = list(tmp_path.rglob("predictions_multiagent_streaming.jsonl"))
    assert len(prediction_paths) == 1
    prediction = json.loads(prediction_paths[0].read_text(encoding="utf-8").splitlines()[0])
    assert prediction["instance_id"] == "owner__repo-1"
    assert prediction["model_name_or_path"] == "fake-model"
    assert prediction["model_patch"].startswith("diff --git")
    assert prediction["model_patch"].endswith("\n")


def test_apply_swebench_harness_results_updates_official_status(tmp_path):
    trace_path = tmp_path / "example.json"
    trace_path.write_text(
        json.dumps(
            {
                "pred": "patch_produced",
                "correct": False,
                "returncode": 0,
                "error": "Prediction patch produced; official SWE-bench harness evaluation was not run.",
                "score_metadata": {"official_evaluation": "not_run"},
            }
        ),
        encoding="utf-8",
    )
    results = [
        {
            "benchmark": "swe_bench_verified",
            "method": "multiagent_streaming",
            "pred": "patch_produced",
            "correct": False,
            "returncode": 0,
            "error": "Prediction patch produced; official SWE-bench harness evaluation was not run.",
            "score_metadata": {
                "instance_id": "owner__repo-1",
                "official_evaluation": "not_run",
            },
            "json_trace_path": str(trace_path),
        }
    ]
    report = {
        "resolved_ids": [],
        "unresolved_ids": [],
        "empty_patch_ids": [],
        "error_ids": ["owner__repo-1"],
    }

    evaluation.apply_swebench_harness_results(
        results,
        method="multiagent_streaming",
        report=report,
        report_path=tmp_path / "report.json",
        artifact_path=tmp_path / "harness.json",
        stdout="owner__repo-1: >>>>> Patch Apply Failed:\nmalformed patch\n\nAll instances run.",
    )

    result = results[0]
    assert result["correct"] is False
    assert result["returncode"] == 1
    assert result["score_metadata"]["official_evaluation"] == "error"
    assert "Patch Apply Failed" in result["error"]
    updated_trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert updated_trace["score_metadata"]["official_evaluation"] == "error"


@pytest.mark.asyncio
async def test_lean_multiagent_workspace_config_seeds_and_returns_final_file(monkeypatch):
    RunnerFakeDockerWorkspace.created = []
    captured_workflow_kwargs = {}

    async def fake_run_workflow(**kwargs):
        captured_workflow_kwargs.update(kwargs)
        workspace = kwargs["docker_workspace"]
        await workspace.write_text(
            "/workspace/Main.lean",
            "import Mathlib\n\ntheorem t : True := by\n  trivial\n",
        )
        return {
            "final_answer": "Agent finished.",
            "run_id": "run",
            "mode": "multi_agent",
            "subagent_mode": "fixed",
            "agent_traces": {},
        }

    monkeypatch.setattr(multiagent_sync, "DockerWorkspace", RunnerFakeDockerWorkspace)
    monkeypatch.setattr(multiagent_sync, "run_workflow", fake_run_workflow)

    row = {
        "id": 1,
        "split": "level1",
        "informal_statement": "Show true.",
        "formal_statement": "import Mathlib\n\ntheorem t : True := by\n  sorry",
        "header": "import Mathlib",
        "topic": "Logic",
        "tag": "Basic",
        "version": "4.28.0",
    }
    workflow_config = ma_proofbench.build_lean_workflow_config(row, argparse.Namespace())
    args = argparse.Namespace(
        max_steps=2,
        total_runtime_timeout=5,
        synthesis_timeout=1,
        workspace_image="python:3.12",
        workspace_command_timeout=30,
        workspace_output_limit=12000,
    )

    raw_output, returncode, trace = await runner.run_multiagent(
        "Complete the Lean proof.",
        UsageLLM([]),
        args,
        workflow_config=workflow_config,
    )

    assert returncode == 0
    assert captured_workflow_kwargs["enable_workspace_tools"] is True
    assert captured_workflow_kwargs["feedback_tool"] is not None
    assert RunnerFakeDockerWorkspace.created[0].files["/workspace/Main.lean"].endswith("trivial\n")
    assert RunnerFakeDockerWorkspace.created[0].cleaned is True
    assert "```lean4\nimport Mathlib\n\ntheorem t : True := by\n  trivial\n```" in raw_output
    assert trace["workspace"]["final_candidate_path"] == "/workspace/Main.lean"


@pytest.mark.asyncio
async def test_single_agent_stops_when_final_answer_is_parseable():
    llm = UsageLLM(
        [
            UsageResponse('{"status": "continue", "final_answer": "", "notes": "Need to inspect the board."}'),
            UsageResponse('{"status": "final", "final_answer": "Final Answer: h3", "notes": "Legal destination found."}'),
        ]
    )
    args = argparse.Namespace(max_steps=3, answer_extractor=lambda text: "h3" if "h3" in text else None)

    result = await runner.run_method("single_agent", "Complete the chess move.", llm, args)

    assert result.raw_output == "Final Answer: h3"
    assert len(result.trace["steps"]) == 2
    assert result.trace["stopped_reason"] == "final_answer_parseable"
    assert result.trace["steps"][1]["parsed_answer"] == "h3"


@pytest.mark.asyncio
async def test_single_agent_respects_max_steps_when_final_answer_is_not_parseable():
    llm = UsageLLM(
        [
            UsageResponse('{"status": "continue", "final_answer": "", "notes": "Still working."}'),
            UsageResponse('{"status": "final", "final_answer": "Final Answer: i9", "notes": "Invalid square."}'),
        ]
    )
    args = argparse.Namespace(max_steps=2, answer_extractor=lambda text: None)

    result = await runner.run_method("single_agent", "Complete the chess move.", llm, args)

    assert result.raw_output == "Final Answer: i9"
    assert len(result.trace["steps"]) == 2
    assert result.trace["stopped_reason"] == "max_steps"
    assert result.trace["steps"][1]["status"] == "final"
    assert result.trace["steps"][1]["parsed_answer"] is None


@pytest.mark.asyncio
async def test_majority_vote_runs_three_independent_single_agents_and_votes():
    llm = UsageLLM(
        [
            UsageResponse('{"status": "final", "final_answer": "Final Answer: A", "notes": "First vote."}'),
            UsageResponse('{"status": "final", "final_answer": "Final Answer: B", "notes": "Second vote."}'),
            UsageResponse('{"status": "final", "final_answer": "Final Answer: B", "notes": "Third vote."}'),
        ]
    )
    args = argparse.Namespace(
        max_steps=1,
        answer_extractor=lambda text: text.rsplit("Final Answer:", 1)[-1].strip()[:1] if "Final Answer:" in text else None,
    )

    result = await runner.run_method("majority_vote", "Question with options.", llm, args)

    assert result.raw_output == "Final Answer: B"
    assert len(llm.prompts) == 3
    assert all("Question with options." in prompt for prompt in llm.prompts)
    assert result.trace["method"] == "majority_vote"
    assert result.trace["agents"] == 3
    assert result.trace["parsed_answers"] == ["A", "B", "B"]
    assert result.trace["voted_answer"] == "B"
    assert result.trace["fallback_used"] is False
    assert len(result.trace["agent_runs"]) == 3
    assert all(agent_run["trace"]["method"] == "single_agent" for agent_run in result.trace["agent_runs"])
    debug = runner.build_multiagent_debug(result.trace)
    assert [step["node"] for step in debug["workflow"]] == ["single_agent_votes", "majority_vote"]


@pytest.mark.asyncio
async def test_majority_vote_tie_uses_first_parsed_answer():
    llm = UsageLLM(
        [
            UsageResponse('{"status": "final", "final_answer": "Final Answer: C", "notes": ""}'),
            UsageResponse('{"status": "final", "final_answer": "Final Answer: B", "notes": ""}'),
            UsageResponse('{"status": "final", "final_answer": "Final Answer: A", "notes": ""}'),
        ]
    )
    args = argparse.Namespace(
        max_steps=1,
        answer_extractor=lambda text: text.rsplit("Final Answer:", 1)[-1].strip()[:1] if "Final Answer:" in text else None,
    )

    result = await runner.run_method("majority_vote", "Question with options.", llm, args)

    assert result.raw_output == "Final Answer: C"
    assert result.trace["parsed_answers"] == ["C", "B", "A"]
    assert result.trace["voted_answer"] == "C"
    assert result.trace["fallback_used"] is False


@pytest.mark.asyncio
async def test_majority_vote_falls_back_to_first_raw_output_when_all_answers_are_unparseable():
    llm = UsageLLM(
        [
            UsageResponse('{"status": "continue", "final_answer": "Unclear first answer", "notes": ""}'),
            UsageResponse('{"status": "continue", "final_answer": "Unclear second answer", "notes": ""}'),
            UsageResponse('{"status": "continue", "final_answer": "Unclear third answer", "notes": ""}'),
        ]
    )
    args = argparse.Namespace(max_steps=1, answer_extractor=lambda text: None)

    result = await runner.run_method("majority_vote", "Question with options.", llm, args)

    assert result.raw_output == "Unclear first answer"
    assert result.trace["parsed_answers"] == [None, None, None]
    assert result.trace["voted_answer"] is None
    assert result.trace["fallback_used"] is True


@pytest.mark.asyncio
async def test_multiagent_debate_uses_three_agents_two_rounds_and_selects_final_answer():
    llm = UsageLLM(
        [
            UsageResponse("Agent 1 round 1 says Final Answer: A"),
            UsageResponse("Agent 2 round 1 says Final Answer: B"),
            UsageResponse("Agent 3 round 1 says Final Answer: B"),
            UsageResponse("Agent 1 round 2 says Final Answer: B"),
            UsageResponse("Agent 2 round 2 says Final Answer: B"),
            UsageResponse("Agent 3 round 2 says Final Answer: C"),
        ]
    )
    args = argparse.Namespace(
        benchmark="mmlu_pro",
        answer_extractor=lambda text: text.rsplit("Final Answer:", 1)[-1].strip()[:1] if "Final Answer:" in text else None,
    )

    result = await runner.run_method("multiagent_debate", "Question with options.", llm, args)

    assert result.raw_output == "Final Answer: B"
    assert len(llm.prompts) == 6
    assert all("Question with options." in prompt for prompt in llm.prompts[:3])
    assert "These are the solutions to the problem from other agents:" in llm.prompts[3]
    assert "Agent 2 round 1 says Final Answer: B" in llm.prompts[3]
    assert "Agent 3 round 1 says Final Answer: B" in llm.prompts[3]
    assert "Agent 1 round 1 says Final Answer: A" in llm.prompts[3]
    assert result.trace["method"] == "multiagent_debate"
    assert result.trace["agents"] == 3
    assert result.trace["rounds"] == 2
    assert result.trace["parsed_final_answers"] == ["B", "B", "C"]
    assert result.trace["majority_answer"] == "B"
    assert len(result.trace["agent_contexts"]) == 3
    assert len(result.trace["agent_contexts"][0]) == 4
    debug = runner.build_multiagent_debug(result.trace)
    assert [step["node"] for step in debug["workflow"]] == ["debate_agents", "debate_answer_selection"]


@pytest.mark.asyncio
async def test_multiagent_debate_math_prompt_matches_upstream_shape():
    llm = UsageLLM(
        [
            UsageResponse(r"Agent 1 round 1 \boxed{1}"),
            UsageResponse(r"Agent 2 round 1 \boxed{2}"),
            UsageResponse(r"Agent 3 round 1 \boxed{2}"),
            UsageResponse(r"Agent 1 round 2 \boxed{2}"),
            UsageResponse(r"Agent 2 round 2 \boxed{2}"),
            UsageResponse(r"Agent 3 round 2 \boxed{3}"),
        ]
    )
    args = argparse.Namespace(
        benchmark="gsm8k",
        answer_extractor=gsm8k.extract_answer,
    )

    result = await runner.run_method("multiagent_debate", "What is 1 + 1?", llm, args)

    assert "Can you solve the following math problem? What is 1 + 1? Explain your reasoning." in llm.prompts[0]
    assert r"Your final answer should be a single numerical number, in the form \boxed{answer}" in llm.prompts[0]
    assert "Can you double check that your answer is correct." not in llm.prompts[0]
    assert "Using the solutions from other agents as additional information" in llm.prompts[3]
    assert "The original math problem is What is 1 + 1?." in llm.prompts[3]
    assert result.raw_output == "Final Answer: 2"


@pytest.mark.asyncio
async def test_run_evaluation_writes_unique_csv_and_json_trace_by_default(monkeypatch, tmp_path):
    benchmark = BenchmarkSpec(
        name="fake",
        display_name="Fake",
        default_output_filename="fake.csv",
        load_items=lambda args: [
            {
                "Question": "Question?",
                "Correct Answer": "A correct answer",
                "Incorrect Answer 1": "Wrong answer 1",
                "Incorrect Answer 2": "Wrong answer 2",
                "Incorrect Answer 3": "Wrong answer 3",
                "Explanation": "Large explanation that should not be duplicated in traces.",
            }
        ],
        build_prompt=lambda row, rng: ("Question?", "A"),
        extract_answer=lambda text: "A",
    )

    async def fake_run_method(method, prompt, llm, args):
        return runner.RunResult(
            raw_output="Final Answer: A",
            returncode=0,
            elapsed_seconds=0.1,
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
            trace={
                "final_answer": "Final Answer: A",
                "orchestrator_plan": {"selected_agents": [{"name": "SolverAgent"}]},
                "event_log": [
                    {
                        "source": "SolverAgent",
                        "target": "broadcast",
                        "event_type": "finding",
                        "content": "Solver shared a useful finding.",
                        "confidence": 0.8,
                    }
                ],
                "agent_traces": {"SolverAgent": {"steps": [{"used_event_ids": ["event-1"]}]}},
            },
        )

    monkeypatch.setattr(evaluation, "get_benchmarks", lambda: {"fake": benchmark})
    monkeypatch.setattr(evaluation, "get_llm", lambda model=None, openai=True, max_tokens=None: "fake-llm")
    monkeypatch.setattr(runner, "run_method", fake_run_method)
    monkeypatch.setattr(runner, "progress", lambda items, desc: items)
    monkeypatch.setattr(runner, "create_run_id", lambda: "run-test", raising=False)
    monkeypatch.setattr(runner, "resolve_auto_openai_model_name", lambda: "openai/gpt-oss-120b")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    args = evaluation.build_parser().parse_args(["--benchmark", "fake", "--methods", "plain_llm", "--output-dir", str(tmp_path)])

    results = await evaluation.run_evaluation(args)

    run_root = tmp_path / "fake" / "gpt-oss-120b" / "run-test"
    output_path = run_root / "fake.csv"
    trace_path = run_root / "examples" / "0000_plain_llm.json"
    run_config_path = run_root / "run_config.json"
    summary_path = run_root / "summary.json"

    assert output_path.exists()
    assert trace_path.exists()
    assert run_config_path.exists()
    assert summary_path.exists()
    assert results[0]["json_trace_path"] == str(trace_path)

    csv_rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))
    assert csv_rows[0]["json_trace_path"] == str(trace_path)

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["method"] == "plain_llm"
    assert trace["settings"]["save_json_traces"] is True
    assert trace["settings"]["resolved_model"] == "openai/gpt-oss-120b"
    assert "row" not in trace
    assert trace["question"] == "Question?"
    assert trace["options"] == {
        "correct": "A correct answer",
        "incorrect": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"],
    }
    assert trace["method_trace"]["orchestrator_plan"]["selected_agents"][0]["name"] == "SolverAgent"
    assert trace["multiagent_debug"]["subagents"] == [{"name": "SolverAgent"}]
    assert trace["multiagent_debug"]["messages"][0]["content"] == "Solver shared a useful finding."
    assert trace["multiagent_debug"]["workflow"] == [
        {"node": "orchestrator", "description": "Created plan and selected subagents."},
        {"node": "SolverAgent", "description": "Ran subagent and published/received messages."},
        {"node": "synthesizer", "description": "Combined subagent outputs and event log into final answer."},
    ]

    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    assert run_config["settings"]["resolved_model"] == "openai/gpt-oss-120b"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["summary"]["plain_llm"]["avg_prompt_tokens"] == 1.0
    assert summary["summary"]["plain_llm"]["avg_completion_tokens"] == 2.0
    assert summary["summary"]["plain_llm"]["avg_total_tokens"] == 3.0
    assert summary["method_averages"]["plain_llm"] == {
        "correct_avg": 1.0,
        "invalid_avg": 0.0,
        "time_avg_seconds": 0.1,
        "total_token_avg": 3.0,
    }


@pytest.mark.asyncio
async def test_run_evaluation_updates_results_summary_and_correctness_matrix_incrementally(monkeypatch, tmp_path):
    benchmark = BenchmarkSpec(
        name="fake",
        display_name="Fake",
        default_output_filename="fake.csv",
        load_items=lambda args: [
            {"Question": "Question 0?", "Correct Answer": "A", "Incorrect Answer 1": "B", "Incorrect Answer 2": "C", "Incorrect Answer 3": "D"},
            {"Question": "Question 1?", "Correct Answer": "A", "Incorrect Answer 1": "B", "Incorrect Answer 2": "C", "Incorrect Answer 3": "D"},
        ],
        build_prompt=lambda row, rng: (row["Question"], "A"),
        extract_answer=lambda text: text.removeprefix("Final Answer: ").strip(),
    )
    calls = []

    async def fake_run_method(method, prompt, llm, args):
        calls.append((method, prompt))
        run_root = tmp_path / "fake" / "gpt-oss-120b" / "run-test"
        output_path = run_root / "fake.csv"
        summary_path = run_root / "summary.json"
        matrix_path = run_root / "correctness.csv"
        if len(calls) == 2:
            assert output_path.exists()
            assert summary_path.exists()
            assert matrix_path.exists()
            csv_rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))
            matrix_rows = list(csv.DictReader(matrix_path.open(newline="", encoding="utf-8")))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            assert len(csv_rows) == 1
            assert matrix_rows == [{"task_id": "0", "plain_llm": "T", "multiagent_streaming": ""}]
            assert summary["method_averages"]["plain_llm"]["correct_avg"] == 1.0
        pred = "A" if method == "plain_llm" else "B"
        return runner.RunResult(
            raw_output=f"Final Answer: {pred}",
            returncode=0,
            elapsed_seconds=0.1,
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            trace={"prompt": prompt, "raw_output": f"Final Answer: {pred}"},
        )

    monkeypatch.setattr(evaluation, "get_benchmarks", lambda: {"fake": benchmark})
    monkeypatch.setattr(evaluation, "get_llm", lambda model=None, openai=True, max_tokens=None: "fake-llm")
    monkeypatch.setattr(runner, "run_method", fake_run_method)
    monkeypatch.setattr(runner, "progress", lambda items, desc: items)
    monkeypatch.setattr(runner, "create_run_id", lambda: "run-test", raising=False)
    monkeypatch.setattr(runner, "resolve_auto_openai_model_name", lambda: "openai/gpt-oss-120b")

    args = evaluation.build_parser().parse_args(
        [
            "--benchmark",
            "fake",
            "--methods",
            "plain_llm,multiagent_streaming",
            "--output-dir",
            str(tmp_path),
        ]
    )

    await evaluation.run_evaluation(args)

    matrix_rows = list(csv.DictReader((tmp_path / "fake" / "gpt-oss-120b" / "run-test" / "correctness.csv").open(newline="", encoding="utf-8")))
    assert matrix_rows == [
        {"task_id": "0", "plain_llm": "T", "multiagent_streaming": "F"},
        {"task_id": "1", "plain_llm": "T", "multiagent_streaming": "F"},
    ]


def test_default_output_path_uses_run_id_to_avoid_overwriting():
    benchmark = gpqa.build_benchmark()
    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa", "--model", "openai/gpt-oss-120b"])

    assert runner.resolve_output_path(benchmark, args, run_id="run-123") == (
        runner.DEFAULT_OUTPUT_DIR / "gpqa" / "gpt-oss-120b" / "run-123" / "gpqa_diamond_results.csv"
    )


def test_gsm8k_default_output_path_uses_run_id_to_avoid_overwriting():
    benchmark = gsm8k.build_benchmark()
    args = evaluation.build_parser().parse_args(["--benchmark", "gsm8k", "--model", "openai/gpt-oss-120b"])

    assert runner.resolve_output_path(benchmark, args, run_id="run-123") == (
        runner.DEFAULT_OUTPUT_DIR / "gsm8k" / "gpt-oss-120b" / "run-123" / "gsm8k_results.csv"
    )


def test_chess_default_output_path_uses_run_id_to_avoid_overwriting():
    benchmark = chess.build_benchmark()
    args = evaluation.build_parser().parse_args(["--benchmark", "chess", "--model", "openai/gpt-oss-120b"])

    assert runner.resolve_output_path(benchmark, args, run_id="run-123") == (
        runner.DEFAULT_OUTPUT_DIR / "chess" / "gpt-oss-120b" / "run-123" / "chess_results.csv"
    )


def test_mmlu_pro_default_output_path_uses_run_id_to_avoid_overwriting():
    benchmark = mmlu_pro.build_benchmark()
    args = evaluation.build_parser().parse_args(["--benchmark", "mmlu_pro", "--model", "openai/gpt-oss-120b"])

    assert runner.resolve_output_path(benchmark, args, run_id="run-123") == (
        runner.DEFAULT_OUTPUT_DIR / "mmlu_pro" / "gpt-oss-120b" / "run-123" / "mmlu_pro_results.csv"
    )


def test_olymmath_default_output_paths_use_run_id_to_avoid_overwriting():
    benchmark = olymmath.build_benchmark()
    lean_benchmark = olymmath.build_lean_benchmark()
    args = evaluation.build_parser().parse_args(["--benchmark", "olymmath", "--model", "openai/gpt-oss-120b"])
    lean_args = evaluation.build_parser().parse_args(["--benchmark", "olymmath_lean", "--model", "openai/gpt-oss-120b"])

    assert runner.resolve_output_path(benchmark, args, run_id="run-123") == (
        runner.DEFAULT_OUTPUT_DIR / "olymmath" / "gpt-oss-120b" / "run-123" / "olymmath_results.csv"
    )
    assert runner.resolve_output_path(lean_benchmark, lean_args, run_id="run-123") == (
        runner.DEFAULT_OUTPUT_DIR / "olymmath_lean" / "gpt-oss-120b" / "run-123" / "olymmath_lean_results.csv"
    )


def test_output_filename_is_written_inside_output_folder():
    benchmark = gpqa.build_benchmark()
    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa", "--model", "openai/gpt-oss-120b", "--output", "result.csv"])

    assert runner.resolve_output_path(benchmark, args, run_id="run-123") == (
        runner.DEFAULT_OUTPUT_DIR / "gpqa" / "gpt-oss-120b" / "run-123" / "result.csv"
    )


def test_model_output_folder_uses_last_provider_path_component():
    benchmark = ma_proofbench.build_benchmark()
    args = evaluation.build_parser().parse_args(
        ["--benchmark", "ma_proofbench", "--model", "google/gemma-4-26B-A4B-it"]
    )

    assert runner.resolve_output_path(benchmark, args, run_id="run-123") == (
        runner.DEFAULT_OUTPUT_DIR / "ma_proofbench" / "gemma-4-26B-A4B-it" / "run-123" / "ma_proofbench_results.csv"
    )


def test_auto_model_resolution_uses_first_api_model(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"data": [{"id": "google/gemma-4-26B-A4B-it"}]}).encode("utf-8")

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(runner, "urlopen", fake_urlopen)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_MODEL_LOOKUP_TIMEOUT", "0.5")

    assert runner.resolve_auto_openai_model_name() == "google/gemma-4-26B-A4B-it"
    assert captured == {"url": "http://localhost:8000/v1/models", "timeout": 0.5}


def test_auto_model_resolution_falls_back_when_api_lookup_fails(monkeypatch, capsys):
    def failing_urlopen(request, timeout):
        raise OSError("server unavailable")

    monkeypatch.setattr(runner, "urlopen", failing_urlopen)
    monkeypatch.setenv("OLLAMA_MODEL", "local/fallback-model")

    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa"])

    assert runner.resolve_model_name(args) == "local/fallback-model"
    assert args.local_model is True
    assert "using local model: local/fallback-model" in capsys.readouterr().out


def test_gpqa_owns_answer_extraction():
    benchmark = gpqa.build_benchmark()

    assert benchmark.extract_answer("Final Answer: C") == "C"


def test_gsm8k_build_prompt_extracts_gold_answer_from_dataset_rationale():
    prompt, gold = gsm8k.build_prompt(
        {
            "question": "Weng earns $12 an hour for babysitting. Yesterday, she did 50 minutes. How much did she earn?",
            "answer": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. #### $10.00",
        },
        random.Random(0),
    )

    assert gold == "10"
    assert "Weng earns $12 an hour" in prompt
    assert "Final Answer: <number>" in prompt


def test_chess_build_prompt_defines_square_output_and_uses_first_target_as_gold():
    prompt, gold = chess.build_prompt(
        {
            "input": "g2g3 f7f5 f1",
            "target": ["h3", "g2"],
        },
        random.Random(0),
    )

    assert gold == "h3"
    assert "g2g3 f7f5 f1" in prompt
    assert "Final Answer: <square>" in prompt
    assert "[a-h][1-8]" in prompt


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ("Final Answer: h3", "h3"),
        ("Answer: G2", "g2"),
        ("The destination square is b8.", "b8"),
        ("h3", "h3"),
    ],
)
def test_chess_extract_answer_returns_normalized_square(raw_output, expected):
    benchmark = chess.build_benchmark()

    assert benchmark.extract_answer(raw_output) == expected


@pytest.mark.parametrize("raw_output", ["Final Answer: i9", "castle kingside", "f1g2"])
def test_chess_extract_answer_rejects_invalid_or_ambiguous_output(raw_output):
    benchmark = chess.build_benchmark()

    assert benchmark.extract_answer(raw_output) is None


def test_chess_score_response_accepts_any_target_square():
    score = chess.score_response(
        {
            "input": "g2g3 f7f5 f1",
            "target": ["h3", "g2"],
        },
        "Final Answer: g2",
        argparse.Namespace(),
    )

    assert score.correct is True
    assert score.pred == "g2"
    assert score.metadata == {
        "output_regex": "[a-h][1-8]",
        "valid_targets": ["h3", "g2"],
    }


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ("Final Answer: 10", "10"),
        ("Final answer: $10.00", "10"),
        ("The answer is 10.0.", "10"),
        ("#### 1,234", "1234"),
        ("Reasoning mentions 3 and 7. Final Answer: 10", "10"),
    ],
)
def test_gsm8k_extract_answer_normalizes_equivalent_numbers(raw_output, expected):
    benchmark = gsm8k.build_benchmark()

    assert benchmark.extract_answer(raw_output) == expected


def test_gsm8k_extract_answer_uses_last_number_as_fallback():
    benchmark = gsm8k.build_benchmark()

    assert benchmark.extract_answer("First compute 12 / 6 = 2. Then add 8 to get 10.") == "10"


def test_gsm8k_question_context_uses_lowercase_dataset_fields():
    context = runner.build_question_context(
        {
            "question": "How many clips did Natalia sell?",
            "answer": "Natalia sold 48+24 = <<48+24=72>>72 clips. #### 72",
        }
    )

    assert context == {
        "question": "How many clips did Natalia sell?",
        "options": {},
        "gold_answer": "72",
    }


def test_mmlu_pro_build_prompt_preserves_dataset_option_order_and_gold_letter():
    prompt, gold = mmlu_pro.build_prompt(
        {
            "question": "Which statement is true?",
            "options": ["Option A", "Option B", "Option C", "Option D", "Option E", "Option F", "Option G", "Option H", "Option I", "Option J"],
            "answer": "H",
            "answer_index": 7,
            "category": "math",
        },
        random.Random(0),
    )

    assert gold == "H"
    assert "A. Option A" in prompt
    assert "H. Option H" in prompt
    assert "J. Option J" in prompt
    assert "Final Answer: <A/B/C/D/E/F/G/H/I/J>" in prompt


def test_mmlu_pro_build_prompt_accepts_test_rows_with_fewer_than_ten_options():
    prompt, gold = mmlu_pro.build_prompt(
        {
            "question": "Which safety label is correct?",
            "options": [
                "Safe practices, Fear, Jealousy, Trivial",
                "Unsafe practices, Distress, Joy, Trivial",
                "Safe practices, Wants, Jealousy, Trivial",
                "Safe practices, Distress, Fear, Trivial",
                "Unsafe practices, Wants, Jealousy, Serious",
                "Safe practices, Distress, Jealousy, Serious",
                "Safe practices, Wants, Fear, Serious",
                "Unsafe practices, Wants, Fear, Trivial",
                "Unsafe practices, Distress, Fear, Serious",
            ],
            "answer": "I",
            "answer_index": 8,
        },
        random.Random(0),
    )

    assert gold == "I"
    assert "I. Unsafe practices, Distress, Fear, Serious" in prompt
    assert "J." not in prompt
    assert "Final Answer: <A/B/C/D/E/F/G/H/I>" in prompt


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ("Final Answer: H", "H"),
        ("Final answer: (H)", "H"),
        ("The answer is option H.", "H"),
        ("Answer: j", "J"),
        ("(C)", "C"),
    ],
)
def test_mmlu_pro_extract_answer_accepts_a_through_j(raw_output, expected):
    benchmark = mmlu_pro.build_benchmark()

    assert benchmark.extract_answer(raw_output) == expected


def test_mmlu_pro_extract_answer_rejects_invalid_or_ambiguous_text():
    benchmark = mmlu_pro.build_benchmark()

    assert benchmark.extract_answer("Final Answer: K") is None
    assert benchmark.extract_answer("I do not know.") is None


def test_mmlu_pro_question_context_uses_options_list():
    context = runner.build_question_context(
        {
            "question": "Which option is correct?",
            "options": ["First", "Second", "Third"],
            "answer": "B",
        }
    )

    assert context == {
        "question": "Which option is correct?",
        "options": {"A": "First", "B": "Second", "C": "Third"},
        "gold_answer": "B",
    }


def test_olymmath_build_prompt_uses_problem_answer_and_subject():
    prompt, gold = olymmath.build_prompt(
        {
            "problem": "Calculate $\\sqrt{9+8\\cos 20^{\\circ }}-\\sec 20^{\\circ }$.",
            "answer": "3",
            "subject": "Algebra",
            "unique_id": "OlymMATH-EASY-2-EN",
        },
        random.Random(0),
    )

    assert gold == "3"
    assert "Subject: Algebra" in prompt
    assert "Calculate $\\sqrt" in prompt
    assert "Final Answer: <answer>" in prompt


@pytest.mark.parametrize(
    ("raw_output", "expected"),
    [
        ("Final Answer: 948", "948"),
        ("The result is clear.\n\\boxed{\\frac{1}{2}}", "1/2"),
        ("Final Answer: 2\\sqrt{10}", "2*sqrt(10)"),
        ("Answer: \\left[ -\\frac{1}{8}, \\frac{1}{8} \\right]", "[-1/8,1/8]"),
    ],
)
def test_olymmath_extract_answer_normalizes_latex_answers(raw_output, expected):
    benchmark = olymmath.build_benchmark()

    assert benchmark.extract_answer(raw_output) == expected


def test_olymmath_score_response_uses_normalized_answer_match():
    score = olymmath.score_response(
        {
            "problem": "Find a value.",
            "answer": "\\frac{1}{2}",
            "subject": "Algebra",
            "unique_id": "OlymMATH-EASY-0-EN",
        },
        "Final Answer: 1/2",
        argparse.Namespace(),
    )

    assert score.correct is True
    assert score.pred == "1/2"
    assert score.metadata["unique_id"] == "OlymMATH-EASY-0-EN"


@pytest.mark.parametrize(
    ("gold", "raw_output"),
    [
        ("\\frac{18}{5}", "Final Answer: 3.6"),
        ("2^{17}", "Final Answer: 131072"),
        ("2\\sqrt{10}", "Final Answer: 6.324555320336759"),
    ],
)
def test_olymmath_score_response_accepts_common_numeric_equivalents(gold, raw_output):
    score = olymmath.score_response(
        {
            "problem": "Find a value.",
            "answer": gold,
            "subject": "Algebra",
            "unique_id": "OlymMATH-EASY-0-EN",
        },
        raw_output,
        argparse.Namespace(),
    )

    assert score.correct is True


def test_olymmath_question_context_uses_problem_and_metadata():
    context = runner.build_question_context(
        {
            "problem": "Find the number of sequences.",
            "answer": "948",
            "subject": "Combinatorics",
            "unique_id": "OlymMATH-EASY-0-EN",
        }
    )

    assert context == {
        "question": "Find the number of sequences.",
        "options": {},
        "unique_id": "OlymMATH-EASY-0-EN",
        "subject": "Combinatorics",
    }


def test_load_local_olymmath_rows_validates_natural_language_rows(tmp_path):
    path = tmp_path / "olymmath.jsonl"
    rows = [
        {
            "problem": "First problem.",
            "answer": "1",
            "subject": "Algebra",
            "unique_id": "OlymMATH-EASY-0-EN",
        },
        {
            "problem": "Second problem.",
            "answer": "2",
            "subject": "Geometry",
            "unique_id": "OlymMATH-EASY-1-EN",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    loaded = olymmath.load_local_rows(path, subset="en-easy", limit=1)

    assert loaded == rows[:1]


def test_olymmath_lean_build_prompt_uses_formal_statement_raw():
    prompt, gold = olymmath.build_lean_prompt(
        {
            "unique_id": "OlymMATH-LEAN-0",
            "subject": "Algebra",
            "formal_statement": "import Mathlib\n\ntheorem to_prove : False := by\n  sorry",
            "formal_statement_raw": "import Mathlib\n\ntheorem to_prove : True := by\n  sorry",
            "en_informal": "Prove true.",
        },
        random.Random(0),
    )

    assert gold == "lean_verifies"
    assert "Prove true." in prompt
    assert "theorem to_prove : True" in prompt
    assert "theorem to_prove : False" not in prompt
    assert "theorem to_prove : True := by" in prompt
    assert "sorry" not in prompt.lower()
    assert "omitted-proof placeholders" in prompt
    assert "Output only the Lean code block" in prompt
    assert "```lean4" in prompt


def test_olymmath_lean_scores_verifier_success(monkeypatch):
    row = {
        "unique_id": "OlymMATH-LEAN-0",
        "subject": "Algebra",
        "formal_statement": "import Mathlib\n\ntheorem to_prove : True := by\n  sorry",
        "formal_statement_raw": "import Mathlib\n\ntheorem to_prove : True := by\n  sorry",
        "en_informal": "Prove true.",
    }
    output = "```lean4\ntheorem to_prove : True := by\n  trivial\n```"
    captured = {}

    def fake_verifier(code, args):
        captured["code"] = code
        return ma_proofbench.LeanVerificationResult(passed=True, verifier_output="ok")

    monkeypatch.setattr(ma_proofbench, "run_lean_verifier", fake_verifier)

    score = olymmath.score_lean_response(row, output, argparse.Namespace())

    assert score.pred == "verified"
    assert score.correct is True
    assert captured["code"] == "import Mathlib\n\ntheorem to_prove : True := by\n  trivial"
    assert score.metadata["verification_passed"] is True


def test_ma_proofbench_build_prompt_matches_paper_general_purpose_prompt():
    prompt, gold = ma_proofbench.build_prompt(
        {
            "id": 1,
            "split": "level1",
            "informal_statement": "Show that sin is Lipschitz.",
            "formal_statement": "import Mathlib\n\ntheorem example_theorem : True := by\n  sorry",
            "header": "import Mathlib",
            "topic": "Real functions",
            "tag": "Functions of one variable",
            "version": "4.28.0",
        },
        random.Random(0),
    )

    assert gold == "lean_verifies"
    assert prompt == (
        "You are an expert in Lean 4 and Mathematics. Please finish the following proof in Lean4 code.\n\n"
        "Do not change the original statement. Copy the final statement to prove exactly.\n"
        "Please include the complete header (including imports and namespaces) so that your code can pass the Lean4 compiler. "
        "Please solve the statement step by step and provide your complete Lean4 code between ```lean4 and ``` after careful reasoning.\n\n"
        "The statement for you to complete is:\n"
        "```lean4\n"
        "import Mathlib\n\n"
        "theorem example_theorem : True := by\n"
        "  sorry\n"
        "```"
    )
    assert "Show that sin is Lipschitz." not in prompt
    assert "Metadata:" not in prompt
    assert "theorem example_theorem : True" in prompt


def test_ma_proofbench_extracts_lean_code_block():
    output = "Reasoning first.\n```lean4\nimport Mathlib\n\ntheorem t : True := by\n  trivial\n```"

    assert ma_proofbench.extract_lean_code(output) == "import Mathlib\n\ntheorem t : True := by\n  trivial"


def test_ma_proofbench_extracts_last_lean_code_block_when_model_repeats_stub():
    output = (
        "The original statement is:\n"
        "```lean4\n"
        "import Mathlib\n\n"
        "theorem t : True := by\n"
        "  sorry\n"
        "```\n"
        "Here is the completed proof:\n"
        "```lean4\n"
        "import Mathlib\n\n"
        "theorem t : True := by\n"
        "  trivial\n"
        "```"
    )

    assert ma_proofbench.extract_lean_code(output) == "import Mathlib\n\ntheorem t : True := by\n  trivial"


def test_ma_proofbench_scores_verifier_success(monkeypatch, tmp_path):
    row = {
        "id": 1,
        "split": "level1",
        "informal_statement": "Show true.",
        "formal_statement": "import Mathlib\n\ntheorem t : True := by\n  sorry",
        "header": "import Mathlib",
        "topic": "Real functions",
        "tag": "Functions of one variable",
        "version": "4.28.0",
    }
    output = "```lean4\nimport Mathlib\n\ntheorem t : True := by\n  trivial\n```"

    monkeypatch.setattr(
        ma_proofbench,
        "run_lean_verifier",
        lambda code, args: ma_proofbench.LeanVerificationResult(passed=True, verifier_output="ok"),
    )

    args = evaluation.build_parser().parse_args(
        ["--benchmark", "ma_proofbench", "--methods", "plain_llm", "--output-dir", str(tmp_path)]
    )
    score = ma_proofbench.score_response(row, output, args)

    assert score.pred == "verified"
    assert score.correct is True
    assert score.metadata["verification_passed"] is True


def test_ma_proofbench_merges_dataset_header_before_verification(monkeypatch, tmp_path):
    captured = {}
    row = {
        "id": 1,
        "split": "level1",
        "informal_statement": "Show true.",
        "formal_statement": "import Mathlib\n\nopen Set\n\ntheorem t : True := by\n  sorry",
        "header": "import Mathlib\n\nopen Set",
        "topic": "Real functions",
        "tag": "Functions of one variable",
        "version": "4.28.0",
    }
    output = "```lean4\ntheorem t : True := by\n  trivial\n```"

    def fake_verifier(code, args):
        captured["code"] = code
        return ma_proofbench.LeanVerificationResult(passed=True, verifier_output="ok")

    monkeypatch.setattr(ma_proofbench, "run_lean_verifier", fake_verifier)

    args = evaluation.build_parser().parse_args(
        ["--benchmark", "ma_proofbench", "--methods", "plain_llm", "--output-dir", str(tmp_path)]
    )
    score = ma_proofbench.score_response(row, output, args)

    assert score.correct is True
    assert captured["code"] == "import Mathlib\n\nopen Set\n\ntheorem t : True := by\n  trivial"
    assert score.metadata["lean_code"] == captured["code"]


def test_ma_proofbench_dispatches_to_kimina_server_verifier(monkeypatch):
    captured = {}

    def fake_kimina_verifier(code, args):
        captured["code"] = code
        captured["host"] = args.kimina_host
        captured["port"] = args.kimina_port
        return ma_proofbench.LeanVerificationResult(
            passed=True,
            verifier_output="complete",
            backend="kimina-server",
        )

    monkeypatch.setattr(ma_proofbench, "run_kimina_server_verifier", fake_kimina_verifier)
    args = evaluation.build_parser().parse_args(
        [
            "--benchmark",
            "ma_proofbench",
            "--methods",
            "plain_llm",
            "--kimina-host",
            "127.0.0.1",
            "--kimina-port",
            "8000",
        ]
    )

    result = ma_proofbench.run_lean_verifier("import Mathlib\n\ntheorem t : True := by\n  trivial", args)

    assert result.passed is True
    assert result.backend == "kimina-server"
    assert captured == {
        "code": "import Mathlib\n\ntheorem t : True := by\n  trivial",
        "host": "127.0.0.1",
        "port": 8000,
    }


def test_ma_proofbench_collects_kimina_complete_result():
    response = {
        "sorries": [],
        "tactics": [],
        "messages": [],
    }

    result = ma_proofbench.collect_kimina_result(
        code="import Mathlib\n\ntheorem t : True := by\n  trivial",
        response=response,
        elapsed_seconds=0.25,
    )

    assert result.passed is True
    assert json.loads(result.verifier_output)["complete"] is True
    assert result.backend == "kimina-server"


def test_ma_proofbench_collects_kimina_sorry_as_incomplete():
    response = {
        "sorries": [{"pos": {"line": 3}}],
        "tactics": [],
        "messages": [],
    }

    result = ma_proofbench.collect_kimina_result(
        code="import Mathlib\n\ntheorem t : True := by\n  sorry",
        response=response,
        elapsed_seconds=0.25,
    )

    payload = json.loads(result.verifier_output)
    assert result.passed is False
    assert payload["pass"] is True
    assert payload["complete"] is False


def test_ma_proofbench_rejects_sorry_before_verifier(tmp_path):
    row = {
        "id": 1,
        "split": "level1",
        "informal_statement": "Show true.",
        "formal_statement": "import Mathlib\n\ntheorem t : True := by\n  sorry",
        "header": "import Mathlib",
        "topic": "Real functions",
        "tag": "Functions of one variable",
        "version": "4.28.0",
    }
    output = "```lean4\nimport Mathlib\n\ntheorem t : True := by\n  sorry\n```"
    args = evaluation.build_parser().parse_args(["--benchmark", "ma_proofbench", "--methods", "plain_llm"])

    score = ma_proofbench.score_response(row, output, args)

    assert score.pred == "contains_sorry"
    assert score.correct is False
    assert "sorry" in score.error


def test_load_local_ma_proofbench_rows_preserves_order_and_filters_level(tmp_path):
    path = tmp_path / "ma_proofbench.jsonl"
    rows = [
        {
            "id": 1,
            "split": "level1",
            "informal_statement": "First.",
            "formal_statement": "import Mathlib\n\ntheorem first : True := by\n  sorry",
            "header": "import Mathlib",
            "topic": "Real functions",
            "tag": "Functions of one variable",
            "version": "4.28.0",
        },
        {
            "id": 2,
            "split": "level2",
            "informal_statement": "Second.",
            "formal_statement": "import Mathlib\n\ntheorem second : True := by\n  sorry",
            "header": "import Mathlib",
            "topic": "Functional analysis",
            "tag": "Banach spaces",
            "version": "4.28.0",
        },
        {
            "id": 3,
            "split": "level1",
            "informal_statement": "Third.",
            "formal_statement": "import Mathlib\n\ntheorem third : True := by\n  sorry",
            "header": "import Mathlib",
            "topic": "Complex analysis",
            "tag": "Holomorphic functions",
            "version": "4.28.0",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    loaded = ma_proofbench.load_local_ma_proofbench_rows(path, level="level1")

    assert [row["id"] for row in loaded] == [1, 3]


@pytest.mark.asyncio
async def test_ma_proofbench_score_hook_is_used_by_runner(monkeypatch, tmp_path):
    row = {
        "id": 1,
        "split": "level1",
        "informal_statement": "Show true.",
        "formal_statement": "import Mathlib\n\ntheorem t : True := by\n  sorry",
        "header": "import Mathlib",
        "topic": "Real functions",
        "tag": "Functions of one variable",
        "version": "4.28.0",
    }

    benchmark = BenchmarkSpec(
        name="fake_proof",
        display_name="Fake Proof",
        default_output_filename="fake_proof.csv",
        load_items=lambda args: [row],
        build_prompt=lambda row, rng: ("Prompt", "lean_verifies"),
        extract_answer=lambda text: None,
        score_response=lambda row, text, args: runner.BenchmarkScore(
            pred="verified",
            correct=True,
            metadata={"verification_passed": True},
        ),
    )

    async def fake_run_method(method, prompt, llm, args):
        return runner.RunResult(raw_output="Lean code", returncode=0, elapsed_seconds=0.1)

    monkeypatch.setattr(evaluation, "get_benchmarks", lambda: {"fake_proof": benchmark})
    monkeypatch.setattr(evaluation, "get_llm", lambda model=None, openai=True, max_tokens=None: "fake-llm")
    monkeypatch.setattr(runner, "run_method", fake_run_method)
    monkeypatch.setattr(runner, "progress", lambda items, desc: items)

    args = evaluation.build_parser().parse_args(
        ["--benchmark", "fake_proof", "--methods", "plain_llm", "--output-dir", str(tmp_path)]
    )

    results = await evaluation.run_evaluation(args)

    assert results[0]["pred"] == "verified"
    assert results[0]["correct"] is True
    assert results[0]["score_metadata"] == {"verification_passed": True}


def test_summarize_results_groups_accuracy_by_method():
    summary = runner.summarize_results(
        [
            {
                "method": "multiagent",
                "correct": True,
                "pred": "A",
                "elapsed_seconds": 1.25,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            {
                "method": "multiagent",
                "correct": False,
                "pred": None,
                "elapsed_seconds": 2.75,
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
            {
                "method": "plain_llm",
                "correct": True,
                "pred": "B",
                "elapsed_seconds": 0.5,
                "prompt_tokens": 20,
                "completion_tokens": 3,
                "total_tokens": 23,
            },
        ]
    )

    assert summary["multiagent"]["total"] == 2
    assert summary["multiagent"]["accuracy"] == 0.5
    assert summary["multiagent"]["invalid_rate"] == 0.5
    assert summary["multiagent"]["elapsed_seconds"] == 4.0
    assert summary["multiagent"]["total_tokens"] == 33
    assert summary["multiagent"]["avg_prompt_tokens"] == 10.5
    assert summary["multiagent"]["avg_completion_tokens"] == 6.0
    assert summary["multiagent"]["avg_total_tokens"] == 16.5
    assert summary["plain_llm"]["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_run_method_records_elapsed_time_and_usage_metadata_tokens():
    llm = UsageLLM(
        [
            UsageResponse(
                content="Final Answer: A",
                usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
            )
        ]
    )
    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa"])

    result = await runner.run_method("plain_llm", "Question?", llm, args)

    assert result.raw_output == "Final Answer: A"
    assert result.returncode == 0
    assert result.elapsed_seconds >= 0
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 4
    assert result.total_tokens == 16


@pytest.mark.asyncio
async def test_run_method_records_response_metadata_token_usage():
    llm = UsageLLM(
        [
            UsageResponse(
                content="Final Answer: B",
                response_metadata={
                    "token_usage": {
                        "prompt_tokens": 9,
                        "completion_tokens": 6,
                        "total_tokens": 15,
                    }
                },
            )
        ]
    )
    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa"])

    result = await runner.run_method("plain_llm", "Question?", llm, args)

    assert result.prompt_tokens == 9
    assert result.completion_tokens == 6
    assert result.total_tokens == 15


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "expected_subagent_mode", "expected_streaming"),
    [
        ("multiagent_streaming", "fixed", True),
        ("multiagent_no_streaming", "fixed", False),
        ("multiagent_dynamic_streaming", "dynamic", True),
        ("multiagent_dynamic_no_streaming", "dynamic", False),
    ],
)
async def test_run_method_passes_subagent_mode_and_streaming_to_workflow(
    monkeypatch,
    method,
    expected_subagent_mode,
    expected_streaming,
):
    captured_kwargs = {}

    async def fake_run_workflow(**kwargs):
        captured_kwargs.update(kwargs)
        return {"final_answer": f"{method} answer"}

    from multi_agent_sync.evaluation.baselines import multiagent_sync

    monkeypatch.setattr(multiagent_sync, "run_workflow", fake_run_workflow)
    args = evaluation.build_parser().parse_args(["--benchmark", "gpqa"])

    result = await runner.run_method(method, "Question?", UsageLLM([]), args)

    assert result.raw_output == f"{method} answer"
    assert captured_kwargs["subagent_mode"] == expected_subagent_mode
    assert captured_kwargs["enable_agent_message_streaming"] is expected_streaming


def test_write_results_csv_includes_timing_and_token_columns(tmp_path):
    output_path = tmp_path / "results.csv"

    runner.write_results_csv(
        output_path,
        [
            {
                "benchmark": "gpqa",
                "method": "plain_llm",
                "index": 0,
                "gold": "A",
                "pred": "A",
                "correct": True,
                "returncode": 0,
                "error": "",
                "elapsed_seconds": 0.125,
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "total_tokens": 16,
                "raw_output": "Final Answer: A",
            }
        ],
    )

    rows = list(csv.DictReader(output_path.open(newline="", encoding="utf-8")))

    assert rows[0]["elapsed_seconds"] == "0.125"
    assert rows[0]["prompt_tokens"] == "12"
    assert rows[0]["completion_tokens"] == "4"
    assert rows[0]["total_tokens"] == "16"
    assert "json_trace_path" in rows[0]


def test_evaluation_main_module_is_importable():
    from multi_agent_sync.evaluation import main as module

    assert callable(module.main)
    assert callable(module.run_evaluation)


def test_load_local_gpqa_csv_rows(tmp_path):
    path = tmp_path / "gpqa.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Question",
                "Correct Answer",
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Question": "Question?",
                "Correct Answer": "Correct",
                "Incorrect Answer 1": "Wrong 1",
                "Incorrect Answer 2": "Wrong 2",
                "Incorrect Answer 3": "Wrong 3",
            }
        )

    rows = gpqa.load_local_gpqa_rows(path, limit=1)

    assert rows == [
        {
            "Question": "Question?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
        }
    ]


def test_load_local_gsm8k_jsonl_rows(tmp_path):
    path = tmp_path / "gsm8k.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "How many clips did Natalia sell?",
                "answer": "Natalia sold 48+24 = <<48+24=72>>72 clips. #### 72",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = gsm8k.load_local_gsm8k_rows(path, limit=1)

    assert rows == [
        {
            "question": "How many clips did Natalia sell?",
            "answer": "Natalia sold 48+24 = <<48+24=72>>72 clips. #### 72",
        }
    ]


def test_load_local_gsm8k_rows_validates_required_fields(tmp_path):
    path = tmp_path / "gsm8k.jsonl"
    path.write_text(json.dumps({"question": "Missing answer"}) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing required field"):
        gsm8k.load_local_gsm8k_rows(path)


def test_load_gsm8k_dataset_uses_openai_main_test_split(monkeypatch, tmp_path, capsys):
    captured = {}
    cache_path = tmp_path / "gsm8k" / "gsm8k_test.jsonl"

    class FakeDataset(list):
        def select(self, selected_range):
            return FakeDataset([self[index] for index in selected_range])

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeDataset(
            [
                {"question": "Question 1?", "answer": "#### 1"},
                {"question": "Question 2?", "answer": "#### 2"},
            ]
        )

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )
    monkeypatch.setattr(gsm8k, "DEFAULT_LOCAL_DATA_FILE", cache_path, raising=False)

    rows = gsm8k.load_gsm8k_dataset(limit=1)

    assert captured == {"args": ("openai/gsm8k", "main"), "kwargs": {"split": "test"}}
    assert rows == [{"question": "Question 1?", "answer": "#### 1"}]
    assert cache_path.exists()
    output = capsys.readouterr().out
    assert "downloading from Hugging Face" in output
    assert "Saved benchmark data file" in output


def test_load_gsm8k_dataset_uses_default_local_file_before_huggingface(monkeypatch, tmp_path, capsys):
    cache_path = tmp_path / "gsm8k" / "gsm8k_test.jsonl"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(json.dumps({"question": "Cached?", "answer": "#### 4"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(gsm8k, "DEFAULT_LOCAL_DATA_FILE", cache_path, raising=False)

    rows = gsm8k.load_gsm8k_dataset(limit=1)

    assert rows == [{"question": "Cached?", "answer": "#### 4"}]
    assert "Using local benchmark data file" in capsys.readouterr().out


def test_load_gpqa_dataset_downloads_and_saves_default_local_file(monkeypatch, tmp_path):
    captured = {}
    cache_path = tmp_path / "gpqa" / "gpqa_diamond.csv"

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return [
            {
                "Question": "Question 1?",
                "Correct Answer": "Correct",
                "Incorrect Answer 1": "Wrong 1",
                "Incorrect Answer 2": "Wrong 2",
                "Incorrect Answer 3": "Wrong 3",
            },
            {
                "Question": "Question 2?",
                "Correct Answer": "Correct 2",
                "Incorrect Answer 1": "Wrong 4",
                "Incorrect Answer 2": "Wrong 5",
                "Incorrect Answer 3": "Wrong 6",
            },
        ]

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )
    monkeypatch.setattr(gpqa, "DEFAULT_LOCAL_DATA_FILE", cache_path, raising=False)

    rows = gpqa.load_gpqa_dataset(limit=1)

    assert captured == {"args": ("Idavidrein/gpqa", "gpqa_diamond"), "kwargs": {"split": "train"}}
    assert rows == [
        {
            "Question": "Question 1?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
        }
    ]
    assert cache_path.exists()


def test_load_local_mmlu_pro_jsonl_rows(tmp_path):
    path = tmp_path / "mmlu_pro.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Which option is correct?",
                "options": ["A option", "B option", "C option", "D option", "E option", "F option", "G option", "H option", "I option", "J option"],
                "answer": "B",
                "answer_index": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = mmlu_pro.load_local_mmlu_pro_rows(path, limit=1)

    assert rows == [
        {
            "question": "Which option is correct?",
            "options": ["A option", "B option", "C option", "D option", "E option", "F option", "G option", "H option", "I option", "J option"],
            "answer": "B",
            "answer_index": 1,
        }
    ]


def test_load_local_mmlu_pro_rows_validates_required_fields(tmp_path):
    path = tmp_path / "mmlu_pro.jsonl"
    path.write_text(json.dumps({"question": "Missing options and answer"}) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing required field"):
        mmlu_pro.load_local_mmlu_pro_rows(path)


def test_load_local_mmlu_pro_rows_accepts_fewer_than_ten_options_when_answer_is_in_range(tmp_path):
    path = tmp_path / "mmlu_pro.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "Which safety label is correct?",
                "options": ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
                "answer": "I",
                "answer_index": 8,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = mmlu_pro.load_local_mmlu_pro_rows(path)

    assert rows[0]["answer"] == "I"
    assert len(rows[0]["options"]) == 9


def test_load_mmlu_pro_dataset_uses_test_split_only(monkeypatch, tmp_path):
    captured = {}
    cache_path = tmp_path / "mmlu_pro" / "mmlu_pro_test.jsonl"

    class FakeDataset(list):
        def select(self, selected_range):
            return FakeDataset([self[index] for index in selected_range])

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeDataset(
            [
                {"question": "Question 1?", "options": ["A"] * 10, "answer": "A", "answer_index": 0},
                {"question": "Question 2?", "options": ["B"] * 10, "answer": "B", "answer_index": 1},
            ]
        )

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )
    monkeypatch.setattr(mmlu_pro, "DEFAULT_LOCAL_DATA_FILE", cache_path, raising=False)

    rows = mmlu_pro.load_mmlu_pro_dataset(limit=1)

    assert captured == {"args": ("TIGER-Lab/MMLU-Pro",), "kwargs": {"split": "test"}}
    assert rows == [{"question": "Question 1?", "options": ["A"] * 10, "answer": "A", "answer_index": 0}]
    assert cache_path.exists()


def test_load_ma_proofbench_dataset_downloads_saves_and_filters_default_local_file(monkeypatch, tmp_path):
    captured = {}
    cache_path = tmp_path / "ma_proofbench" / "ma_proofbench_test.jsonl"

    def fake_load_dataset(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return [
            {
                "id": "one",
                "split": "level1",
                "informal_statement": "Informal 1",
                "formal_statement": "theorem one : True := by trivial",
                "header": "import Mathlib",
                "topic": "logic",
                "tag": "test",
                "version": "v1",
            },
            {
                "id": "two",
                "split": "level2",
                "informal_statement": "Informal 2",
                "formal_statement": "theorem two : True := by trivial",
                "header": "import Mathlib",
                "topic": "logic",
                "tag": "test",
                "version": "v1",
            },
        ]

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )
    monkeypatch.setattr(ma_proofbench, "DEFAULT_LOCAL_DATA_FILE", cache_path, raising=False)

    rows = ma_proofbench.load_ma_proofbench_dataset(limit=1, level="level2")

    assert captured == {"args": ("openbmb/MA-ProofBench",), "kwargs": {"split": "test"}}
    assert [row["id"] for row in rows] == ["two"]
    assert cache_path.exists()


def test_load_olymmath_dataset_downloads_and_saves_all_default_local_files(monkeypatch, tmp_path, capsys):
    captured = []

    def fake_load_dataset(*args, **kwargs):
        captured.append((args, kwargs))
        config = args[1]
        if config == "lean":
            row = {
                "unique_id": "lean-1",
                "subject": "algebra",
                "formal_statement": "theorem lean_one : True := by trivial",
            }
        else:
            row = {"problem": f"{config} problem?", "answer": "1", "subject": "algebra", "unique_id": f"{config}-1"}
        return [row, dict(row, unique_id=f"{row['unique_id']}-second")]

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )
    local_data_dir = tmp_path / "OlymMATH"
    monkeypatch.setattr(olymmath, "DEFAULT_LOCAL_DATA_DIR", local_data_dir, raising=False)

    rows = olymmath.load_olymmath_dataset(subset="en-hard", limit=1)

    assert captured == [
        (("RUC-AIBOX/OlymMATH", "en-easy"), {"split": "test"}),
        (("RUC-AIBOX/OlymMATH", "en-hard"), {"split": "test"}),
        (("RUC-AIBOX/OlymMATH", "zh-easy"), {"split": "test"}),
        (("RUC-AIBOX/OlymMATH", "zh-hard"), {"split": "test"}),
        (("RUC-AIBOX/OlymMATH", "lean"), {"split": "test"}),
    ]
    assert rows == [{"problem": "en-hard problem?", "answer": "1", "subject": "algebra", "unique_id": "en-hard-1"}]
    assert (local_data_dir / "OlymMATH-EN-EASY.jsonl").exists()
    assert (local_data_dir / "OlymMATH-EN-HARD.jsonl").exists()
    assert (local_data_dir / "OlymMATH-ZH-EASY.jsonl").exists()
    assert (local_data_dir / "OlymMATH-ZH-HARD.jsonl").exists()
    assert (local_data_dir / "OlymMATH-LEAN.jsonl").exists()
    output = capsys.readouterr().out
    assert "OlymMATH local data incomplete" in output
    assert "Downloading OlymMATH data file from Hugging Face" in output


def test_load_gpqa_dataset_falls_back_to_default_local_file_when_hub_is_gated(monkeypatch, tmp_path):
    path = tmp_path / "gpqa" / "gpqa_diamond.csv"
    path.parent.mkdir(parents=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Question",
                "Correct Answer",
                "Incorrect Answer 1",
                "Incorrect Answer 2",
                "Incorrect Answer 3",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Question": "Local question?",
                "Correct Answer": "Correct",
                "Incorrect Answer 1": "Wrong 1",
                "Incorrect Answer 2": "Wrong 2",
                "Incorrect Answer 3": "Wrong 3",
            }
        )

    def gated_load_dataset(*args, **kwargs):
        raise RuntimeError("Dataset is gated. You must be authenticated to access it.")

    monkeypatch.setitem(
        __import__("sys").modules,
        "datasets",
        types.SimpleNamespace(load_dataset=gated_load_dataset),
    )
    monkeypatch.setattr(gpqa, "DEFAULT_LOCAL_DATA_FILE", path, raising=False)

    rows = gpqa.load_gpqa_dataset(limit=1)

    assert rows == [
        {
            "Question": "Local question?",
            "Correct Answer": "Correct",
            "Incorrect Answer 1": "Wrong 1",
            "Incorrect Answer 2": "Wrong 2",
            "Incorrect Answer 3": "Wrong 3",
        }
    ]


def test_gated_dataset_error_has_actionable_message():
    error = gpqa.build_dataset_access_error(RuntimeError("Dataset is gated. You must be authenticated to access it."))

    assert "GPQA is a gated Hugging Face dataset" in error
    assert "HF_TOKEN" in error
    assert "--data-file" in error
