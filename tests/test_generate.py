from pathlib import Path
from types import SimpleNamespace

import pytest

from multi_agent_sync import generate


class ConfigManagerStub:
    def __init__(self, expt=None, helma=None, vllm=None):
        self.sections = {"expt": expt or {}, "helma": helma or {}, "vllm": vllm or {}}
        self.config = SimpleNamespace(engine_ready_timeout=1800)
        self.path = Path("/project/server.yaml")

    def section(self, name):
        return self.sections[name]


def test_server_ready_timeout_reads_vllm_section_with_default():
    assert generate.server_ready_timeout(ConfigManagerStub()) == 1800.0
    assert generate.server_ready_timeout(ConfigManagerStub(vllm={"engine_ready_timeout": 42})) == 42.0


def test_server_ready_timeout_rejects_invalid_value():
    with pytest.raises(ValueError, match="vllm.engine_ready_timeout"):
        generate.server_ready_timeout(ConfigManagerStub(vllm={"engine_ready_timeout": 0}))


def test_load_helma_config_builds_slurm_arguments():
    manager = ConfigManagerStub(
        helma={
            "job_name": "gpt-oss",
            "nodes": 1,
            "gres": "gpu:4",
            "partition": "h100",
            "output": "out_single_node.out",
            "time": "4:00:00",
        }
    )

    config = generate.load_helma_config(manager)

    assert config.slurm_arguments() == (
        "--job-name=gpt-oss",
        "--nodes=1",
        "--gres=gpu:4",
        "--partition=h100",
        "--output=out_single_node.out",
        "--time=4:00:00",
    )


@pytest.mark.parametrize(
    ("helma", "message"),
    [
        ({"nodes": 0}, "helma.nodes"),
        ({"nodes": 1}, "helma.job_name"),
    ],
)
def test_load_helma_config_rejects_invalid_values(helma, message):
    with pytest.raises(ValueError, match=message):
        generate.load_helma_config(ConfigManagerStub(helma=helma))


def test_load_experiment_config_validates_and_deduplicates_lists():
    manager = ConfigManagerStub(
        {
            "benchmark": ["gsm8k", "gpqa", "gsm8k"],
            "methods": ["multiagent_dynamic_streaming", "plain_llm", "plain_llm"],
            "limit": 2,
        }
    )

    config = generate.load_experiment_config(manager)

    assert config.benchmarks == ("gsm8k", "gpqa")
    assert config.methods == ("multiagent_dynamic_streaming", "plain_llm")
    assert config.limit == 2
    assert config.repeats == 1
    assert config.max_steps is None
    assert config.think_mode is None
    assert config.single_agent_min_steps is None
    assert config.single_agent_max_steps is None
    assert config.debate_rounds is None
    assert config.max_tokens is None
    assert config.resume_run is None
    assert config.benchmark_data_dir is None
    assert config.output_dir == Path("/project/output")


def test_load_experiment_config_resolves_benchmark_data_dir_relative_to_yaml():
    manager = ConfigManagerStub(
        {
            "benchmark": ["gsm8k"],
            "methods": ["plain_llm"],
            "benchmark_data_dir": "benchmark-data",
        }
    )

    config = generate.load_experiment_config(manager)

    assert config.benchmark_data_dir == Path("/project/benchmark-data")


def test_load_experiment_config_resolves_output_dir_relative_to_yaml():
    manager = ConfigManagerStub(
        {
            "benchmark": ["gsm8k"],
            "methods": ["plain_llm"],
            "output_dir": "output",
        }
    )

    config = generate.load_experiment_config(manager)

    assert config.output_dir == Path("/project/output")


def test_shell_environment_exports_includes_benchmark_data_dir():
    manager = ConfigManagerStub(
        {
            "benchmark": ["gsm8k"],
            "methods": ["plain_llm"],
            "benchmark_data_dir": "/data/benchmark data",
        }
    )

    config = generate.load_experiment_config(manager)

    assert generate.shell_environment_exports(config) == ("export BENCHMARK_DATA_DIR='/data/benchmark data'",)


def test_load_experiment_config_resolves_resume_run_relative_to_yaml():
    manager = ConfigManagerStub(
        {
            "benchmark": ["gsm8k"],
            "methods": ["plain_llm"],
            "resume_run": "output/gsm8k/model/run-id",
        }
    )

    config = generate.load_experiment_config(manager)

    assert config.resume_run == Path("/project/output/gsm8k/model/run-id")


@pytest.mark.parametrize(
    ("expt", "message"),
    [
        ({"benchmark": [], "methods": ["plain_llm"]}, "expt.benchmark"),
        ({"benchmark": ["unknown"], "methods": ["plain_llm"]}, "Unknown benchmark"),
        ({"benchmark": ["gsm8k"], "methods": []}, "expt.methods"),
        ({"benchmark": ["gsm8k"], "methods": ["unknown"]}, "Unknown method"),
        ({"benchmark": ["gsm8k"], "methods": ["plain_llm"], "limit": -1}, "expt.limit"),
        ({"benchmark": ["gsm8k"], "methods": ["plain_llm"], "repeats": 0}, "expt.repeats"),
        ({"benchmark": ["gsm8k"], "methods": ["plain_llm"], "max_steps": 0}, "expt.max_steps"),
        ({"benchmark": ["gsm8k"], "methods": ["plain_llm"], "think_mode": "false"}, "expt.think_mode"),
        ({"benchmark": ["gsm8k"], "methods": ["plain_llm"], "max_tokens": 0}, "expt.max_tokens"),
        (
            {
                "benchmark": ["gsm8k"],
                "methods": ["plain_llm"],
                "single_agent_min_steps": 6,
                "single_agent_max_steps": 5,
            },
            "single_agent_min_steps",
        ),
        ({"benchmark": ["gsm8k"], "methods": ["plain_llm"], "benchmark_data_dir": ""}, "expt.benchmark_data_dir"),
        ({"benchmark": ["gsm8k"], "methods": ["plain_llm"], "output_dir": ""}, "expt.output_dir"),
        (
            {"benchmark": ["gsm8k", "gpqa"], "methods": ["plain_llm"], "resume_run": "output/run"},
            "one configured benchmark",
        ),
    ],
)
def test_load_experiment_config_rejects_invalid_values(expt, message):
    with pytest.raises(ValueError, match=message):
        generate.load_experiment_config(ConfigManagerStub(expt))


@pytest.mark.asyncio
async def test_run_experiments_starts_one_server_for_all_benchmarks(monkeypatch):
    events = []
    manager = ConfigManagerStub(
        {
            "benchmark": ["gsm8k", "gpqa"],
            "methods": ["multiagent_dynamic_streaming", "plain_llm"],
            "limit": 1,
            "repeats": 2,
            "max_steps": 5,
            "think_mode": False,
            "single_agent_min_steps": 5,
            "single_agent_max_steps": 5,
            "debate_rounds": 5,
            "max_tokens": 12000,
        }
    )

    class Server:
        def stop(self):
            events.append(("stop",))

    def fake_spinup_server(received_manager, *, timeout):
        events.append(("start", received_manager, timeout))
        return Server()

    async def fake_run_evaluation(args):
        events.append(
            (
                "run",
                args.benchmark,
                args.methods,
                args.limit,
                args.max_steps,
                args.think_mode,
                args.single_agent_min_steps,
                args.single_agent_max_steps,
                args.debate_rounds,
                args.max_tokens,
                args.output_dir,
                args.resume_run,
            )
        )
        return []

    monkeypatch.setattr(generate, "get_config_manager", lambda: lambda path: manager)
    monkeypatch.setattr(generate, "spinup_server", fake_spinup_server)
    monkeypatch.setattr(generate.evaluation, "run_evaluation", fake_run_evaluation)

    await generate.run_experiments("server.yaml")

    assert events == [
        ("start", manager, 1800.0),
        ("run", "gsm8k", "multiagent_dynamic_streaming,plain_llm", 1, 5, False, 5, 5, 5, 12000, "/project/output", None),
        ("run", "gsm8k", "multiagent_dynamic_streaming,plain_llm", 1, 5, False, 5, 5, 5, 12000, "/project/output", None),
        ("run", "gpqa", "multiagent_dynamic_streaming,plain_llm", 1, 5, False, 5, 5, 5, 12000, "/project/output", None),
        ("run", "gpqa", "multiagent_dynamic_streaming,plain_llm", 1, 5, False, 5, 5, 5, 12000, "/project/output", None),
        ("stop",),
    ]


def test_print_evaluation_matrix_includes_optional_runtime_args(monkeypatch, capsys):
    manager = ConfigManagerStub(
        {
            "benchmark": ["gsm8k"],
            "methods": ["plain_llm"],
            "limit": 1,
            "max_steps": 5,
            "think_mode": False,
            "single_agent_min_steps": 5,
            "single_agent_max_steps": 5,
            "debate_rounds": 5,
            "max_tokens": 12000,
        }
    )

    monkeypatch.setattr(generate, "get_config_manager", lambda: lambda path: manager)

    generate.main(["server.yaml", "--print-evaluation-matrix"])

    assert capsys.readouterr().out == (
        "gsm8k\tplain_llm\t1\t/project/output\t-\t5\tfalse\t5\t5\t5\t12000\n"
    )


@pytest.mark.asyncio
async def test_run_experiments_passes_configured_resume_run(monkeypatch):
    captured = {}
    manager = ConfigManagerStub(
        {
            "benchmark": ["gsm8k"],
            "methods": ["plain_llm"],
            "limit": 1,
            "resume_run": "output/gsm8k/model/run-id",
        }
    )

    class Server:
        def stop(self):
            pass

    async def fake_run_evaluation(args):
        captured["resume_run"] = args.resume_run
        return []

    monkeypatch.setattr(generate, "get_config_manager", lambda: lambda path: manager)
    monkeypatch.setattr(generate, "spinup_server", lambda manager, timeout: Server())
    monkeypatch.setattr(generate.evaluation, "run_evaluation", fake_run_evaluation)

    await generate.run_experiments("server.yaml")

    assert captured["resume_run"] == "/project/output/gsm8k/model/run-id"
