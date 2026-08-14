from types import SimpleNamespace

import pytest

from multi_agent_sync import generate


class ConfigManagerStub:
    def __init__(self, expt=None, helma=None):
        self.sections = {"expt": expt or {}, "helma": helma or {}}
        self.config = SimpleNamespace(engine_ready_timeout=1800)

    def section(self, name):
        return self.sections[name]


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


@pytest.mark.parametrize(
    ("expt", "message"),
    [
        ({"benchmark": [], "methods": ["plain_llm"]}, "expt.benchmark"),
        ({"benchmark": ["unknown"], "methods": ["plain_llm"]}, "Unknown benchmark"),
        ({"benchmark": ["gsm8k"], "methods": []}, "expt.methods"),
        ({"benchmark": ["gsm8k"], "methods": ["unknown"]}, "Unknown method"),
        ({"benchmark": ["gsm8k"], "methods": ["plain_llm"], "limit": -1}, "expt.limit"),
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
        }
    )

    class Server:
        def stop(self):
            events.append(("stop",))

    def fake_spinup_server(received_manager, *, timeout):
        events.append(("start", received_manager, timeout))
        return Server()

    async def fake_run_evaluation(args):
        events.append(("run", args.benchmark, args.methods, args.limit))
        return []

    monkeypatch.setattr(generate, "get_config_manager", lambda: lambda path: manager)
    monkeypatch.setattr(generate, "spinup_server", fake_spinup_server)
    monkeypatch.setattr(generate.evaluation, "run_evaluation", fake_run_evaluation)

    await generate.run_experiments("server.yaml")

    assert events == [
        ("start", manager, 1800.0),
        ("run", "gsm8k", "multiagent_dynamic_streaming,plain_llm", 1),
        ("run", "gpqa", "multiagent_dynamic_streaming,plain_llm", 1),
        ("stop",),
    ]
