import pytest

from multi_agent_sync import main as cli


@pytest.mark.asyncio
async def test_async_main_passes_total_runtime_timeout_to_workflow(monkeypatch, tmp_path):
    captured_kwargs = {}
    captured_llm_kwargs = {}

    async def fake_run_workflow(**kwargs):
        captured_kwargs.update(kwargs)
        return {"final_answer": "Done."}

    def fake_get_llm(model=None, openai=True):
        captured_llm_kwargs["model"] = model
        captured_llm_kwargs["openai"] = openai
        return "fake-llm"

    monkeypatch.setattr(cli, "get_llm", fake_get_llm)
    monkeypatch.setattr(cli, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(cli, "save_run_artifacts", lambda state, root_dir: tmp_path / "run")

    args = cli.build_parser().parse_args(
        ["--total-runtime-timeout", "600", "--agent-runtime-timeout", "700", "Calculate", "2+2"]
    )

    await cli.async_main(args)

    assert captured_kwargs["task"] == "Calculate 2+2"
    assert captured_kwargs["total_runtime_timeout"] == 600.0
    assert captured_kwargs["agent_runtime_timeout"] == 700.0
    assert captured_kwargs["think_mode"] is True
    assert captured_llm_kwargs["openai"] is True


def test_main_parser_rejects_disabled_local_model_entrypoint():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--local-model", "--model", "llama3.2", "Calculate", "2+2"])


@pytest.mark.asyncio
async def test_async_main_passes_dynamic_subagent_mode_to_workflow(monkeypatch, tmp_path):
    captured_kwargs = {}

    async def fake_run_workflow(**kwargs):
        captured_kwargs.update(kwargs)
        return {"final_answer": "Done."}

    monkeypatch.setattr(cli, "get_llm", lambda model=None, openai=True: "fake-llm")
    monkeypatch.setattr(cli, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(cli, "save_run_artifacts", lambda state, root_dir: tmp_path / "run")

    args = cli.build_parser().parse_args(["--subagent-mode", "dynamic", "Calculate", "2+2"])

    await cli.async_main(args)

    assert captured_kwargs["subagent_mode"] == "dynamic"


@pytest.mark.asyncio
async def test_async_main_passes_agent_early_stop_flag_to_workflow(monkeypatch, tmp_path):
    captured_kwargs = {}

    async def fake_run_workflow(**kwargs):
        captured_kwargs.update(kwargs)
        return {"final_answer": "Done."}

    monkeypatch.setattr(cli, "get_llm", lambda model=None, openai=True: "fake-llm")
    monkeypatch.setattr(cli, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(cli, "save_run_artifacts", lambda state, root_dir: tmp_path / "run")

    args = cli.build_parser().parse_args(["--allow-agent-early-stop", "Calculate", "2+2"])

    await cli.async_main(args)

    assert captured_kwargs["allow_agent_early_stop"] is True


@pytest.mark.asyncio
async def test_async_main_passes_disabled_think_mode_to_workflow(monkeypatch, tmp_path):
    captured_kwargs = {}

    async def fake_run_workflow(**kwargs):
        captured_kwargs.update(kwargs)
        return {"final_answer": "Done."}

    monkeypatch.setattr(cli, "get_llm", lambda model=None, openai=True: "fake-llm")
    monkeypatch.setattr(cli, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(cli, "save_run_artifacts", lambda state, root_dir: tmp_path / "run")

    args = cli.build_parser().parse_args(["--no-think-mode", "Calculate", "2+2"])

    await cli.async_main(args)

    assert captured_kwargs["think_mode"] is False


@pytest.mark.asyncio
async def test_async_main_creates_and_cleans_docker_workspace(monkeypatch, tmp_path):
    captured_create_kwargs = {}
    captured_workflow_kwargs = {}
    cleanup_called = False

    class FakeDockerWorkspace:
        @classmethod
        async def create(cls, **kwargs):
            captured_create_kwargs.update(kwargs)
            return cls()

        async def cleanup(self):
            nonlocal cleanup_called
            cleanup_called = True

    async def fake_run_workflow(**kwargs):
        captured_workflow_kwargs.update(kwargs)
        return {"final_answer": "Done."}

    monkeypatch.setattr(cli, "DockerWorkspace", FakeDockerWorkspace)
    monkeypatch.setattr(cli, "get_llm", lambda model=None, openai=True: "fake-llm")
    monkeypatch.setattr(cli, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(cli, "save_run_artifacts", lambda state, root_dir: tmp_path / "run")

    args = cli.build_parser().parse_args(
        [
            "--docker-workspace",
            "--workspace-image",
            "python:3.12",
            "--workspace-source",
            str(tmp_path),
            "Fix",
            "the",
            "repo",
        ]
    )

    await cli.async_main(args)

    assert captured_create_kwargs["image"] == "python:3.12"
    assert captured_create_kwargs["source_path"] == str(tmp_path)
    assert captured_workflow_kwargs["enable_workspace_tools"] is True
    assert captured_workflow_kwargs["docker_workspace"] is not None
    assert cleanup_called is True
