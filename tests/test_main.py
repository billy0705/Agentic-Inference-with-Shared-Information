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

    args = cli.build_parser().parse_args(["--total-runtime-timeout", "600", "Calculate", "2+2"])

    await cli.async_main(args)

    assert captured_kwargs["task"] == "Calculate 2+2"
    assert captured_kwargs["total_runtime_timeout"] == 600.0
    assert captured_llm_kwargs["openai"] is True


@pytest.mark.asyncio
async def test_async_main_can_use_local_model(monkeypatch, tmp_path):
    captured_llm_kwargs = {}

    async def fake_run_workflow(**kwargs):
        return {"final_answer": "Done."}

    def fake_get_llm(model=None, openai=True):
        captured_llm_kwargs["model"] = model
        captured_llm_kwargs["openai"] = openai
        return "fake-llm"

    monkeypatch.setattr(cli, "get_llm", fake_get_llm)
    monkeypatch.setattr(cli, "run_workflow", fake_run_workflow)
    monkeypatch.setattr(cli, "save_run_artifacts", lambda state, root_dir: tmp_path / "run")

    args = cli.build_parser().parse_args(["--local-model", "--model", "llama3.2", "Calculate", "2+2"])

    await cli.async_main(args)

    assert captured_llm_kwargs == {"model": "llama3.2", "openai": False}


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
