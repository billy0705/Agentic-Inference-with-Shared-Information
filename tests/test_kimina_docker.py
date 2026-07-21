import pytest

from multi_agent_sync.evaluation.kimina_docker import KiminaDockerServer


@pytest.mark.asyncio
async def test_kimina_docker_server_does_not_start_when_server_is_available(monkeypatch):
    docker_calls = []
    server = KiminaDockerServer(host_port=8001)

    async def fake_docker(*args, check=True):
        docker_calls.append(args)
        return 0, "", ""

    async def fake_is_available():
        return True

    monkeypatch.setattr(server, "is_available", fake_is_available)
    monkeypatch.setattr(server, "_docker", fake_docker)

    await server.ensure_running()

    assert docker_calls == []
    assert server.started_container is False


@pytest.mark.asyncio
async def test_kimina_docker_server_reuses_existing_stopped_container(monkeypatch):
    docker_calls = []
    availability = iter([False, True])
    server = KiminaDockerServer(host_port=8001, startup_timeout=1.0, poll_interval=0.0)

    async def fake_is_available():
        return next(availability)

    async def fake_docker(*args, check=True):
        docker_calls.append(args)
        return 0, "container-id", ""

    monkeypatch.setattr(server, "is_available", fake_is_available)
    monkeypatch.setattr(server, "_docker", fake_docker)

    await server.ensure_running()

    assert docker_calls == [("start", "multi-agent-kimina-lean-server")]
    assert server.started_container is True


@pytest.mark.asyncio
async def test_kimina_docker_server_runs_new_container_on_host_port_8001(monkeypatch):
    docker_calls = []
    availability = iter([False, True])
    server = KiminaDockerServer(host_port=8001, startup_timeout=1.0, poll_interval=0.0)

    async def fake_is_available():
        return next(availability)

    async def fake_docker(*args, check=True):
        docker_calls.append(args)
        if args[0] == "start":
            return 1, "", "No such container"
        return 0, "container-id", ""

    monkeypatch.setattr(server, "is_available", fake_is_available)
    monkeypatch.setattr(server, "_docker", fake_docker)

    await server.ensure_running()

    assert docker_calls == [
        ("start", "multi-agent-kimina-lean-server"),
        (
            "run",
            "-d",
            "--name",
            "multi-agent-kimina-lean-server",
            "-p",
            "8001:8000",
            "projectnumina/kimina-lean-server:2.0.0",
        )
    ]
    assert server.started_container is True


@pytest.mark.asyncio
async def test_kimina_docker_server_cleanup_removes_only_started_container(monkeypatch):
    docker_calls = []
    server = KiminaDockerServer()

    async def fake_docker(*args, check=True):
        docker_calls.append((args, check))
        return 0, "", ""

    monkeypatch.setattr(server, "_docker", fake_docker)

    await server.cleanup()
    server.started_container = True
    await server.cleanup()

    assert docker_calls == [(("rm", "-f", "multi-agent-kimina-lean-server"), False)]
