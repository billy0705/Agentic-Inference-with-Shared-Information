from __future__ import annotations

import pytest

from multi_agent_sync.tools.bash import BashTool
from multi_agent_sync.workspace.docker import BashResult, DockerWorkspace


class FakeDockerWorkspace(DockerWorkspace):
    def __init__(self) -> None:
        super().__init__(container_name="fake-container")
        self.commands: list[str] = []

    async def run_bash(self, command: str, *, timeout_seconds: float | None = None) -> BashResult:
        self.commands.append(command)
        return BashResult(
            command=command,
            exit_code=0,
            stdout="inside docker\n",
            stderr="",
            timed_out=False,
            container_name=self.container_name,
        )


@pytest.mark.asyncio
async def test_bash_tool_runs_command_through_docker_workspace_only():
    workspace = FakeDockerWorkspace()
    tool = BashTool(workspace)

    result = await tool.run("pwd")

    assert workspace.commands == ["pwd"]
    assert result.stdout == "inside docker\n"
    assert result.container_name == "fake-container"


def test_bash_tool_rejects_non_docker_workspace_to_prevent_local_fallback():
    with pytest.raises(TypeError, match="DockerWorkspace"):
        BashTool(object())


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /workspace",
        "rm -fr /workspace/",
        "rm -rf .",
        "rm -rf ./",
        "rm -rf *",
        "rm -rf .*",
        "rm -rf /workspace/*",
        "rm -rf /workspace/.*",
        "cd /tmp && rm -rf /workspace",
    ],
)
def test_docker_workspace_rejects_commands_that_destroy_workspace_root(command):
    workspace = DockerWorkspace(container_name="fake-container")

    rejection = workspace.validate_bash_command(command)

    assert rejection is not None
    assert "Refusing to run destructive workspace command" in rejection


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /tmp/build",
        "rm -rf astropy/build",
        "git status --short && git rev-parse HEAD",
        "echo 'rm -rf /workspace'",
    ],
)
def test_docker_workspace_allows_non_workspace_root_cleanup(command):
    workspace = DockerWorkspace(container_name="fake-container")

    assert workspace.validate_bash_command(command) is None
