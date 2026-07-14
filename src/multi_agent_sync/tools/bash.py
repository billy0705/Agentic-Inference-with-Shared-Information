from __future__ import annotations

from multi_agent_sync.workspace.docker import BashResult, DockerWorkspace


class BashTool:
    def __init__(self, workspace: DockerWorkspace) -> None:
        if not isinstance(workspace, DockerWorkspace):
            raise TypeError("BashTool requires a DockerWorkspace; local shell fallback is not allowed.")
        self.workspace = workspace

    async def run(self, command: str, *, timeout_seconds: float | None = None) -> BashResult:
        return await self.workspace.run_bash(command, timeout_seconds=timeout_seconds)
