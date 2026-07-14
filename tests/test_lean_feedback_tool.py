from __future__ import annotations

from dataclasses import dataclass

import pytest

from multi_agent_sync.evaluation.lean_feedback import LeanVerifierTool
from multi_agent_sync.evaluation.ma_proofbench import LeanVerificationResult
from multi_agent_sync.workspace.docker import BashResult, DockerWorkspace


class FakeDockerWorkspace(DockerWorkspace):
    def __init__(self, files: dict[str, str]) -> None:
        super().__init__(container_name="lean-fake-container")
        self.files = dict(files)
        self.commands: list[str] = []

    async def run_bash(self, command: str, *, timeout_seconds: float | None = None) -> BashResult:
        self.commands.append(command)
        if command == "cat /workspace/Main.lean":
            return BashResult(
                command=command,
                exit_code=0,
                stdout=self.files["/workspace/Main.lean"],
                stderr="",
                timed_out=False,
                container_name=self.container_name,
            )
        return BashResult(
            command=command,
            exit_code=1,
            stdout="",
            stderr="missing file",
            timed_out=False,
            container_name=self.container_name,
        )


@dataclass
class CapturingVerifier:
    code: str = ""

    def __call__(self, code: str) -> LeanVerificationResult:
        self.code = code
        return LeanVerificationResult(
            passed=False,
            verifier_output='{"errors": [{"data": "unknown tactic"}]}',
            returncode=1,
            backend="kimina-server",
        )


@pytest.mark.asyncio
async def test_lean_verifier_tool_reads_candidate_from_docker_and_calls_verifier():
    workspace = FakeDockerWorkspace({"/workspace/Main.lean": "theorem t : True := by\n  trivial\n"})
    verifier = CapturingVerifier()
    tool = LeanVerifierTool(workspace=workspace, verify_code=verifier)

    result = await tool.run(path="/workspace/Main.lean")

    assert workspace.commands == ["cat /workspace/Main.lean"]
    assert verifier.code == "theorem t : True := by\n  trivial\n"
    assert result["passed"] is False
    assert result["backend"] == "kimina-server"
    assert "unknown tactic" in result["verifier_output"]
