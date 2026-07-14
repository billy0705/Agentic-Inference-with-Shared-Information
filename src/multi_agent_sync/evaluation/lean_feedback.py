from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from multi_agent_sync.workspace.docker import DockerWorkspace


@dataclass
class LeanVerifierTool:
    workspace: DockerWorkspace
    verify_code: Callable[[str], Any]
    default_path: str = "/workspace/Main.lean"
    name: str = "verify_candidate"

    async def run(self, *, path: str | None = None) -> dict[str, Any]:
        candidate_path = path or self.default_path
        code = await self.workspace.read_text(candidate_path)
        verification = self.verify_code(code)
        return {
            "tool": self.name,
            "path": candidate_path,
            "passed": bool(getattr(verification, "passed", False)),
            "backend": getattr(verification, "backend", ""),
            "returncode": getattr(verification, "returncode", None),
            "verifier_output": getattr(verification, "verifier_output", str(verification)),
        }
