from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class BashResult:
    command: str
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    container_name: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class DockerWorkspace:
    container_name: str
    workdir: str = "/workspace"
    output_char_limit: int = 12000
    default_timeout_seconds: float = 60.0

    @classmethod
    async def create(
        cls,
        *,
        image: str,
        source_path: str | Path | None = None,
        name_prefix: str = "multi-agent-sync",
        workdir: str = "/workspace",
        output_char_limit: int = 12000,
        default_timeout_seconds: float = 60.0,
    ) -> "DockerWorkspace":
        container_name = f"{name_prefix}-{uuid4().hex[:12]}"
        workspace = cls(
            container_name=container_name,
            workdir=workdir,
            output_char_limit=output_char_limit,
            default_timeout_seconds=default_timeout_seconds,
        )
        await workspace._docker("run", "-d", "--name", container_name, "-w", workdir, image, "sleep", "infinity")
        await workspace._docker("exec", container_name, "mkdir", "-p", workdir)
        if source_path is not None:
            source = Path(source_path).expanduser().resolve()
            if not source.exists():
                raise RuntimeError(f"Workspace source path does not exist: {source}")
            await workspace._docker("cp", f"{source}/.", f"{container_name}:{workdir}")
        return workspace

    async def run_bash(self, command: str, *, timeout_seconds: float | None = None) -> BashResult:
        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-w",
            self.workdir,
            self.container_name,
            "bash",
            "-lc",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
            timed_out = False
            exit_code = int(process.returncode or 0)
        except asyncio.TimeoutError:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
            timed_out = True
            exit_code = 124

        return BashResult(
            command=command,
            exit_code=exit_code,
            stdout=self._decode_and_limit(stdout_bytes),
            stderr=self._decode_and_limit(stderr_bytes),
            timed_out=timed_out,
            container_name=self.container_name,
        )

    async def cleanup(self) -> None:
        await self._docker("rm", "-f", self.container_name, check=False)

    async def _docker(self, *args: str, check: bool = True) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = self._decode_and_limit(stdout_bytes)
        stderr = self._decode_and_limit(stderr_bytes)
        exit_code = int(process.returncode or 0)
        if check and exit_code != 0:
            raise RuntimeError(f"docker {' '.join(args)} failed with exit code {exit_code}: {stderr or stdout}")
        return exit_code, stdout, stderr

    def _decode_and_limit(self, value: bytes) -> str:
        text = value.decode("utf-8", errors="replace")
        if len(text) <= self.output_char_limit:
            return text
        omitted = len(text) - self.output_char_limit
        return f"{text[: self.output_char_limit]}\n...[truncated {omitted} chars]"
