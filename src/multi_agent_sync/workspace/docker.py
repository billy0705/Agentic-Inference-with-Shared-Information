from __future__ import annotations

import asyncio
import os
import shlex
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

    _SHELL_SEPARATORS = {";", "&&", "||", "|", "&"}

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
        rejection = self.validate_bash_command(command)
        if rejection is not None:
            return BashResult(
                command=command,
                exit_code=126,
                stdout="",
                stderr=rejection,
                timed_out=False,
                container_name=self.container_name,
            )

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

    def validate_bash_command(self, command: str) -> str | None:
        tokens = self._tokenize_bash_command(command)
        if tokens is None:
            return None

        expect_command = True
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token in self._SHELL_SEPARATORS:
                expect_command = True
                index += 1
                continue
            if not expect_command:
                index += 1
                continue
            if self._is_shell_assignment(token):
                index += 1
                continue
            if os.path.basename(token) != "rm":
                expect_command = False
                index += 1
                continue

            operands = self._rm_operands(tokens[index + 1 :])
            dangerous_target = next((target for target in operands if self._is_workspace_destroy_target(target)), None)
            if dangerous_target is not None:
                return (
                    "Refusing to run destructive workspace command: "
                    f"`rm` targets `{dangerous_target}` from Docker workdir `{self.workdir}`. "
                    "Use git cleanup commands or remove a specific subdirectory instead."
                )
            expect_command = False
            index += 1
        return None

    def _tokenize_bash_command(self, command: str) -> list[str] | None:
        try:
            lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            lexer.commenters = ""
            return list(lexer)
        except ValueError:
            return None

    def _rm_operands(self, tokens: list[str]) -> list[str]:
        operands: list[str] = []
        parsing_options = True
        for token in tokens:
            if token in self._SHELL_SEPARATORS:
                break
            if parsing_options and token == "--":
                parsing_options = False
                continue
            if parsing_options and token.startswith("-") and token != "-":
                continue
            operands.append(token)
        return operands

    def _is_workspace_destroy_target(self, target: str) -> bool:
        normalized_workdir = self.workdir.rstrip("/") or "/"
        normalized_target = target.rstrip("/") or "/"
        if normalized_target in {".", normalized_workdir, "$PWD", "${PWD}"}:
            return True
        if target in {"*", ".*", "./*", "./.*"}:
            return True
        return target in {f"{normalized_workdir}/*", f"{normalized_workdir}/.*"}

    def _is_shell_assignment(self, token: str) -> bool:
        if "=" not in token or token.startswith("="):
            return False
        name, _value = token.split("=", 1)
        return name.replace("_", "A").isalnum() and not name[0].isdigit()

    async def read_text(self, path: str) -> str:
        result = await self.run_bash(f"cat {shlex.quote(path)}")
        if result.exit_code != 0:
            raise RuntimeError(f"Could not read {path} from Docker workspace: {result.stderr or result.stdout}")
        return result.stdout

    async def write_text(self, path: str, content: str) -> None:
        quoted_path = shlex.quote(path)
        quoted_parent = shlex.quote(str(Path(path).parent))
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "-i",
            "-w",
            self.workdir,
            self.container_name,
            "bash",
            "-lc",
            f"mkdir -p {quoted_parent} && cat > {quoted_path}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate(content.encode("utf-8"))
        exit_code = int(process.returncode or 0)
        if exit_code != 0:
            stdout = self._decode_and_limit(stdout_bytes)
            stderr = self._decode_and_limit(stderr_bytes)
            raise RuntimeError(f"Could not write {path} in Docker workspace: {stderr or stdout}")

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
