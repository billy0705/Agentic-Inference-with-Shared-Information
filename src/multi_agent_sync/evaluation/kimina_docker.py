from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_KIMINA_DOCKER_IMAGE = "projectnumina/kimina-lean-server:2.0.0"
DEFAULT_KIMINA_DOCKER_CONTAINER = "multi-agent-kimina-lean-server"
DEFAULT_KIMINA_CONTAINER_PORT = 8000


@dataclass
class KiminaDockerServer:
    host: str = "127.0.0.1"
    host_port: int = 8001
    image: str = DEFAULT_KIMINA_DOCKER_IMAGE
    container_name: str = DEFAULT_KIMINA_DOCKER_CONTAINER
    container_port: int = DEFAULT_KIMINA_CONTAINER_PORT
    startup_timeout: float = 120.0
    poll_interval: float = 1.0
    started_container: bool = False

    async def ensure_running(self) -> None:
        if await self.is_available():
            return

        start_exit_code, _, _ = await self._docker("start", self.container_name, check=False)
        if start_exit_code == 0:
            self.started_container = True
            await self.wait_until_available()
            return

        await self._docker(
            "run",
            "-d",
            "--name",
            self.container_name,
            "-p",
            f"{self.host_port}:{self.container_port}",
            self.image,
        )
        self.started_container = True
        await self.wait_until_available()

    async def cleanup(self) -> None:
        if self.started_container:
            await self._docker("rm", "-f", self.container_name, check=False)

    async def wait_until_available(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        last_error = "Kimina Lean Server did not respond."
        while time.monotonic() < deadline:
            if await self.is_available():
                return
            await asyncio.sleep(self.poll_interval)
        raise RuntimeError(
            f"Kimina Lean Server did not become available at http://{self.host}:{self.host_port} "
            f"within {self.startup_timeout:g}s. Last error: {last_error}"
        )

    async def is_available(self) -> bool:
        return await asyncio.to_thread(self._is_available_sync)

    def _is_available_sync(self) -> bool:
        request = Request(f"http://{self.host}:{self.host_port}/docs", method="GET")
        try:
            with urlopen(request, timeout=2) as response:
                return 200 <= int(response.status) < 500
        except (OSError, URLError):
            return False

    async def _docker(self, *args: str, check: bool = True) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate()
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        exit_code = int(process.returncode or 0)
        if check and exit_code != 0:
            raise RuntimeError(f"docker {' '.join(args)} failed with exit code {exit_code}: {stderr or stdout}")
        return exit_code, stdout, stderr
