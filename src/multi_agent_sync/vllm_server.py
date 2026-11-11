from __future__ import annotations

import copy
from dataclasses import replace
import os
from pathlib import Path
from typing import Any


def add_spinup_server_arguments(parser: Any) -> None:
    parser.add_argument(
        "--spinup-server",
        "--spinup_server",
        dest="spinup_server",
        action="store_true",
        help="Start the configured vLLM server for this run and stop it afterward.",
    )
    parser.add_argument(
        "--server-config",
        type=Path,
        default=None,
        help="Start vLLM with this launcher configuration.",
    )
    parser.add_argument(
        "--server-startup-timeout",
        type=float,
        default=1800.0,
        help="Seconds to wait for a spun-up vLLM server. Defaults to 1800.",
    )


def should_spinup_server(args: Any) -> bool:
    return bool(args.spinup_server or args.server_config is not None)


def resolve_server_config(args: Any) -> Path:
    return args.server_config or Path("server.yaml")


def _optional_positive_number(section: dict[str, Any], key: str) -> int | float | None:
    value = section.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"'vllm.{key}' must be a positive number")
    return value


def _without_options(arguments: tuple[str, ...], options: set[str]) -> tuple[str, ...]:
    filtered: list[str] = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in options:
            index += 1
            if index < len(arguments) and not arguments[index].startswith("--"):
                index += 1
            continue
        filtered.append(argument)
        index += 1
    return tuple(filtered)


def apply_vllm_launch_overrides(manager: Any) -> None:
    vllm = manager.section("vllm")
    overrides: list[str] = []

    tensor_parallel_size = _optional_positive_number(vllm, "tensor_parallel_size")
    if tensor_parallel_size is not None:
        overrides.extend(("--tensor-parallel-size", str(tensor_parallel_size)))

    max_model_len = _optional_positive_number(vllm, "max_model_len")
    if max_model_len is not None:
        overrides.extend(("--max-model-len", str(max_model_len)))

    gpu_memory_utilization = _optional_positive_number(vllm, "gpu_memory_utilization")
    if gpu_memory_utilization is None:
        gpu_memory_utilization = _optional_positive_number(vllm, "gpu_memory_utilisation")
    if gpu_memory_utilization is not None:
        overrides.extend(("--gpu-memory-utilization", str(gpu_memory_utilization)))

    if overrides:
        overridden_options = {argument for argument in overrides if argument.startswith("--")}
        arguments = _without_options(manager.config.arguments, overridden_options)
        manager.config = replace(manager.config, arguments=(*arguments, *overrides))


def spinup_server(config_path: Any, *, timeout: float = 1800.0) -> Any:
    """Start vLLM, configure this process to use it, and wait until it is ready."""

    try:
        from vllm_server_launcher import ConfigManager
    except ImportError as exc:
        raise RuntimeError(
            "--spinup-server requires vllm-server-launcher. Install the internal "
            "package in this environment first."
        ) from exc

    manager = config_path if isinstance(config_path, ConfigManager) else ConfigManager(config_path)
    subagent_manager = build_subagent_manager(manager)
    servers: list[Any] = []
    try:
        apply_vllm_launch_overrides(manager)
        servers.append(_start_with_visible_devices(manager))
        os.environ["OPENAI_MODEL"] = manager.config.model_name
        os.environ["OPENAI_BASE_URL"] = f"{servers[0].base_url}/v1"
        if subagent_manager is not None:
            apply_vllm_launch_overrides(subagent_manager)
            servers.append(_start_with_visible_devices(subagent_manager))
            os.environ[SUBAGENT_MODEL_ENV] = subagent_manager.config.model_name
            os.environ[SUBAGENT_BASE_URL_ENV] = f"{servers[1].base_url}/v1"
        else:
            os.environ.pop(SUBAGENT_MODEL_ENV, None)
            os.environ.pop(SUBAGENT_BASE_URL_ENV, None)
        for server in servers:
            server.wait_until_ready(timeout=timeout)
    except BaseException:
        for server in servers:
            server.stop()
        raise
    return servers[0] if len(servers) == 1 else ServerGroup(servers)


SUBAGENT_SECTION = "vllm_subagent"
SUBAGENT_MODEL_ENV = "OPENAI_SUBAGENT_MODEL"
SUBAGENT_BASE_URL_ENV = "OPENAI_SUBAGENT_BASE_URL"


class ServerGroup:
    """Stops several vLLM servers together; mirrors the single-server interface used here."""

    def __init__(self, servers: list[Any]) -> None:
        self.servers = servers

    @property
    def base_url(self) -> str:
        return self.servers[0].base_url

    def stop(self) -> None:
        for server in reversed(self.servers):
            server.stop()


def build_subagent_manager(manager: Any) -> Any | None:
    """Return a launcher manager for the optional ``vllm_subagent`` server, or None.

    The section accepts the same keys as ``vllm``. ``sif_path`` and
    ``engine_ready_timeout`` are inherited from ``vllm`` when omitted; ``port``
    is required and must differ from the main server; ``log_path`` defaults to
    ``vllm-server-subagent.log``.
    """

    subagent = manager.section(SUBAGENT_SECTION)
    if not subagent:
        return None
    main = manager.section("vllm")
    if "port" not in subagent:
        raise ValueError(f"'{SUBAGENT_SECTION}.port' is required")
    if subagent["port"] == main.get("port", 8000):
        raise ValueError(f"'{SUBAGENT_SECTION}.port' must differ from 'vllm.port'")
    merged = {
        "sif_path": main.get("sif_path"),
        "engine_ready_timeout": main.get("engine_ready_timeout"),
        "log_path": "vllm-server-subagent.log",
        **subagent,
    }
    sub_manager = copy.copy(manager)
    # Re-parse with the subagent section in place of ``vllm`` so relative paths
    # still resolve against the original YAML file.
    sub_manager._data = {**manager._data, "vllm": merged}
    sub_manager.config = sub_manager._parse_launcher_config()
    return sub_manager


def _start_with_visible_devices(manager: Any) -> Any:
    """Start a server, pinning it to ``vllm.cuda_visible_devices`` when configured."""

    devices = manager.section("vllm").get("cuda_visible_devices")
    if devices is None:
        return manager.start_server()
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
    try:
        return manager.start_server()
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
