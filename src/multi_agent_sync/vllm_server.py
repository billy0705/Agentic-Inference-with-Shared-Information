from __future__ import annotations

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
    apply_vllm_launch_overrides(manager)
    server = manager.start_server()
    os.environ["OPENAI_MODEL"] = manager.config.model_name
    os.environ["OPENAI_BASE_URL"] = f"{server.base_url}/v1"
    try:
        server.wait_until_ready(timeout=timeout)
    except BaseException:
        server.stop()
        raise
    return server
