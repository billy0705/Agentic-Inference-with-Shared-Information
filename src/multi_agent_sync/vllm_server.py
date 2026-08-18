from __future__ import annotations

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
    server = manager.start_server()
    os.environ["OPENAI_MODEL"] = manager.config.model_name
    os.environ["OPENAI_BASE_URL"] = f"{server.base_url}/v1"
    try:
        server.wait_until_ready(timeout=timeout)
    except BaseException:
        server.stop()
        raise
    return server
