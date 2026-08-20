from dataclasses import dataclass
import pytest

from multi_agent_sync.vllm_server import apply_vllm_launch_overrides


@dataclass(frozen=True)
class LauncherConfigStub:
    arguments: tuple[str, ...] = ()


class ConfigManagerStub:
    def __init__(self, vllm, arguments=("--trust-remote-code",)):
        self._vllm = vllm
        self.config = LauncherConfigStub(arguments)

    def section(self, name):
        assert name == "vllm"
        return dict(self._vllm)


def test_apply_vllm_launch_overrides_passes_config_values_as_launcher_arguments():
    manager = ConfigManagerStub({"tensor_parallel_size": 4, "max_model_len": 64000, "gpu_memory_utilization": 0.95})

    apply_vllm_launch_overrides(manager)

    assert manager.config.arguments == (
        "--trust-remote-code",
        "--tensor-parallel-size",
        "4",
        "--max-model-len",
        "64000",
        "--gpu-memory-utilization",
        "0.95",
    )


def test_apply_vllm_launch_overrides_accepts_gpu_memory_utilisation_alias():
    manager = ConfigManagerStub({"gpu_memory_utilisation": 0.9})

    apply_vllm_launch_overrides(manager)

    assert manager.config.arguments[-2:] == ("--gpu-memory-utilization", "0.9")


def test_apply_vllm_launch_overrides_replaces_existing_model_arguments():
    manager = ConfigManagerStub(
        {"tensor_parallel_size": 4, "max_model_len": 64000, "gpu_memory_utilization": 0.95},
        (
            "--tensor-parallel-size",
            "1",
            "--reasoning-parser",
            "gemma4",
            "--max-model-len",
            "21000",
            "--gpu-memory-utilization",
            "0.90",
        ),
    )

    apply_vllm_launch_overrides(manager)

    assert manager.config.arguments == (
        "--reasoning-parser",
        "gemma4",
        "--tensor-parallel-size",
        "4",
        "--max-model-len",
        "64000",
        "--gpu-memory-utilization",
        "0.95",
    )


@pytest.mark.parametrize(
    "vllm",
    [
        {"tensor_parallel_size": 0},
        {"max_model_len": 0},
        {"gpu_memory_utilization": 0},
        {"gpu_memory_utilisation": "0.95"},
    ],
)
def test_apply_vllm_launch_overrides_rejects_invalid_values(vllm):
    with pytest.raises(ValueError, match="vllm"):
        apply_vllm_launch_overrides(ConfigManagerStub(vllm))
