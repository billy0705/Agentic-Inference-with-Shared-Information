"""Evaluation benchmark runners for multi_agent_sync."""

import sys

from multi_agent_sync.evaluation import baselines, benchmarks, main, runner
from multi_agent_sync.evaluation.benchmarks import chess, gpqa, gsm8k, ma_proofbench, mmlu_pro, olymmath

sys.modules[__name__ + ".chess"] = chess
sys.modules[__name__ + ".gpqa"] = gpqa
sys.modules[__name__ + ".gsm8k"] = gsm8k
sys.modules[__name__ + ".ma_proofbench"] = ma_proofbench
sys.modules[__name__ + ".mmlu_pro"] = mmlu_pro
sys.modules[__name__ + ".olymmath"] = olymmath

__all__ = [
    "baselines",
    "benchmarks",
    "chess",
    "gpqa",
    "gsm8k",
    "ma_proofbench",
    "main",
    "mmlu_pro",
    "olymmath",
    "runner",
]
