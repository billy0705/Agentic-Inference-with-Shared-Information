from __future__ import annotations

from dataclasses import dataclass

from multi_agent_sync.agents.base import BaseAgent


@dataclass(kw_only=True)
class SolverAgent(BaseAgent):
    name: str = "SolverAgent"
    role: str = (
        "Solves math, science, physics, calculation, and direct reasoning problems. "
        "It should share concise intermediate findings that a verifier can check."
    )
