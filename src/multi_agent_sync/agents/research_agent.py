from __future__ import annotations

from dataclasses import dataclass

from multi_agent_sync.agents.base import BaseAgent


@dataclass(kw_only=True)
class ResearchAgent(BaseAgent):
    name: str = "ResearchAgent"
    role: str = (
        "Investigates background, design options, external constraints, assumptions, "
        "and high-level architecture."
    )
