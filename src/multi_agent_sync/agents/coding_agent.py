from __future__ import annotations

from dataclasses import dataclass

from multi_agent_sync.agents.base import BaseAgent


@dataclass(kw_only=True)
class CodingAgent(BaseAgent):
    name: str = "CodingAgent"
    role: str = (
        "Thinks about implementation plan, modules, code structure, APIs, dependencies, "
        "and executable prototype."
    )
