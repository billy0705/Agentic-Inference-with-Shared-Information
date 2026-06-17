from __future__ import annotations

from dataclasses import dataclass, field

from multi_agent_sync.agents.base import BaseAgent


@dataclass(kw_only=True)
class VerifierAgent(BaseAgent):
    name: str = "VerifierAgent"
    role: str = (
        "Checks another agent's reasoning for numerical correctness, unit conversion, "
        "missing assumptions, contradictions, and overclaiming."
    )
    reactive_steps_enabled: bool = True
    max_reactive_steps: int = 1
    reactive_event_types: set[str] = field(default_factory=lambda: {"finding", "critique", "warning"})

    async def publish_finding(self, content: str, confidence: float = 0.7, step_index: int | None = None):
        return await self.publish_event(
            "critique",
            content,
            confidence=confidence,
            metadata={"step_index": step_index} if step_index is not None else {},
        )
