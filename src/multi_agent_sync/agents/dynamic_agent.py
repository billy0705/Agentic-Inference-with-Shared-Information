from __future__ import annotations

from dataclasses import dataclass

from multi_agent_sync.agents.base import BaseAgent


@dataclass(kw_only=True)
class DynamicAgent(BaseAgent):
    async def publish_finding(self, content: str, confidence: float = 0.7, step_index: int | None = None):
        if self.critical_debate:
            return await self.publish_event(
                "critique",
                content,
                confidence=confidence,
                metadata={"step_index": step_index} if step_index is not None else {},
            )
        return await super().publish_finding(content, confidence=confidence, step_index=step_index)
