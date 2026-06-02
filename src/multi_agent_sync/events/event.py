from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

EventType = Literal[
    "task_started",
    "plan_created",
    "agent_started",
    "finding",
    "message_received",
    "question",
    "warning",
    "critique",
    "error",
    "agent_done",
    "final_summary",
]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class AgentEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    source: str
    target: str | None = "broadcast"
    event_type: EventType
    content: str
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utc_now_iso)
