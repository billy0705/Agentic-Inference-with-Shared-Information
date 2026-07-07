from __future__ import annotations

from typing import Any


async def run_plain_llm(prompt: str, llm: Any) -> tuple[str, int]:
    response = await llm.ainvoke(prompt)
    return getattr(response, "content", str(response)), 0
