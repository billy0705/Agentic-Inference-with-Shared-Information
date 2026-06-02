from __future__ import annotations

import os

from langchain_ollama import ChatOllama


def get_llm(model: str | None = None) -> ChatOllama:
    return ChatOllama(
        model=model or os.getenv("OLLAMA_MODEL", "qwen3:4b"),
        temperature=0.2,
    )
