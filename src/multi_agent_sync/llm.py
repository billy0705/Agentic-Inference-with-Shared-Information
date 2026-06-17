from __future__ import annotations

import os
from typing import Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_llm(model: str | None = None, openai: bool = True) -> Any:
    if openai:
        llm = ChatOpenAI(
            model=model or os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b"),
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            temperature=0,
        )
    else:
        llm = ChatOllama(
            model=model or os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            temperature=0.2,
        )
    return llm
