from __future__ import annotations

import os
from typing import Any

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


DEFAULT_OPENAI_BASE_URL = "http://localhost:8000/v1"


def get_openai_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).rstrip("/")


def get_llm(model: str | None = None, openai: bool = True, max_tokens: int = 4096) -> Any:
    if openai:
        llm = ChatOpenAI(
            model=model or os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b"),
            base_url=get_openai_base_url(),
            api_key="EMPTY",
            temperature=0,
            max_completion_tokens=max_tokens,
        )
    else:
        llm = ChatOllama(
            model=model or os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            temperature=0.2,
            num_predict=max_tokens,
        )
    return llm
