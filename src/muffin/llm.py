import os
from typing import Any

from langchain_openai import ChatOpenAI


DEFAULT_OPENAI_BASE_URL = "http://localhost:8000/v1"


def get_openai_base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL).rstrip("/")


def get_llm(
    model: str | None = None,
    max_tokens: int = 4096,
    base_url: str | None = None,
) -> Any:
    return ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b"),
        base_url=(base_url or get_openai_base_url()).rstrip("/"),
        api_key="EMPTY",
        temperature=0,
        max_completion_tokens=max_tokens,
    )
