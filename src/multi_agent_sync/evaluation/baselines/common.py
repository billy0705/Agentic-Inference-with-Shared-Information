from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any


@dataclass
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens = add_optional_ints(self.prompt_tokens, other.prompt_tokens)
        self.completion_tokens = add_optional_ints(self.completion_tokens, other.completion_tokens)
        self.total_tokens = add_optional_ints(self.total_tokens, other.total_tokens)


class MeteredLLM:
    def __init__(self, llm: Any) -> None:
        self._llm = llm
        self.usage = TokenUsage()

    async def ainvoke(self, prompt: str) -> Any:
        response = await self._llm.ainvoke(prompt)
        self.usage.add(extract_token_usage(response))
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._llm, name)


def add_optional_ints(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_token_usage(response: Any) -> TokenUsage:
    usage_metadata = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}

    prompt_tokens = coerce_int(
        usage_metadata.get("input_tokens")
        or usage_metadata.get("prompt_tokens")
        or token_usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
    )
    completion_tokens = coerce_int(
        usage_metadata.get("output_tokens")
        or usage_metadata.get("completion_tokens")
        or token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
    )
    total_tokens = coerce_int(usage_metadata.get("total_tokens") or token_usage.get("total_tokens"))
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def extract_answer_with_args(output: str, args: argparse.Namespace) -> str | None:
    answer_extractor = getattr(args, "answer_extractor", None)
    if callable(answer_extractor):
        return answer_extractor(output)
    return output.strip() or None


def most_frequent_present_answer(answers: list[str | None]) -> str | None:
    present_answers = [answer for answer in answers if answer is not None]
    if not present_answers:
        return None

    most_frequent = present_answers[0]
    highest_count = 0
    for answer in present_answers:
        count = present_answers.count(answer)
        if count > highest_count:
            highest_count = count
            most_frequent = answer
    return most_frequent
