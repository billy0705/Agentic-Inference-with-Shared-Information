from __future__ import annotations

import argparse
from typing import Any


from multi_agent_sync.token_usage import TokenUsage, add_optional_ints, coerce_int, extract_token_usage


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
