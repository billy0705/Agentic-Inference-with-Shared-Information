from __future__ import annotations

import argparse
from typing import Any

from multi_agent_sync.evaluation.baselines.common import extract_answer_with_args, most_frequent_present_answer
from multi_agent_sync.token_usage import extract_token_usage


DEBATE_AGENT_COUNT = 3
DEBATE_ROUNDS = 2
MATH_DEBATE_BENCHMARKS = {"gsm8k", "olymmath"}
MULTIPLE_CHOICE_DEBATE_BENCHMARKS = {"gpqa", "mmlu_pro"}


async def run_multiagent_debate(prompt: str, llm: Any, args: argparse.Namespace) -> tuple[str, int, dict[str, Any]]:
    benchmark_name = str(getattr(args, "benchmark", "") or "")
    prompt_style = debate_prompt_style(benchmark_name)
    agent_contexts = [
        [{"role": "user", "content": build_debate_initial_prompt(prompt, prompt_style)}]
        for _ in range(DEBATE_AGENT_COUNT)
    ]
    agent_names = [f"DebateAgent{agent_index + 1}" for agent_index in range(DEBATE_AGENT_COUNT)]
    agent_traces: dict[str, dict[str, Any]] = {
        agent_name: {
            "assignment": {
                "method": "multiagent_debate",
                "agent": agent_index + 1,
                "prompt_style": prompt_style,
            },
            "steps": [],
            "event_receipts": [],
            "unused_received_events": [],
        }
        for agent_index, agent_name in enumerate(agent_names)
    }
    rounds: list[dict[str, Any]] = []

    for round_index in range(DEBATE_ROUNDS):
        round_trace: dict[str, Any] = {"round": round_index + 1, "agent_responses": []}
        for agent_index, agent_context in enumerate(agent_contexts):
            if round_index != 0:
                other_agent_contexts = agent_contexts[:agent_index] + agent_contexts[agent_index + 1 :]
                agent_context.append(
                    {
                        "role": "user",
                        "content": build_debate_round_prompt(
                            other_agent_contexts,
                            prompt,
                            2 * round_index - 1,
                            prompt_style,
                        ),
                    }
                )

            rendered_prompt = render_debate_context(agent_context)
            response = await llm.ainvoke(rendered_prompt)
            content = getattr(response, "content", str(response))
            assistant_message = {"role": "assistant", "content": content}
            agent_context.append(assistant_message)
            agent_traces[agent_names[agent_index]]["steps"].append(
                {
                    "step": round_index + 1,
                    "prompt": rendered_prompt,
                    "raw_response": content,
                    "parsed_output": {
                        "round": round_index + 1,
                        "output": content,
                    },
                    "published_events": [],
                    "inbox_events": [],
                    "used_event_ids": [],
                    "token_usage": extract_token_usage(response).as_dict(),
                }
            )
            round_trace["agent_responses"].append(
                {
                    "agent": agent_index + 1,
                    "agent_name": agent_names[agent_index],
                    "prompt": rendered_prompt,
                    "response": content,
                }
            )
        rounds.append(round_trace)

    final_outputs = [context[-1]["content"] for context in agent_contexts]
    parsed_final_answers = [extract_answer_with_args(output, args) for output in final_outputs]
    majority_answer = most_frequent_present_answer(parsed_final_answers)
    raw_output = f"Final Answer: {majority_answer}" if majority_answer is not None else final_outputs[0]
    agent_outputs = dict(zip(agent_names, final_outputs, strict=True))
    for agent_name, final_output in agent_outputs.items():
        agent_traces[agent_name]["final_output"] = final_output

    return raw_output, 0, {
        "method": "multiagent_debate",
        "prompt": prompt,
        "prompt_style": prompt_style,
        "agents": DEBATE_AGENT_COUNT,
        "rounds": DEBATE_ROUNDS,
        "agent_outputs": agent_outputs,
        "agent_traces": agent_traces,
        "agent_contexts": agent_contexts,
        "round_traces": rounds,
        "final_outputs": final_outputs,
        "parsed_final_answers": parsed_final_answers,
        "majority_answer": majority_answer,
        "raw_output": raw_output,
    }


def debate_prompt_style(benchmark_name: str) -> str:
    if benchmark_name in MATH_DEBATE_BENCHMARKS:
        return "math"
    if benchmark_name in MULTIPLE_CHOICE_DEBATE_BENCHMARKS:
        return "multiple_choice"
    return "generic"


def build_debate_initial_prompt(prompt: str, prompt_style: str) -> str:
    if prompt_style == "math":
        return (
            f"Can you solve the following math problem? {prompt} Explain your reasoning.\n"
            "Your final answer should be a single numerical number, in the form \\boxed{answer}, "
            "at the end of your response.\n"
        )
    if prompt_style == "multiple_choice":
        return (
            f"Can you answer the following question as accurately as possible? {prompt} "
            "Explain your answer, putting the answer in the form (X) at the end of your response."
        )
    return prompt


def build_debate_round_prompt(
    other_agent_contexts: list[list[dict[str, str]]],
    prompt: str,
    response_index: int,
    prompt_style: str,
) -> str:
    if not other_agent_contexts:
        if prompt_style == "math":
            return (
                "Can you double check that your answer is correct. Please reiterate your answer, "
                "with your final answer a single numerical number, in the form \\boxed{answer}."
            )
        return "Can you double check that your answer is correct. Put your final answer in the form (X) at the end of your response."

    prefix = "These are the solutions to the problem from other agents: "
    for agent_context in other_agent_contexts:
        agent_response = agent_context[response_index]["content"]
        prefix += f"\n\n One agent solution: ```{agent_response}```"

    if prompt_style == "math":
        return (
            prefix
            + "\n\n Using the solutions from other agents as additional information, can you provide your answer to the math problem? \n"
            + f" The original math problem is {prompt}.\n"
            + "Your final answer should be a single numerical number, in the form \\boxed{answer}, at the end of your response."
        )
    if prompt_style == "multiple_choice":
        return (
            prefix
            + "\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? "
            + "Examine your solution and that other agents step by step.\n"
            + "Put your answer in the form (X) at the end of your response."
        )
    return (
        prefix
        + "\n\n Using the reasoning from other agents as additional advice, can you give an updated answer? "
        + "Examine your solution and that other agents step by step."
    )


def render_debate_context(agent_context: list[dict[str, str]]) -> str:
    return "\n\n".join(f"{message['role']}: {message['content']}" for message in agent_context)
