from __future__ import annotations


MATH_DEBATE_BENCHMARKS = {"gsm8k", "olymmath"}
MULTIPLE_CHOICE_DEBATE_BENCHMARKS = {"gpqa", "mmlu_pro"}


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
