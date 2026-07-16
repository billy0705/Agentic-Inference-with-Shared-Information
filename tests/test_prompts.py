import inspect
import random

import pytest

from multi_agent_sync.agents.research_agent import ResearchAgent
from multi_agent_sync.evaluation import gpqa, gsm8k, hotpotqa, mmlu_pro
from multi_agent_sync.events.in_memory_streamer import InMemoryEventStreamer
from multi_agent_sync.graph import nodes
from multi_agent_sync.orchestrator import orchestrator


def test_render_prompt_loads_package_template():
    from multi_agent_sync.prompts import render_prompt

    prompt = render_prompt(
        "evaluation/gpqa_question.j2",
        question="What is the answer?",
        option_lines=["A. First", "B. Second"],
    )

    assert "Question:\nWhat is the answer?" in prompt
    assert "A. First\nB. Second" in prompt
    assert prompt == prompt.strip()


def test_gpqa_prompt_is_rendered_from_template():
    prompt, correct_label = gpqa.build_prompt(
        {
            "Question": "Which option is correct?",
            "Correct Answer": "Correct answer",
            "Incorrect Answer 1": "Wrong answer 1",
            "Incorrect Answer 2": "Wrong answer 2",
            "Incorrect Answer 3": "Wrong answer 3",
        },
        random.Random(0),
    )

    assert correct_label in {"A", "B", "C", "D"}
    assert "Which option is correct?" in prompt
    assert "Final Answer: <A/B/C/D>" in prompt
    assert "You are answering a difficult graduate-level science multiple-choice question." not in inspect.getsource(
        gpqa.build_prompt
    )


def test_gsm8k_prompt_is_rendered_from_template():
    prompt, gold = gsm8k.build_prompt(
        {
            "question": "Natalia sold 48 clips in April and half as many in May. How many did she sell?",
            "answer": "Natalia sold 48/2 = <<48/2=24>>24 in May. #### 72",
        },
        random.Random(0),
    )

    assert gold == "72"
    assert "Natalia sold 48 clips" in prompt
    assert "Final Answer: <number>" in prompt
    assert "You are solving a grade-school math word problem." not in inspect.getsource(gsm8k.build_prompt)


def test_hotpotqa_prompt_is_rendered_from_template():
    prompt, gold = hotpotqa.build_prompt(
        {
            "question": "What city is the capital of France?",
            "answer": "Paris",
            "context": {
                "title": ["France", "Paris"],
                "sentences": [
                    ["France is a country in Europe.", "Its capital is Paris."],
                    ["Paris is the capital and most populous city of France."],
                ],
            },
        },
        random.Random(0),
    )

    assert gold == "paris"
    assert "What city is the capital of France?" in prompt
    assert "Title: France" in prompt
    assert "Title: Paris" in prompt
    assert "Final Answer: <short answer>" in prompt
    assert "You are answering a HotpotQA multi-hop question." not in inspect.getsource(hotpotqa.build_prompt)


def test_mmlu_pro_prompt_is_rendered_from_template():
    prompt, gold = mmlu_pro.build_prompt(
        {
            "question": "Which option is correct?",
            "options": ["First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth", "Tenth"],
            "answer": "J",
            "answer_index": 9,
            "category": "science",
        },
        random.Random(0),
    )

    assert gold == "J"
    assert "Category: science" in prompt
    assert "A. First" in prompt
    assert "J. Tenth" in prompt
    assert "Final Answer: <A/B/C/D/E/F/G/H/I/J>" in prompt
    assert "You are answering a challenging multiple-choice question from MMLU-Pro." not in inspect.getsource(
        mmlu_pro.build_prompt
    )


@pytest.mark.asyncio
async def test_agent_prompt_is_rendered_from_template():
    agent = ResearchAgent(
        run_id="run-prompts",
        task="Build a chess website",
        assigned_subtask="Research synchronization",
        llm=None,
        event_streamer=InMemoryEventStreamer(),
    )

    prompt = await agent.build_prompt(step_index=1, relevant_events=[])

    expected_sections = [
        "[GLOBAL STATIC PREFIX]",
        "[AGENT STATIC PREFIX]",
        "[STEP DYNAMIC SUFFIX]",
    ]
    section_positions = [prompt.index(section) for section in expected_sections]

    assert section_positions == sorted(section_positions)
    assert "[GLOBAL STATIC PREFIX]\nBuild a chess website" in prompt
    assert "Agent name:\nResearchAgent" in prompt
    assert "Assigned subtask:\nResearch synchronization" in prompt
    assert "Current step:\n1 of 3" in prompt
    assert "Local notes:\n- None yet." in prompt
    assert "Recent relevant events:\n- No relevant external findings yet." in prompt
    assert "Respond with concise summaries only." in prompt
    assert "Respond with concise summaries only." not in inspect.getsource(ResearchAgent.build_prompt)


def test_graph_prompts_are_not_embedded_in_node_functions():
    assert "Answer the user task directly with one concise response." not in inspect.getsource(nodes.direct_answer_node)
    assert "You are the Synthesizer for a LangGraph multi-agent prototype." not in inspect.getsource(nodes.synthesizer_node)


def test_orchestrator_prompt_is_rendered_from_template():
    prompt = orchestrator.build_orchestrator_prompt(
        "Build a chess app",
        {
            "ResearchAgent": object(),
            "CodingAgent": object(),
        },
    )

    assert "You are the model-based orchestrator for a local LangGraph multi-agent system." in prompt
    assert "Build a chess app" in prompt
    assert '"mode": "multi_agent"' in prompt
    assert '"mode": "direct"' in prompt
    assert "Only use direct mode for easy factual tasks." in prompt
    assert "You are the model-based orchestrator" not in inspect.getsource(orchestrator.build_orchestrator_prompt)
