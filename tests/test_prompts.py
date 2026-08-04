import inspect
import random

import pytest

from multi_agent_sync.agents.research_agent import ResearchAgent
from multi_agent_sync.evaluation import gpqa, gsm8k, hotpotqa, mmlu_pro
from multi_agent_sync.events.in_memory_streamer import InMemoryEventStreamer
from multi_agent_sync.graph import nodes
from multi_agent_sync.orchestrator import orchestrator
from multi_agent_sync.prompts import render_prompt


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
    assert "Shared findings:\n- No shared findings yet." in prompt
    assert "Check shared findings before answering." in prompt
    assert "Shared findings may include an answer/candidate plus a short reason; verify them before adopting." in prompt
    assert "<one useful finding for other agents; may include your current answer/candidate and one short reason" in prompt
    assert "Recent relevant events" not in prompt
    assert "Respond with concise summaries only." in prompt
    assert "ANSWER_CHOICE:" in prompt
    assert "ANSWER_REASON:" in prompt
    assert "NEXT_STEP:" not in prompt
    assert "CONFIDENCE:" not in prompt
    assert "confidence" not in prompt.lower()
    assert "Respond with concise summaries only." not in inspect.getsource(ResearchAgent.build_prompt)


def test_tool_agent_prompt_allows_sharing_candidate_and_short_reason():
    prompt = render_prompt(
        "agents/tool_step.j2",
        agent_name="CodingAgent",
        task="Fix a failing test",
        role="Implementer",
        description="",
        rules=[],
        critical_debate=False,
        assigned_subtask="Find the candidate fix.",
        workspace_access="write",
        is_reactive=False,
        reactive_reason="",
        step_index=1,
        max_steps=3,
        notes="- None yet.",
        events="- No shared findings yet.",
        tool_observations="- None yet.",
        feedback_tool_name="",
    )

    assert "Shared findings may include an answer/candidate plus a short reason; verify them before adopting." in prompt
    assert "<one useful finding for other agents; may include your current answer/candidate and one short reason" in prompt


def test_graph_prompts_are_not_embedded_in_node_functions():
    assert "Answer the user task directly with one concise response." not in inspect.getsource(nodes.direct_answer_node)
    assert "You are the Summarizer for a LangGraph multi-agent prototype." not in inspect.getsource(nodes.synthesizer_node)
    assert "Do not solve the task again." not in inspect.getsource(nodes.synthesizer_node)


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
    assert "You may choose direct mode when the task can be answered well by one direct response." in prompt
    assert "The reason must explain why direct or multi_agent was selected." in prompt
    assert "You are the model-based orchestrator" not in inspect.getsource(orchestrator.build_orchestrator_prompt)


def test_dynamic_orchestrator_prompt_allows_direct_and_requires_detailed_subagents():
    prompt = orchestrator.build_dynamic_orchestrator_prompt("Analyze this benchmark result.")

    assert len(prompt) < 3200
    assert "You are the dynamic subagent orchestrator" in prompt
    assert "Default to multi_agent mode." in prompt
    assert "You may choose direct mode only when you are truthfully 100% certain" in prompt
    assert "The reason must explain truthfully why direct or multi_agent was selected." in prompt
    assert "Never claim 100% certainty for a normal benchmark question unless the answer is explicitly given in the prompt." in prompt
    assert "Diverse subagents" in prompt
    assert "If multi_agent, create 2 to 4 subagents." in prompt
    assert "dependency order" not in prompt
    assert '"depends_on"' not in prompt
    assert "Do not create overlapping roles" in prompt
    assert "evidence, check, or perspective" in prompt
    assert '"direct_certainty": "100_percent"' in prompt
    assert '"mode": "direct"' in prompt
    assert '"role": "specific expertise and responsibility for this subagent"' in prompt
    assert '"description": "unique evidence, constraints, checks, or perspective' in prompt
    assert "Analyze this benchmark result." in prompt


def test_ordered_dynamic_orchestrator_prompt_requests_dependency_order():
    prompt = orchestrator.build_dynamic_orchestrator_prompt(
        "Analyze this benchmark result.",
        ordered_step_one=True,
    )

    assert "define the step-one dependency DAG" in prompt
    assert '"depends_on": []' in prompt
    assert "Do not create self-dependencies, unknown references, or cycles." in prompt
    assert "If multi_agent, create 2 to 4 subagents." not in prompt
