from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from multi_agent_sync.events.in_memory_streamer import InMemoryEventStreamer
from multi_agent_sync.graph.nodes import direct_answer_node, orchestrator_node, route_after_orchestrator, run_multi_agent_runtime_node, synthesizer_node
from multi_agent_sync.graph.state import GraphState
from multi_agent_sync.streaming.console import ConsoleEventStreamer


def build_workflow():
    graph = StateGraph(GraphState)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("direct_answer", direct_answer_node)
    graph.add_node("run_multi_agent_runtime", run_multi_agent_runtime_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_edge(START, "orchestrator")
    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "direct_answer": "direct_answer",
            "run_multi_agent_runtime": "run_multi_agent_runtime",
        },
    )
    graph.add_edge("direct_answer", END)
    graph.add_edge("run_multi_agent_runtime", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()


async def run_workflow(
    task: str,
    llm: Any | None = None,
    subagent_mode: str = "fixed",
    max_steps_per_agent: int = 3,
    total_runtime_timeout: float = 600.0,
    synthesis_timeout: float = 60.0,
    enable_agent_message_streaming: bool = True,
    stream_to_console: bool = True,
    no_color: bool = False,
) -> GraphState:
    workflow = build_workflow()
    event_streamer = InMemoryEventStreamer()
    if stream_to_console:
        ConsoleEventStreamer(event_streamer, no_color=no_color)
    initial_state: GraphState = {
        "task": task,
        "run_id": str(uuid4()),
        "mode": "",
        "subagent_mode": subagent_mode,
        "task_type": "",
        "reason": "",
        "plan": [],
        "selected_agents": [],
        "assignments": [],
        "orchestrator_plan": {},
        "event_log": [],
        "agent_outputs": {},
        "agent_traces": {},
        "final_answer": "",
        "event_streamer": event_streamer,
        "max_steps_per_agent": max_steps_per_agent,
        "total_runtime_timeout": total_runtime_timeout,
        "synthesis_timeout": synthesis_timeout,
        "enable_agent_message_streaming": enable_agent_message_streaming,
        "stream_to_console": stream_to_console,
        "no_color": no_color,
    }
    if llm is not None:
        initial_state["llm"] = llm
    return await workflow.ainvoke(initial_state)
