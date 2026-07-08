import asyncio
from dataclasses import dataclass

import pytest

from multi_agent_sync.agents.coding_agent import CodingAgent
from multi_agent_sync.agents.critic_agent import CriticAgent
from multi_agent_sync.agents.research_agent import ResearchAgent
from multi_agent_sync.events.event import AgentEvent
from multi_agent_sync.events.in_memory_streamer import InMemoryEventStreamer
from multi_agent_sync.tools.bash import BashTool
from multi_agent_sync.tracing.trace import TraceLogger
from multi_agent_sync.workspace.docker import BashResult, DockerWorkspace


@dataclass
class FakeResponse:
    content: str


class FakeLLM:
    async def ainvoke(self, prompt: str) -> FakeResponse:
        if "ResearchAgent" in prompt:
            return FakeResponse(
                "SUMMARY:\nReal-time chess needs synchronized moves.\n"
                "SHARE_FINDING:\nReal-time chess needs move synchronization across clients.\n"
                "CONFIDENCE:\n0.9\n"
                "LOCAL_NOTES:\nPrefer evented game state updates."
            )
        if "CodingAgent" in prompt:
            return FakeResponse(
                "SUMMARY:\nThe prototype needs a clean module structure.\n"
                "SHARE_FINDING:\nUse WebSocket-style channels for real-time game state sync.\n"
                "CONFIDENCE:\n0.82\n"
                "LOCAL_NOTES:\nExpose a small API around move updates."
            )
        if "CriticAgent" in prompt:
            return FakeResponse(
                "SUMMARY:\nThe plan needs validation safeguards.\n"
                "SHARE_FINDING:\nServer-side move validation is required to prevent illegal or cheating moves.\n"
                "CONFIDENCE:\n0.88\n"
                "LOCAL_NOTES:\nCheck concurrency and validation failure paths."
            )
        return FakeResponse("SUMMARY:\nNo-op.\nSHARE_FINDING:\n\nCONFIDENCE:\n0.5\nLOCAL_NOTES:\n")


class ToolLoopLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            assert "Docker workspace" in prompt
            return FakeResponse(
                'ACTION:\n{"tool": "bash", "command": "printf hello"}\n'
                "SHARE_FINDING:\nInspecting the workspace from Docker.\n"
                "CONFIDENCE:\n0.8\n"
                "LOCAL_NOTES:\nNeed the command output before finishing."
            )
        assert "printf hello" in prompt
        assert "hello from docker" in prompt
        return FakeResponse(
            "FINAL:\nThe Docker command output was observed and the agent can finish.\n"
            "SHARE_FINDING:\nObserved Docker command output successfully.\n"
            "CONFIDENCE:\n0.9\n"
            "LOCAL_NOTES:\nTool loop completed."
        )


class AgentFakeDockerWorkspace(DockerWorkspace):
    def __init__(self) -> None:
        super().__init__(container_name="agent-fake-container")
        self.commands: list[str] = []

    async def run_bash(self, command: str, *, timeout_seconds: float | None = None) -> BashResult:
        self.commands.append(command)
        return BashResult(
            command=command,
            exit_code=0,
            stdout="hello from docker\n",
            stderr="",
            timed_out=False,
            container_name=self.container_name,
        )


class VerifyCandidateLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            assert "verify_candidate" in prompt
            return FakeResponse(
                'ACTION:\n{"tool": "verify_candidate", "path": "/workspace/Main.lean"}\n'
                "SHARE_FINDING:\nChecking the current Lean candidate with the verifier.\n"
                "CONFIDENCE:\n0.8\n"
                "LOCAL_NOTES:\nNeed verifier feedback."
            )
        assert "unknown tactic" in prompt
        return FakeResponse(
            "FINAL:\nVerifier feedback was received and used.\n"
            "SHARE_FINDING:\nLean verifier feedback was available to the agent.\n"
            "CONFIDENCE:\n0.9\n"
            "LOCAL_NOTES:\nDone."
        )


class FakeLeanFeedbackTool:
    name = "verify_candidate"

    async def run(self, *, path: str | None = None):
        return {
            "tool": "verify_candidate",
            "path": path,
            "passed": False,
            "backend": "kimina-server",
            "returncode": 1,
            "verifier_output": '{"errors": [{"data": "unknown tactic"}]}',
        }


def build_agents(streamer: InMemoryEventStreamer, run_id: str = "run-agent-test"):
    llm = FakeLLM()
    kwargs = {
        "run_id": run_id,
        "task": "Build a prototype chess website",
        "llm": llm,
        "event_streamer": streamer,
        "max_steps": 2,
        "max_runtime_seconds": 5,
        "step_delay_seconds": 0.01,
    }
    return [
        ResearchAgent(assigned_subtask="Research real-time chess architecture", **kwargs),
        CodingAgent(assigned_subtask="Plan implementation modules", **kwargs),
        CriticAgent(assigned_subtask="Review safety and race conditions", **kwargs),
    ]


@pytest.mark.asyncio
async def test_agents_publish_receive_and_complete_under_timeout():
    streamer = InMemoryEventStreamer()
    research_agent, coding_agent, critic_agent = build_agents(streamer)

    for agent in (research_agent, coding_agent, critic_agent):
        agent.subscribe()

    await asyncio.wait_for(
        asyncio.gather(research_agent.run(), coding_agent.run(), critic_agent.run()),
        timeout=5,
    )
    await streamer.drain(timeout=1)

    events = await streamer.get_events(run_id="run-agent-test")
    research_findings = [event for event in events if event.source == "ResearchAgent" and event.event_type == "finding"]
    coding_received = [event for event in coding_agent.observed_events if event.source == "ResearchAgent"]
    critic_findings = [event for event in critic_agent.observed_events if event.event_type == "finding"]
    critiques = [event for event in events if event.source == "CriticAgent" and event.event_type in {"critique", "warning", "finding"}]
    done_events = [event for event in events if event.event_type == "agent_done"]

    assert research_findings
    assert coding_received
    assert critic_findings
    assert critiques
    assert len(done_events) == 3


@pytest.mark.asyncio
async def test_agent_ignores_own_events_and_duplicate_event_ids():
    streamer = InMemoryEventStreamer()
    trace_logger = TraceLogger()
    agent = ResearchAgent(
        run_id="run-duplicates",
        task="Build a prototype chess website",
        assigned_subtask="Research architecture",
        llm=FakeLLM(),
        event_streamer=streamer,
        trace_logger=trace_logger,
        max_steps=1,
        max_events_per_agent=5,
    )
    trace_logger.start_agent(agent.name, {"agent_name": agent.name})

    own_event = AgentEvent(run_id="run-duplicates", source="ResearchAgent", event_type="finding", content="Own finding")
    external_event = AgentEvent(run_id="run-duplicates", source="CodingAgent", event_type="finding", content="External finding")

    await agent.handle_event(own_event)
    await agent.handle_event(external_event)
    await agent.handle_event(external_event)

    assert agent.observed_events == [external_event]
    assert agent.seen_event_ids == {external_event.event_id}
    receipts = trace_logger.export()["ResearchAgent"]["event_receipts"]
    assert [receipt["ignored_reason"] for receipt in receipts] == [
        "self_event",
        None,
        "duplicate_event",
    ]
    assert receipts[1]["accepted"] is True
    assert receipts[1]["inbox_size_after"] == 1


@pytest.mark.asyncio
async def test_agent_stops_processing_events_after_done():
    streamer = InMemoryEventStreamer()
    trace_logger = TraceLogger()
    agent = ResearchAgent(
        run_id="run-after-done",
        task="Build a prototype chess website",
        assigned_subtask="Research architecture",
        llm=FakeLLM(),
        event_streamer=streamer,
        trace_logger=trace_logger,
        max_steps=1,
        step_delay_seconds=0,
    )

    await agent.run()
    late_event = AgentEvent(run_id="run-after-done", source="CodingAgent", event_type="warning", content="Late warning")
    await agent.handle_event(late_event)

    assert late_event not in agent.observed_events
    receipt = trace_logger.export()["ResearchAgent"]["event_receipts"][-1]
    assert receipt["event_id"] == late_event.event_id
    assert receipt["accepted"] is False
    assert receipt["ignored_reason"] == "agent_done"


@pytest.mark.asyncio
async def test_coding_agent_uses_docker_bash_observation_in_tool_loop():
    streamer = InMemoryEventStreamer()
    trace_logger = TraceLogger()
    workspace = AgentFakeDockerWorkspace()
    agent = CodingAgent(
        run_id="run-tool-loop",
        task="Inspect a file and finish.",
        assigned_subtask="Use bash in Docker to inspect the workspace.",
        llm=ToolLoopLLM(),
        event_streamer=streamer,
        trace_logger=trace_logger,
        bash_tool=BashTool(workspace),
        workspace_access="write",
        max_steps=2,
        step_delay_seconds=0,
    )

    output = await agent.run()

    assert workspace.commands == ["printf hello"]
    assert "The Docker command output was observed" in output
    trace = trace_logger.export()["CodingAgent"]
    first_step = trace["steps"][0]
    second_step = trace["steps"][1]
    assert first_step["parsed_output"]["tool_action"]["command"] == "printf hello"
    assert first_step["parsed_output"]["tool_result"]["stdout"] == "hello from docker\n"
    assert second_step["parsed_output"]["status"] == "final"


@pytest.mark.asyncio
async def test_coding_agent_can_request_lean_verifier_feedback_tool():
    streamer = InMemoryEventStreamer()
    trace_logger = TraceLogger()
    workspace = AgentFakeDockerWorkspace()
    agent = CodingAgent(
        run_id="run-lean-feedback",
        task="Complete a Lean proof.",
        assigned_subtask="Edit Main.lean and verify it.",
        llm=VerifyCandidateLLM(),
        event_streamer=streamer,
        trace_logger=trace_logger,
        bash_tool=BashTool(workspace),
        feedback_tool=FakeLeanFeedbackTool(),
        workspace_access="write",
        max_steps=2,
        step_delay_seconds=0,
    )

    output = await agent.run()

    assert "Verifier feedback was received" in output
    trace = trace_logger.export()["CodingAgent"]
    first_step = trace["steps"][0]
    assert first_step["parsed_output"]["tool_action"]["tool"] == "verify_candidate"
    assert first_step["parsed_output"]["tool_result"]["backend"] == "kimina-server"
    assert "unknown tactic" in first_step["parsed_output"]["tool_result"]["verifier_output"]
