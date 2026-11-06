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
    usage_metadata: dict[str, int] | None = None
    response_metadata: dict | None = None


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


@pytest.mark.asyncio
async def test_agent_prompt_respects_configured_shared_finding_limit():
    streamer = InMemoryEventStreamer()
    agent = ResearchAgent(
        run_id="run-shared-limit",
        task="Compare findings.",
        assigned_subtask="Use recent shared context.",
        llm=FakeLLM(),
        event_streamer=streamer,
        shared_finding_limit=2,
    )
    for index in range(4):
        await streamer.publish(
            AgentEvent(
                run_id="run-shared-limit",
                source=f"Agent{index}",
                event_type="finding",
                content=f"finding-{index}",
            )
        )

    prompt = await agent.build_prompt(step_index=1)

    assert "finding-0" not in prompt
    assert "finding-1" not in prompt
    assert "finding-2" in prompt
    assert "finding-3" in prompt


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


class FinalGuardRetryLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return FakeResponse(
                "FINAL:\nI am done without editing files.\n"
                "SHARE_FINDING:\nAttempting to finish.\n"
                "CONFIDENCE:\n0.7\n"
                "LOCAL_NOTES:\nNeed guard result."
            )
        if len(self.prompts) == 2:
            assert "final_guard_failed" in prompt
            assert "git diff is empty" in prompt
            return FakeResponse(
                'ACTION:\n{"tool": "bash", "command": "printf edited"}\n'
                "SHARE_FINDING:\nApplying an edit after guard feedback.\n"
                "CONFIDENCE:\n0.8\n"
                "LOCAL_NOTES:\nEdit command issued."
            )
        assert "printf edited" in prompt
        return FakeResponse(
            "FINAL:\nThe repository files were edited in Docker.\n"
            "SHARE_FINDING:\nWorkspace edit is now present.\n"
            "CONFIDENCE:\n0.9\n"
            "LOCAL_NOTES:\nDone."
        )


class FakeFinalGuardTool:
    name = "fake_final_guard"

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, *, final_response: str | None = None):
        self.calls += 1
        if self.calls == 1:
            return {
                "tool": self.name,
                "passed": False,
                "error": "final_guard_failed",
                "message": "git diff is empty",
                "stderr": "git diff is empty",
            }
        return {
            "tool": self.name,
            "passed": True,
        }


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


class LocalNotesOnlyLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return FakeResponse(
                "SUMMARY:\nLong reasoning summary that should not be copied into the next local notes prompt.\n"
                "SHARE_FINDING:\nCandidate A is currently strongest.\n"
                "LOCAL_NOTES:\nANSWER_CHOICE: A\nANSWER_REASON: Candidate A best matches the evidence."
            )
        assert "ANSWER_CHOICE: A" in prompt
        assert "ANSWER_REASON: Candidate A best matches the evidence." in prompt
        assert "NEXT_STEP:" not in prompt
        assert "Long reasoning summary that should not be copied" not in prompt
        return FakeResponse(
            "FINAL:\nFinal Answer: A\n"
            "SHARE_FINDING:\nFinal candidate is A.\n"
            "LOCAL_NOTES:\nANSWER_CHOICE: A\nANSWER_REASON: Candidate A remains the supported final answer."
        )


class AlwaysFinalLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: str) -> FakeResponse:
        self.prompts.append(prompt)
        return FakeResponse(
            "FINAL:\nFinal Answer: A\n"
            "SHARE_FINDING:\n\n"
            "LOCAL_NOTES:\nANSWER_CHOICE: A\nANSWER_REASON: Candidate A is selected."
        )


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
async def test_agent_step_trace_records_token_usage_from_llm_response_metadata():
    class UsageLLM:
        async def ainvoke(self, prompt: str) -> FakeResponse:
            return FakeResponse(
                "FINAL:\nToken usage is recorded.\n"
                "SHARE_FINDING:\n\n"
                "CONFIDENCE:\n0.8\n"
                "LOCAL_NOTES:\nDone.",
                usage_metadata={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            )

    streamer = InMemoryEventStreamer()
    trace_logger = TraceLogger()
    agent = ResearchAgent(
        run_id="run-token-usage",
        task="Record token usage.",
        assigned_subtask="Finish once.",
        llm=UsageLLM(),
        event_streamer=streamer,
        trace_logger=trace_logger,
        max_steps=1,
        step_delay_seconds=0,
    )

    await agent.run()

    step = trace_logger.export()["ResearchAgent"]["steps"][0]
    assert step["token_usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }
    assert "confidence" not in step["parsed_output"]


@pytest.mark.asyncio
async def test_agent_prompt_reuses_only_short_local_notes_not_previous_summary():
    streamer = InMemoryEventStreamer()
    trace_logger = TraceLogger()
    llm = LocalNotesOnlyLLM()
    agent = ResearchAgent(
        run_id="run-local-notes",
        task="Choose the best option.",
        assigned_subtask="Pick an answer and revise once.",
        llm=llm,
        event_streamer=streamer,
        trace_logger=trace_logger,
        max_steps=2,
        step_delay_seconds=0,
    )

    output = await agent.run()

    assert "Final Answer: A" in output
    assert len(llm.prompts) == 2
    assert "Long reasoning summary that should not be copied" not in llm.prompts[1]
    assert agent.local_notes == [
        "ANSWER_CHOICE: A\nANSWER_REASON: Candidate A best matches the evidence.",
        "ANSWER_CHOICE: A\nANSWER_REASON: Candidate A remains the supported final answer.",
    ]


@pytest.mark.asyncio
async def test_agent_runs_configured_steps_when_early_stop_is_disabled():
    streamer = InMemoryEventStreamer()
    trace_logger = TraceLogger()
    llm = AlwaysFinalLLM()
    agent = ResearchAgent(
        run_id="run-no-early-stop",
        task="Choose the best option.",
        assigned_subtask="Pick an answer.",
        llm=llm,
        event_streamer=streamer,
        trace_logger=trace_logger,
        max_steps=3,
        step_delay_seconds=0,
        allow_agent_early_stop=False,
    )

    await agent.run()

    assert len(llm.prompts) == 3
    assert len(trace_logger.export()["ResearchAgent"]["steps"]) == 3


@pytest.mark.asyncio
async def test_agent_can_stop_before_configured_steps_when_early_stop_is_enabled():
    streamer = InMemoryEventStreamer()
    trace_logger = TraceLogger()
    llm = AlwaysFinalLLM()
    agent = ResearchAgent(
        run_id="run-early-stop",
        task="Choose the best option.",
        assigned_subtask="Pick an answer.",
        llm=llm,
        event_streamer=streamer,
        trace_logger=trace_logger,
        max_steps=3,
        step_delay_seconds=0,
        allow_agent_early_stop=True,
    )

    await agent.run()

    assert len(llm.prompts) == 1
    assert len(trace_logger.export()["ResearchAgent"]["steps"]) == 1


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
async def test_final_guard_rejects_final_and_allows_extra_tool_steps():
    streamer = InMemoryEventStreamer()
    trace_logger = TraceLogger()
    workspace = AgentFakeDockerWorkspace()
    guard = FakeFinalGuardTool()
    llm = FinalGuardRetryLLM()
    agent = CodingAgent(
        run_id="run-final-guard",
        task="Edit files before finishing.",
        assigned_subtask="Use Docker edits before final answer.",
        llm=llm,
        event_streamer=streamer,
        trace_logger=trace_logger,
        bash_tool=BashTool(workspace),
        final_guard_tool=guard,
        workspace_access="write",
        max_steps=2,
        step_delay_seconds=0,
    )

    output = await agent.run()

    assert "repository files were edited" in output
    assert workspace.commands == ["printf edited"]
    assert guard.calls == 2
    trace = trace_logger.export()["CodingAgent"]
    assert trace["steps"][0]["parsed_output"]["status"] == "continue"
    assert trace["steps"][0]["parsed_output"]["tool_result"]["error"] == "final_guard_failed"
    assert trace["steps"][1]["parsed_output"]["tool_action"]["command"] == "printf edited"
    assert trace["steps"][2]["parsed_output"]["status"] == "final"


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
