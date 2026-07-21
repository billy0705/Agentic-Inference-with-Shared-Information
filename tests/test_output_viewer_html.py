from pathlib import Path


VIEWER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "output_viewer.html"


def test_static_output_viewer_has_local_folder_reader_and_trace_parser():
    assert VIEWER_PATH.exists()
    html = VIEWER_PATH.read_text(encoding="utf-8")

    assert "Output Viewer" in html
    assert "showDirectoryPicker" in html
    assert "webkitdirectory" in html
    assert "DEFAULT_OUTPUT_ROOT = '../output/'" in html
    assert "loadDefaultOutputFolder" in html
    assert "isFileProtocol" in html
    assert "configureProtocolMode" in html
    assert "Default unavailable on file://" in html
    assert "Browser blocks automatic default loading for file:// pages" in html
    assert "if (!isFileProtocol()) loadDefaultOutputFolder();" in html
    assert "parseOutputFiles" in html
    assert "method_trace" in html
    assert "agent_traces" in html
    assert "event_receipts" in html
    assert "token_usage" in html
    assert "formatTokenUsage" in html
    assert "Token usage" in html
    assert "normalizeSingleAgentTrace" in html
    assert "normalizeMajorityVoteTrace" in html
    assert "normalizeDebateTrace" in html
    assert "normalizeSynthesizerTrace" in html
    assert "synthesizer_trace" in html
    assert "methodTrace.method === 'single_agent'" in html
    assert "methodTrace.method === 'majority_vote'" in html
    assert "methodTrace.method === 'multiagent_debate'" in html
    assert "SingleAgent" in html
    assert "VoterAgent" in html
    assert "DebateAgent" in html
    assert "Synthesizer" in html
    assert "normalizeSingleAgentStep(step, methodTrace)" in html
    assert "step.prompt || methodTrace.prompt" in html
    assert "open_prompt" in html
    assert "ACTION:" in html
    assert "normalizeOrchestrator" in html
    assert "runsForModelFilter" in html
    assert "runsForMethodFilter" in html
    assert "runMethods" in html


def test_static_output_viewer_exposes_required_filters_and_tabs():
    assert VIEWER_PATH.exists()
    html = VIEWER_PATH.read_text(encoding="utf-8")

    for control_id in ("benchmarkFilter", "modelFilter", "methodFilter", "runFilter", "exampleFilter"):
        assert f'id="{control_id}"' in html

    assert '<select id="agentFilter"></select>' not in html
    assert '<select id="eventAgentFilter"></select>' not in html
    assert "renderAgentFilter" in html
    assert "renderEventAgentFilter" in html
    assert "setAgentFilter" in html
    assert "setEventAgentFilter" in html

    for tab in ("Overview", "Orchestrator", "Prompt", "Output", "Agents", "Events", "Tools", "Raw"):
        assert tab in html

    assert "renderEventFlow" in html
    assert "All subagents" in html
    assert "All event agents" in html
