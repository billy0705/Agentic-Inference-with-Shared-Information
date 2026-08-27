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
    assert "buildLazyExamples" in html
    assert "scheduleExampleTraceLoad" in html
    assert "Reading metadata files" in html
    assert "Promise.all(candidates" not in html
    assert "method_trace" in html
    assert "agent_traces" in html
    assert "event_receipts" in html
    assert "token_usage" in html
    assert "formatTokenUsage" in html
    assert "Token usage" in html
    assert "normalizeSingleAgentTrace" in html
    assert "normalizeDirectTrace" in html
    assert "normalizeMajorityVoteTrace" in html
    assert "normalizeDebateTrace" in html
    assert "normalizeSummarizerTrace" in html
    assert "synthesizer_trace" in html
    assert "Summarizer trace present" in html
    assert "method_trace.synthesizer_trace" in html
    assert "renderSummarizer" in html
    assert "summarizer: normalizeSummarizerTrace(methodTrace)" in html
    assert "agents.push(synthesizer)" not in html
    assert "methodTrace.method === 'single_agent'" in html
    assert "methodTrace.method === 'majority_vote'" in html
    assert "methodTrace.method === 'multiagent_debate'" in html
    assert "SingleAgent" in html
    assert "DirectAgent" in html
    assert "VoterAgent" in html
    assert "DebateAgent" in html
    assert "Summarizer" in html
    assert "normalizeSingleAgentStep(step, methodTrace)" in html
    assert "step.prompt || methodTrace.prompt" in html
    assert "open_prompt" in html
    assert "ACTION:" in html
    assert "normalizeOrchestrator" in html
    assert "normalizeDynamicOrchestration" in html
    assert "renderDynamicOrchestration" in html
    assert "renderDynamicOrchestrationRound" in html
    assert "renderDynamicOrchestrationPlan" in html
    assert "renderDynamicOrchestrationPool" in html
    assert "renderDynamicOrchestrationReportRows" in html
    assert "renderDynamicOrchestrationStepUsage" in html
    assert "dynamic_orchestration_trace" in html
    assert "Dynamic Orchestration" in html
    assert "Workflow plan" in html
    assert "Fixed subagent pool" in html
    assert "Current step" in html
    assert "Step result" in html
    assert "Step assignments" in html
    assert "Reporting agents" in html
    assert "Support-only agents" in html
    assert "report_policy" in html
    assert "support_excerpt" in html
    assert "dynamic_orchestration_trace" in html
    assert "Dynamic Orchestration" in html
    assert "Routing decision reason" in html
    assert "Agent description" in html
    assert "runsForModelFilter" in html
    assert "runsForMethodFilter" in html
    assert "runMethods" in html
    assert "Accuracy Overview" in html
    assert "Run Analysis" in html
    assert "renderRunAnalysisPanel" in html
    assert "analysisSelectionContext" in html
    assert "score_metadata: normalizeScoreMetadata" in html
    assert "normalizeScoreMetadata" in html
    assert "acceptedAnswersForExample" in html
    assert "valid_targets" in html
    assert "formatAcceptedAnswers" in html
    assert "selectedAnalysisBenchmark" in html
    assert "selectedAnalysisModel" in html
    assert "setAnalysisBenchmark" in html
    assert "setAnalysisModel" in html
    assert "BASELINE_METHOD_DEFAULTS" in html
    assert "single_agent" in html
    assert "majority_vote" in html
    assert "multiagent_debate" in html
    assert "Baseline methods" in html
    assert "Baseline correct, current wrong" in html
    assert "Selected baselines correct" not in html
    assert "setAnalysisBaselineMethod" in html
    assert "Baseline sources" in html
    assert "runAnalysisRows" in html
    assert "classifyAnalysisIssue" in html
    assert "renderAnalysisIssueRow" in html
    assert "baseline correct, current wrong" in html
    assert "current wrong, selected baselines wrong or missing" in html
    assert "current correct, selected baselines wrong" in html
    assert "current correct, selected baseline also correct" in html
    assert "Error distribution" in html
    assert "renderErrorDistributionChart" in html
    assert "Round majority and agreement" in html
    assert "renderConsensusTrendChart" in html
    assert "analysisRoundMetric: 'correct_majority'" in html
    assert "analysisRoundMetricOptions" in html
    assert "selectedAnalysisRoundMetric" in html
    assert "setAnalysisRoundMetric" in html
    assert "Round bar" in html
    assert "Correct majority" in html
    assert "Same choice" in html
    assert "Same correct choice" in html
    assert "MAX_ANALYSIS_ROUNDS = 3" in html
    assert "analysisRoundIsVisible" in html
    assert "Round-level answer analysis" in html
    assert "correct majority" in html
    assert "wrong majority" in html
    assert "same correct choice" in html
    assert "barWidth(row[metric.key], row.total)" in html
    assert "barWidth(row.correct_majority, maxTotal)" not in html
    assert "round answer correct" not in html
    assert "round answer wrong" not in html
    assert "correct_subagents" in html
    assert "Decision changed after shared findings" not in html
    assert "Same choice after more rounds" not in html
    assert "agents changed after shared info" in html
    assert "final agents all agree" in html
    assert "analysisRoundRows" in html
    assert "parenthesized" in html
    assert "text.match(/\\(([A-Za-z][A-Za-z0-9_.-]{0,11})\\)/)" in html
    assert "analysisDecisionShift" in html
    assert "analysisConsensusTrend" in html
    assert "Shared findings impact" in html
    assert "renderSharedFindingImpactChart" in html
    assert "renderAnalysisTraceLoadingNotice" in html
    assert "scheduleAnalysisTraceLoad" in html
    assert "unloadedAnalysisTraceExamples" in html
    assert "Loading shared finding traces" in html
    assert "analyzeSharedFindings" in html
    assert "classifySharedFinding" in html
    assert "Shared findings details" in html
    assert "What agents shared" in html
    assert "Answer changes after receiving shared findings" in html
    assert "Changed after shared" in html
    assert "wrong -> correct" in html
    assert "correct -> wrong" in html
    assert "wrong -> wrong" in html
    assert "correct -> correct" in html
    assert "Changed ending wrong" not in html
    assert "changed ending wrong" not in html
    assert "wrong subagent consensus" in html
    assert "wrong tie-break / final selection" in html
    assert "buildFailureExplanation" in html
    assert "Why our method failed" in html
    assert "Why wrong subagents did not choose gold" in html
    assert "latestAgentCandidates" in html
    assert "candidateRationale" in html
    assert "wrong subagent consensus ignored a correct minority" in html
    assert "tie-break or final selector did not use the correct subagent" in html
    assert "subagent majority chose the wrong answer" not in html
    assert "final selected answer is wrong" not in html
    assert "candidate majority wrong" not in html
    assert "candidate selected wrong" not in html
    assert "no majority, summarizer wrong" in html
    assert "summarizer overrode correct aggregation" in html
    assert "analysis_target_method" in html
    assert "candidate_aggregation" in html
    assert "setAnalysisMethod" in html
    assert "Latest Benchmark Accuracy" in html
    assert "accuracyBenchmarkGroups" in html
    assert "comparisonFingerprint" in html
    assert "compatibleAccuracyRuns" in html
    assert "comparisonMethodRows" in html
    assert "source_run_id" in html
    assert "renderSourceCell" in html
    assert "source-chip" in html
    assert "Full source run" in html
    assert "source: current" in html
    assert "source: reused" in html
    assert "benchmark-block" in html
    assert "setAccuracyModel" in html
    assert "isExcludedAccuracyRun" in html
    assert "isFakeLabel" in html
    assert "dynamicStreamingRouteBreakdown" in html
    assert "dynamicOrchestrationRoundBreakdown" in html
    assert "renderDynamicOrchestrationRoundBreakdown" in html
    assert "Dynamic orchestration rounds" in html
    assert "renderRouteBreakdown" in html
    assert "Dynamic route split" in html
    assert "Question Results" in html
    assert 'id="questionMatrixPanel"' not in html
    assert "renderQuestionMatrix" in html
    assert "renderQuestionMatrix(currentRun())" in html
    assert "questionResultRows" in html
    assert "questionInvalidLocations" in html
    assert "Invalid locations" in html
    assert "invalid.method" in html
    assert "selectExampleByIndexAndMethod" in html
    assert "result-cell" in html
    assert "result-cell compact" in html
    assert "same question id" in html
    assert "example-status-correct" in html
    assert "example-status-wrong" in html
    assert "exampleSelectClass" in html
    assert " ? 'correct' : 'wrong'" in html
    assert "multiagent_dynamic_streaming" in html
    assert "bar-fill" in html


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
    assert "renderEventTypeFilter" in html
    assert "setEventTypeFilter" in html
    assert 'eventType: \'communication\'' in html
    assert "Communication only" in html
    assert "Lifecycle only" in html
    assert "All event types" in html
    assert "eventTypeIsVisible" in html
    assert "renderPageTabs" in html
    assert 'page: \'accuracy\'' in html
    assert "analysisBenchmark: ''" in html
    assert "analysisModels: {}" in html
    assert 'analysisMethod: \'multiagent_dynamic_streaming\'' in html
    assert "analysisBaselineMethods: []" in html
    assert "analysisRoundMetric: 'correct_majority'" in html
    assert 'id="analysisBenchmarkSelect"' in html
    assert 'id="analysisModelSelect"' in html

    for tab in ("Overview", "Orchestrator", "Prompt", "Output", "Agents", "Summarizer", "Events", "Tools", "Raw"):
        assert tab in html

    assert "renderEventFlow" in html
    assert "All subagents" in html
    assert "All event agents" in html
