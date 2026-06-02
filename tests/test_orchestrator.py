from multi_agent_sync.orchestrator.orchestrator import classify_task, create_orchestrator_plan


def test_calculation_uses_solver_and_verifier_without_coding_agent():
    plan = create_orchestrator_plan("Calculate the acceleration of a 2 kg object under a 10 N force.")

    assert plan["mode"] == "multi_agent"
    assert plan["task_type"] == "calculation"
    assert [assignment["agent_name"] for assignment in plan["assignments"]] == ["SolverAgent", "VerifierAgent"]


def test_simple_qa_uses_direct_mode_with_no_assignments():
    plan = create_orchestrator_plan("What is an API?")

    assert plan["mode"] == "direct"
    assert plan["task_type"] == "simple_qa"
    assert plan["assignments"] == []


def test_debugging_uses_coding_and_critic_only():
    plan = create_orchestrator_plan("Debug this failing pytest error in my API module.")

    assert plan["mode"] == "multi_agent"
    assert plan["task_type"] == "debugging_task"
    assert [assignment["agent_name"] for assignment in plan["assignments"]] == ["CodingAgent", "CriticAgent"]


def test_classifier_exposes_expected_task_types():
    assert classify_task("Draft a short release announcement.") == "writing_task"
    assert classify_task("Design the architecture for a multi-tenant SaaS API.") == "architecture_design"
    assert classify_task("Research tradeoffs between Redis Streams and Kafka.") == "research_project"
