from __future__ import annotations

from multi_agent_sync.agents.coding_agent import CodingAgent
from multi_agent_sync.agents.critic_agent import CriticAgent
from multi_agent_sync.agents.research_agent import ResearchAgent
from multi_agent_sync.agents.solver_agent import SolverAgent
from multi_agent_sync.agents.verifier_agent import VerifierAgent

AGENT_REGISTRY = {
    "ResearchAgent": ResearchAgent,
    "CodingAgent": CodingAgent,
    "CriticAgent": CriticAgent,
    "SolverAgent": SolverAgent,
    "VerifierAgent": VerifierAgent,
}
