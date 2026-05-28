"""
RAG Evaluation Reasoners

Multi-reasoner architectures for comprehensive RAG evaluation:
- Faithfulness: Adversarial debate (Prosecutor vs Defender)
- Relevance: Multi-jury consensus voting
- Hallucination: Hybrid ML+LLM chain-of-verification
- Constitutional: Configurable principles-based evaluation
"""

from agentfield import AgentRouter

# Create the main router
router = AgentRouter(tags=["rag-evaluation"])

# Import and register all reasoner modules
from .faithfulness import register_faithfulness_reasoners
from .relevance import register_relevance_reasoners
from .hallucination import register_hallucination_reasoners
from .constitutional import register_constitutional_reasoners
from .orchestrator import register_orchestrator_reasoners

# Register all reasoners with the router
register_faithfulness_reasoners(router)
register_relevance_reasoners(router)
register_hallucination_reasoners(router)
register_constitutional_reasoners(router)
register_orchestrator_reasoners(router)

__all__ = ["router"]
