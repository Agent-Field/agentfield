"""
RAG Evaluation Agent Node

Multi-reasoner evaluation system for RAG-generated responses featuring:
- Faithfulness: Adversarial debate pattern (Prosecutor vs Defender)
- Relevance: Multi-jury consensus voting
- Hallucination: Hybrid ML+LLM chain-of-verification
- Constitutional: Configurable principles-based evaluation

Production-ready backend AI for enterprise RAG evaluation.
"""

import os

# Load .env file for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

from agentfield import Agent, AIConfig
from reasoners import router


# Initialize agent
app = Agent(
    node_id="rag-evaluation",
    version="1.0.0",
    description="Multi-reasoner RAG evaluation with adversarial debate, jury consensus, and hybrid ML+LLM verification",
    agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080"),
    ai_config=AIConfig(
        model=os.getenv("AI_MODEL", "openrouter/deepseek/deepseek-chat-v3-0324"),
    ),
)

# Include the evaluation router with all metric reasoners
app.include_router(router)


if __name__ == "__main__":
    # Start the agent server
    port_env = os.getenv("PORT")
    if port_env is None:
        app.run(auto_port=True, host="0.0.0.0")
    else:
        app.run(port=int(port_env), host="0.0.0.0")
