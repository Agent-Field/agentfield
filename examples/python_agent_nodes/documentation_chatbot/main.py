"""Simplified Documentation chatbot with parallel retrieval and self-aware synthesis."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from agentfield import AIConfig, Agent
from agentfield.logger import log_info

if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

from embedding import embed_texts
from routers import (
    ingestion_router,
    qa_router,
    query_router,
    retrieval_router,
)

app = Agent(
    node_id="documentation-chatbot",
    agentfield_server=f"{os.getenv('AGENTFIELD_SERVER')}",
    api_key=os.getenv("AGENTFIELD_API_KEY"),
    ai_config=AIConfig(
        model=os.getenv("AI_MODEL", "openrouter/openai/gpt-4o-mini"),
    ),
)

for router in (
    query_router,
    ingestion_router,
    retrieval_router,
    qa_router,
):
    app.include_router(router)


def _warmup_embeddings() -> None:
    """Warm up the embedding model on startup."""
    try:
        embed_texts(["doc-chatbot warmup"])
        log_info("FastEmbed model warmed up for documentation chatbot")
    except Exception as exc:  # pragma: no cover - best-effort
        log_info(f"FastEmbed warmup failed: {exc}")


if __name__ == "__main__":
    _warmup_embeddings()

    print("📚 Simplified Documentation Chatbot Agent")
    print("🧠 Node ID: documentation-chatbot")
    print(f"🌐 Control Plane: {app.agentfield_server}")
    print("\n🎯 Architecture: 3-Agent Parallel System + Document-Level Retrieval")
    print("  1. Query Planner → Generates diverse search queries")
    print("  2. Parallel Retrievers → Concurrent vector search")
    print("  3. Self-Aware Synthesizer → Answer + confidence assessment")
    print("\n✨ Features:")
    print("  - Parallel retrieval for 3x speed improvement")
    print("  - Self-aware synthesis (no separate review)")
    print("  - Max 1 refinement iteration (prevents loops)")
    print("  - Document-level context (full pages vs isolated chunks)")
    print("  - Smart document ranking (frequency + relevance scoring)")
    port_env = os.getenv("PORT")
    if port_env is None:
        app.run(auto_port=True, host="::")
    else:
        app.run(port=int(port_env), host="::")
