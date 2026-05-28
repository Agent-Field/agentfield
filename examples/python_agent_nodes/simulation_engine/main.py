"""Generalized Multi-Agent Simulation System.

A domain-agnostic simulation engine that can model any enterprise scenario using
LLM-powered multi-reasoner architecture with maximum parallelism.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from dotenv import load_dotenv

from agentfield import AIConfig, Agent

load_dotenv()

if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

from routers import (
    aggregation_router,
    decision_router,
    entity_router,
    scenario_router,
    simulation_router,
)

app = Agent(
    node_id="simulation-enginepy",
    agentfield_server=f"{os.getenv('AGENTFIELD_SERVER', 'http://localhost:8080')}",
    dev_mode=True,
    ai_config=AIConfig(
        model="openrouter/deepseek/deepseek-v3.1-terminus",  # LiteLLM auto-detects provider from model name
        api_key=os.getenv("OPENROUTER_API_KEY"),  # or set OPENAI_API_KEY env var
    ),
)

# Register all routers
for router in (
    scenario_router,
    entity_router,
    decision_router,
    aggregation_router,
    simulation_router,
):
    app.include_router(router)


if __name__ == "__main__":
    print("🎯 Generalized Multi-Agent Simulation System")
    print("🧠 Node ID: simulation-engine")
    print(f"🌐 Control Plane: {app.agentfield_server}")
    print("\n📊 Architecture: Multi-Reasoner Parallel System")
    print("  1. Scenario Analysis → Decompose scenario and build factor graph")
    print(
        "  2. Entity Generation → Create diverse entity population (batched parallel)"
    )
    print(
        "  3. Decision Simulation → Simulate decisions for all entities (parallel batches)"
    )
    print("  4. Aggregation → Analyze results and generate insights")
    print("\n✨ Key Features:")
    print("  - Scalable to large populations (1000+ entities)")
    print("  - Optimized batching (5 entities per AI call)")
    print("  - Parallel decision simulation (20 concurrent)")
    print("  - Intelligent data sampling for analysis")
    print("  - Domain-agnostic (works for any enterprise scenario)")

    port_env = os.getenv("PORT")
    if port_env is None:
        app.run(auto_port=True, host="localhost")
    else:
        app.run(port=int(port_env), host="localhost")
