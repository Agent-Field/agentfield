"""Deep Research Agent with Recursive Planning."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from agentfield import AIConfig, Agent

if __package__ in (None, ""):
    current_dir = Path(__file__).resolve().parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

from routers import planning_router, research_router

app = Agent(
    node_id="deep-research",
    agentfield_server=f"{os.getenv('AGENTFIELD_SERVER', 'http://localhost:8080')}",
    ai_config=AIConfig(
        model=os.getenv("AI_MODEL", "openrouter/deepseek/deepseek-v3.1-terminus"),
    ),
)

app.include_router(planning_router)
app.include_router(research_router)


if __name__ == "__main__":
    search_provider = os.getenv("SEARCH_PROVIDER", "tavily").strip().lower() or "tavily"
    search_provider_label = {
        "tavily": "Tavily",
        "parallel": "Parallel Search",
    }.get(search_provider, search_provider)
    print("🔬 Deep Research Agent")
    print("🧠 Node ID: deep-research")
    print(f"🌐 Control Plane: {app.agentfield_server}")
    print("\n🎯 Architecture: Recursive Planning + Research Execution")
    print("  1. Recursive Task Decomposition → Breaks research into subtasks")
    print("  2. Topological Graph → Identifies dependencies and parallelization")
    print(f"  3. Research Execution → {search_provider_label} with citation tracking")
    print("  4. Findings Synthesis → Structured results with sources")
    print("\n✨ Features:")
    print("  - Recursive task breakdown with configurable depth")
    print("  - Automatic dependency detection")
    print("  - Parallel execution planning")
    print(f"  - Web search integration ({search_provider_label})")
    print("  - Citation tracking and structured findings")
    print("  - Elegant and simple AgentField primitives")

    port_env = os.getenv("PORT")
    if port_env is None:
        app.run(auto_port=True, host="::")
    else:
        app.run(port=int(port_env), host="::")
