"""GitHub Copilot CLI as an AgentField node.

Boot entry point. Registers the ``ask``, ``plan``, ``review`` and
``run_task`` reasoners on a single :class:`agentfield.Agent` and starts
the HTTP server via ``app.run(auto_port=True)`` (same convention as the
other Python examples).
"""

from __future__ import annotations

import os

from agentfield import Agent

from reasoners import register


app = Agent(
    node_id=os.getenv("AGENT_NODE_ID", "copilot"),
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
)

register(app)


if __name__ == "__main__":
    print("🤖 GitHub Copilot CLI agent")
    print(f"📍 Node: {app.node_id}")
    print(f"🌐 Control Plane: {os.getenv('AGENTFIELD_URL', 'http://localhost:8080')}")
    print(f"🧠 Default model: {os.getenv('COPILOT_MODEL', 'gpt-5')}")
    isolate = os.getenv("AGENTFIELD_COPILOT_ISOLATE", "").strip().lower() in (
        "1", "true", "yes", "on"
    )
    print(f"🔒 Isolation (config_dir): {'per-node' if isolate else 'shared ~/.copilot'}")
    app.run(auto_port=True)
