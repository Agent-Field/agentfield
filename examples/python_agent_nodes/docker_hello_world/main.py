"""
Docker/Kubernetes-friendly Hello World (Python)

This example is designed to validate the full AgentField execution path:

client -> control plane (/api/v1/execute) -> agent callback URL -> response

It is intentionally deterministic (no LLM credentials required).
"""

import os
import sys
import threading
import time

from agentfield import Agent


app = Agent(
    node_id=os.getenv("AGENT_NODE_ID", "demo-python-agent"),
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:8080"),
    dev_mode=True,
)


@app.reasoner()
async def hello(name: str = "AgentField") -> dict:
    return {"greeting": f"Hello, {name}!", "node_id": app.node_id}


@app.reasoner()
async def demo_echo(message: str = "Hello!") -> dict:
    return {"echo": message, "node_id": app.node_id}


def _log_heartbeat() -> None:
    n = 0
    node = app.node_id
    while True:
        print(f"[{node}] demo stdout heartbeat {n}", flush=True)
        print(f"[{node}] demo stderr heartbeat {n}", file=sys.stderr, flush=True)
        n += 1
        time.sleep(3)


if __name__ == "__main__":
    threading.Thread(target=_log_heartbeat, daemon=True).start()
    port = int(os.getenv("PORT", "8001"))
    # For containerized runs, set AGENT_CALLBACK_URL so the control plane can call back:
    #   AGENT_CALLBACK_URL=http://<service-name>:<port>
    app.run(host="0.0.0.0", port=port, auto_port=False)

