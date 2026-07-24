"""
Multi-Version Agent Example (Python)

Demonstrates multi-version agent support using the composite primary key
(id, version). All agents share the same node_id but register with different
versions, creating separate rows in the control plane.

The execute endpoint transparently routes across versioned agents using
weighted round-robin when no default (unversioned) agent exists.

Usage:
    # Start control plane first, then:
    cd examples/python_agent_nodes/multi_version
    python main.py
"""

import asyncio
import os
import signal
import sys
import time
from typing import List

import httpx
from agentfield import Agent


CP_URL = os.getenv("AGENTFIELD_URL", "http://localhost:8080")
AGENT_ID = "mv-demo-py"
BASE_PORT = 9200


def create_agent(version: str, port: int) -> Agent:
    """Create an agent instance for a specific version."""
    app = Agent(
        node_id=AGENT_ID,
        agentfield_server=CP_URL,
        version=version,
        dev_mode=False,
        callback_url=f"http://localhost:{port}",
    )

    @app.reasoner()
    async def echo(message: str = "") -> dict:
        """Echo reasoner present on every version."""
        return {
            "agent": AGENT_ID,
            "version": version,
            "echoed": message,
        }

    if version == "2.0.0":

        @app.reasoner()
        async def v2_feature(data: str = "") -> dict:
            """Extra capability only available in v2."""
            return {
                "agent": AGENT_ID,
                "version": version,
                "feature": "Only available in v2",
                "input": data,
            }

    return app


async def validate_registration():
    """Validate that both versions registered and routing works."""
    print("\n--- Validating multi-version registration ---\n")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # List all nodes and check that both versions are registered
        try:
            res = await client.get(f"{CP_URL}/api/v1/nodes?show_all=true")
            res.raise_for_status()
            data = res.json()
            nodes = data.get("nodes") or data.get("agents") or []
            agent_nodes = [
                n for n in nodes if (n.get("id") or n.get("node_id")) == AGENT_ID
            ]
            print(f'[Nodes] Found {len(agent_nodes)} versions of "{AGENT_ID}":')
            for n in agent_nodes:
                nid = n.get("id") or n.get("node_id")
                print(
                    f"  - id={nid}, version={n.get('version')}, base_url={n.get('base_url')}"
                )
        except Exception as e:
            print(f"Failed to list nodes: {e}")
            return

        # Execute against the shared ID - the CP will route via round-robin
        print(f"\n[Execute] Sending requests to {AGENT_ID}.echo:")
        for i in range(4):
            try:
                res = await client.post(
                    f"{CP_URL}/api/v1/execute/{AGENT_ID}.echo",
                    json={"input": {"message": f"request-{i}"}},
                    headers={"Content-Type": "application/json"},
                )
                body = res.json()
                result = body.get("result") or {}
                payload_version = (
                    result.get("version") if isinstance(result, dict) else None
                )
                routed_version = (
                    res.headers.get("X-Routed-Version")
                    or payload_version
                    or "(unknown)"
                )
                print(
                    f"  Request {i}: routed to version={routed_version}, result={result}"
                )
            except Exception as e:
                print(f"  Request {i}: failed - {e}")

    print("\n--- Validation complete ---\n")


def run_agent_in_thread(app: Agent, port: int):
    """Run an agent's serve() in a background thread."""
    import threading

    def target():
        app.serve(port=port, host="0.0.0.0")

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return t


def main():
    versions = [
        {"version": "1.0.0", "port": BASE_PORT},
        {"version": "2.0.0", "port": BASE_PORT + 1},
    ]

    print("Multi-version agent example (Python)")
    print(f"  Control plane: {CP_URL}")
    print(f"  Agent ID:      {AGENT_ID}")
    print(
        "  Versions:      " + ", ".join(f"{v['version']}@:{v['port']}" for v in versions) + "\n"
    )

    # Create and start all agents in background threads
    agents: List[Agent] = []
    for spec in versions:
        app = create_agent(spec["version"], spec["port"])
        agents.append(app)
        run_agent_in_thread(app, spec["port"])
        print(f"  Started {AGENT_ID} v{spec['version']} on port {spec['port']}")

    # Give the CP a moment to process registrations
    print("\n  Waiting for registrations to propagate...")
    time.sleep(3)

    # Validate
    asyncio.run(validate_registration())

    # Keep running so heartbeats continue
    print("All agents running. Press Ctrl+C to stop.\n")
    try:
        signal.pause()
    except (KeyboardInterrupt, AttributeError):
        # AttributeError: signal.pause() not available on Windows
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    print("Shutting down.")


if __name__ == "__main__":
    main()
