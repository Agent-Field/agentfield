"""
Functional smoke test for the TypeScript SDK.

Spins up a Node-based agent (using the packaged TS SDK) and executes it through
the real control plane to validate registration + execution wiring.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from contextlib import asynccontextmanager

import pytest

from utils import unique_node_id


def _get_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


@asynccontextmanager
async def run_ts_agent(node_id: str):
    """
    Launch the TS agent as a subprocess and ensure cleanup.
    """
    port = _get_free_port()
    callback_host = os.environ.get("TEST_AGENT_CALLBACK_HOST", "test-runner")
    env = os.environ.copy()
    env.update(
        {
            "TS_AGENT_ID": node_id,
            "TS_AGENT_PORT": str(port),
            "TS_AGENT_PUBLIC_URL": f"http://{callback_host}:{port}",
            "TS_AGENT_BIND_HOST": "0.0.0.0",
        }
    )
    env.setdefault("NODE_PATH", "/usr/local/lib/node_modules:/usr/lib/node_modules")

    # Resolve the agent script location relative to the tests/functional root
    script_path = os.path.join(os.path.dirname(__file__), "..", "ts_agents", "echo-agent.mjs")

    process = await asyncio.create_subprocess_exec(
        "node",
        script_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        yield port, process
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()


async def _wait_for_registration(http_client, node_id: str, process, timeout: float = 30.0):
    deadline = asyncio.get_event_loop().time() + timeout
    last_error = None
    while asyncio.get_event_loop().time() < deadline:
        if process.returncode is not None:
            stdout, stderr = await process.communicate()
            raise AssertionError(
                f"TS agent exited early with code {process.returncode}. "
                f"stdout: {stdout.decode()} stderr: {stderr.decode()}"
            )
        try:
            resp = await http_client.get(f"/api/v1/nodes/{node_id}")
            if resp.status_code == 200:
                return resp.json()
            last_error = resp.text
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        await asyncio.sleep(0.5)

    raise AssertionError(f"Node {node_id} did not register in time. Last error: {last_error}")


async def _wait_for_port_ready(port: int, process, host: str = "127.0.0.1", timeout: float = 15.0):
    """
    Ensure the TS agent has actually bound its HTTP listener before hitting it via the control plane.
    This avoids a race where the agent registers/heartbeats before its server starts listening.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if process.returncode is not None:
            stdout, stderr = await process.communicate()
            raise AssertionError(
                f"TS agent exited early before opening port {port}. "
                f"stdout: {stdout.decode()} stderr: {stderr.decode()}"
            )
        try:
            reader, writer = await asyncio.open_connection(host=host, port=port)
            writer.close()
            await writer.wait_closed()
            return
        except (ConnectionRefusedError, OSError):
            await asyncio.sleep(0.2)

    raise AssertionError(f"TS agent did not open port {host}:{port} in time")


@pytest.mark.functional
@pytest.mark.asyncio
async def test_typescript_agent_registers_and_executes(async_http_client):
    node_id = unique_node_id("ts-agent")

    async with run_ts_agent(node_id) as (port, process):
        registration = await _wait_for_registration(async_http_client, node_id, process)
        assert registration["id"] == node_id
        assert any(r["id"] == "echo" for r in registration.get("reasoners", []))

        # Avoid race where control plane calls the agent before its listener is ready.
        await _wait_for_port_ready(port, process)

        # Execute via control plane
        resp = await async_http_client.post(
            f"/api/v1/reasoners/{node_id}.echo",
            json={"input": {"message": "hello-ts"}},
            timeout=30.0,
        )

        # Collect logs if execution fails for easier debugging
        if resp.status_code != 200:
            async def _drain_process(proc: asyncio.subprocess.Process, timeout: float = 5.0):
                """
                Capture stdout/stderr without hanging the test if the agent stays alive.
                We explicitly terminate on timeout so we don't block on a long-lived heartbeat loop.
                """
                try:
                    return await asyncio.wait_for(proc.communicate(), timeout=timeout)
                except asyncio.TimeoutError:
                    proc.terminate()
                    try:
                        return await asyncio.wait_for(proc.communicate(), timeout=timeout)
                    except asyncio.TimeoutError:
                        proc.kill()
                        return await asyncio.wait_for(proc.communicate(), timeout=timeout)

            stdout, stderr = await _drain_process(process)
            print("TS agent stdout:", stdout.decode("utf-8"), file=sys.stderr)
            print("TS agent stderr:", stderr.decode("utf-8"), file=sys.stderr)

        assert resp.status_code == 200, resp.text
        body = resp.json()
        result = body.get("result", {})
        assert result.get("echoed") == "hello-ts"
        assert result.get("workflowId"), "workflowId missing in response"
        assert result.get("runId"), "runId missing in response"
