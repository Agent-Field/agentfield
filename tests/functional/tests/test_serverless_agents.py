"""
End-to-end validation for serverless agent nodes across SDKs.

Each test spins up a lightweight serverless handler (Python, TypeScript, Go),
registers it through the control plane's `/nodes/register-serverless` endpoint,
and executes a reasoner via the normal execution gateway to ensure discovery,
invocation, and parent/child call wiring all work without heartbeats.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple

import pytest
import uvicorn
from agentfield import Agent
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from utils import run_go_agent, unique_node_id

TEST_BIND_HOST = os.environ.get("TEST_AGENT_BIND_HOST", "0.0.0.0")
TEST_CALLBACK_HOST = os.environ.get("TEST_AGENT_CALLBACK_HOST", "test-runner")


def _get_free_port(host: str = TEST_BIND_HOST) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


async def _wait_for_port(host: str, port: int, timeout: float = 15.0, process=None):
    deadline = asyncio.get_event_loop().time() + timeout
    last_error: Optional[BaseException] = None
    while asyncio.get_event_loop().time() < deadline:
        if process and process.returncode is not None:
            stdout, stderr = await process.communicate()
            raise AssertionError(
                f"Process exited early (code {process.returncode}). "
                f"stdout={stdout.decode()} stderr={stderr.decode()}"
            )
        try:
            reader, writer = await asyncio.open_connection(host=host, port=port)
            writer.close()
            await writer.wait_closed()
            return
        except (ConnectionRefusedError, OSError) as exc:  # noqa: PERF203
            last_error = exc
            await asyncio.sleep(0.2)
    raise AssertionError(f"Port {host}:{port} did not open in time: {last_error}")


async def _register_serverless(async_http_client, invocation_url: str, *, retries: int = 6):
    # Prefer CLI to match DX; fall back to HTTP only if CLI is unavailable.
    cli_result = await _register_serverless_via_cli(invocation_url)
    if cli_result.get("ok"):
        return {"source": "cli"}
    if cli_result.get("error") == "missing-cli":
        # Fall back to API registration for environments without af on PATH
        pass
    elif cli_result.get("error"):
        raise AssertionError(f"af nodes register-serverless failed: {cli_result}")

    last_error = None
    for attempt in range(retries):
        resp = await async_http_client.post(
            "/api/v1/nodes/register-serverless",
            json={"invocation_url": invocation_url},
            timeout=20.0,
        )
        if resp.status_code in (200, 201):
            return resp.json()
        last_error = resp.text
        await asyncio.sleep(0.5)
    raise AssertionError(f"Failed to register serverless agent at {invocation_url}: {last_error}")


async def _register_serverless_via_cli(invocation_url: str):
    env = os.environ.copy()
    env.setdefault("AGENTFIELD_SERVER", env.get("CONTROL_PLANE_URL", "http://localhost:8080"))
    token = env.get("AGENTFIELD_TOKEN")

    cmd = [
        "af",
        "nodes",
        "register-serverless",
        "--url",
        invocation_url,
        "--server",
        env["AGENTFIELD_SERVER"],
        "--json",
    ]
    if token:
        cmd.extend(["--token", token])

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
    except FileNotFoundError:
        return {"ok": False, "error": "missing-cli"}

    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": "cli-failed",
            "code": proc.returncode,
            "stderr": stderr.decode(),
            "stdout": stdout.decode(),
        }

    if stdout:
        try:
            json.loads(stdout.decode())
        except json.JSONDecodeError:
            # best-effort parse, not fatal
            pass
    return {"ok": True}


@asynccontextmanager
async def run_python_serverless_agent(node_id: str, control_plane_url: str) -> AsyncIterator[str]:
    """
    Start a lightweight FastAPI wrapper that delegates to Agent.handle_serverless.
    """
    app = Agent(
        node_id=node_id,
        agentfield_server=control_plane_url,
        auto_register=False,
        dev_mode=True,
    )

    @app.reasoner()
    async def hello(name: str = "AgentField") -> dict:  # type: ignore[return-type]
        ctx = app.ctx
        return {
            "greeting": f"Hello, {name}!",
            "run_id": getattr(ctx, "workflow_id", None),
            "execution_id": getattr(ctx, "execution_id", None),
            "parent_execution_id": getattr(ctx, "parent_execution_id", None),
        }

    @app.reasoner()
    async def relay(target: str, message: str = "ping") -> dict:  # type: ignore[return-type]
        downstream = await app.call(target, message=message)
        return {"downstream": downstream, "parent_execution_id": getattr(app.ctx, "execution_id", None)}

    fastapi_app = FastAPI()

    @fastapi_app.get("/discover")
    async def discover():
        return await asyncio.to_thread(app.handle_serverless, {"path": "/discover"})

    @fastapi_app.post("/execute")
    async def execute(request: Request):
        payload = await request.json()
        result = await asyncio.to_thread(app.handle_serverless, {"path": "/execute", **payload})
        status = result.get("statusCode", 200)
        body = result.get("body", result)
        return JSONResponse(content=body, status_code=status)

    port = _get_free_port()
    config = uvicorn.Config(
        app=fastapi_app,
        host=TEST_BIND_HOST,
        port=port,
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(config)
    loop = asyncio.new_event_loop()

    def run_server():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    await asyncio.sleep(0.5)

    try:
        yield f"http://{TEST_CALLBACK_HOST}:{port}"
    finally:
        server.should_exit = True
        if loop.is_running():
            loop.call_soon_threadsafe(lambda: None)
        thread.join(timeout=10)


@asynccontextmanager
async def run_ts_serverless_agent(node_id: str, control_plane_url: str) -> AsyncIterator[Tuple[str, asyncio.subprocess.Process]]:
    port = _get_free_port()
    env = os.environ.copy()
    env.update(
        {
            "TS_AGENT_ID": node_id,
            "TS_AGENT_PORT": str(port),
            "TS_AGENT_BIND_HOST": TEST_BIND_HOST,
            "AGENTFIELD_SERVER": control_plane_url,
        }
    )
    env.setdefault("NODE_PATH", "/usr/local/lib/node_modules:/usr/lib/node_modules")
    script_path = Path(__file__).resolve().parent.parent / "ts_agents" / "serverless-agent.mjs"

    process = await asyncio.create_subprocess_exec(
        "node",
        str(script_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    try:
        await _wait_for_port("127.0.0.1", port, process=process)
        yield f"http://{TEST_CALLBACK_HOST}:{port}", process
    finally:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()


@asynccontextmanager
async def run_go_serverless_agent(node_id: str, control_plane_url: str) -> AsyncIterator[str]:
    port = _get_free_port()
    env = {
        **os.environ,
        "AGENT_NODE_ID": node_id,
        "AGENTFIELD_URL": control_plane_url,
        "PORT": str(port),
        "AGENTFIELD_TOKEN": os.environ.get("AGENTFIELD_TOKEN", ""),
    }

    async with run_go_agent("serverless", env=env) as proc:
        await _wait_for_port("127.0.0.1", port, process=proc.process)
        yield f"http://{TEST_CALLBACK_HOST}:{port}"


@pytest.mark.functional
@pytest.mark.asyncio
async def test_python_serverless_agent_registers_and_executes(async_http_client, control_plane_url):
    node_id = unique_node_id("py-svless")

    async with run_python_serverless_agent(node_id, control_plane_url) as invocation_url:
        await _register_serverless(async_http_client, invocation_url)

        resp = await async_http_client.post(
            f"/api/v1/reasoners/{node_id}.hello",
            json={"input": {"name": "Lambda"}},
            timeout=30.0,
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        result = body.get("result", {})
        assert result.get("greeting") == "Hello, Lambda!"
        assert result.get("execution_id"), "execution_id should propagate to serverless reasoner"


@pytest.mark.functional
@pytest.mark.asyncio
async def test_serverless_python_chain_calls(async_http_client, control_plane_url):
    child_id = unique_node_id("py-svless-child")
    parent_id = unique_node_id("py-svless-parent")

    async with run_python_serverless_agent(child_id, control_plane_url) as child_url:
        await _register_serverless(async_http_client, child_url)

        async with run_python_serverless_agent(parent_id, control_plane_url) as parent_url:
            await _register_serverless(async_http_client, parent_url)

            resp = await async_http_client.post(
                f"/api/v1/reasoners/{parent_id}.relay",
                json={"input": {"target": f"{child_id}.hello", "message": "hi-child"}},
                timeout=40.0,
            )
            assert resp.status_code == 200, resp.text
            result = resp.json().get("result", {})
            assert result.get("downstream", {}).get("greeting") == "Hello, hi-child!"
            assert result.get("parent_execution_id"), "parent execution id should be set on relay reasoner"


@pytest.mark.functional
@pytest.mark.asyncio
async def test_typescript_serverless_agent(async_http_client, control_plane_url):
    node_id = unique_node_id("ts-svless")

    async with run_ts_serverless_agent(node_id, control_plane_url) as (invocation_url, process):
        await _register_serverless(async_http_client, invocation_url)

        resp = await async_http_client.post(
            f"/api/v1/reasoners/{node_id}.hello",
            json={"input": {"name": "TS Lambda"}},
            timeout=30.0,
        )

        if resp.status_code != 200:
            stdout, stderr = await process.communicate()
            print("TS serverless stdout:", stdout.decode(), file=sys.stderr)
            print("TS serverless stderr:", stderr.decode(), file=sys.stderr)

        assert resp.status_code == 200, resp.text
        result = resp.json().get("result", {})
        assert result.get("greeting") == "Hello, TS Lambda!"
        assert result.get("execution_id")


@pytest.mark.functional
@pytest.mark.asyncio
async def test_go_serverless_agent(async_http_client, control_plane_url):
    node_id = unique_node_id("go-svless")

    async with run_go_serverless_agent(node_id, control_plane_url) as invocation_url:
        await _register_serverless(async_http_client, invocation_url)

        resp = await async_http_client.post(
            f"/api/v1/reasoners/{node_id}.hello",
            json={"input": {"name": "gopher"}},
            timeout=30.0,
        )
        assert resp.status_code == 200, resp.text
        result = resp.json().get("result", {})
        assert result.get("greeting") == "Hello, gopher!"
        assert result.get("execution_id")
