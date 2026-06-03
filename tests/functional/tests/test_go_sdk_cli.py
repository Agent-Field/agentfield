import asyncio
import json
import os
import time

import httpx
import pytest

from utils import get_go_agent_binary, run_go_agent, unique_node_id


async def _wait_for_registration(client, node_id: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        resp = await client.get(f"/api/v1/nodes/{node_id}")
        last_status = resp.status_code
        if resp.status_code == 200:
            return
        await asyncio.sleep(1)
    raise AssertionError(f"Node {node_id} not registered after {timeout}s (last status={last_status})")


async def _wait_for_workflow_run(client, run_id: str, *, expected_reasoners: set[str], timeout: float = 45.0):
    """Poll workflow run detail endpoint until expected reasoners appear."""
    # Give the database a moment to commit the executions
    await asyncio.sleep(0.5)

    deadline = time.time() + timeout
    last_body = None
    while time.time() < deadline:
        resp = await client.get(f"/api/ui/v2/workflow-runs/{run_id}")
        if resp.status_code == 200:
            body = resp.json()
            last_body = body
            executions = body.get("executions", [])
            reasoners = {node.get("reasoner_id") for node in executions}
            if expected_reasoners.issubset(reasoners):
                return executions
        await asyncio.sleep(1)
    raise AssertionError(
        f"Workflow run {run_id} missing expected nodes after {timeout}s: {json.dumps(last_body, indent=2)}"
    )


async def _wait_for_workflow_run_completed(
    client, run_id: str, *, expected_reasoners: set[str], timeout: float = 45.0
):
    """Poll workflow run detail endpoint until expected reasoners appear and are terminal."""
    timeline = await _wait_for_workflow_run(client, run_id, expected_reasoners=expected_reasoners, timeout=timeout)

    deadline = time.time() + timeout
    last_body = None
    while time.time() < deadline:
        reasoners = {node.get("reasoner_id") for node in timeline}
        statuses = {node.get("status") for node in timeline}
        if expected_reasoners.issubset(reasoners) and statuses == {"succeeded"}:
            return timeline

        await asyncio.sleep(1)
        resp = await client.get(f"/api/ui/v2/workflow-runs/{run_id}")
        if resp.status_code == 200:
            body = resp.json()
            last_body = body
            timeline = body.get("executions", [])

    raise AssertionError(
        f"Workflow run {run_id} did not reach terminal success after {timeout}s: {json.dumps(last_body, indent=2)}"
    )


async def _wait_for_agent_health(url: str, timeout: float = 20.0):
    deadline = time.time() + timeout
    async with httpx.AsyncClient(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                resp = await client.get(url)
                if resp.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    raise AssertionError(f"Agent did not become healthy at {url}")


async def _resolve_workflow_id_from_execution(client, execution_id: str, timeout: float = 15.0) -> str:
    """Query execution status API to retrieve canonical workflow/run ID."""
    deadline = time.time() + timeout
    last_status = None
    while time.time() < deadline:
        resp = await client.get(f"/api/v1/executions/{execution_id}")
        last_status = resp.status_code
        if resp.status_code == 200:
            data = resp.json()
            workflow_id = data.get("workflow_id") or data.get("run_id") or data.get("runId")
            if workflow_id:
                return workflow_id
        await asyncio.sleep(0.5)
    raise AssertionError(
        f"Unable to resolve workflow id from execution {execution_id}, last status={last_status}"
    )


async def _wait_for_execution_logs(client, execution_id: str, timeout: float = 15.0):
    """Poll structured execution-log API until entries appear for an execution."""
    deadline = time.time() + timeout
    last_body = None
    while time.time() < deadline:
        resp = await client.get(f"/api/ui/v1/executions/{execution_id}/logs", params={"tail": "200"})
        assert resp.status_code == 200, f"execution logs failed: {resp.status_code} {resp.text}"
        body = resp.json()
        last_body = body
        entries = body.get("entries", [])
        if entries:
            return entries
        await asyncio.sleep(0.5)
    raise AssertionError(
        f"Execution {execution_id} produced no structured logs after {timeout}s: {json.dumps(last_body, indent=2)}"
    )


async def _wait_for_node_logs_containing(client, node_id: str, marker: str, timeout: float = 15.0):
    """Poll raw node-log proxy until a marker string appears in the visible tail."""
    deadline = time.time() + timeout
    last_joined = ""
    while time.time() < deadline:
        resp = await client.get(
            f"/api/ui/v1/nodes/{node_id}/logs",
            params={"tail_lines": "10000"},
            timeout=30.0,
        )
        assert resp.status_code == 200, f"node logs failed: {resp.status_code} {resp.text}"
        lines = [ln for ln in resp.text.strip().split("\n") if ln.strip()]
        if lines:
            last_joined = "\n".join(lines)
            if marker in last_joined:
                return lines
        await asyncio.sleep(0.5)
    raise AssertionError(f"Node {node_id} logs never contained marker {marker!r}: {last_joined}")


@pytest.mark.functional
@pytest.mark.asyncio
async def test_go_sdk_cli_and_control_plane(async_http_client, control_plane_url):
    """
    Verify Go SDK hello-world example works as both CLI and control-plane node:
    - CLI invocation prints greeting without control-plane dependency.
    - Server mode registers and executes via control plane while nested reasoners call each other locally,
      still producing parent/child workflow edges in the control plane DAG.
    """
    node_id = unique_node_id("go-hello-agent")

    binary = None
    try:
        binary = get_go_agent_binary("hello")
    except FileNotFoundError:
        pytest.skip("Go agent binary not built; ensure go_agents are compiled in test image")

    # Start Go agent server pointed at control plane.
    env_server = {
        **os.environ,
        "AGENTFIELD_URL": control_plane_url,
        "AGENT_NODE_ID": node_id,
        "AGENT_LISTEN_ADDR": ":8001",
        # Control plane reaches the agent via docker network name.
        "AGENT_PUBLIC_URL": "http://test-runner:8001",
    }
    async with run_go_agent("hello", args=["serve"], env=env_server):
        await _wait_for_registration(async_http_client, node_id)
        await _wait_for_agent_health("http://127.0.0.1:8001/health")

        # Execute via control plane to build a workflow DAG: demo_echo -> say_hello -> add_emoji.
        payload = {"input": {"message": "Hello, Agentfield!"}}
        resp = await async_http_client.post(
            f"/api/v1/execute/{node_id}.demo_echo", json=payload, timeout=30.0
        )
        assert resp.status_code == 200, f"execute failed: {resp.status_code} {resp.text}"

        body = resp.json()
        execution_id = body.get("execution_id")
        assert execution_id, f"execution_id missing in response: {body}"

        workflow_id = await _resolve_workflow_id_from_execution(async_http_client, execution_id)
        # Validate result shape
        result = body.get("result") or {}
        assert result.get("name") == "Hello, Agentfield!"
        assert "greeting" in result

        timeline = await _wait_for_workflow_run_completed(
            async_http_client,
            workflow_id,
            expected_reasoners={"demo_echo", "say_hello", "add_emoji"},
            timeout=30.0,
        )
        id_by_reasoner = {node["reasoner_id"]: node["execution_id"] for node in timeline}
        parent_by_reasoner = {node["reasoner_id"]: node.get("parent_execution_id") for node in timeline}

        assert parent_by_reasoner.get("demo_echo") in (None, "", "null")
        assert parent_by_reasoner.get("say_hello") == id_by_reasoner["demo_echo"]
        assert parent_by_reasoner.get("add_emoji") == id_by_reasoner["say_hello"]

        # CLI invocation should still work without control plane.
        env_cli = {k: v for k, v in os.environ.items() if k not in {"AGENTFIELD_URL", "AGENT_PUBLIC_URL"}}
        env_cli["AGENT_NODE_ID"] = f"{node_id}-cli"
        cli_proc = await asyncio.create_subprocess_exec(
            binary,
            "--set",
            "message=CLI Functional",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env_cli,
        )
        stdout, stderr = await cli_proc.communicate()
        assert cli_proc.returncode == 0, f"CLI failed rc={cli_proc.returncode} stderr={stderr.decode()}"
        assert "Hello, CLI Functional" in stdout.decode()


@pytest.mark.functional
@pytest.mark.asyncio
async def test_go_sdk_local_calls_emit_workflow_events(async_http_client, control_plane_url):
    """
    Ensure Go SDK local composition (CallLocal) still emits workflow events so the control plane builds a full DAG.
    """
    try:
        get_go_agent_binary("hello")
    except FileNotFoundError:
        pytest.skip("Go agent binary not built; ensure go_agents are compiled in test image")

    node_id = unique_node_id("go-local-dag")

    env_server = {
        **os.environ,
        "AGENTFIELD_URL": control_plane_url,
        "AGENT_NODE_ID": node_id,
        "AGENT_LISTEN_ADDR": ":8001",
        "AGENT_PUBLIC_URL": "http://test-runner:8001",
    }

    async with run_go_agent("hello", args=["serve"], env=env_server):
        await _wait_for_registration(async_http_client, node_id)
        await _wait_for_agent_health("http://127.0.0.1:8001/health")

        payload = {"input": {"message": "Local DAG"}}
        resp = await async_http_client.post(
            f"/api/v1/execute/{node_id}.demo_echo", json=payload, timeout=30.0
        )
        assert resp.status_code == 200, f"execute failed: {resp.status_code} {resp.text}"

        body = resp.json()
        execution_id = body.get("execution_id")
        assert execution_id, f"execution_id missing in response: {body}"

        workflow_id = await _resolve_workflow_id_from_execution(async_http_client, execution_id)
        timeline = await _wait_for_workflow_run_completed(
            async_http_client,
            workflow_id,
            expected_reasoners={"demo_echo", "say_hello", "add_emoji"},
            timeout=30.0,
        )

        assert len(timeline) == 3
        id_by_reasoner = {node["reasoner_id"]: node["execution_id"] for node in timeline}
        parent_by_reasoner = {node["reasoner_id"]: node.get("parent_execution_id") for node in timeline}

        assert parent_by_reasoner.get("demo_echo") in (None, "", "null")
        assert parent_by_reasoner.get("say_hello") == id_by_reasoner["demo_echo"]
        assert parent_by_reasoner.get("add_emoji") == id_by_reasoner["say_hello"]

        for node in timeline:
            assert node.get("agent_node_id") == node_id
            assert node.get("workflow_depth") in (0, 1, 2)


@pytest.mark.functional
@pytest.mark.asyncio
async def test_go_sdk_execution_observability_surfaces_structured_and_process_logs(async_http_client, control_plane_url):
    """
    Validate execution observability in the canonical functional harness:
    - structured execution logs are queryable from the control plane for the execution
    - raw node process logs still expose mirrored NDJSON carrying the same execution correlation
    """
    try:
        get_go_agent_binary("hello")
    except FileNotFoundError:
        pytest.skip("Go agent binary not built; ensure go_agents are compiled in test image")

    node_id = unique_node_id("go-observability")
    env_server = {
        **os.environ,
        "AGENTFIELD_URL": control_plane_url,
        "AGENT_NODE_ID": node_id,
        "AGENT_LISTEN_ADDR": ":8001",
        "AGENT_PUBLIC_URL": "http://test-runner:8001",
    }

    async with run_go_agent("hello", args=["serve"], env=env_server):
        await _wait_for_registration(async_http_client, node_id)
        await _wait_for_agent_health("http://127.0.0.1:8001/health")

        resp = await async_http_client.post(
            f"/api/v1/execute/{node_id}.demo_echo",
            json={"input": {"message": "Execution observability"}},
            timeout=30.0,
        )
        assert resp.status_code == 200, f"execute failed: {resp.status_code} {resp.text}"

        body = resp.json()
        execution_id = body.get("execution_id")
        assert execution_id, f"execution_id missing in response: {body}"

        workflow_id = await _resolve_workflow_id_from_execution(async_http_client, execution_id)
        entries = await _wait_for_execution_logs(async_http_client, execution_id)

        assert all(entry.get("execution_id") == execution_id for entry in entries)
        assert any(entry.get("workflow_id") == workflow_id for entry in entries)
        assert any(entry.get("agent_node_id") == node_id for entry in entries)
        assert any(entry.get("source") == "sdk.runtime" for entry in entries)
        assert any(entry.get("event_type") for entry in entries)
        assert any(entry.get("system_generated") is True for entry in entries)
        assert any(entry.get("message") for entry in entries)

        proxy_lines = await _wait_for_node_logs_containing(async_http_client, node_id, execution_id)
        parsed = [json.loads(line) for line in proxy_lines]
        assert parsed, "expected mirrored structured NDJSON lines in raw node logs"
        assert any(row.get("line", "").find(execution_id) >= 0 for row in parsed)
        assert any(row.get("line", "").find(workflow_id) >= 0 for row in parsed)
        assert any(row.get("line", "").find("sdk.runtime") >= 0 for row in parsed)
