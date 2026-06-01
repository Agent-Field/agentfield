"""
UI proxy for agent NDJSON process logs (GET /api/ui/v1/nodes/:id/logs).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from agentfield import node_logs
from utils import run_agent_server, unique_node_id


@pytest.mark.functional
@pytest.mark.asyncio
async def test_ui_proxies_node_process_logs(async_http_client, make_test_agent):
    node_id = unique_node_id("logs-proxy-agent")
    agent = make_test_agent(node_id=node_id)

    @agent.reasoner()
    async def noisy():
        print("log-proxy-marker-stdout", flush=True)
        node_logs.get_ring().append("stdout", "log-proxy-marker-stdout", 64 * 1024)
        return {"ok": True}

    async with run_agent_server(agent):
        await asyncio.sleep(1.5)

        exec_res = await async_http_client.post(
            f"/api/v1/execute/{node_id}.noisy",
            json={"input": {}},
            timeout=60.0,
        )
        assert exec_res.status_code == 200, exec_res.text

        deadline = asyncio.get_running_loop().time() + 5.0
        last_joined = ""
        while True:
            logs_res = await async_http_client.get(
                "/api/ui/v1/nodes/" + node_id + "/logs",
                params={"tail_lines": "10000"},
                timeout=30.0,
            )
            assert logs_res.status_code == 200, logs_res.text

            lines = [ln for ln in logs_res.text.strip().split("\n") if ln.strip()]
            assert lines, "expected NDJSON lines from log proxy"

            parsed = [json.loads(ln) for ln in lines]
            assert all("seq" in row and "stream" in row and "line" in row for row in parsed)

            last_joined = "\n".join(lines)
            if "log-proxy-marker-stdout" in last_joined:
                break

            if asyncio.get_running_loop().time() >= deadline:
                pytest.fail(f"expected marker in proxied logs, got tail: {last_joined}")

            await asyncio.sleep(0.5)
