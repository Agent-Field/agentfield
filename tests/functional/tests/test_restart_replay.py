import asyncio
import uuid
from typing import Any, Dict

import pytest

from agents.restart_replay_agent import create_agent, reset_state, snapshot_counts
from utils import run_agent_server, unique_node_id


async def _poll_execution(client, execution_id: str, *, timeout: float = 180.0) -> Dict[str, Any]:
    deadline = asyncio.get_running_loop().time() + timeout
    last_payload: Dict[str, Any] = {}

    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(f"/api/v1/executions/{execution_id}", timeout=10.0)
        assert response.status_code == 200, response.text
        payload = response.json()
        last_payload = payload
        status = payload.get("status")
        if status in {"succeeded", "failed", "cancelled", "timeout"}:
            return payload
        await asyncio.sleep(2)

    pytest.fail(f"execution {execution_id} did not finish in time; last={last_payload}")


async def _get_lightweight_dag(client, run_id: str) -> Dict[str, Any]:
    response = await client.get(
        f"/api/ui/v1/workflows/{run_id}/dag?lightweight=1",
        timeout=20.0,
    )
    assert response.status_code == 200, response.text
    return response.json()


def _find_node(dag: Dict[str, Any], reasoner_id: str, status: str | None = None) -> Dict[str, Any]:
    for node in dag.get("timeline", []):
        if node.get("reasoner_id") != reasoner_id:
            continue
        if status is not None and node.get("status") != status:
            continue
        return node
    pytest.fail(f"missing DAG node reasoner={reasoner_id!r} status={status!r}: {dag}")


@pytest.mark.functional
@pytest.mark.openrouter
@pytest.mark.slow
@pytest.mark.asyncio
async def test_restart_reuses_successful_calls_and_continues_complex_openrouter_graph(
    openrouter_config,
    async_http_client,
):
    reset_state()
    node_id = unique_node_id("restart-replay-agent")
    agent = create_agent(node_id=node_id, ai_config=openrouter_config)
    scenario_id = f"restart-{uuid.uuid4().hex[:8]}"
    question = "Can AgentField restart a failed dynamic graph without repeating upstream calls?"

    async with run_agent_server(agent, startup_delay=2.0, registration_delay=2.0):
        start_response = await async_http_client.post(
            f"/api/v1/execute/async/{node_id}.root_investigation",
            json={
                "input": {
                    "scenario_id": scenario_id,
                    "question": question,
                    "model": openrouter_config.model,
                }
            },
            timeout=20.0,
        )
        assert start_response.status_code == 202, start_response.text
        start_body = start_response.json()
        source_run_id = start_body["run_id"]

        first = await _poll_execution(
            async_http_client,
            start_body["execution_id"],
            timeout=180.0,
        )
        assert first["status"] == "failed", first

        first_counts = snapshot_counts()
        assert first_counts["root_investigation"] == 1
        assert first_counts["plan_scope"] == 1
        assert first_counts["assess_dimension:technical"] == 1
        assert first_counts["assess_dimension:product"] == 1
        assert first_counts["synthesize"] == 1
        assert first_counts.get("verify_recovery", 0) == 0

        failed_dag = await _get_lightweight_dag(async_http_client, source_run_id)
        failed_node = _find_node(failed_dag, "synthesize", "failed")

        restart_response = await async_http_client.post(
            f"/api/v1/executions/{failed_node['execution_id']}/restart",
            json={"reuse": "succeeded-before", "reason": "functional restart replay test"},
            timeout=20.0,
        )
        assert restart_response.status_code == 202, restart_response.text
        restart_body = restart_response.json()
        assert restart_body["source_run_id"] == source_run_id
        assert restart_body["source_execution_id"] == failed_node["execution_id"]
        assert restart_body["replay_mode"] == "succeeded-before"

        restarted = await _poll_execution(
            async_http_client,
            restart_body["execution_id"],
            timeout=180.0,
        )
        assert restarted["status"] == "succeeded", restarted
        result = restarted.get("result") or {}
        assert result.get("scenario_id") == scenario_id
        assert "verdict" in result
        assert result["synthesis"]["replay_source_run_id"] == source_run_id

        final_counts = snapshot_counts()
        assert final_counts["root_investigation"] == 2
        assert final_counts["plan_scope"] == 1
        assert final_counts["assess_dimension:technical"] == 1
        assert final_counts["assess_dimension:product"] == 1
        assert final_counts["synthesize"] == 2
        assert final_counts["verify_recovery"] == 1

        restarted_dag = await _get_lightweight_dag(async_http_client, restart_body["run_id"])
        timeline = restarted_dag.get("timeline", [])
        reused = [
            node
            for node in timeline
            if node.get("reuse", {}).get("source_execution_id")
        ]
        assert {node["reasoner_id"] for node in reused} >= {
            "plan_scope",
            "assess_dimension",
        }
        assert _find_node(restarted_dag, "synthesize", "succeeded")
        assert _find_node(restarted_dag, "verify_recovery", "succeeded")
