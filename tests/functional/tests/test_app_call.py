import pytest

from agents.call_chain_agents import (
    ORCHESTRATOR_SPEC,
    WORKER_SPEC,
    create_orchestrator_agent,
    create_worker_agent,
)
from utils import run_agent_server, unique_node_id


@pytest.mark.functional
@pytest.mark.asyncio
async def test_cross_agent_app_call_workflow(async_http_client):
    worker = create_worker_agent(node_id=unique_node_id(WORKER_SPEC.default_node_id))
    orchestrator = create_orchestrator_agent(
        node_id=unique_node_id(ORCHESTRATOR_SPEC.default_node_id),
        target_node_id=worker.node_id,
    )

    async with run_agent_server(worker), run_agent_server(orchestrator):
        payload = {"input": {"text": "AgentField rocks"}}

        response = await async_http_client.post(
            f"/api/v1/reasoners/{orchestrator.node_id}.delegate_pipeline",
            json=payload,
            timeout=30.0,
        )

        assert response.status_code == 200, response.text
        body = response.json()
        result = body["result"]

        assert result["original"] == "AgentField rocks"
        delegated = result["delegated"]
        assert delegated["upper"] == "AGENTFIELD ROCKS"
        assert delegated["length"] == len("AgentField rocks")
        assert result["tokens"] == 2
