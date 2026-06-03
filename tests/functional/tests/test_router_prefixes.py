import pytest

from agents.router_prefix_agent import AGENT_SPEC, create_agent as create_router_agent
from utils import run_agent_server, unique_node_id


@pytest.mark.functional
@pytest.mark.asyncio
async def test_router_prefix_registration_and_execution(async_http_client):
    agent = create_router_agent(node_id=unique_node_id(AGENT_SPEC.default_node_id))

    async with run_agent_server(agent):
        node_response = await async_http_client.get(f"/api/v1/nodes/{agent.node_id}")
        assert node_response.status_code == 200
        node_data = node_response.json()

        reasoner_ids = {r["id"] for r in node_data.get("reasoners", [])}
        assert {"tools_echo", "tools_status"} <= reasoner_ids

        echo_response = await async_http_client.post(
            f"/api/v1/execute/{agent.node_id}.tools_echo",
            json={"input": {"message": "router check"}},
            timeout=20.0,
        )
        assert echo_response.status_code == 200
        echo_result = echo_response.json()["result"]
        assert echo_result["message"] == "router check"
        assert echo_result["length"] == len("router check")

        status_response = await async_http_client.post(
            f"/api/v1/reasoners/{agent.node_id}.tools_status",
            json={"input": {}},
            timeout=20.0,
        )
        assert status_response.status_code == 200
        status_result = status_response.json()["result"]
        assert status_result["node_id"] == agent.node_id
        assert status_result["router_prefix"] == "tools"
        assert "tools_echo" in status_result["reasoners"]
