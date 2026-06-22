"""
Functional smoke tests for Agentic Resource Discovery (ARD).
"""

from __future__ import annotations

import pytest

from utils import run_agent_server, unique_node_id


@pytest.mark.functional
@pytest.mark.asyncio
async def test_ard_catalog_publish_and_registry_routes(make_test_agent, async_http_client):
    agent = make_test_agent(node_id=unique_node_id("ard-agent"))

    @agent.reasoner(tags=["legal", "contracts"])
    async def ard_review_contract(text: str) -> dict:
        return {"status": "reviewed", "text": text}

    async with run_agent_server(agent):
        dashboard_resp = await async_http_client.get("/api/ui/v1/ard", timeout=30.0)
        assert dashboard_resp.status_code == 200, dashboard_resp.text
        dashboard = dashboard_resp.json()
        assert dashboard["summary"]["ard_enabled"] is True
        assert dashboard["summary"]["catalog_published"] is True

        publication = next(
            item
            for item in dashboard["publications"]
            if item["node_id"] == agent.node_id and item["target_id"] == "ard_review_contract"
        )
        publication.update(
            {
                "published": True,
                "display_name": "ARD Contract Reviewer",
                "description": "Reviews contract text for functional ARD discovery.",
                "tags": ["legal", "contracts"],
                "representative_queries": ["review this MSA", "find risky clauses"],
                "artifact_type": "application/openapi+json",
            }
        )

        publish_resp = await async_http_client.put(
            "/api/ui/v1/ard/publications",
            json=publication,
            timeout=30.0,
        )
        assert publish_resp.status_code == 200, publish_resp.text

        catalog_resp = await async_http_client.get(
            "/.well-known/ai-catalog.json",
            timeout=30.0,
        )
        assert catalog_resp.status_code == 200, catalog_resp.text
        catalog = catalog_resp.json()
        assert catalog["specVersion"]
        assert catalog["host"]["displayName"] == "AgentField Functional Control Plane"
        assert catalog["host"]["identifier"] == "did:web:functional.agentfield.local"
        entries = catalog["entries"]
        entry = next(item for item in entries if item["displayName"] == "ARD Contract Reviewer")
        assert entry["identifier"].startswith("urn:ai:functional.agentfield.local:agentfield:")
        assert entry["identifier"].endswith(":reasoner:ard_review_contract")
        assert bool(entry.get("url")) ^ bool(entry.get("data"))
        assert entry["type"] == "application/openapi+json"

        artifact_resp = await async_http_client.get(entry["url"], timeout=30.0)
        assert artifact_resp.status_code == 200, artifact_resp.text
        artifact = artifact_resp.json()
        assert artifact["info"]["title"] == "ARD Contract Reviewer"

        search_resp = await async_http_client.post(
            "/api/v1/ard/search",
            json={"query": {"text": "contract"}, "pageSize": 10},
            timeout=30.0,
        )
        assert search_resp.status_code == 200, search_resp.text
        search_payload = search_resp.json()
        assert any(result["identifier"] == entry["identifier"] for result in search_payload["results"])

        agents_resp = await async_http_client.get("/api/v1/ard/agents?pageSize=5", timeout=30.0)
        assert agents_resp.status_code == 200, agents_resp.text
        agents_payload = agents_resp.json()
        assert any(item["identifier"] == entry["identifier"] for item in agents_payload["items"])

        explore_resp = await async_http_client.post(
            "/api/v1/ard/explore",
            json={
                "query": {"text": "contract"},
                "resultType": {"facets": [{"field": "tags"}]},
            },
            timeout=30.0,
        )
        assert explore_resp.status_code == 200, explore_resp.text
        explore_payload = explore_resp.json()
        tag_values = {bucket["value"] for bucket in explore_payload["facets"]["tags"]["buckets"]}
        assert "contracts" in tag_values
