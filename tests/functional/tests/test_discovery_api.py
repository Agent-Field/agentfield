"""
Functional test for the Discovery API and Python SDK wrapper.
"""

from __future__ import annotations

import asyncio

import pytest

from utils import run_agent_server, unique_node_id


@pytest.mark.functional
@pytest.mark.asyncio
async def test_discovery_endpoint_and_python_sdk(make_test_agent, async_http_client):
    agent_primary = make_test_agent(node_id=unique_node_id("discovery-agent"))
    agent_secondary = make_test_agent(node_id=unique_node_id("insights-agent"))

    @agent_primary.reasoner(tags=["research", "ml"])
    async def deep_research(query: str, depth: int = 2) -> dict:
        return {"query": query, "depth": depth, "result": "ok"}

    @agent_primary.skill(tags=["web", "search"])
    def web_search(query: str, num_results: int = 3) -> dict:
        return {"query": query, "num_results": num_results}

    @agent_secondary.reasoner(tags=["analysis", "research"])
    async def global_research(topic: str, region: str = "global") -> dict:
        return {"topic": topic, "region": region}

    @agent_secondary.skill(tags=["web", "scraping"])
    def web_scraper(url: str) -> dict:
        return {"url": url, "status": "ok"}

    async with run_agent_server(agent_primary), run_agent_server(agent_secondary):
        node_filter = f"{agent_primary.node_id},{agent_secondary.node_id}"
        # JSON response with both agents, schemas, and pagination alias usage.
        response = await async_http_client.get(
            "/api/v1/discovery/capabilities",
            params={
                "node_ids": node_filter,
                "include_input_schema": "true",
                "include_output_schema": "true",
                "include_examples": "true",
                "limit": "2",
                "offset": "0",
            },
            timeout=30.0,
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["total_agents"] == 2
        assert payload["total_reasoners"] == 2
        assert payload["total_skills"] == 2
        assert payload["pagination"]["has_more"] is False

        agent_ids = {cap["agent_id"] for cap in payload["capabilities"]}
        assert {agent_primary.node_id, agent_secondary.node_id} == agent_ids
        for cap in payload["capabilities"]:
            for reasoner in cap["reasoners"]:
                assert reasoner.get("input_schema") is not None
                assert reasoner.get("output_schema") is not None

        # Wildcard reasoner filtering.
        reasoner_resp = await async_http_client.get(
            "/api/v1/discovery/capabilities",
            params={
                "reasoner": "*research*",
                "include_input_schema": "true",
                "node_ids": node_filter,
            },
            timeout=15.0,
        )
        reasoner_payload = reasoner_resp.json()
        assert reasoner_payload["total_reasoners"] == 2

        # Skill + tag filtering and pagination offset.
        skill_resp = await async_http_client.get(
            "/api/v1/discovery/capabilities",
            params={
                "skill": "web_*",
                "tags": "web*",
                "node_ids": node_filter,
                "limit": "1",
                "offset": "1",
            },
            timeout=15.0,
        )
        skill_payload = skill_resp.json()
        assert skill_payload["total_skills"] == 2
        assert skill_payload["pagination"]["has_more"] is False
        assert len(skill_payload["capabilities"]) == 1
        assert len(skill_payload["capabilities"][0]["skills"]) == 1

        # Compact format for lightweight clients.
        compact_resp = await async_http_client.get(
            "/api/v1/discovery/capabilities",
            params={"format": "compact", "tags": "research", "node_ids": node_filter},
            timeout=15.0,
        )
        assert compact_resp.status_code == 200, compact_resp.text
        compact_payload = compact_resp.json()
        assert len(compact_payload["reasoners"]) == 2
        assert any(
            item["id"] == "global_research" for item in compact_payload["reasoners"]
        )

        # XML format for LLM prompts.
        xml_resp = await async_http_client.get(
            "/api/v1/discovery/capabilities",
            params={"format": "xml"},
            timeout=15.0,
        )
        assert xml_resp.status_code == 200
        xml_body = xml_resp.text
        assert agent_primary.node_id in xml_body and agent_secondary.node_id in xml_body

        loop = asyncio.get_event_loop()

        # Python SDK wrapper (synchronous call via executor) - JSON format.
        sdk_json = await loop.run_in_executor(
            None,
            lambda: agent_primary.discover(
                reasoner="*research*",
                include_input_schema=True,
                agent_ids=[agent_primary.node_id, agent_secondary.node_id],
            ),
        )
        assert sdk_json.json is not None
        assert sdk_json.json.total_reasoners == 2

        # Python SDK wrapper for XML/compact outputs.
        sdk_compact = await loop.run_in_executor(
            None,
            lambda: agent_primary.discover(
                format="compact",
                tags=["research"],
                agent_ids=[agent_primary.node_id, agent_secondary.node_id],
            ),
        )
        assert sdk_compact.compact is not None
        assert any(cap.id == "deep_research" for cap in sdk_compact.compact.reasoners)

        sdk_xml = await loop.run_in_executor(
            None,
            lambda: agent_primary.discover(
                format="xml",
                agent_ids=[agent_primary.node_id, agent_secondary.node_id],
            ),
        )
        assert sdk_xml.xml is not None
        assert agent_secondary.node_id in sdk_xml.xml
