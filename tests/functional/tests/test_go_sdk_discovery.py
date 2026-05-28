import asyncio
import json
import os

import pytest

from utils import get_go_agent_binary, run_go_agent, unique_node_id
from tests.test_go_sdk_cli import _wait_for_agent_health, _wait_for_registration


async def _run_discovery_cli(binary: str, env: dict[str, str]) -> dict:
    proc = await asyncio.create_subprocess_exec(
        binary,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise AssertionError(
            f"discovery cli failed rc={proc.returncode} stderr={stderr.decode()}"
        )
    return json.loads(stdout)


@pytest.mark.functional
@pytest.mark.asyncio
async def test_go_sdk_discovery_flow(async_http_client, control_plane_url):
    try:
        get_go_agent_binary("hello")
        discovery_binary = get_go_agent_binary("discovery")
    except FileNotFoundError:
        pytest.skip("Go discovery binaries not available in test image")

    node_id = unique_node_id("go-discovery-agent")
    env_server = {
        **os.environ,
        "AGENTFIELD_URL": control_plane_url,
        "AGENT_NODE_ID": node_id,
        "AGENT_LISTEN_ADDR": ":8101",
        "AGENT_PUBLIC_URL": "http://test-runner:8101",
    }

    async with run_go_agent("hello", args=["serve"], env=env_server):
        await _wait_for_registration(async_http_client, node_id)
        await _wait_for_agent_health("http://127.0.0.1:8101/health")

        base_env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"AGENTFIELD_URL", "AGENT_PUBLIC_URL"}
        }
        base_env.update(
            {
                "AGENTFIELD_URL": control_plane_url,
                "AGENT_NODE_ID": unique_node_id("go-discovery-client"),
            }
        )

        # JSON format filtered to the Go agent with schema inclusion.
        env_json = {
            **base_env,
            "DISCOVERY_AGENT": node_id,
            "DISCOVERY_INCLUDE_INPUT_SCHEMA": "true",
            "DISCOVERY_INCLUDE_OUTPUT_SCHEMA": "true",
            "DISCOVERY_FORMAT": "json",
        }
        json_payload = await _run_discovery_cli(discovery_binary, env_json)
        assert json_payload["format"] == "json"
        totals = json_payload.get("totals", {})
        assert totals.get("agents") == 1
        assert totals.get("reasoners") >= 1
        caps = json_payload.get("capabilities", [])
        assert caps and caps[0]["agent_id"] == node_id
        reasoner_ids = {item["id"] for item in caps[0]["reasoners"]}
        assert {"demo_echo", "say_hello", "add_emoji"}.issubset(reasoner_ids)

        # Compact format filtered by reasoner pattern.
        env_compact = {
            **base_env,
            "DISCOVERY_AGENT": node_id,
            "DISCOVERY_REASONER_PATTERN": "say_*",
            "DISCOVERY_FORMAT": "compact",
        }
        compact_payload = await _run_discovery_cli(discovery_binary, env_compact)
        assert compact_payload["format"] == "compact"
        compact_totals = compact_payload.get("totals", {})
        assert compact_totals.get("reasoners") == 1
        targets = {item["target"] for item in compact_payload.get("reasoners", [])}
        assert f"{node_id}:say_hello" in targets

        # XML format sanity check.
        env_xml = {**base_env, "DISCOVERY_FORMAT": "xml"}
        xml_payload = await _run_discovery_cli(discovery_binary, env_xml)
        assert xml_payload["format"] == "xml"
        xml_body = xml_payload.get("xml", "")
        assert node_id in xml_body
