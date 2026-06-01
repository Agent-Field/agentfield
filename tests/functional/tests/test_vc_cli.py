import asyncio
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest

from utils import run_agent_server, unique_node_id


async def _wait_for_vc_chain(
    client,
    workflow_id: str,
    *,
    minimum_components: int = 1,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Poll the workflow VC chain endpoint until the expected component count appears."""
    deadline = asyncio.get_running_loop().time() + timeout
    last_payload: Dict[str, Any] | None = None

    while True:
        response = await client.get(f"/api/v1/did/workflow/{workflow_id}/vc-chain")
        if response.status_code == 200:
            last_payload = response.json()
            components = last_payload.get("component_vcs", [])
            if len(components) >= minimum_components:
                return last_payload

        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(
                f"Timed out waiting for VC chain for workflow {workflow_id}: {last_payload}"
            )

        await asyncio.sleep(1.0)


async def _run_vc_verify(vc_file: Path, *, expect_success: bool = True) -> Dict[str, Any]:
    """Execute `af vc verify` and parse the JSON payload."""
    process = await asyncio.create_subprocess_exec(
        "af",
        "vc",
        "verify",
        str(vc_file),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={
            **os.environ,
            "AGENTFIELD_HOME": os.environ.get("AGENTFIELD_HOME", "/tmp/agentfield-cli"),
        },
    )
    stdout, stderr = await process.communicate()

    stdout_text = stdout.decode()
    stderr_text = stderr.decode()

    try:
        payload = json.loads(stdout_text) if stdout_text.strip() else {}
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise AssertionError(
            f"CLI output was not valid JSON. stdout={stdout_text} stderr={stderr_text}"
        ) from exc

    if expect_success:
        assert (
            process.returncode == 0
        ), f"`af vc verify` failed unexpectedly: rc={process.returncode}, stderr={stderr_text}, stdout={stdout_text}"
    else:
        assert (
            process.returncode != 0
        ), "`af vc verify` succeeded unexpectedly when tampering should fail"

    return payload


@pytest.mark.functional
@pytest.mark.asyncio
async def test_vc_cli_verifies_workflow_chain(make_test_agent, async_http_client, tmp_path):
    """
    Validate the documentation workflow (docs/core-concepts/identity-and-trust) that exports
    a workflow VC chain and verifies it offline via `af vc verify`.
    """
    agent = make_test_agent(node_id=unique_node_id("vc-cli-agent"))

    @agent.reasoner()
    async def attest_event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a deterministic payload so hashing + VC generation is stable."""
        return {
            "event_type": event_type,
            "payload": payload,
            "fingerprint": f"{event_type}:{sorted(payload.keys())}",
        }

    async with run_agent_server(agent):
        response = await async_http_client.post(
            f"/api/v1/reasoners/{agent.node_id}.attest_event",
            json={
                "input": {
                    "event_type": "vc_cli_test",
                    "payload": {"severity": "info", "component": "vc-cli"},
                }
            },
            timeout=30.0,
        )
        assert response.status_code == 200, response.text

        headers = response.headers
        workflow_id = headers.get("X-Workflow-ID") or headers.get("x-workflow-id")
        execution_id = headers.get("X-Execution-ID") or headers.get("x-execution-id")
        assert workflow_id, "Workflow ID header missing from response"
        assert execution_id, "Execution ID header missing from response"

        vc_chain = await _wait_for_vc_chain(async_http_client, workflow_id, minimum_components=1)

        component_execution_ids = {vc["execution_id"] for vc in vc_chain["component_vcs"]}
        assert execution_id in component_execution_ids, "Execution VC not found in chain"

        vc_file = tmp_path / f"{workflow_id}-vc-chain.json"
        vc_file.write_text(json.dumps(vc_chain, indent=2))

        cli_result = await _run_vc_verify(vc_file, expect_success=True)

        assert cli_result["valid"] is True
        assert cli_result["signature_valid"] is True
        assert cli_result["workflow_id"] == workflow_id
        assert cli_result["summary"]["total_components"] >= 1
        assert (
            cli_result["summary"]["valid_components"]
            == cli_result["summary"]["total_components"]
        )

        # Tamper with an execution VC payload and expect verification to fail
        exec_tampered = copy.deepcopy(vc_chain)
        exec_tampered["component_vcs"][0]["vc_document"]["credentialSubject"]["execution"][
            "status"
        ] = "tampered"
        exec_tampered_file = tmp_path / f"{workflow_id}-vc-chain-exec-tampered.json"
        exec_tampered_file.write_text(json.dumps(exec_tampered, indent=2))

        tampered_exec_result = await _run_vc_verify(exec_tampered_file, expect_success=False)
        assert tampered_exec_result["valid"] is False

        # Tamper with the workflow-level VC document as well
        workflow_tampered = copy.deepcopy(vc_chain)
        workflow_doc = workflow_tampered["workflow_vc"]["vc_document"]
        workflow_doc["credentialSubject"]["status"] = "tampered"
        workflow_tampered_file = tmp_path / f"{workflow_id}-vc-chain-workflow-tampered.json"
        workflow_tampered_file.write_text(json.dumps(workflow_tampered, indent=2))

        tampered_workflow_result = await _run_vc_verify(
            workflow_tampered_file, expect_success=False
        )
        assert tampered_workflow_result["valid"] is False
