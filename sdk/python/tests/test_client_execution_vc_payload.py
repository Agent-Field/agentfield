"""
Unit tests for parent_vc_id propagation in execution VC payload.
Tests that the SDK includes parent_vc_id in the execution_context when posting to /api/v1/execution/vc.
"""

import pytest
from unittest.mock import patch, MagicMock
from agentfield.vc_generator import VCGenerator
from agentfield.execution_context import ExecutionContext
from datetime import datetime, timezone


@pytest.mark.unit
def test_vc_generator_includes_parent_vc_id_in_payload():
    """
    Verify that VCGenerator.generate_execution_vc() includes parent_vc_id
    in the execution_context payload sent to the control plane.
    """
    vc_gen = VCGenerator("http://localhost:8080")
    vc_gen.set_enabled(True)

    # Create an execution context with parent_vc_id
    exec_context = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="test_reasoner",
        parent_vc_id="vc_trigger_webhook_stripe_payment_123",
        caller_did="did:test:caller",
        target_did="did:test:target",
        agent_node_did="did:test:agent",
        session_id="sess-1",
        run_id="run-1",
    )
    # Add timestamp required by generate_execution_vc
    exec_context.timestamp = datetime.now(timezone.utc)

    input_data = {"event": "payment.succeeded"}
    output_data = {"processed": True}

    # Mock the requests.post call to capture the payload
    with patch("agentfield.vc_generator.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "vc_id": "vc-generated-1",
            "execution_id": "exec-1",
            "workflow_id": "wf-1",
            "session_id": "sess-1",
            "issuer_did": "did:issuer",
            "target_did": "did:test:target",
            "caller_did": "did:test:caller",
            "vc_document": {},
            "signature": "sig",
            "input_hash": "hash_in",
            "output_hash": "hash_out",
            "status": "success",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        mock_post.return_value = mock_response

        vc_gen.generate_execution_vc(
            execution_context=exec_context,
            input_data=input_data,
            output_data=output_data,
            status="success",
        )

        # Verify the POST was called
        assert mock_post.called

        # Extract the payload sent in the POST request
        call_args = mock_post.call_args
        posted_json = call_args.kwargs.get("json") or call_args[1].get("json")

        # Verify execution_context is in the payload
        assert "execution_context" in posted_json
        exec_ctx_payload = posted_json["execution_context"]

        # Verify parent_vc_id is included in execution_context
        assert "parent_vc_id" in exec_ctx_payload
        assert exec_ctx_payload["parent_vc_id"] == "vc_trigger_webhook_stripe_payment_123"


@pytest.mark.unit
def test_vc_generator_omits_parent_vc_id_when_none():
    """
    Verify that VCGenerator includes parent_vc_id even when None
    (the control plane will handle the null/optional value).
    """
    vc_gen = VCGenerator("http://localhost:8080")
    vc_gen.set_enabled(True)

    # Create an execution context WITHOUT parent_vc_id
    exec_context = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="test_reasoner",
        caller_did="did:test:caller",
        target_did="did:test:target",
        agent_node_did="did:test:agent",
        session_id="sess-1",
        run_id="run-1",
    )
    exec_context.timestamp = datetime.now(timezone.utc)

    input_data = {"event": "test"}
    output_data = {"result": "ok"}

    with patch("agentfield.vc_generator.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "vc_id": "vc-generated-2",
            "execution_id": "exec-1",
            "workflow_id": "wf-1",
            "session_id": "sess-1",
            "issuer_did": "did:issuer",
            "target_did": "did:test:target",
            "caller_did": "did:test:caller",
            "vc_document": {},
            "signature": "sig",
            "input_hash": "hash_in",
            "output_hash": "hash_out",
            "status": "success",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        mock_post.return_value = mock_response

        vc_gen.generate_execution_vc(
            execution_context=exec_context,
            input_data=input_data,
            output_data=output_data,
            status="success",
        )

        call_args = mock_post.call_args
        posted_json = call_args.kwargs.get("json") or call_args[1].get("json")
        exec_ctx_payload = posted_json["execution_context"]

        # Verify parent_vc_id is present and None
        assert "parent_vc_id" in exec_ctx_payload
        assert exec_ctx_payload["parent_vc_id"] is None


@pytest.mark.unit
def test_vc_generator_posts_to_correct_endpoint():
    """
    Verify that VCGenerator posts to /api/v1/execution/vc endpoint.
    """
    vc_gen = VCGenerator("http://localhost:8080")
    vc_gen.set_enabled(True)

    exec_context = ExecutionContext(
        workflow_id="wf-1",
        execution_id="exec-1",
        agent_instance=None,
        reasoner_name="test_reasoner",
        parent_vc_id="vc_trigger_test",
        caller_did="did:test:caller",
        target_did="did:test:target",
        agent_node_did="did:test:agent",
        session_id="sess-1",
        run_id="run-1",
    )
    exec_context.timestamp = datetime.now(timezone.utc)

    with patch("agentfield.vc_generator.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "vc_id": "vc-generated-3",
            "execution_id": "exec-1",
            "workflow_id": "wf-1",
            "session_id": "sess-1",
            "issuer_did": "did:issuer",
            "target_did": "did:test:target",
            "caller_did": "did:test:caller",
            "vc_document": {},
            "signature": "sig",
            "input_hash": "hash_in",
            "output_hash": "hash_out",
            "status": "success",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        mock_post.return_value = mock_response

        vc_gen.generate_execution_vc(
            execution_context=exec_context,
            input_data={"test": "data"},
            output_data={"result": "success"},
            status="success",
        )

        # Verify the endpoint
        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", "")
        assert "/api/v1/execution/vc" in url
