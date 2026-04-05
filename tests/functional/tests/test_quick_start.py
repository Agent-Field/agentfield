"""
Functional tests covering the README and docs Quick Start flows.

These tests make sure both public entry points stay accurate by:
1. Spinning up the router-based `demo_echo` agent that ships with `af init`
2. Running the OpenRouter-powered summarization agent from the README
3. Driving both agents entirely through the control plane APIs (`/execute`, `/reasoners`)
"""

import os
import queue
import threading
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

import pytest

from agents.docs_quick_start_agent import (
    AGENT_SPEC as DOCS_QUICK_START_SPEC,
    create_agent as create_docs_quick_start_agent,
)
from agents.quick_start_agent import create_agent as create_readme_quick_start_agent
from utils import run_agent_server


QUICK_START_URL = os.environ.get("TEST_QUICK_START_URL")
TEST_BIND_HOST = os.environ.get("TEST_AGENT_BIND_HOST", "127.0.0.1")
TEST_CALLBACK_HOST = os.environ.get("TEST_AGENT_CALLBACK_HOST", "127.0.0.1")
README_NODE_ID = "researcher"
DOCS_NODE_ID = "my-agent"
DEMO_MESSAGE = "Hello, Agentfield!"

EXAMPLE_DOMAIN_HTML = """<!doctype html>
<html>
<head>
    <title>Example Domain</title>
    <meta charset="utf-8" />
    <meta http-equiv="Content-type" content="text/html; charset=utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
</head>
<body>
    <div>
        <h1>Example Domain</h1>
        <p>This domain is for use in illustrative examples in documents. You may use this
        domain in literature without prior coordination or asking for permission.</p>
        <p><a href="https://www.iana.org/domains/example">More information...</a></p>
    </div>
</body>
</html>
"""


def _start_example_domain_server() -> Tuple[ThreadingHTTPServer, threading.Thread, str]:
    """
    Spin up a lightweight HTTP server that serves the Example Domain HTML used in docs.
    """

    class ExampleDomainHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(EXAMPLE_DOMAIN_HTML.encode("utf-8"))

        def log_message(self, *_):
            # Silence default logging noise from BaseHTTPRequestHandler
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), ExampleDomainHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}"


def _start_webhook_capture_server() -> Tuple[ThreadingHTTPServer, threading.Thread, str, "queue.Queue[dict]"]:
    """
    Spin up a lightweight HTTP server that captures execution webhooks.
    """

    deliveries: "queue.Queue[dict]" = queue.Queue()

    class WebhookCaptureHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)

            payload = json.loads(body.decode("utf-8"))
            deliveries.put(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')

        def log_message(self, *_):
            return

    server = ThreadingHTTPServer((TEST_BIND_HOST, 0), WebhookCaptureHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _, port = server.server_address
    return server, thread, f"http://{TEST_CALLBACK_HOST}:{port}/webhook", deliveries


@pytest.mark.functional
@pytest.mark.asyncio
async def test_docs_quick_start_demo_echo_flow(async_http_client):
    """
    Validate the `/docs/quick-start` instructions (demo_echo router + /execute endpoint).
    """
    node_id = DOCS_NODE_ID
    assert node_id == DOCS_QUICK_START_SPEC.default_node_id

    agent = create_docs_quick_start_agent(node_id=node_id)

    async with run_agent_server(agent):
        nodes_response = await async_http_client.get(f"/api/v1/nodes/{node_id}")
        assert nodes_response.status_code == 200, nodes_response.text

        node_data = nodes_response.json()
        assert node_data["id"] == node_id

        reasoner_ids = [r.get("id") for r in node_data.get("reasoners", [])]
        assert any(
            rid in {"demo_echo", "echo"} for rid in reasoner_ids
        ), f"Reasoner IDs {reasoner_ids} did not include demo_echo/echo"

        execution_request = {"input": {"message": DEMO_MESSAGE}}

        execution_response = await async_http_client.post(
            f"/api/v1/execute/{node_id}.demo_echo",
            json=execution_request,
            timeout=30.0,
        )

        assert execution_response.status_code == 200, execution_response.text
        payload = execution_response.json()

        assert payload["status"] == "succeeded"
        assert payload["execution_id"]
        assert payload["run_id"]
        assert payload["duration_ms"] >= 0
        assert payload["finished_at"]

        result = payload["result"]
        assert result["original"] == DEMO_MESSAGE
        assert result["echoed"] == DEMO_MESSAGE
        assert result["length"] == len(DEMO_MESSAGE)

        print("✓ Docs Quick Start demo_echo flow succeeded")


@pytest.mark.functional
@pytest.mark.asyncio
async def test_docs_quick_start_execution_webhook_contract(async_http_client):
    """
    Validate execution webhook delivery for the quick-start demo_echo flow.
    """
    node_id = DOCS_NODE_ID
    agent = create_docs_quick_start_agent(node_id=node_id)
    webhook_server = None
    webhook_thread = None

    try:
        async with run_agent_server(agent):
            webhook_server, webhook_thread, webhook_url, deliveries = _start_webhook_capture_server()

            execution_response = await async_http_client.post(
                f"/api/v1/execute/{node_id}.demo_echo",
                json={
                    "input": {"message": DEMO_MESSAGE},
                    "context": {"analysis_group": "demo.short_form"},
                    "webhook": {"url": webhook_url},
                },
                timeout=30.0,
            )

            assert execution_response.status_code == 200, execution_response.text
            response_payload = execution_response.json()
            assert response_payload["status"] == "succeeded"
            assert response_payload["webhook_registered"] is True

            webhook_payload = deliveries.get(timeout=10)
            assert webhook_payload["event"] == "execution.completed"
            assert webhook_payload["execution_id"] == response_payload["execution_id"]
            assert webhook_payload["workflow_id"] == response_payload["run_id"]
            assert webhook_payload["agent_node_id"] == node_id
            assert webhook_payload["reasoner_id"] == "demo_echo"
            assert webhook_payload["status"] == "succeeded"
            assert webhook_payload["target"] == f"{node_id}.demo_echo"
            assert webhook_payload["type"] == "reasoner"
            assert webhook_payload["started_at"]
            assert webhook_payload["completed_at"]
            assert webhook_payload["timestamp"]
            assert webhook_payload["duration_ms"] >= 0
            assert webhook_payload["retry_count"] == 0
            assert webhook_payload["context"] == {"analysis_group": "demo.short_form"}
            assert webhook_payload["result"]["original"] == DEMO_MESSAGE
            assert webhook_payload["result"]["echoed"] == DEMO_MESSAGE
            assert webhook_payload.get("error_category") in (None, "")
            assert webhook_payload.get("status_reason") in (None, "")

            print("✓ Docs Quick Start execution webhook contract succeeded")
    finally:
        if webhook_server:
            webhook_server.shutdown()
        if webhook_thread:
            webhook_thread.join(timeout=5)


@pytest.mark.functional
@pytest.mark.openrouter
@pytest.mark.flaky(reruns=2, reruns_delay=5, condition=True, reason="OpenRouter API can intermittently timeout")
@pytest.mark.asyncio
async def test_readme_quick_start_summarize_flow(
    openrouter_config,
    async_http_client,
):
    """
    Validate the README Quick Start instructions end-to-end.

    This spins up the canonical README agent (fetch_url + summarize), registers it
    as `researcher`, submits a request through `/api/v1/execute/researcher.summarize`,
    and ensures the summarization result matches the documentation.
    """
    content_server: Optional[ThreadingHTTPServer] = None
    content_thread: Optional[threading.Thread] = None

    # Determine which URL to summarize. Default to local Example Domain server
    # to avoid relying on outbound internet access, but allow overriding via env.
    if QUICK_START_URL:
        target_url = QUICK_START_URL
    else:
        content_server, content_thread, target_url = _start_example_domain_server()

    node_id = README_NODE_ID
    agent = create_readme_quick_start_agent(openrouter_config, node_id=node_id)

    async with run_agent_server(agent):
        nodes_response = await async_http_client.get(f"/api/v1/nodes/{agent.node_id}")
        assert nodes_response.status_code == 200, nodes_response.text

        node_data = nodes_response.json()
        assert node_data["id"] == agent.node_id
        assert "summarize" in [r["id"] for r in node_data.get("reasoners", [])]

        execution_request = {"input": {"url": target_url}}

        execution_response = await async_http_client.post(
            f"/api/v1/execute/{agent.node_id}.summarize",
            json=execution_request,
            timeout=90.0,
        )

        assert execution_response.status_code == 200, execution_response.text
        result_data = execution_response.json()

        assert "result" in result_data
        result = result_data["result"]

        assert result["url"] == target_url
        summary_text = result["summary"]
        assert summary_text, "Summary should not be empty"
        assert len(summary_text.split()) >= 5, "Summary should contain multiple words"

        snippet = result.get("content_snippet", "")
        assert "Example Domain" in snippet, "Snippet should contain fetched page content"
        assert len(snippet) > 0

        assert result_data["duration_ms"] > 0

        print("✓ README Quick Start summarize flow succeeded")

    if content_server:
        content_server.shutdown()
        if content_thread:
            content_thread.join(timeout=5)
