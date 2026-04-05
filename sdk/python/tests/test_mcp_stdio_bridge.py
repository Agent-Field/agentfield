"""
Tests for agentfield.mcp_stdio_bridge — StdioMCPBridge.
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentfield.mcp_stdio_bridge import StdioMCPBridge, PendingRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bridge(**kwargs) -> StdioMCPBridge:
    config = {
        "run": "echo hello",
        "working_dir": "/tmp",
        "environment": {},
    }
    return StdioMCPBridge(server_config=config, port=9000, **kwargs)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestStdioMCPBridgeInit:
    def test_default_state(self):
        bridge = _make_bridge()
        assert bridge.port == 9000
        assert bridge.running is False
        assert bridge.initialized is False
        assert bridge.pending_requests == {}
        assert bridge._request_id_counter == 0
        assert bridge.request_timeout == 30.0

    def test_dev_mode_stored(self):
        bridge = _make_bridge(dev_mode=True)
        assert bridge.dev_mode is True

    def test_server_config_stored(self):
        config = {"run": "python mcp.py", "working_dir": "/app", "environment": {"X": "1"}}
        bridge = StdioMCPBridge(server_config=config, port=9001)
        assert bridge.server_config == config


# ---------------------------------------------------------------------------
# _get_next_request_id
# ---------------------------------------------------------------------------


class TestGetNextRequestId:
    def test_increments_from_zero(self):
        bridge = _make_bridge()
        assert bridge._get_next_request_id() == 1
        assert bridge._get_next_request_id() == 2
        assert bridge._get_next_request_id() == 3


# ---------------------------------------------------------------------------
# stop — cancels pending requests
# ---------------------------------------------------------------------------


class TestStopCancelsPending:
    async def test_stop_cancels_pending_requests(self):
        bridge = _make_bridge()
        bridge.running = True

        future = asyncio.get_event_loop().create_future()
        bridge.pending_requests["1"] = PendingRequest(
            future=future, timestamp=asyncio.get_event_loop().time()
        )

        # No actual process — patch the writer/process to None so stop() doesn't choke
        bridge.stdin_writer = None
        bridge.process = None
        bridge.server_task = None
        bridge.stdio_reader_task = None

        await bridge.stop()

        assert bridge.running is False
        assert bridge.pending_requests == {}
        assert future.done()
        assert isinstance(future.exception(), Exception)

    async def test_stop_sets_running_false(self):
        bridge = _make_bridge()
        bridge.running = True
        bridge.stdin_writer = None
        bridge.process = None
        bridge.server_task = None
        bridge.stdio_reader_task = None

        await bridge.stop()
        assert bridge.running is False


# ---------------------------------------------------------------------------
# _send_stdio_notification — requires stdin_writer
# ---------------------------------------------------------------------------


class TestSendNotification:
    async def test_raises_when_not_initialized(self):
        bridge = _make_bridge()
        with pytest.raises(RuntimeError, match="not initialized"):
            await bridge._send_stdio_notification("test/method", {})

    async def test_sends_jsonrpc_notification(self):
        bridge = _make_bridge()

        mock_writer = AsyncMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        bridge.stdin_writer = mock_writer

        await bridge._send_stdio_notification("notifications/test", {"key": "value"})

        mock_writer.write.assert_called_once()
        written = mock_writer.write.call_args[0][0].decode("utf-8")
        data = json.loads(written.strip())
        assert data["jsonrpc"] == "2.0"
        assert data["method"] == "notifications/test"
        assert "id" not in data  # notifications have no id


# ---------------------------------------------------------------------------
# _send_stdio_request — requires stdin_writer
# ---------------------------------------------------------------------------


class TestSendRequest:
    async def test_raises_when_not_initialized(self):
        bridge = _make_bridge()
        with pytest.raises(RuntimeError):
            await bridge._send_stdio_request("tools/list", {})

    async def test_request_written_with_correct_format(self):
        bridge = _make_bridge()

        mock_writer = AsyncMock()
        written_data = []
        mock_writer.write = MagicMock(side_effect=lambda b: written_data.append(b))
        mock_writer.drain = AsyncMock()
        bridge.stdin_writer = mock_writer

        # Manually resolve the future so _send_stdio_request returns
        async def resolve_future(*args, **kwargs):
            await asyncio.sleep(0)
            req_id = str(bridge._request_id_counter)
            pending = bridge.pending_requests.get(req_id)
            if pending and not pending.future.done():
                pending.future.set_result({"jsonrpc": "2.0", "id": int(req_id), "result": {}})

        mock_writer.drain.side_effect = resolve_future

        await bridge._send_stdio_request("tools/list", {})

        assert len(written_data) == 1
        payload = json.loads(written_data[0].decode("utf-8").strip())
        assert payload["method"] == "tools/list"
        assert payload["jsonrpc"] == "2.0"
        assert "id" in payload


# ---------------------------------------------------------------------------
# _handle_stdio_response — correlates responses
# ---------------------------------------------------------------------------


class TestHandleStdioResponse:
    async def test_resolves_pending_future(self):
        bridge = _make_bridge()

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        bridge.pending_requests["42"] = PendingRequest(
            future=future, timestamp=loop.time()
        )

        response = {"jsonrpc": "2.0", "id": 42, "result": {"tools": []}}
        await bridge._handle_stdio_response(response)

        assert future.done()
        assert future.result() == response
        assert "42" not in bridge.pending_requests

    async def test_ignores_notification_without_id(self):
        bridge = _make_bridge()
        # No pending requests should be affected
        notification = {"jsonrpc": "2.0", "method": "notifications/something"}
        await bridge._handle_stdio_response(notification)
        assert bridge.pending_requests == {}

    async def test_ignores_unknown_request_id(self):
        bridge = _make_bridge()
        # No matching pending request — should not raise
        response = {"jsonrpc": "2.0", "id": 999, "result": {}}
        await bridge._handle_stdio_response(response)


# ---------------------------------------------------------------------------
# _cleanup_expired_requests
# ---------------------------------------------------------------------------


class TestCleanupExpiredRequests:
    async def test_removes_expired_pending_requests(self):
        bridge = _make_bridge()
        bridge.request_timeout = 0.001  # very short

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        bridge.pending_requests["1"] = PendingRequest(
            future=future,
            timestamp=loop.time() - 10,  # 10 seconds ago → expired
        )

        await bridge._cleanup_expired_requests()

        assert "1" not in bridge.pending_requests
        assert future.done()
        assert isinstance(future.exception(), asyncio.TimeoutError)

    async def test_keeps_non_expired_requests(self):
        bridge = _make_bridge()
        bridge.request_timeout = 30.0

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        bridge.pending_requests["1"] = PendingRequest(
            future=future,
            timestamp=loop.time(),  # just added
        )

        await bridge._cleanup_expired_requests()

        assert "1" in bridge.pending_requests
        assert not future.done()

        # cleanup
        future.cancel()


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    async def test_returns_false_when_not_running(self):
        bridge = _make_bridge()
        bridge.running = False
        assert await bridge.health_check() is False

    async def test_returns_false_when_no_process(self):
        bridge = _make_bridge()
        bridge.running = True
        bridge.process = None
        assert await bridge.health_check() is False

    async def test_returns_false_when_process_exited(self):
        bridge = _make_bridge()
        bridge.running = True
        mock_process = MagicMock()
        mock_process.returncode = 1  # process has exited
        bridge.process = mock_process
        assert await bridge.health_check() is False


# ---------------------------------------------------------------------------
# _start_stdio_process — error handling
# ---------------------------------------------------------------------------


class TestStartStdioProcess:
    async def test_returns_false_when_no_run_command(self):
        bridge = StdioMCPBridge(
            server_config={"run": "", "working_dir": "/tmp", "environment": {}},
            port=9000,
        )
        result = await bridge._start_stdio_process()
        assert result is False

    async def test_returns_false_on_subprocess_error(self):
        bridge = _make_bridge()
        with patch("asyncio.create_subprocess_shell", side_effect=OSError("not found")):
            result = await bridge._start_stdio_process()
        assert result is False


# ---------------------------------------------------------------------------
# _handle_list_tools and _handle_call_tool
# ---------------------------------------------------------------------------


class TestHandleTools:
    async def test_handle_list_tools_returns_tools(self):
        bridge = _make_bridge()

        mock_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": [{"name": "calculator"}]},
        }
        with patch.object(bridge, "_send_stdio_request", new_callable=AsyncMock, return_value=mock_response):
            result = await bridge._handle_list_tools({})

        assert result == {"tools": [{"name": "calculator"}]}

    async def test_handle_list_tools_raises_on_error(self):
        bridge = _make_bridge()
        error_response = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "not found"}}
        with patch.object(bridge, "_send_stdio_request", new_callable=AsyncMock, return_value=error_response):
            with pytest.raises(RuntimeError, match="Tools list failed"):
                await bridge._handle_list_tools({})

    async def test_handle_call_tool_requires_name(self):
        bridge = _make_bridge()
        with pytest.raises(ValueError, match="Tool name is required"):
            await bridge._handle_call_tool({})

    async def test_handle_call_tool_returns_result(self):
        bridge = _make_bridge()
        mock_response = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "42"}]},
        }
        with patch.object(bridge, "_send_stdio_request", new_callable=AsyncMock, return_value=mock_response):
            result = await bridge._handle_call_tool({"name": "calculator", "arguments": {"x": 1}})

        assert result == {"content": [{"type": "text", "text": "42"}]}

    async def test_handle_call_tool_raises_on_error(self):
        bridge = _make_bridge()
        error_response = {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "failed"}}
        with patch.object(bridge, "_send_stdio_request", new_callable=AsyncMock, return_value=error_response):
            with pytest.raises(RuntimeError, match="Tool call failed"):
                await bridge._handle_call_tool({"name": "calc"})
