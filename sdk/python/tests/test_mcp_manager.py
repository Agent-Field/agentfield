"""
Tests for agentfield.mcp_manager — MCPManager lifecycle.
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch


from agentfield.mcp_manager import MCPManager, MCPServerConfig, MCPServerProcess


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(tmp_dir: str = "/tmp", dev_mode: bool = False) -> MCPManager:
    return MCPManager(agent_directory=tmp_dir, dev_mode=dev_mode)


def _make_config(alias: str = "test-server", transport: str = "http") -> MCPServerConfig:
    return MCPServerConfig(
        alias=alias,
        run_command="echo hello",
        working_dir="/tmp",
        environment={},
        transport=transport,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestMCPManagerInit:
    def test_default_attributes(self):
        mgr = _make_manager()
        assert mgr.servers == {}
        assert mgr.stdio_bridges == {}
        assert mgr.port_range_start == 8100
        assert mgr.used_ports == set()

    def test_dev_mode_stored(self):
        mgr = MCPManager(agent_directory="/tmp", dev_mode=True)
        assert mgr.dev_mode is True

    def test_agent_directory_stored(self):
        mgr = MCPManager(agent_directory="/var/app")
        assert mgr.agent_directory == "/var/app"


# ---------------------------------------------------------------------------
# discover_mcp_servers
# ---------------------------------------------------------------------------


class TestDiscoverMCPServers:
    def test_returns_empty_when_no_mcp_dir(self):
        mgr = _make_manager(tmp_dir="/nonexistent/path")
        servers = mgr.discover_mcp_servers()
        assert servers == []

    def test_discovers_server_from_config_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            mcp_dir = os.path.join(tmp, "packages", "mcp", "my-server")
            os.makedirs(mcp_dir)
            config = {
                "alias": "my-mcp",
                "run": "node server.js",
                "environment": {"FOO": "bar"},
                "transport": "stdio",
            }
            with open(os.path.join(mcp_dir, "config.json"), "w") as f:
                json.dump(config, f)

            mgr = MCPManager(agent_directory=tmp)
            servers = mgr.discover_mcp_servers()

        assert len(servers) == 1
        assert servers[0].alias == "my-mcp"
        assert servers[0].run_command == "node server.js"
        assert servers[0].transport == "stdio"

    def test_skips_dir_without_config_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            mcp_dir = os.path.join(tmp, "packages", "mcp", "broken-server")
            os.makedirs(mcp_dir)
            # No config.json

            mgr = MCPManager(agent_directory=tmp)
            servers = mgr.discover_mcp_servers()

        assert servers == []

    def test_uses_directory_name_as_alias_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            mcp_dir = os.path.join(tmp, "packages", "mcp", "fallback-alias")
            os.makedirs(mcp_dir)
            config = {"run": "python server.py"}
            with open(os.path.join(mcp_dir, "config.json"), "w") as f:
                json.dump(config, f)

            mgr = MCPManager(agent_directory=tmp)
            servers = mgr.discover_mcp_servers()

        assert servers[0].alias == "fallback-alias"


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------


class TestPortAllocation:
    def test_get_next_available_port_returns_int(self):
        mgr = _make_manager()
        port = mgr._get_next_available_port()
        assert isinstance(port, int)
        assert port >= 8100

    def test_used_ports_tracked(self):
        mgr = _make_manager()
        port = mgr._get_next_available_port()
        assert port in mgr.used_ports

    def test_no_duplicate_ports(self):
        mgr = _make_manager()
        p1 = mgr._get_next_available_port()
        p2 = mgr._get_next_available_port()
        assert p1 != p2


# ---------------------------------------------------------------------------
# _detect_transport
# ---------------------------------------------------------------------------


class TestDetectTransport:
    def test_http_transport(self):
        mgr = _make_manager()
        config = _make_config(transport="http")
        assert mgr._detect_transport(config) == "http"

    def test_stdio_transport(self):
        mgr = _make_manager()
        config = _make_config(transport="stdio")
        assert mgr._detect_transport(config) == "stdio"


# ---------------------------------------------------------------------------
# start_server — HTTP path (mocked subprocess)
# ---------------------------------------------------------------------------


class TestStartHTTPServer:
    async def test_start_http_server_success(self):
        mgr = _make_manager()
        config = _make_config(transport="http")

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # still running

        with patch("subprocess.Popen", return_value=mock_process), \
             patch.object(mgr, "_get_next_available_port", return_value=8200), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mgr._start_http_server(config)

        assert result is True
        assert "test-server" in mgr.servers
        assert mgr.servers["test-server"].status == "running"

    async def test_start_http_server_process_dies(self):
        mgr = _make_manager()
        config = _make_config(transport="http")

        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # process exited

        with patch("subprocess.Popen", return_value=mock_process), \
             patch.object(mgr, "_get_next_available_port", return_value=8201), \
             patch("asyncio.sleep", new_callable=AsyncMock):
            result = await mgr._start_http_server(config)

        assert result is False

    async def test_start_http_server_exception_returns_false(self):
        mgr = _make_manager()
        config = _make_config(transport="http")

        with patch("subprocess.Popen", side_effect=OSError("no such file")), \
             patch.object(mgr, "_get_next_available_port", return_value=8202):
            result = await mgr._start_http_server(config)

        assert result is False


# ---------------------------------------------------------------------------
# stop_server — HTTP path
# ---------------------------------------------------------------------------


class TestStopServer:
    async def test_stop_http_server_terminates_process(self):
        mgr = _make_manager()
        mock_process = MagicMock()
        mock_process.poll.return_value = None

        server_proc = MCPServerProcess(
            config=_make_config(),
            process=mock_process,
            port=8200,
            status="running",
        )
        mgr.servers["test-server"] = server_proc

        result = await mgr.stop_server("test-server")

        assert result is True
        mock_process.terminate.assert_called_once()

    async def test_stop_nonexistent_server_returns_false(self):
        mgr = _make_manager()
        result = await mgr.stop_server("nonexistent")
        assert result is False

    async def test_stop_releases_port(self):
        mgr = _make_manager()
        mgr.used_ports.add(8300)
        server_proc = MCPServerProcess(
            config=_make_config(),
            process=None,
            port=8300,
            status="stopped",
        )
        mgr.servers["test-server"] = server_proc

        await mgr.stop_server("test-server")
        assert 8300 not in mgr.used_ports


# ---------------------------------------------------------------------------
# get_server_status
# ---------------------------------------------------------------------------


class TestGetServerStatus:
    def test_returns_none_for_unknown_alias(self):
        mgr = _make_manager()
        assert mgr.get_server_status("unknown") is None

    def test_returns_http_server_status(self):
        mgr = _make_manager()
        server_proc = MCPServerProcess(
            config=_make_config(),
            process=None,
            port=8200,
            status="running",
        )
        mgr.servers["test-server"] = server_proc
        status = mgr.get_server_status("test-server")
        assert status is not None
        assert status["transport"] == "http"
        assert status["port"] == 8200
        assert status["status"] == "running"

    def test_returns_stdio_bridge_status(self):
        mgr = _make_manager()
        mock_bridge = MagicMock()
        mock_bridge.port = 8300
        mock_bridge.running = True
        mock_bridge.initialized = True
        mgr.stdio_bridges["stdio-server"] = mock_bridge

        status = mgr.get_server_status("stdio-server")
        assert status is not None
        assert status["transport"] == "stdio"
        assert status["status"] == "running"


# ---------------------------------------------------------------------------
# start_server dispatch
# ---------------------------------------------------------------------------


class TestStartServerDispatch:
    async def test_stdio_transport_calls_start_stdio_server(self):
        mgr = _make_manager()
        config = _make_config(transport="stdio")

        with patch.object(mgr, "_start_stdio_server", new_callable=AsyncMock, return_value=True) as mock_start:
            result = await mgr.start_server(config)

        mock_start.assert_called_once_with(config)
        assert result is True

    async def test_http_transport_calls_start_http_server(self):
        mgr = _make_manager()
        config = _make_config(transport="http")

        with patch.object(mgr, "_start_http_server", new_callable=AsyncMock, return_value=True) as mock_start:
            result = await mgr.start_server(config)

        mock_start.assert_called_once_with(config)
        assert result is True


# ---------------------------------------------------------------------------
# shutdown_all
# ---------------------------------------------------------------------------


class TestShutdownAll:
    async def test_shutdown_all_stops_all_servers(self):
        mgr = _make_manager()

        # Add an HTTP server
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        server_proc = MCPServerProcess(
            config=_make_config("http-srv"),
            process=mock_process,
            port=8400,
            status="running",
        )
        mgr.servers["http-srv"] = server_proc

        await mgr.shutdown_all()

        # After shutdown_all, the process MUST have been terminated
        mock_process.terminate.assert_called()
        # Server entry should be cleaned up (removed from dict)
        assert "http-srv" not in mgr.servers or mgr.servers["http-srv"].status != "running"
