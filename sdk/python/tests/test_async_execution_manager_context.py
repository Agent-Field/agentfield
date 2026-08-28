import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentfield.async_config import AsyncConfig
from agentfield.async_execution_manager import AsyncExecutionManager
from agentfield.execution_context import (
    ExecutionContext,
    reset_execution_context,
    set_execution_context,
)
from agentfield.logger import AgentFieldLogger


class RecordingControlPlane:
    def __init__(self):
        self.calls = []

    async def post_execution_logs(self, execution_id, record):
        self.calls.append((execution_id, record))


@pytest.mark.asyncio
async def test_manager_background_logs_do_not_inherit_reasoner_context(monkeypatch):
    control_plane = RecordingControlPlane()
    public_logger = AgentFieldLogger("test.manager.context")
    public_logger._cp_client = control_plane
    monkeypatch.setattr("agentfield.async_execution_manager.logger", public_logger)

    manager = AsyncExecutionManager(
        "http://control",
        AsyncConfig(enable_performance_logging=False, enable_event_stream=False),
    )
    manager.connection_manager = SimpleNamespace(start=AsyncMock(), close=AsyncMock())
    manager.result_cache = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    background_logged = asyncio.Event()

    async def polling_loop():
        public_logger.info("Polling loop stopped")
        background_logged.set()
        await asyncio.Event().wait()

    async def cleanup_loop():
        await asyncio.Event().wait()

    manager._polling_loop = polling_loop
    manager._cleanup_loop = cleanup_loop

    execution_context = ExecutionContext.create_new("node-1", "reasoner")
    token = set_execution_context(execution_context)
    try:
        public_logger.info("reasoner is running")
        await manager.start()
    finally:
        reset_execution_context(token)

    await background_logged.wait()
    await asyncio.sleep(0)
    await manager.stop()
    await asyncio.sleep(0)

    assert [call[0] for call in control_plane.calls] == [execution_context.execution_id]
    assert control_plane.calls[0][1]["message"] == "reasoner is running"
