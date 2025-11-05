import asyncio
from haxen_sdk.connection_manager import (
    ConnectionManager,
    ConnectionConfig,
    ConnectionState,
)


class FakeClient:
    async def register_agent_with_status(self, **kwargs):
        return False, None  # simulate failure so start enters reconnection


class FakeHaxenHandler:
    async def send_enhanced_heartbeat(self):
        return True


class FakeAgent:
    def __init__(self):
        self.client = FakeClient()
        self.haxen_handler = FakeHaxenHandler()
        self.node_id = "n"
        self.reasoners = []
        self.skills = []
        self.base_url = "http://agent"
        self._current_status = None


def test_start_enters_reconnecting_and_stop_quick(monkeypatch):
    agent = FakeAgent()
    cfg = ConnectionConfig(retry_interval=0.01, health_check_interval=0.01)
    mgr = ConnectionManager(agent, cfg)

    async def fake_reconnect_loop(self):
        # Simulate a quick state flip then exit
        self.state = ConnectionState.RECONNECTING
        await asyncio.sleep(0)

    # Monkeypatch the reconnection loop to avoid long-running task
    monkeypatch.setattr(
        ConnectionManager, "_reconnection_loop", fake_reconnect_loop, raising=False
    )

    async def run():
        ok = await mgr.start()
        assert ok is False
        # After failure, state may be set to DEGRADED by _on_connection_failure,
        # and reconnection task is scheduled. Accept either.
        assert mgr.state in (ConnectionState.RECONNECTING, ConnectionState.DEGRADED)
        await mgr.stop()

    asyncio.run(run())
