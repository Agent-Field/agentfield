"""Structured execution logs must never wreck the loop that carries them (#620).

``AgentFieldLogger`` forwards structured records to the control plane in the
background. It used to hand-roll that dispatch: an un-retained
``loop.create_task`` when a loop was running, and a daemon thread that built a
throwaway event loop *per record* when one was not. Both branches were wrong,
and the second one was actively destructive — the throwaway loop drove the
client's shared ``httpx.AsyncClient`` and then closed underneath it.

The behaviours asserted here are the contract for that dispatch; the loop
affinity of the shared HTTP client is asserted alongside them because it is the
same defect seen from the client's side.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from agentfield.client import AgentFieldClient
from agentfield.execution_context import ExecutionContext
from agentfield.logger import AgentFieldLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _context(execution_id: str) -> ExecutionContext:
    return ExecutionContext(
        workflow_id="wf-1",
        execution_id=execution_id,
        run_id="run-1",
        agent_instance=None,
        reasoner_name="sample_reasoner",
        agent_node_id="node-1",
    )


async def _wait_for(predicate, timeout: float = 5.0) -> bool:
    """Poll ``predicate`` from the running loop until it holds or time runs out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return bool(predicate())


def _emit_from_a_thread_without_a_loop(logger: AgentFieldLogger, execution_id: str):
    """Emit a structured record from a thread that has no event loop."""
    finished = threading.Event()

    def _run() -> None:
        try:
            logger.log_execution(
                "record from sync code",
                event_type="log.info",
                execution_context=_context(execution_id),
            )
        finally:
            finished.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=5)
    return finished


class _RecordingCpClient:
    """Control-plane client double that records where each dispatch ran."""

    def __init__(self):
        self.calls = []
        self.loops = []
        self.gate = None

    async def post_execution_logs(self, execution_id, entries):
        self.loops.append(asyncio.get_running_loop())
        if self.gate is not None:
            await self.gate.wait()
        self.calls.append((execution_id, entries))


class _FailingCpClient:
    def __init__(self):
        self.attempts = 0

    async def post_execution_logs(self, execution_id, entries):
        self.attempts += 1
        raise RuntimeError("control plane unreachable")


class _LoopBoundAsyncClient:
    """httpx.AsyncClient double that reproduces the real loop affinity.

    Verified against httpx 0.28.1 / httpcore 1.0.9: once a request has been
    made, the pooled connection lives on the loop that made it. Driving the
    same client from another loop raises ``Event loop is closed`` when that
    loop has since closed, and ``... is bound to a different event loop``
    while it is still alive.
    """

    def __init__(self, sink, **kwargs):
        self.headers = dict(kwargs.get("headers") or {})
        self.is_closed = False
        self.owning_loop = None
        self._sink = sink

    async def _bind(self) -> None:
        loop = asyncio.get_running_loop()
        if self.owning_loop is None:
            self.owning_loop = loop
        elif self.owning_loop is not loop:
            if self.owning_loop.is_closed():
                raise RuntimeError("Event loop is closed")
            raise RuntimeError(
                "<asyncio.locks.Event object> is bound to a different event loop"
            )

    async def post(self, url, **kwargs):
        await self._bind()
        self._sink.append(url)
        return SimpleNamespace(status_code=200, text="", json=lambda: {})

    async def aclose(self) -> None:
        self.is_closed = True


class _HttpxStub:
    """Stand-in ``httpx`` module handing out :class:`_LoopBoundAsyncClient`."""

    def __init__(self):
        self.requested_urls = []
        self.clients = []

    def AsyncClient(self, **kwargs):  # noqa: N802 - mirrors the httpx API
        client = _LoopBoundAsyncClient(self.requested_urls, **kwargs)
        self.clients.append(client)
        return client


@pytest.fixture
def httpx_stub(monkeypatch):
    stub = _HttpxStub()
    monkeypatch.setattr("agentfield.client.httpx", None)
    monkeypatch.setattr(
        "agentfield.client._ensure_httpx", lambda force_reload=False: stub
    )
    return stub


# ---------------------------------------------------------------------------
# C1 — a log emitted inside a running loop
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_log_from_async_context_is_delivered_on_the_running_loop(capsys):
    cp = _RecordingCpClient()
    logger = AgentFieldLogger("test.dispatch.async")
    logger._cp_client = cp

    logger.log_execution(
        "hello",
        event_type="log.info",
        execution_context=_context("exec-async"),
    )

    assert await _wait_for(lambda: len(cp.calls) == 1)
    execution_id, record = cp.calls[0]
    assert execution_id == "exec-async"
    assert record["message"] == "hello"
    assert cp.loops[0] is asyncio.get_running_loop()
    capsys.readouterr()


@pytest.mark.unit
async def test_pending_dispatch_survives_a_garbage_collection(capsys):
    """The in-flight dispatch must not be collected before it completes (#902)."""
    import gc

    cp = _RecordingCpClient()
    cp.gate = asyncio.Event()
    logger = AgentFieldLogger("test.dispatch.retention")
    logger._cp_client = cp

    logger.log_execution(
        "slow record",
        event_type="log.info",
        execution_context=_context("exec-retained"),
    )

    # Let the dispatch start and park on the gate, then collect aggressively.
    assert await _wait_for(lambda: len(cp.loops) == 1)
    gc.collect()
    await asyncio.sleep(0)

    cp.gate.set()
    assert await _wait_for(lambda: len(cp.calls) == 1)
    capsys.readouterr()


@pytest.mark.unit
async def test_failed_async_dispatch_is_not_reported_as_unretrieved(capsys):
    """A dispatch failure must not surface asyncio's unhandled-exception warning."""
    import gc

    unhandled = []
    asyncio.get_running_loop().set_exception_handler(
        lambda _loop, context: unhandled.append(context)
    )

    cp = _FailingCpClient()
    logger = AgentFieldLogger("test.dispatch.async.failure")
    logger._cp_client = cp

    logger.log_execution(
        "doomed",
        event_type="log.info",
        execution_context=_context("exec-doomed"),
    )

    assert await _wait_for(lambda: cp.attempts == 1)
    await asyncio.sleep(0.05)
    gc.collect()
    await asyncio.sleep(0.05)

    assert unhandled == []
    capsys.readouterr()


# ---------------------------------------------------------------------------
# C2/C3 — a log emitted with no running loop
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_log_from_sync_context_is_delivered_without_blocking(capsys):
    cp = _RecordingCpClient()
    logger = AgentFieldLogger("test.dispatch.sync")
    logger._cp_client = cp

    finished = _emit_from_a_thread_without_a_loop(logger, "exec-sync")

    assert finished.is_set(), "emitting a log must not block the calling thread"
    assert await _wait_for(lambda: len(cp.calls) == 1)
    assert cp.calls[0][0] == "exec-sync"
    capsys.readouterr()


@pytest.mark.unit
async def test_sync_context_logs_share_one_loop_that_stays_open(capsys):
    """No throwaway loop per record: every sync dispatch runs on one live loop."""
    cp = _RecordingCpClient()
    logger = AgentFieldLogger("test.dispatch.sync.loops")
    logger._cp_client = cp

    for index in range(3):
        _emit_from_a_thread_without_a_loop(logger, f"exec-sync-{index}")

    assert await _wait_for(lambda: len(cp.calls) == 3)
    assert len(set(id(loop) for loop in cp.loops)) == 1, (
        "each record got its own event loop"
    )
    assert not cp.loops[0].is_closed(), (
        "the dispatch loop was closed underneath the shared client"
    )
    capsys.readouterr()


@pytest.mark.unit
async def test_sync_context_log_leaves_the_agent_loop_able_to_use_the_client(
    httpx_stub, capsys
):
    """The scenario from #620: a log emitted before the agent's first request.

    The dispatch used to run on a throwaway loop that then closed, stranding
    the shared client's connection pool; the agent's next request on its own
    loop died with ``RuntimeError: Event loop is closed`` and the call was
    silently dropped.
    """
    cp_client = AgentFieldClient(base_url="http://control-plane.invalid")
    logger = AgentFieldLogger("test.dispatch.realclient")
    logger._cp_client = cp_client

    _emit_from_a_thread_without_a_loop(logger, "exec-sync-first")
    assert await _wait_for(lambda: len(httpx_stub.requested_urls) == 1)

    # Now the agent's own loop posts through the same client.
    await cp_client.post_execution_logs("exec-main", {"message": "from the agent"})

    assert httpx_stub.requested_urls == [
        "http://control-plane.invalid/api/v1/executions/exec-sync-first/logs",
        "http://control-plane.invalid/api/v1/executions/exec-main/logs",
    ]
    capsys.readouterr()


@pytest.mark.unit
async def test_sync_context_log_after_the_agent_loop_is_still_delivered(
    httpx_stub, capsys
):
    """The mirror scenario: the agent's loop claims the client first.

    The background dispatch used to die with ``... is bound to a different
    event loop`` and the record was swallowed by post_execution_logs().
    """
    cp_client = AgentFieldClient(base_url="http://control-plane.invalid")
    logger = AgentFieldLogger("test.dispatch.realclient.reverse")
    logger._cp_client = cp_client

    await cp_client.post_execution_logs("exec-main", {"message": "from the agent"})
    assert len(httpx_stub.requested_urls) == 1

    _emit_from_a_thread_without_a_loop(logger, "exec-sync-second")

    assert await _wait_for(lambda: len(httpx_stub.requested_urls) == 2)
    assert httpx_stub.requested_urls[1] == (
        "http://control-plane.invalid/api/v1/executions/exec-sync-second/logs"
    )
    capsys.readouterr()


# ---------------------------------------------------------------------------
# C4/C5 — failure isolation and no-op cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_dispatch_failure_never_reaches_the_caller(capsys):
    class _ExplodingCpClient:
        def post_execution_logs(self, execution_id, entries):
            raise RuntimeError("client is broken")

    logger = AgentFieldLogger("test.dispatch.explode")
    logger._cp_client = _ExplodingCpClient()

    record = logger.log_execution(
        "still returns",
        event_type="log.info",
        execution_context=_context("exec-explode"),
    )

    assert record["message"] == "still returns"
    capsys.readouterr()


@pytest.mark.unit
async def test_no_dispatch_without_a_client_or_an_execution_id(capsys):
    cp = _RecordingCpClient()

    detached = AgentFieldLogger("test.dispatch.noclient")
    detached._cp_client = None
    detached.log_execution(
        "nowhere to go",
        event_type="log.info",
        execution_context=_context("exec-none"),
    )

    unscoped = AgentFieldLogger("test.dispatch.nocontext")
    unscoped._cp_client = cp
    unscoped.log_execution("no execution", event_type="log.info")

    await asyncio.sleep(0.1)
    assert cp.calls == []
    capsys.readouterr()


# ---------------------------------------------------------------------------
# C6 — the shared HTTP client is never handed to a second loop
# ---------------------------------------------------------------------------


@pytest.mark.unit
async def test_async_http_client_is_never_shared_across_loops(httpx_stub):
    cp_client = AgentFieldClient(base_url="http://control-plane.invalid")

    mine = await cp_client.get_async_http_client()
    assert await cp_client.get_async_http_client() is mine

    from_other_loop = {}

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            from_other_loop["client"] = loop.run_until_complete(
                cp_client.get_async_http_client()
            )
        finally:
            loop.close()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=5)

    assert from_other_loop["client"] is not mine
    # The original loop keeps its own client.
    assert await cp_client.get_async_http_client() is mine


@pytest.mark.unit
async def test_async_http_client_slot_is_reclaimed_when_its_loop_dies(httpx_stub):
    """A client whose creating loop has closed is never handed out again."""
    cp_client = AgentFieldClient(base_url="http://control-plane.invalid")

    orphaned = {}

    def _worker() -> None:
        loop = asyncio.new_event_loop()
        try:
            orphaned["client"] = loop.run_until_complete(
                cp_client.get_async_http_client()
            )
        finally:
            loop.close()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=5)

    fresh = await cp_client.get_async_http_client()
    assert fresh is not orphaned["client"]
