from __future__ import annotations

import asyncio

import pytest

from agentfield.async_config import AsyncConfig
from agentfield.execution_state import ExecutionState, ExecutionStatus
from agentfield.result_cache import CacheEntry, CacheMetrics, ResultCache


def test_cache_entry_and_metrics_properties(monkeypatch):
    times = iter([110.0, 115.0, 120.0, 130.0, 140.0])
    monkeypatch.setattr("agentfield.result_cache.time.time", lambda: next(times))

    entry = CacheEntry(value="v", created_at=100.0, accessed_at=105.0, ttl=8.0)
    metrics = CacheMetrics(hits=3, misses=1, created_at=90.0)

    assert entry.age == 10.0
    assert entry.time_since_access == 10.0
    assert entry.is_expired is True
    entry.touch()
    assert entry.access_count == 1
    assert metrics.hit_rate == 75.0
    assert metrics.uptime == 50.0


def test_cache_contains_delete_clear_repr_and_execution_helpers(monkeypatch):
    cfg = AsyncConfig(enable_result_caching=True, result_cache_ttl=30.0, result_cache_max_size=3)
    cache = ResultCache(cfg)

    cache.set_execution_result("exec-1", {"ok": True})
    assert cache.get_execution_result("exec-1") == {"ok": True}
    assert "exec:exec-1" in cache

    cache.set("alpha", 1)
    cache.set("beta", 2)
    assert cache.delete("alpha") is True
    assert cache.delete("missing") is False
    assert cache.get_keys(pattern="exec:") == ["exec:exec-1"]
    assert "size=2/3" in repr(cache)

    cache.clear()
    assert len(cache) == 0
    assert cache.metrics.size == 0


def test_cache_execution_state_and_stats(monkeypatch):
    cfg = AsyncConfig(enable_result_caching=True, result_cache_ttl=30.0)
    cache = ResultCache(cfg)

    successful = ExecutionState(
        execution_id="exec-1",
        target="node.reasoner",
        input_data={},
        status=ExecutionStatus.SUCCEEDED,
        result={"done": True},
    )
    unsuccessful = ExecutionState(
        execution_id="exec-2",
        target="node.reasoner",
        input_data={},
        status=ExecutionStatus.FAILED,
        result=None,
    )

    cache.cache_execution_state(successful)
    cache.cache_execution_state(unsuccessful)
    cache.get("exec:exec-1")

    stats = cache.get_stats()

    assert cache.get_execution_result("exec-1") == {"done": True}
    assert cache.get_execution_result("exec-2") is None
    assert stats["size"] == 1
    assert stats["average_access_count"] >= 1.0
    assert stats["average_age"] >= 0.0
    assert stats["enabled"] is True


def test_cleanup_expired_and_disabled_cache_behaviour(monkeypatch):
    cfg = AsyncConfig(enable_result_caching=False, result_cache_ttl=1.0)
    cache = ResultCache(cfg)

    assert cache.get("missing") is None
    assert cache.metrics.misses == 1

    enabled = ResultCache(AsyncConfig(enable_result_caching=True, result_cache_ttl=1.0))
    enabled._cache["old"] = CacheEntry(value=1, created_at=0.0, accessed_at=0.0, ttl=0.5)
    enabled._cache["new"] = CacheEntry(value=2, created_at=10.0, accessed_at=10.0, ttl=5.0)

    monkeypatch.setattr("agentfield.result_cache.time.time", lambda: 2.0)
    removed = enabled._cleanup_expired()

    assert removed == 1
    assert list(enabled._cache) == ["new"]
    assert enabled.metrics.expirations == 1


@pytest.mark.asyncio
async def test_start_stop_and_cleanup_loop_error_paths(monkeypatch):
    cfg = AsyncConfig(
        enable_result_caching=True,
        result_cache_ttl=0.01,
        cleanup_interval=0.01,
        enable_performance_logging=True,
    )
    cache = ResultCache(cfg)

    sleep_calls = {"count": 0}
    debug_messages = []
    error_messages = []

    async def fake_sleep(delay):
        sleep_calls["count"] += 1
        if sleep_calls["count"] == 1:
            return None
        raise asyncio.CancelledError()

    def fake_cleanup():
        if sleep_calls["count"] == 1:
            raise RuntimeError("cleanup broke")
        return 0

    monkeypatch.setattr("agentfield.result_cache.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(cache, "_cleanup_expired", fake_cleanup)
    monkeypatch.setattr("agentfield.result_cache.logger.debug", lambda msg: debug_messages.append(msg))
    monkeypatch.setattr("agentfield.result_cache.logger.error", lambda msg: error_messages.append(msg))

    await cache.start()
    await cache._cleanup_task
    await cache.stop()

    assert cache._shutdown_event is None
    assert cache.metrics.size == 0
    assert any("Cache cleanup error" in msg for msg in error_messages)


@pytest.mark.asyncio
async def test_start_with_caching_disabled_and_context_manager():
    cache = ResultCache(AsyncConfig(enable_result_caching=False))

    async with cache as active_cache:
        assert active_cache is cache
        assert cache._cleanup_task is None

    assert len(cache) == 0
