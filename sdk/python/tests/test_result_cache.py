import time
from agentfield.result_cache import ResultCache
from agentfield.async_config import AsyncConfig
import asyncio


def test_cache_set_get_and_metrics():
    cfg = AsyncConfig(enable_result_caching=True, result_cache_ttl=5.0)
    cache = ResultCache(cfg)
    assert cache.get("k1") is None
    cache.set("k1", 123)
    assert cache.get("k1") == 123
    stats = cache.get_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1


def test_cache_ttl_expiration():
    cfg = AsyncConfig(enable_result_caching=True, result_cache_ttl=0.1)
    cache = ResultCache(cfg)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    time.sleep(0.15)
    assert cache.get("k") is None
    # expirations metric should increase
    stats = cache.get_stats()
    assert stats["expirations"] >= 1


def test_cache_lru_eviction():
    cfg = AsyncConfig(
        enable_result_caching=True, result_cache_ttl=5.0, result_cache_max_size=2
    )
    cache = ResultCache(cfg)
    cache.set("a", 1)
    cache.set("b", 2)
    # Access 'a' so 'b' becomes LRU
    assert cache.get("a") == 1
    cache.set("c", 3)  # should evict LRU ('b')
    assert cache.get("a") == 1
    assert cache.get("b") is None
    assert cache.get("c") == 3


def test_cache_async_context_and_cleanup_loop():
    # Use very small TTL and cleanup interval
    cfg = AsyncConfig(
        enable_result_caching=True, result_cache_ttl=0.05, cleanup_interval=0.02
    )
    cache = ResultCache(cfg)

    async def run():
        async with cache:
            cache.set("z", 9)
            assert cache.get("z") == 9
            await asyncio.sleep(0.1)  # allow ttl to expire and cleanup to run
            # Accessing should return None after cleanup removes it
            return cache.get("z")

    out = asyncio.run(run())
    assert out is None
