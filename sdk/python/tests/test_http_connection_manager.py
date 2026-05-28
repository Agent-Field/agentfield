"""
Tests for http_connection_manager.py async connection pooling.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from agentfield.http_connection_manager import (
    ConnectionManager,
    ConnectionMetrics,
    ConnectionHealth,
)
from agentfield.exceptions import AgentFieldClientError


def test_connection_metrics():
    """Test ConnectionMetrics recording and calculations."""
    metrics = ConnectionMetrics()

    assert metrics.total_requests == 0
    assert metrics.success_rate == 0.0

    metrics.record_request(success=True)
    assert metrics.total_requests == 1
    assert metrics.successful_requests == 1
    assert metrics.failed_requests == 0
    assert metrics.success_rate == 100.0

    metrics.record_request(success=False)
    assert metrics.total_requests == 2
    assert metrics.successful_requests == 1
    assert metrics.failed_requests == 1
    assert metrics.success_rate == 50.0

    metrics.record_request(success=False, timeout=True)
    assert metrics.timeout_requests == 1


def test_connection_health():
    """Test ConnectionHealth status tracking."""
    health = ConnectionHealth()

    assert health.is_healthy is True
    assert health.consecutive_failures == 0

    health.mark_unhealthy("Connection failed")
    assert health.is_healthy is False
    assert health.consecutive_failures == 1
    assert health.last_error == "Connection failed"

    health.mark_healthy()
    assert health.is_healthy is True
    assert health.consecutive_failures == 0
    assert health.last_error is None


@pytest.mark.asyncio
async def test_connection_manager_init():
    """Test ConnectionManager initialization."""
    manager = ConnectionManager()
    assert manager.config is not None
    assert manager._session is None
    assert manager._closed is False
    assert manager.metrics is not None
    assert manager.health is not None


@pytest.mark.asyncio
async def test_connection_manager_start_close():
    """Test ConnectionManager start and close lifecycle."""
    manager = ConnectionManager()

    await manager.start()
    assert manager._session is not None
    assert not manager._closed

    await manager.close()
    assert manager._closed
    assert manager._session is None


@pytest.mark.asyncio
async def test_connection_manager_context_manager():
    """Test ConnectionManager as async context manager."""
    async with ConnectionManager() as manager:
        assert manager._session is not None
        assert not manager._closed

    assert manager._closed


@pytest.mark.asyncio
async def test_connection_manager_double_start():
    """Test that starting an already started manager raises error."""
    manager = ConnectionManager()
    await manager.start()

    with pytest.raises(AgentFieldClientError, match="already started"):
        await manager.start()

    await manager.close()


@pytest.mark.asyncio
async def test_connection_manager_start_after_close():
    """Test that starting after close raises error."""
    manager = ConnectionManager()
    await manager.start()
    await manager.close()

    with pytest.raises(AgentFieldClientError, match="closed"):
        await manager.start()


@pytest.mark.asyncio
async def test_connection_manager_get_session():
    """Test getting session from manager."""
    manager = ConnectionManager()
    await manager.start()

    async with manager.get_session() as session:
        assert session is not None
        assert session == manager._session

    await manager.close()


@pytest.mark.asyncio
async def test_connection_manager_get_session_not_started():
    """Test getting session before start raises error."""
    manager = ConnectionManager()

    with pytest.raises(AgentFieldClientError, match="not started"):
        async with manager.get_session():
            pass


# Note: request() method is tested indirectly through batch_request and other integration tests
# Direct unit testing of aiohttp session.request is complex due to async context managers


@pytest.mark.asyncio
async def test_connection_manager_request_timeout():
    """Test request timeout handling."""
    manager = ConnectionManager()
    await manager.start()

    with patch.object(
        manager._session, "request", new_callable=AsyncMock
    ) as mock_request:
        mock_request.side_effect = asyncio.TimeoutError()

        with pytest.raises(asyncio.TimeoutError):
            await manager.request("GET", "http://example.com")

        assert manager.metrics.timeout_requests == 1
        assert manager.health.is_healthy is False

    await manager.close()


@pytest.mark.asyncio
async def test_connection_manager_batch_request():
    """Test batch request execution."""
    manager = ConnectionManager()
    await manager.start()

    mock_response = AsyncMock()
    mock_response.status = 200

    with patch.object(manager, "request", new_callable=AsyncMock) as mock_request:
        mock_request.return_value = mock_response

        requests = [
            {"method": "GET", "url": "http://example.com/1"},
            {"method": "GET", "url": "http://example.com/2"},
            {"method": "GET", "url": "http://example.com/3"},
        ]

        results = await manager.batch_request(requests)

        assert len(results) == 3
        assert mock_request.call_count == 3

    await manager.close()


@pytest.mark.asyncio
async def test_connection_manager_health_check():
    """Test health check functionality."""
    manager = ConnectionManager()
    await manager.start()

    is_healthy = await manager.health_check()
    assert is_healthy is True

    await manager.close()

    is_healthy = await manager.health_check()
    assert is_healthy is False


@pytest.mark.asyncio
async def test_connection_manager_properties():
    """Test ConnectionManager properties."""
    manager = ConnectionManager()

    # Initially not closed (but not started either)
    assert manager.is_closed is False
    # Health is True by default
    assert manager.is_healthy is True

    await manager.start()
    # After start, still not closed
    assert manager.is_closed is False
    # Health should still be True after start
    assert manager.is_healthy is True

    await manager.close()
    # After close, should be closed
    assert manager.is_closed is True
    # After close, health should be False
    assert manager.is_healthy is False


def test_connection_manager_repr():
    """Test ConnectionManager string representation."""
    manager = ConnectionManager()
    repr_str = repr(manager)

    assert "ConnectionManager" in repr_str
    assert "pool_size" in repr_str
