from __future__ import annotations

import random

import pytest

from haxen_sdk.rate_limiter import RateLimitError, StatelessRateLimiter


class DummyHTTPError(Exception):
    status_code = 429

    def __init__(self, message: str = "rate limited", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.response = type(
            "Resp",
            (),
            {"status_code": self.status_code, "headers": {"Retry-After": retry_after}},
        )()


@pytest.mark.unit
def test_is_rate_limit_error_detects_status_code():
    limiter = StatelessRateLimiter()

    assert limiter._is_rate_limit_error(DummyHTTPError())
    assert limiter._is_rate_limit_error(type("LiteLLMRateLimitError", (), {})())
    assert not limiter._is_rate_limit_error(RuntimeError("other error"))


@pytest.mark.unit
def test_calculate_backoff_prefers_retry_after(monkeypatch):
    limiter = StatelessRateLimiter(base_delay=1.0, jitter_factor=0.0, max_delay=10.0)

    assert limiter._calculate_backoff_delay(0, retry_after=2.5) == pytest.approx(2.5)

    # When retry_after is missing we fall back to exponential backoff.
    delay = limiter._calculate_backoff_delay(1, retry_after=None)
    assert delay == pytest.approx(2.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_with_retry_eventual_success(monkeypatch):
    limiter = StatelessRateLimiter(max_retries=3, base_delay=0.01, jitter_factor=0.0)
    attempts = {"count": 0, "sleeps": []}

    async def fake_sleep(delay):
        attempts["sleeps"].append(delay)

    monkeypatch.setattr("haxen_sdk.rate_limiter.asyncio.sleep", fake_sleep)

    async def flaky_call():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise DummyHTTPError(retry_after=0.2)
        return "ok"

    result = await limiter.execute_with_retry(flaky_call)

    assert result == "ok"
    assert attempts["count"] == 3
    assert len(attempts["sleeps"]) == 2
    assert attempts["sleeps"][0] == pytest.approx(0.2)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_with_retry_gives_up(monkeypatch):
    limiter = StatelessRateLimiter(max_retries=1, base_delay=0.01, jitter_factor=0.0)

    async def fake_sleep(delay):
        pass

    monkeypatch.setattr("haxen_sdk.rate_limiter.asyncio.sleep", fake_sleep)

    async def always_fail():
        raise DummyHTTPError()

    with pytest.raises(RateLimitError):
        await limiter.execute_with_retry(always_fail)


@pytest.mark.unit
def test_calculate_backoff_applies_jitter_and_max_cap():
    limiter = StatelessRateLimiter(base_delay=0.5, jitter_factor=0.3, max_delay=1.0)
    limiter._container_seed = 42

    attempt = 4
    expected_base = limiter.max_delay
    rng = random.Random(limiter._container_seed + attempt)
    jitter_range = expected_base * limiter.jitter_factor
    expected_delay = max(0.1, expected_base + rng.uniform(-jitter_range, jitter_range))

    delay = limiter._calculate_backoff_delay(attempt)
    assert delay == pytest.approx(expected_delay)


@pytest.mark.unit
def test_extract_retry_after_uses_attribute_fallback():
    limiter = StatelessRateLimiter()

    class Response:
        headers = {"Retry-After": "invalid"}

    class Error:
        response = Response()
        retry_after = "7"

        def __str__(self):
            return "rate limit"

    assert limiter._extract_retry_after(Error()) == pytest.approx(7.0)

    class ErrorWithoutRetry:
        response = Response()
        retry_after = object()

        def __str__(self):
            return "rate limit"

    assert limiter._extract_retry_after(ErrorWithoutRetry()) is None


@pytest.mark.unit
def test_is_rate_limit_error_detects_message_and_status():
    limiter = StatelessRateLimiter()

    class MsgError(Exception):
        status_code = 503

        def __str__(self):
            return "temporarily rate-limited by server"

    assert limiter._is_rate_limit_error(MsgError())

    class PlainError(Exception):
        def __str__(self):
            return "all good"

    assert limiter._is_rate_limit_error(PlainError()) is False


@pytest.mark.unit
def test_update_circuit_breaker_resets_on_success():
    limiter = StatelessRateLimiter()
    limiter._circuit_open_time = 1.0
    limiter._consecutive_failures = 5

    limiter._update_circuit_breaker(success=True)

    assert limiter._circuit_open_time is None
    assert limiter._consecutive_failures == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_circuit_breaker_blocks_and_recovers(monkeypatch):
    limiter = StatelessRateLimiter(
        max_retries=0,
        base_delay=0.01,
        jitter_factor=0.0,
        circuit_breaker_threshold=1,
        circuit_breaker_timeout=5,
    )

    async def fake_sleep(delay):
        return None

    monkeypatch.setattr("haxen_sdk.rate_limiter.asyncio.sleep", fake_sleep)

    class Clock:
        def __init__(self, value: float):
            self.value = value

        def time(self) -> float:
            return self.value

        def advance(self, seconds: float) -> None:
            self.value += seconds

    clock = Clock(100.0)
    monkeypatch.setattr("haxen_sdk.rate_limiter.time.time", clock.time)

    async def always_limit():
        raise DummyHTTPError()

    with pytest.raises(RateLimitError) as first_error:
        await limiter.execute_with_retry(always_limit)

    assert "Last error" in str(first_error.value)
    assert limiter._circuit_open_time == pytest.approx(clock.time())

    was_called = False

    async def should_not_run():
        nonlocal was_called
        was_called = True
        return "ok"

    with pytest.raises(RateLimitError) as open_error:
        await limiter.execute_with_retry(should_not_run)

    assert was_called is False
    assert "Circuit breaker is open" in str(open_error.value)

    clock.advance(10.0)

    async def succeed():
        return "ok"

    result = await limiter.execute_with_retry(succeed)

    assert result == "ok"
    assert limiter._consecutive_failures == 0
    assert limiter._circuit_open_time is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_rate_limit_error_is_reraised():
    limiter = StatelessRateLimiter()

    async def blow_up():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        await limiter.execute_with_retry(blow_up)
