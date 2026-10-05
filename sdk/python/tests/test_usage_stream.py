from types import SimpleNamespace
import pytest
from agentfield.cost_tracker import CostTracker
from agentfield.usage_stream import UsageTrackingStream


@pytest.mark.asyncio
async def test_stream_receipt_not_consumed_eagerly_and_recorded_once():
    consumed = []

    async def source():
        consumed.append(1)
        yield SimpleNamespace(usage=None, content="hello")
        yield SimpleNamespace(
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        )

    tracker = CostTracker()
    stream = UsageTrackingStream(
        source(), tracker, "openrouter/deepseek/deepseek-v4", "openrouter"
    )
    assert consumed == []
    chunks = [c async for c in stream]
    await stream.aclose()
    assert chunks[0].content == "hello"
    entries = tracker.serialize()["entries"]
    assert len(entries) == 1
    assert entries[0]["total_tokens"] == 12
    assert entries[0]["routing_provider"] == "openrouter"


@pytest.mark.asyncio
async def test_abandoned_stream_marks_unknown_not_zero_usage():
    async def source():
        yield SimpleNamespace(usage=None)
        raise AssertionError("must not consume abandoned stream")

    tracker = CostTracker()
    stream = UsageTrackingStream(
        source(), tracker, "openrouter/qwen/qwen", "openrouter"
    )
    await stream.__anext__()
    await stream.aclose()
    assert tracker.serialize()["entries"][0]["usage_status"] == "missing"


@pytest.mark.asyncio
async def test_latest_usage_receipt_wins_over_intermediate_zero():
    async def source():
        yield SimpleNamespace(
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        yield SimpleNamespace(
            usage={"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}
        )

    tracker = CostTracker()
    stream = UsageTrackingStream(
        source(), tracker, "openrouter/qwen/qwen3", "openrouter"
    )
    async for _ in stream:
        pass
    assert len(tracker.serialize()["entries"]) == 1
    assert tracker.serialize()["total_tokens"] == 100


@pytest.mark.asyncio
async def test_early_close_records_last_observed_receipt_once():
    async def source():
        yield SimpleNamespace(
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        )
        raise AssertionError("abandoned stream was drained")

    tracker = CostTracker()
    stream = UsageTrackingStream(
        source(), tracker, "openrouter/qwen/qwen3", "openrouter"
    )
    await stream.__anext__()
    await stream.aclose()
    await stream.aclose()
    assert tracker.serialize()["total_tokens"] == 12
    assert len(tracker.serialize()["entries"]) == 1


@pytest.mark.asyncio
async def test_async_context_manager_protocol_is_preserved():
    class Source:
        def __init__(self):
            self.entered = False
            self.exited = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def __aenter__(self):
            self.entered = True
            return self

        async def __aexit__(self, exc_type, exc, tb):
            self.exited = True
            return False

    source = Source()
    tracker = CostTracker()
    stream = UsageTrackingStream(source, tracker, "openrouter/qwen/qwen3", "openrouter")
    async with stream as entered:
        assert entered is stream
        assert source.entered
    assert source.exited
    assert tracker.serialize()["entries"][0]["usage_status"] == "missing"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_tokens", ["broken", 10**400, -1, float("nan")])
async def test_malformed_usage_marks_missing_without_corrupting_output(bad_tokens):
    async def source():
        yield SimpleNamespace(content="hello", usage={"prompt_tokens": bad_tokens})

    tracker = CostTracker()
    stream = UsageTrackingStream(
        source(), tracker, "openrouter/qwen/qwen3", "openrouter"
    )
    assert [chunk.content async for chunk in stream] == ["hello"]
    assert tracker.serialize()["entries"][0]["usage_status"] == "missing"


@pytest.mark.asyncio
async def test_tracker_failure_does_not_change_successful_stream():
    class BrokenTracker:
        def record(self, **kwargs):
            raise RuntimeError("optional accounting failed")

    async def source():
        yield SimpleNamespace(
            content="hello", usage={"prompt_tokens": 1, "completion_tokens": 1}
        )

    stream = UsageTrackingStream(
        source(), BrokenTracker(), "openrouter/qwen/qwen3", "openrouter"
    )
    assert [chunk.content async for chunk in stream] == ["hello"]


@pytest.mark.asyncio
async def test_stream_keeps_adapter_provider_separate_from_openrouter_route():
    async def source():
        yield SimpleNamespace(usage={"prompt_tokens": 3, "completion_tokens": 2})

    tracker = CostTracker()
    stream = UsageTrackingStream(source(), tracker, "gpt-4o", "openai", "openrouter")
    async for _ in stream:
        pass
    entry = tracker.serialize()["entries"][0]
    assert entry["provider"] == "openai"
    assert entry["routing_provider"] == "openrouter"
