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
        yield SimpleNamespace(usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12})
    tracker = CostTracker()
    stream = UsageTrackingStream(source(), tracker, "openrouter/deepseek/deepseek-v4", "openrouter")
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
    stream = UsageTrackingStream(source(), tracker, "openrouter/qwen/qwen", "openrouter")
    await stream.__anext__()
    await stream.aclose()
    assert tracker.serialize()["entries"][0]["usage_status"] == "missing"
