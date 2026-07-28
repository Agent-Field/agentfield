"""Tests for token/cost usage capture, serialization, and transport.

Covers:
- usage recorded even when cost is None (tokens never gated on pricing);
- Anthropic-native and OpenAI/LiteLLM usage-shape extraction, plus cache tokens;
- provider-native (OpenRouter) and litellm cost resolution + cost_source;
- CostTracker.serialize() contract shape;
- harness token capture from a fake Claude Code result message;
- per-execution tracker isolation (concurrency) and contextvar reset;
- the execution envelope carries the correct top-level "usage" object.
"""

from __future__ import annotations

import asyncio
import types
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest  # pyright: ignore[reportMissingImports]

from agentfield.cost_tracker import (
    USAGE_ENVELOPE_KEY,
    CostTracker,
    derive_provider,
    get_current_cost_tracker,
    reset_current_cost_tracker,
    set_current_cost_tracker,
)
from agentfield.multimodal_response import (
    _extract_usage,
    _resolve_cost,
    detect_multimodal_response,
)


# ---------------------------------------------------------------------------
# CostTracker: recording without pricing + serialize() contract
# ---------------------------------------------------------------------------


class TestCostTrackerUsage:
    def test_records_tokens_when_cost_is_none(self):
        tracker = CostTracker()
        tracker.record(
            model="anthropic/claude-opus-4-8",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=None,
        )
        assert tracker.call_count == 1
        assert tracker.total_tokens == 150
        # Unknown cost must not blow up totals.
        assert tracker.total_cost_usd == 0.0

    def test_derive_provider_from_slug(self):
        assert derive_provider("anthropic/claude-opus-4-8") == "anthropic"
        assert derive_provider("openrouter/anthropic/claude") == "openrouter"
        assert derive_provider("openai/gpt-4o") == "openai"
        assert derive_provider("gpt-4o") is None
        assert derive_provider(None) is None

    def test_record_auto_derives_provider(self):
        tracker = CostTracker()
        tracker.record(model="anthropic/claude-opus-4-8", prompt_tokens=1)
        entry = tracker.serialize()["entries"][0]
        assert entry["provider"] == "anthropic"

    def test_serialize_contract_shape(self):
        tracker = CostTracker()
        tracker.record(
            model="claude-opus-4-8",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
            reasoner_name="my_reasoner",
            provider="anthropic",
            cache_read_tokens=10,
            cache_creation_tokens=5,
            cost_source="litellm",
        )
        usage = tracker.serialize()
        assert usage["total_cost_usd"] == 0.01
        assert usage["total_input_tokens"] == 100
        assert usage["total_output_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert len(usage["entries"]) == 1
        entry = usage["entries"][0]
        assert entry == {
            "source": "llm",
            "provider": "anthropic",
            "model": "claude-opus-4-8",
            "harness": None,
            "reasoner": "my_reasoner",
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 10,
            "cache_creation_tokens": 5,
            "total_tokens": 150,
            "cost_usd": 0.01,
            "cost_source": "litellm",
        }

    def test_serialize_total_cost_null_when_all_unknown(self):
        tracker = CostTracker()
        tracker.record(model="m", prompt_tokens=10, completion_tokens=5, cost_usd=None)
        usage = tracker.serialize()
        # Contract: number|null — null when no entry had a known cost.
        assert usage["total_cost_usd"] is None
        assert usage["total_input_tokens"] == 10
        assert usage["total_output_tokens"] == 5

    def test_serialize_partial_costs_sum_known_only(self):
        tracker = CostTracker()
        tracker.record(model="m", prompt_tokens=10, cost_usd=None)
        tracker.record(model="m", prompt_tokens=20, cost_usd=0.02)
        usage = tracker.serialize()
        assert usage["total_cost_usd"] == 0.02

    def test_summary_backward_compatible(self):
        tracker = CostTracker()
        tracker.record(model="gpt-4", total_tokens=100, cost_usd=0.003)
        summary = tracker.summary()
        # Existing keys unchanged.
        assert set(summary.keys()) == {
            "total_cost_usd",
            "total_tokens",
            "total_calls",
            "by_model",
        }
        assert summary["by_model"]["gpt-4"]["calls"] == 1

    def test_harness_entry_serialization(self):
        tracker = CostTracker()
        tracker.record(
            model="claude-opus-4-8",
            prompt_tokens=200,
            completion_tokens=80,
            source="harness",
            harness="claude_code",
            provider="anthropic",
            cost_usd=0.05,
            cost_source="provider",
        )
        entry = tracker.serialize()["entries"][0]
        assert entry["source"] == "harness"
        assert entry["harness"] == "claude_code"
        assert entry["cost_source"] == "provider"


# ---------------------------------------------------------------------------
# multimodal_response: usage-shape robustness + cost resolution
# ---------------------------------------------------------------------------


class TestUsageExtraction:
    def test_openai_shape(self):
        u = types.SimpleNamespace(
            prompt_tokens=100, completion_tokens=50, total_tokens=150
        )
        usage = _extract_usage(u)
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert "cache_read_tokens" not in usage

    def test_anthropic_native_shape(self):
        u = types.SimpleNamespace(input_tokens=200, output_tokens=90)
        usage = _extract_usage(u)
        assert usage["prompt_tokens"] == 200
        assert usage["completion_tokens"] == 90
        assert usage["total_tokens"] == 290

    def test_anthropic_cache_tokens(self):
        u = types.SimpleNamespace(
            input_tokens=200,
            output_tokens=90,
            cache_read_input_tokens=30,
            cache_creation_input_tokens=12,
        )
        usage = _extract_usage(u)
        assert usage["cache_read_tokens"] == 30
        assert usage["cache_creation_tokens"] == 12

    def test_litellm_nested_cached_tokens(self):
        u = types.SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=types.SimpleNamespace(cached_tokens=25),
        )
        usage = _extract_usage(u)
        assert usage["cache_read_tokens"] == 25

    def test_dict_usage_shape(self):
        usage = _extract_usage(
            {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12}
        )
        assert usage["prompt_tokens"] == 5
        assert usage["completion_tokens"] == 7


class TestCostResolution:
    def _resp(self, model="gpt-4o", usage=None, hidden=None):
        resp = types.SimpleNamespace(model=model)
        if hidden is not None:
            resp._hidden_params = hidden
        return resp, usage

    def test_openrouter_native_cost_wins(self):
        usage = types.SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, cost=0.0042
        )
        resp = types.SimpleNamespace(model="openrouter/openai/gpt-4o")
        cost, source = _resolve_cost(resp, usage)
        assert cost == 0.0042
        assert source == "provider"

    def test_hidden_params_response_cost(self):
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        resp = types.SimpleNamespace(
            model="gpt-4o", _hidden_params={"response_cost": 0.009}
        )
        cost, source = _resolve_cost(resp, usage)
        assert cost == 0.009
        assert source == "litellm"

    def test_litellm_completion_cost_fallback(self):
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        resp = types.SimpleNamespace(model="gpt-4o")
        mock_litellm = MagicMock()
        mock_litellm.completion_cost.return_value = 0.0035
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            cost, source = _resolve_cost(resp, usage)
        assert cost == 0.0035
        assert source == "litellm"

    def test_no_cost_returns_none(self):
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        resp = types.SimpleNamespace(model="gpt-4o")
        mock_litellm = MagicMock()
        mock_litellm.completion_cost.side_effect = Exception("unknown model")
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            cost, source = _resolve_cost(resp, usage)
        assert cost is None
        assert source is None


def _fake_response(usage, model="gpt-4o"):
    message = types.SimpleNamespace(content="hi", audio=None, images=None)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice], model=model, usage=usage, data=None)


class TestDetectMultimodalUsage:
    def test_tokens_preserved_when_cost_fails(self):
        usage = types.SimpleNamespace(input_tokens=200, output_tokens=90)
        resp = _fake_response(usage, model="anthropic/claude-opus-4-8")
        mock_litellm = MagicMock()
        mock_litellm.completion_cost.side_effect = Exception("no price")
        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            result = detect_multimodal_response(resp)
        # Cost unknown, but tokens survived.
        assert result.cost_usd is None
        assert result.cost_source is None
        assert result.usage["prompt_tokens"] == 200
        assert result.usage["completion_tokens"] == 90

    def test_openrouter_native_cost_read_through(self):
        usage = types.SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, cost=0.0012
        )
        resp = _fake_response(usage, model="openrouter/openai/gpt-4o")
        result = detect_multimodal_response(resp)
        assert result.cost_usd == 0.0012
        assert result.cost_source == "provider"


# ---------------------------------------------------------------------------
# OpenRouter usage-accounting injection
# ---------------------------------------------------------------------------


class TestOpenRouterUsageAccounting:
    def test_injects_usage_include_for_openrouter(self):
        from agentfield.openrouter_attribution import (
            apply_openrouter_usage_accounting,
        )

        params: dict[str, Any] = {"model": "openrouter/openai/gpt-4o"}
        apply_openrouter_usage_accounting(params)
        assert params["extra_body"]["usage"] == {"include": True}

    def test_noop_for_non_openrouter(self):
        from agentfield.openrouter_attribution import (
            apply_openrouter_usage_accounting,
        )

        params: dict[str, Any] = {"model": "openai/gpt-4o"}
        apply_openrouter_usage_accounting(params)
        assert "extra_body" not in params

    def test_get_litellm_params_wires_usage_accounting(self):
        from agentfield.types import AIConfig

        params = AIConfig(model="openrouter/openai/gpt-4o").get_litellm_params()
        assert params["extra_body"]["usage"] == {"include": True}


# ---------------------------------------------------------------------------
# Harness: Claude Code result-message token capture
# ---------------------------------------------------------------------------


class _AsyncStream:
    def __init__(self, items: list[Any]):
        self._items = items

    def __aiter__(self):
        async def _gen():
            for item in self._items:
                yield item

        return _gen()


@pytest.mark.asyncio
async def test_claude_provider_parses_usage_object(monkeypatch):
    from agentfield.harness.providers.claude import ClaudeCodeProvider

    class FakeClaudeAgentOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    def fake_query(*, prompt: str, options: Any):
        _ = (prompt, options)
        return _AsyncStream(
            [
                {
                    "type": "result",
                    "result": "done",
                    "session_id": "s",
                    "total_cost_usd": 0.02,
                    "num_turns": 2,
                    "model": "claude-opus-4-8",
                    "usage": {
                        "input_tokens": 1200,
                        "output_tokens": 340,
                        "cache_read_input_tokens": 800,
                        "cache_creation_input_tokens": 50,
                    },
                }
            ]
        )

    fake_sdk = ModuleType("claude_agent_sdk")
    setattr(fake_sdk, "ClaudeAgentOptions", FakeClaudeAgentOptions)
    setattr(fake_sdk, "query", fake_query)
    monkeypatch.setitem(__import__("sys").modules, "claude_agent_sdk", fake_sdk)

    raw = await ClaudeCodeProvider().execute("hi", {})
    assert raw.metrics.input_tokens == 1200
    assert raw.metrics.output_tokens == 340
    assert raw.metrics.cache_read_tokens == 800
    assert raw.metrics.cache_creation_tokens == 50
    assert raw.metrics.model == "claude-opus-4-8"


def test_accumulate_metrics_sums_tokens():
    from agentfield.harness._result import Metrics, RawResult
    from agentfield.harness._runner import _accumulate_metrics

    raws = [
        RawResult(metrics=Metrics(input_tokens=100, output_tokens=40, model="m")),
        RawResult(metrics=Metrics(input_tokens=50, output_tokens=20)),
    ]
    _cost, _turns, _sid, _msgs, tokens = _accumulate_metrics(raws)
    assert tokens["input_tokens"] == 150
    assert tokens["output_tokens"] == 60
    assert tokens["model"] == "m"


def test_extract_token_usage_codex_shape():
    from agentfield.harness._cli import extract_token_usage

    events = [
        {"type": "thread.started", "thread_id": "t"},
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 500,
                "output_tokens": 120,
                "cached_input_tokens": 200,
            },
        },
    ]
    tokens = extract_token_usage(events)
    assert tokens["input_tokens"] == 500
    assert tokens["output_tokens"] == 120
    assert tokens["cache_read_tokens"] == 200


# ---------------------------------------------------------------------------
# Per-execution tracker isolation (contextvar) + concurrency
# ---------------------------------------------------------------------------


class TestTrackerIsolation:
    def test_set_get_reset(self):
        assert get_current_cost_tracker() is None
        tracker = CostTracker()
        token = set_current_cost_tracker(tracker)
        try:
            assert get_current_cost_tracker() is tracker
        finally:
            reset_current_cost_tracker(token)
        assert get_current_cost_tracker() is None

    @pytest.mark.asyncio
    async def test_concurrent_executions_do_not_cross_contaminate(self):
        async def run(model: str, count: int) -> CostTracker:
            tracker = CostTracker()
            token = set_current_cost_tracker(tracker)
            try:
                for _ in range(count):
                    await asyncio.sleep(0)  # yield so tasks interleave
                    current = get_current_cost_tracker()
                    assert current is tracker
                    current.record(model=model, prompt_tokens=1, total_tokens=1)
            finally:
                reset_current_cost_tracker(token)
            return tracker

        t1, t2 = await asyncio.gather(run("a", 5), run("b", 3))
        assert t1.call_count == 5
        assert t2.call_count == 3
        # Each tracker only saw its own model.
        assert {e["model"] for e in t1.serialize()["entries"]} == {"a"}
        assert {e["model"] for e in t2.serialize()["entries"]} == {"b"}


# ---------------------------------------------------------------------------
# Envelope transport: sync body wrap + async status payload
# ---------------------------------------------------------------------------


class TestEnvelopeTransport:
    def _agent(self):
        from agentfield.agent import Agent

        return Agent(node_id="usage-test-agent")

    def test_sync_merges_usage_sibling_into_dict(self):
        agent = self._agent()
        tracker = CostTracker()
        tracker.record(
            model="anthropic/claude-opus-4-8",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_usd=0.01,
            cost_source="litellm",
        )
        body = agent._wrap_sync_result_with_usage({"answer": 42}, tracker)
        # Usage merged under the reserved namespaced envelope key as a
        # top-level sibling; result keys preserved.
        assert body["answer"] == 42
        assert body[USAGE_ENVELOPE_KEY]["total_input_tokens"] == 100
        assert body[USAGE_ENVELOPE_KEY]["entries"][0]["provider"] == "anthropic"

    def test_sync_preserves_user_usage_key(self):
        # A user result that legitimately returns its own "usage" key is
        # payload, not transport — it must never be overwritten or moved.
        agent = self._agent()
        tracker = CostTracker()
        tracker.record(model="m", prompt_tokens=1, total_tokens=1)
        body = agent._wrap_sync_result_with_usage(
            {"answer": 42, "usage": {"user": "data"}}, tracker
        )
        assert body["usage"] == {"user": "data"}
        assert body[USAGE_ENVELOPE_KEY]["entries"]

    def test_sync_non_dict_result_unchanged(self):
        agent = self._agent()
        tracker = CostTracker()
        tracker.record(model="m", prompt_tokens=1, total_tokens=1)
        # Non-dict results cannot carry a top-level usage key; returned as-is.
        assert agent._wrap_sync_result_with_usage([1, 2, 3], tracker) == [1, 2, 3]
        assert agent._wrap_sync_result_with_usage("text", tracker) == "text"

    def test_sync_no_wrap_when_empty(self):
        agent = self._agent()
        tracker = CostTracker()
        result = {"answer": 42}
        # No entries -> raw result returned unchanged (backward compatible).
        assert agent._wrap_sync_result_with_usage(result, tracker) is result

    def test_usage_summary_or_none(self):
        agent = self._agent()
        assert agent._usage_summary_or_none(None) is None
        empty = CostTracker()
        assert agent._usage_summary_or_none(empty) is None
        tracker = CostTracker()
        tracker.record(model="m", prompt_tokens=1, total_tokens=1)
        summary = agent._usage_summary_or_none(tracker)
        assert summary is not None
        assert summary["entries"]

    def test_record_harness_usage_into_current_tracker(self):
        from agentfield.harness._result import HarnessResult

        agent = self._agent()
        tracker = CostTracker()
        token = set_current_cost_tracker(tracker)
        try:
            result = HarnessResult(
                result="done",
                cost_usd=0.03,
                input_tokens=1000,
                output_tokens=200,
                cache_read_tokens=500,
                total_tokens=1200,
                model="claude-opus-4-8",
            )
            agent._record_harness_usage(result, provider="claude-code")
        finally:
            reset_current_cost_tracker(token)

        entries = tracker.serialize()["entries"]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["source"] == "harness"
        assert entry["harness"] == "claude_code"
        assert entry["input_tokens"] == 1000
        assert entry["output_tokens"] == 200
        assert entry["cache_read_tokens"] == 500
        assert entry["cost_usd"] == 0.03

    def test_record_harness_usage_noop_when_empty(self):
        from agentfield.harness._result import HarnessResult

        agent = self._agent()
        tracker = CostTracker()
        token = set_current_cost_tracker(tracker)
        try:
            agent._record_harness_usage(
                HarnessResult(result="done", cost_usd=None), provider="codex"
            )
        finally:
            reset_current_cost_tracker(token)
        assert tracker.call_count == 0


class TestEnvelopeEndToEnd:
    """Drive the FastAPI endpoint and assert usage lands in both transports."""

    @pytest.mark.asyncio
    async def test_async_callback_payload_has_usage(self, monkeypatch):
        import httpx

        from agentfield.agent import Agent
        from agentfield.client import AgentFieldClient

        agent = Agent(
            node_id="usage-e2e",
            agentfield_server="http://control",
            auto_register=False,
        )

        @agent.reasoner()
        async def spend(value: int) -> dict:
            # Simulate an LLM call recording usage into the current tracker.
            tracker = get_current_cost_tracker()
            assert tracker is not None
            tracker.record(
                model="anthropic/claude-opus-4-8",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                cost_usd=0.01,
                cost_source="litellm",
            )
            return {"value": value}

        recorded: list[dict] = []

        class DummyResponse:
            status_code = 200

            def json(self):
                return {}

        async def fake_request(self, method, url, **kwargs):
            recorded.append({"url": url, "json": kwargs.get("json")})
            return DummyResponse()

        monkeypatch.setattr(AgentFieldClient, "_async_request", fake_request)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=agent), base_url="http://agent"
        ) as client:
            response = await client.post(
                "/reasoners/spend",
                json={"value": 7},
                headers={"X-Execution-ID": "exec-abc"},
            )

        assert response.status_code == 202
        await asyncio.sleep(0.1)

        status_calls = [e for e in recorded if "/executions/" in e["url"]]
        assert status_calls
        payload = status_calls[-1]["json"]
        assert payload["status"] == "succeeded"
        assert payload["result"] == {"value": 7}
        usage = payload["usage"]
        assert usage["total_input_tokens"] == 100
        assert usage["total_output_tokens"] == 50
        assert usage["total_cost_usd"] == 0.01
        assert usage["entries"][0]["provider"] == "anthropic"
        assert usage["entries"][0]["cost_source"] == "litellm"

    @pytest.mark.asyncio
    async def test_sync_body_wrapped_with_usage(self):
        import httpx

        from agentfield.agent import Agent

        # No agentfield_server -> sync path (no async callback).
        agent = Agent(node_id="usage-e2e-sync", auto_register=False)

        @agent.reasoner()
        async def spend(value: int) -> dict:
            tracker = get_current_cost_tracker()
            assert tracker is not None
            tracker.record(
                model="openai/gpt-4o",
                prompt_tokens=10,
                completion_tokens=4,
                total_tokens=14,
                cost_usd=0.001,
            )
            return {"value": value}

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=agent), base_url="http://agent"
        ) as client:
            response = await client.post("/reasoners/spend", json={"value": 9})

        assert response.status_code == 200
        body = response.json()
        # Usage is merged under the reserved namespaced envelope key as a
        # top-level sibling of the result object (the control plane strips
        # exactly that key back out); the rest is the original result
        # unchanged — a user-owned "usage" key would be untouched.
        assert body["value"] == 9
        assert body[USAGE_ENVELOPE_KEY]["total_input_tokens"] == 10
        assert body[USAGE_ENVELOPE_KEY]["entries"][0]["provider"] == "openai"
        rest = {k: v for k, v in body.items() if k != USAGE_ENVELOPE_KEY}
        assert rest == {"value": 9}

    @pytest.mark.asyncio
    async def test_sync_body_unwrapped_when_no_usage(self):
        import httpx

        from agentfield.agent import Agent

        agent = Agent(node_id="usage-e2e-none", auto_register=False)

        @agent.reasoner()
        async def noop(value: int) -> dict:
            return {"value": value}

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=agent), base_url="http://agent"
        ) as client:
            response = await client.post("/reasoners/noop", json={"value": 3})

        assert response.status_code == 200
        # Backward compatible: no usage -> raw result body, no envelope.
        assert response.json() == {"value": 3}
