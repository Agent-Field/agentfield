"""Regression tests for the silent litellm deadlock fix.

These tests pin down the failure mode that motivated PR #384:

  1. ``litellm.acompletion`` hangs forever waiting on a half-closed httpx
     socket.
  2. ``asyncio.wait_for`` cancels the parent coroutine but the underlying
     httpx pool is left in a bad state.
  3. Every subsequent ``acompletion`` call grabs the same stale connection
     and hangs forever — silent deadlock; py-spy shows all asyncio worker
     threads idle and zero active Python frames anywhere.

The fix has three load-bearing pieces, each tested below:

  * ``litellm_params['timeout']`` is set so litellm/httpx aborts the socket
    itself when it does honor the parameter.
  * An ``asyncio.wait_for`` safety net at 2× the configured timeout fires
    even if litellm ignores its own timeout (it currently does).
  * On timeout, ``_reset_litellm_http_clients`` clears every known
    module-level cache so the next call opens a fresh pool, breaking the
    deadlock cycle.

If any of these regress, an extract_all_entities-style workload will start
hanging silently in production again. Please don't delete these tests
without re-running an end-to-end deep-research session against the real
litellm pool.
"""

from __future__ import annotations

import asyncio
import copy
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from agentfield.agent_ai import AgentAI, _reset_litellm_http_clients
from tests.helpers import StubAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _DeadlockTestConfig:
    """Minimal AI config that exercises the litellm + rate-limiter path."""

    def __init__(self):
        self.model = "openai/gpt-4"
        self.temperature = 0.1
        self.max_tokens = 100
        self.top_p = 1.0
        self.stream = False
        self.response_format = "auto"
        self.fallback_models = []
        self.final_fallback_model = None
        self.enable_rate_limit_retry = False  # bypass retries; we want to see raw behavior
        self.rate_limit_max_retries = 0
        self.rate_limit_base_delay = 0.0
        self.rate_limit_max_delay = 0.0
        self.rate_limit_jitter_factor = 0.0
        self.rate_limit_circuit_breaker_threshold = 1
        self.rate_limit_circuit_breaker_timeout = 1
        self.auto_inject_memory = []
        self.model_limits_cache = {}
        self.audio_model = "tts-1"
        self.vision_model = "dall-e-3"
        self.video_model = "fal-ai/minimax-video/image-to-video"
        self.fal_api_key = None

    def copy(self, deep=False):
        return copy.deepcopy(self)

    async def get_model_limits(self, model=None):
        return {"context_length": 1000, "max_output_tokens": 100}

    def get_litellm_params(self, **overrides):
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stream": self.stream,
        }
        params.update(overrides)
        return params


@pytest.fixture
def fast_timeout_agent():
    """Stub agent whose `llm_call_timeout` is short enough to test in real time."""
    agent = StubAgent()
    agent.ai_config = _DeadlockTestConfig()
    agent.memory = SimpleNamespace()
    # 0.2s timeout → asyncio safety net at 0.4s. Tests run in well under a second.
    agent.async_config = SimpleNamespace(
        llm_call_timeout=0.2,
        connection_pool_size=4,
        connection_pool_per_host=4,
    )
    return agent


def _install_litellm_stub(monkeypatch, acompletion_side_effect):
    """Install a fake `litellm` module with a controllable `acompletion`."""
    module = types.ModuleType("litellm")
    module.acompletion = acompletion_side_effect

    # Cached client attributes that `_reset_litellm_http_clients` should wipe.
    # Pre-populate them so we can assert they're cleared post-timeout.
    module.module_level_client = MagicMock(name="module_level_client")
    module.module_level_aclient = MagicMock(name="module_level_aclient")
    module.aclient_session = MagicMock(name="aclient_session")
    module.client_session = MagicMock(name="client_session")
    module.in_memory_llm_clients_cache = MagicMock(name="in_memory_llm_clients_cache")
    module.in_memory_llm_clients_cache.clear = MagicMock(name="cache_clear")

    utils_module = types.ModuleType("utils")
    utils_module.get_max_tokens = lambda model: 8192
    utils_module.token_counter = lambda model, messages: 10
    utils_module.trim_messages = lambda messages, model, max_tokens: messages
    module.utils = utils_module

    monkeypatch.setitem(sys.modules, "litellm", module)
    monkeypatch.setitem(sys.modules, "litellm.utils", utils_module)
    monkeypatch.setattr("agentfield.agent_ai.litellm", module, raising=False)
    return module


def _make_chat_response(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, audio=None))]
    )


# ---------------------------------------------------------------------------
# 1. _reset_litellm_http_clients unit behavior
# ---------------------------------------------------------------------------


def test_reset_litellm_http_clients_clears_known_caches(monkeypatch):
    """The reset helper must clear every module-level client attribute we
    know litellm uses to pool connections. If litellm renames or removes one,
    this test will catch it before production does."""

    fake_litellm = types.ModuleType("litellm")
    cleared = {"cache_called": False}

    class _ClearableCache:
        def clear(self):
            cleared["cache_called"] = True

    fake_litellm.module_level_client = object()
    fake_litellm.module_level_aclient = object()
    fake_litellm.aclient_session = object()
    fake_litellm.client_session = object()
    fake_litellm.in_memory_llm_clients_cache = _ClearableCache()

    _reset_litellm_http_clients(fake_litellm)

    assert fake_litellm.module_level_client is None
    assert fake_litellm.module_level_aclient is None
    assert fake_litellm.aclient_session is None
    assert fake_litellm.client_session is None
    assert cleared["cache_called"] is True, (
        "in_memory_llm_clients_cache.clear() must be called so the next "
        "litellm.acompletion gets a fresh client pool."
    )


def test_reset_litellm_http_clients_tolerates_missing_attrs():
    """Litellm versions vary; the reset must not raise on missing attributes."""
    fake_litellm = types.ModuleType("litellm")
    # Empty module — none of the cache attrs exist.
    _reset_litellm_http_clients(fake_litellm)  # must not raise


def test_reset_litellm_http_clients_tolerates_none_module():
    """Defensive: passing None must be a no-op, not a crash."""
    _reset_litellm_http_clients(None)  # must not raise


# ---------------------------------------------------------------------------
# 2. End-to-end deadlock-recovery via _make_litellm_call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hanging_acompletion_triggers_timeout_and_pool_reset(
    monkeypatch, fast_timeout_agent
):
    """The smoking gun. Reproduces the production deadlock in miniature:

      1. acompletion hangs (asyncio.Event that never sets, like a half-closed
         httpx socket).
      2. The asyncio.wait_for safety net at 2 × llm_call_timeout fires.
      3. _reset_litellm_http_clients is invoked (cached clients become None).
      4. A subsequent acompletion call returns successfully (the fix worked).
    """
    call_count = {"n": 0}
    never_set = asyncio.Event()  # never set → simulates a hung HTTP read

    async def acompletion_side_effect(**params):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call hangs forever, like a half-closed socket.
            await never_set.wait()
            return _make_chat_response("never reached")
        # Second call (after the pool reset) succeeds.
        return _make_chat_response("recovered")

    stub_module = _install_litellm_stub(monkeypatch, acompletion_side_effect)
    ai = AgentAI(fast_timeout_agent)
    monkeypatch.setattr(ai, "_ensure_model_limits_cached", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.detect_input_type", lambda value: "text"
    )
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.serialize_result", lambda value: value
    )

    # 1) First call must time out via the safety net (2× llm_call_timeout = 0.4s),
    #    not hang forever.
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await asyncio.wait_for(ai.ai("hello"), timeout=2.0)

    # 2) Pool reset must have fired.
    assert stub_module.module_level_aclient is None, (
        "litellm.module_level_aclient should be cleared after a timeout — "
        "without this, the next call grabs the stuck client and deadlocks."
    )
    assert stub_module.module_level_client is None
    assert stub_module.aclient_session is None
    assert stub_module.client_session is None

    # 3) The next call must succeed (i.e., we are not in a permanent
    #    deadlocked state). This is the actual production-relevant assertion:
    #    one slow request must not poison every subsequent request.
    never_set.set()  # let any lingering coroutine unblock for clean shutdown
    result = await asyncio.wait_for(ai.ai("hello again"), timeout=2.0)
    assert hasattr(result, "text")
    assert result.text == "recovered"
    assert call_count["n"] == 2


@pytest.mark.asyncio
async def test_litellm_params_includes_request_timeout(monkeypatch, fast_timeout_agent):
    """litellm should always be called with an explicit `timeout` parameter
    matching `async_config.llm_call_timeout`. If litellm gains proper timeout
    support in a future version, this is what makes us pick it up — and even
    today, it's the only thing that lets httpx abort the socket cleanly."""
    captured: Dict[str, Any] = {}

    async def acompletion_side_effect(**params):
        captured.update(params)
        return _make_chat_response("ok")

    _install_litellm_stub(monkeypatch, acompletion_side_effect)
    ai = AgentAI(fast_timeout_agent)
    monkeypatch.setattr(ai, "_ensure_model_limits_cached", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.detect_input_type", lambda value: "text"
    )
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.serialize_result", lambda value: value
    )

    await ai.ai("hello")

    assert "timeout" in captured, (
        "litellm.acompletion must be called with a `timeout` parameter so "
        "httpx can abort the socket itself, not just our asyncio coroutine."
    )
    assert captured["timeout"] == fast_timeout_agent.async_config.llm_call_timeout


@pytest.mark.asyncio
async def test_safety_net_fires_within_two_times_llm_call_timeout(
    monkeypatch, fast_timeout_agent
):
    """Bound the worst-case wall-clock time. If the safety-net multiplier
    regresses (e.g., someone bumps it to 10× or removes it), production hangs
    that *do* happen will be invisible for many minutes — exactly the bug we
    were chasing.

    With llm_call_timeout=0.2s the cancel must land well under 1.0s.
    """
    never_set = asyncio.Event()

    async def acompletion_side_effect(**params):
        await never_set.wait()
        return _make_chat_response("never")

    _install_litellm_stub(monkeypatch, acompletion_side_effect)
    ai = AgentAI(fast_timeout_agent)
    monkeypatch.setattr(ai, "_ensure_model_limits_cached", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.detect_input_type", lambda value: "text"
    )
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.serialize_result", lambda value: value
    )

    loop = asyncio.get_event_loop()
    started = loop.time()
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        # Outer wait_for is generous; the SDK's own safety net should fire first.
        await asyncio.wait_for(ai.ai("hello"), timeout=2.0)
    elapsed = loop.time() - started

    # llm_call_timeout=0.2 → safety net 0.4s. Allow generous slack for CI.
    assert elapsed < 1.5, (
        f"Safety net took {elapsed:.2f}s — expected < 1.5s. Something has "
        f"regressed the asyncio.wait_for(timeout=2× llm_call_timeout) logic."
    )

    never_set.set()


# ---------------------------------------------------------------------------
# 3. Realistic concurrency scenarios
#
# These mirror the production failure shape: extract_all_entities runs
# ~10 ai() calls in parallel via asyncio.gather. If 2 of them hang on a
# half-closed httpx socket, the original bug poisoned the connection pool
# for *every* subsequent call. The tests below verify that:
#
#   - Mixed parallel batches (some succeed, some hang) settle correctly.
#   - The pool reset triggered by the hung calls does not corrupt the
#     successful ones still in flight.
#   - A follow-up batch after a hang round goes through cleanly — i.e.
#     the recovery is durable, not just one-shot.
# ---------------------------------------------------------------------------


def _ai_with_hang_pattern(monkeypatch, fast_timeout_agent, hang_predicate):
    """Build an AgentAI whose litellm.acompletion hangs whenever
    `hang_predicate(call_index)` returns True, and otherwise returns a
    successful response tagged with the call index. Returns (ai, stub_module,
    counters)."""
    counters = {"started": 0, "succeeded": 0, "hung_release_event": asyncio.Event()}

    async def acompletion_side_effect(**params):
        counters["started"] += 1
        idx = counters["started"]
        if hang_predicate(idx):
            # Wait until the test releases us, OR until cancellation lands.
            try:
                await counters["hung_release_event"].wait()
            except asyncio.CancelledError:
                raise
            return _make_chat_response(f"late-{idx}")
        counters["succeeded"] += 1
        return _make_chat_response(f"ok-{idx}")

    stub_module = _install_litellm_stub(monkeypatch, acompletion_side_effect)
    ai = AgentAI(fast_timeout_agent)
    monkeypatch.setattr(ai, "_ensure_model_limits_cached", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.detect_input_type", lambda value: "text"
    )
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.serialize_result", lambda value: value
    )
    return ai, stub_module, counters


@pytest.mark.asyncio
async def test_parallel_hangs_some_succeed_some_recover_via_pool_reset(
    monkeypatch, fast_timeout_agent
):
    """The actual production scenario in miniature.

    Spawn 10 concurrent ai() calls (mimics extract_all_entities's
    asyncio.gather over batched prompts). Two of them hang on a "stale
    socket". The remaining 8 must return successfully — the hung calls'
    safety-net firing must not corrupt unrelated in-flight requests. The
    pool reset must run, and a follow-up batch must go through cleanly.
    """
    # Calls #3 and #7 hang; everything else returns instantly.
    hung_indices = {3, 7}
    ai, stub_module, counters = _ai_with_hang_pattern(
        monkeypatch, fast_timeout_agent, lambda idx: idx in hung_indices
    )

    # Batch 1 — 10 parallel calls, 2 of which will hang.
    results = await asyncio.gather(
        *(ai.ai(f"prompt-{i}") for i in range(10)),
        return_exceptions=True,
    )
    successes = [r for r in results if not isinstance(r, BaseException)]
    timeouts = [r for r in results if isinstance(r, (TimeoutError, asyncio.TimeoutError))]
    other_errors = [
        r
        for r in results
        if isinstance(r, BaseException)
        and not isinstance(r, (TimeoutError, asyncio.TimeoutError))
    ]

    assert len(successes) == 8, (
        f"Expected 8 of 10 parallel calls to succeed; got {len(successes)} "
        f"successes, {len(timeouts)} timeouts, {len(other_errors)} other errors. "
        f"This means a hung call's safety-net firing corrupted unrelated "
        f"in-flight requests — the bug we are trying to prevent."
    )
    assert len(timeouts) == 2, (
        f"Expected 2 hung calls to surface as TimeoutError; got {len(timeouts)}. "
        f"Other errors: {other_errors!r}"
    )
    assert other_errors == []

    # Pool reset must have fired (at least once — possibly twice for the two hangs).
    assert stub_module.module_level_aclient is None
    assert stub_module.module_level_client is None
    assert stub_module.in_memory_llm_clients_cache.clear.called

    # Batch 2 — release the still-pending hung futures so they don't leak,
    # then run another 5 calls. All must succeed (recovery is durable).
    counters["hung_release_event"].set()
    # Reset the predicate so no further calls hang.
    counters["started"] = 0  # restart numbering for the second batch
    # Now reinstall a side effect that always succeeds. We do this by
    # swapping acompletion outright — simpler than threading state.
    async def always_ok(**params):
        return _make_chat_response("post-recovery")

    stub_module.acompletion = always_ok

    follow_up = await asyncio.gather(*(ai.ai(f"recovery-{i}") for i in range(5)))
    assert len(follow_up) == 5
    for r in follow_up:
        assert hasattr(r, "text")
        assert r.text == "post-recovery"


@pytest.mark.asyncio
async def test_cascading_sequential_hangs_each_recover_independently(
    monkeypatch, fast_timeout_agent
):
    """Three calls in a row each hang then time out, then a fourth call
    succeeds. Verifies the reset is durable across multiple consecutive
    failures — not just the first one. If reset accidentally caches state
    (e.g. someone adds a `_already_reset` flag), this catches it.
    """
    counters = {"n": 0, "resets_observed": 0}

    async def acompletion_side_effect(**params):
        counters["n"] += 1
        if counters["n"] <= 3:
            never = asyncio.Event()
            await never.wait()  # hang forever
            return _make_chat_response("never")
        return _make_chat_response("finally")

    stub_module = _install_litellm_stub(monkeypatch, acompletion_side_effect)
    ai = AgentAI(fast_timeout_agent)
    monkeypatch.setattr(ai, "_ensure_model_limits_cached", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.detect_input_type", lambda value: "text"
    )
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.serialize_result", lambda value: value
    )

    # Calls 1, 2, 3 must each time out.
    for attempt in range(1, 4):
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await asyncio.wait_for(ai.ai(f"attempt-{attempt}"), timeout=2.0)
        # After each timeout, the cache must be reset. We can re-set the
        # mock attrs between calls to confirm each timeout independently
        # triggered a fresh reset (not just the first one).
        assert stub_module.module_level_aclient is None, (
            f"Pool reset did not fire on attempt {attempt} — reset is not "
            f"idempotent across consecutive failures."
        )
        # Reinstall to detect the next reset.
        stub_module.module_level_aclient = MagicMock(name="reinstalled")

    # Call 4 must succeed.
    result = await ai.ai("attempt-4")
    assert hasattr(result, "text")
    assert result.text == "finally"
    assert counters["n"] == 4


@pytest.mark.asyncio
async def test_fallback_model_used_after_primary_hangs(monkeypatch, fast_timeout_agent):
    """When `fallback_models` is configured and the primary model hangs,
    the safety net should fire on the primary, the pool should reset, and
    the fallback model should be invoked successfully. This exercises the
    interplay between `_make_litellm_call` and `_execute_with_fallbacks`."""
    fast_timeout_agent.ai_config.fallback_models = ["openai/gpt-3.5"]
    call_log: List[str] = []

    async def acompletion_side_effect(**params):
        model = params["model"]
        call_log.append(model)
        if model == fast_timeout_agent.ai_config.model:  # primary hangs
            await asyncio.Event().wait()
            return _make_chat_response("never")
        # Fallback succeeds.
        return _make_chat_response(f"served-by-{model}")

    stub_module = _install_litellm_stub(monkeypatch, acompletion_side_effect)
    ai = AgentAI(fast_timeout_agent)
    monkeypatch.setattr(ai, "_ensure_model_limits_cached", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.detect_input_type", lambda value: "text"
    )
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.serialize_result", lambda value: value
    )

    result = await asyncio.wait_for(ai.ai("hello"), timeout=2.0)

    assert call_log == [fast_timeout_agent.ai_config.model, "openai/gpt-3.5"], (
        f"Expected primary then fallback in order; got {call_log}. The "
        f"fallback path is broken when the primary times out."
    )
    assert hasattr(result, "text")
    assert result.text == "served-by-openai/gpt-3.5"
    # Pool reset must have run (primary timeout triggered it before fallback).
    assert stub_module.module_level_aclient is None


# ---------------------------------------------------------------------------
# 4. Reset robustness — concurrency and defensive paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_resets_are_safe(monkeypatch):
    """If multiple coroutines hit `_reset_litellm_http_clients` at the same
    instant (which happens when several parallel ai() calls all time out
    together), the reset must be safe to call concurrently. Setting a None
    twice is harmless; clearing an already-cleared dict is harmless. This
    test pins down that property so a future "optimization" can't break it.
    """
    fake_litellm = types.ModuleType("litellm")
    cleared_count = {"n": 0}

    class _CountingCache:
        def clear(self):
            cleared_count["n"] += 1

    fake_litellm.module_level_client = object()
    fake_litellm.module_level_aclient = object()
    fake_litellm.aclient_session = object()
    fake_litellm.client_session = object()
    fake_litellm.in_memory_llm_clients_cache = _CountingCache()

    # Run 20 resets concurrently. None should raise; all should observe the
    # final cleared state.
    async def reset():
        _reset_litellm_http_clients(fake_litellm)

    await asyncio.gather(*(reset() for _ in range(20)))

    assert fake_litellm.module_level_aclient is None
    assert fake_litellm.module_level_client is None
    assert fake_litellm.aclient_session is None
    assert fake_litellm.client_session is None
    assert cleared_count["n"] == 20  # every reset called clear() — that's fine


def test_reset_swallows_exceptions_from_broken_cache():
    """If `in_memory_llm_clients_cache.clear()` itself raises (e.g. some
    third-party plugin replaced the cache with a broken object), the reset
    must NOT propagate — the caller is already raising TimeoutError and
    swallowing the original cause would mask the deadlock recovery flow.
    """
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.module_level_aclient = object()

    class _ExplodingCache:
        def clear(self):
            raise RuntimeError("kaboom")

    fake_litellm.in_memory_llm_clients_cache = _ExplodingCache()

    # Must not raise.
    _reset_litellm_http_clients(fake_litellm)
    # Other attrs should still have been processed despite the cache exploding.
    assert fake_litellm.module_level_aclient is None


def test_reset_does_not_clobber_unrelated_module_attrs():
    """The reset should ONLY touch the documented client/cache attributes.
    If someone later changes the implementation to do something broad like
    `for attr in dir(litellm_module): ...`, this test catches the regression
    by checking that unrelated attributes (config flags, helper functions,
    submodules) survive untouched.
    """
    fake_litellm = types.ModuleType("litellm")
    fake_litellm.module_level_aclient = object()
    fake_litellm.suppress_debug_info = True
    fake_litellm.set_verbose = False
    fake_litellm.api_key = "sk-keep-me"

    def sentinel_callback():
        return None

    fake_litellm.success_callback = sentinel_callback

    _reset_litellm_http_clients(fake_litellm)

    assert fake_litellm.module_level_aclient is None  # cleared
    # Everything else must survive.
    assert fake_litellm.suppress_debug_info is True
    assert fake_litellm.set_verbose is False
    assert fake_litellm.api_key == "sk-keep-me"
    assert fake_litellm.success_callback is sentinel_callback


# ---------------------------------------------------------------------------
# 5. Tool-calling loop path
#
# `_tool_loop_completion` has its own copy of the timeout + reset logic
# because the tool loop calls litellm differently. The fix MUST apply to
# both paths — the regular `_make_litellm_call` AND the tool-loop variant.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_calling_loop_recovers_from_hang(monkeypatch, fast_timeout_agent):
    """Tool-calling loop path: same hang, same fix.

    Calls `ai(..., tools=[...])` which routes through `execute_tool_call_loop`
    → `_tool_loop_completion` → `_make_call`. The hang must trigger the
    same safety net + pool reset as the regular path.
    """
    call_count = {"n": 0}
    never_set = asyncio.Event()

    async def acompletion_side_effect(**params):
        call_count["n"] += 1
        if call_count["n"] == 1:
            await never_set.wait()
            return _make_chat_response("never")
        # Second call returns a "no tool call needed" response so the loop ends.
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="recovered-via-tool-loop",
                        audio=None,
                        tool_calls=None,
                    )
                )
            ]
        )

    stub_module = _install_litellm_stub(monkeypatch, acompletion_side_effect)

    # Stub out the tool-call loop machinery so we can drive _tool_loop_completion
    # directly without needing real tool schemas.
    async def fake_loop(*, agent, messages, tools, config, needs_lazy_hydration,
                       litellm_params, make_completion):
        params = {**litellm_params, "messages": messages}
        resp = await make_completion(params)
        return resp, SimpleNamespace(total_turns=1)

    monkeypatch.setattr(
        "agentfield.tool_calling.execute_tool_call_loop", fake_loop, raising=False
    )
    monkeypatch.setattr(
        "agentfield.tool_calling._build_tool_config",
        lambda tools, agent: ([], SimpleNamespace(max_turns=5, max_tool_calls=10), False),
        raising=False,
    )

    ai = AgentAI(fast_timeout_agent)
    monkeypatch.setattr(ai, "_ensure_model_limits_cached", lambda: asyncio.sleep(0))
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.detect_input_type", lambda value: "text"
    )
    monkeypatch.setattr(
        "agentfield.agent_ai.AgentUtils.serialize_result", lambda value: value
    )

    # First invocation: tool-loop path hangs and times out.
    with pytest.raises((TimeoutError, asyncio.TimeoutError)):
        await asyncio.wait_for(ai.ai("hello", tools=[]), timeout=2.0)

    # Pool reset must have fired from the tool-loop path too.
    assert stub_module.module_level_aclient is None, (
        "Tool-calling loop's `_tool_loop_completion._make_call` did not reset "
        "the litellm pool on timeout. Both the regular and tool-loop paths "
        "must apply the same fix."
    )

    # Second invocation through the tool loop must succeed and not raise.
    never_set.set()
    await ai.ai("hello again", tools=[])
    assert call_count["n"] == 2
