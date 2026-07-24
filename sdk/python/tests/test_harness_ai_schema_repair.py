"""Regression tests for AI-based schema repair in the harness runner.

Background
----------
Before this change, when an opencode harness call produced text output that
didn't validate against the requested pydantic schema, the runner would
retry the *entire* opencode subprocess — re-running the full agentic loop,
re-reading the repo, etc. — just to coerce the output into valid JSON.
On a real review that's a 30-minute round-trip for a problem that's
typically "the model produced a valid answer but with a stray trailing
comma."

`_ai_schema_repair` short-circuits that: when the harness call did produce
non-empty text but it didn't validate, we make ONE cheap LLM call (no
tools, no repo) that just reformats the existing text into valid JSON.
That's seconds vs. tens of minutes.

These tests pin the load-bearing invariants:
  - empty output → repair is skipped (the user's explicit guidance: if
    there's no output we can't repair from nothing)
  - non-empty malformed output → repair is attempted and, on success,
    avoids the expensive harness retry path
  - repair failure → falls through to the existing harness retry path
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from agentfield.harness._runner import _ai_schema_repair


class _Schema(BaseModel):
    name: str
    count: int


def _make_litellm_response(content: str):
    """Minimal stand-in for a litellm.acompletion response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


@pytest.mark.asyncio
async def test_ai_schema_repair_skips_when_text_is_empty():
    """Empty text = nothing to reformat — must NOT call the LLM.

    User's explicit constraint: 'if it actually failed and there was no
    output, then we can't do an AI-based JSON repair.' Important to pin
    because calling the LLM with empty content would (a) waste a request,
    and (b) risk the model hallucinating fake data.
    """
    options = {"model": "openrouter/moonshotai/kimi-k2.6"}

    # If litellm gets called, this fixture would record it. Instead we
    # patch acompletion to raise — if it's reached, the test fails loud.
    async def _explode(*args, **kwargs):
        raise AssertionError("litellm must NOT be called for empty input")

    with patch("litellm.acompletion", side_effect=_explode):
        result = await _ai_schema_repair("", _Schema, options)

    assert result is None, "empty text must short-circuit to None without LLM call"


@pytest.mark.asyncio
async def test_ai_schema_repair_skips_when_text_is_whitespace_only():
    """Whitespace-only counts as empty. Same reasoning."""
    options = {"model": "openrouter/moonshotai/kimi-k2.6"}

    async def _explode(*args, **kwargs):
        raise AssertionError("litellm must NOT be called for whitespace input")

    with patch("litellm.acompletion", side_effect=_explode):
        result = await _ai_schema_repair("   \n\t  ", _Schema, options)

    assert result is None


@pytest.mark.asyncio
async def test_ai_schema_repair_skips_when_no_model_configured():
    """No model = no idea what to call. Bail rather than guess.

    The harness retry path is the safety net for this case; we don't want
    to silently route to a default model that the caller hasn't authorized.
    """
    options: dict = {}  # no "model" key

    async def _explode(*args, **kwargs):
        raise AssertionError("litellm must NOT be called without a model")

    with patch("litellm.acompletion", side_effect=_explode):
        result = await _ai_schema_repair("some text here", _Schema, options)

    assert result is None


@pytest.mark.asyncio
async def test_ai_schema_repair_succeeds_on_valid_json_response():
    """Happy path: text was malformed, the repair LLM returns valid JSON,
    the result parses against the schema. This is the case that saves the
    30-minute harness retry."""
    options = {"model": "openrouter/moonshotai/kimi-k2.6"}

    async def _fake_acompletion(*args, **kwargs):
        # The repair call returns a clean JSON object matching _Schema.
        return _make_litellm_response('{"name": "test", "count": 42}')

    with patch("litellm.acompletion", side_effect=_fake_acompletion):
        result = await _ai_schema_repair(
            "name: test, count: 42 (this was malformed)", _Schema, options
        )

    assert result is not None
    assert result.name == "test"
    assert result.count == 42


@pytest.mark.asyncio
async def test_ai_schema_repair_returns_none_when_llm_fails():
    """LLM exception → None, so the caller falls through to the existing
    harness retry path. Repair is best-effort — we never let it block the
    fallback flow."""
    options = {"model": "openrouter/moonshotai/kimi-k2.6"}

    async def _fail(*args, **kwargs):
        raise RuntimeError("simulated litellm failure")

    with patch("litellm.acompletion", side_effect=_fail):
        result = await _ai_schema_repair("some text", _Schema, options)

    assert result is None


@pytest.mark.asyncio
async def test_ai_schema_repair_returns_none_when_repair_output_unparseable():
    """If the repair LLM itself returns garbage that doesn't parse against
    the schema, we return None — same fall-through behavior as a hard
    failure. We don't recurse repair-on-repair."""
    options = {"model": "openrouter/moonshotai/kimi-k2.6"}

    async def _bad_response(*args, **kwargs):
        # Repair LLM returns something that doesn't match the schema at all.
        return _make_litellm_response("this is just prose, no JSON")

    with patch("litellm.acompletion", side_effect=_bad_response):
        result = await _ai_schema_repair("some text", _Schema, options)

    assert result is None


@pytest.mark.asyncio
async def test_ai_schema_repair_respects_timeout():
    """A wedged repair call must NOT block the harness indefinitely.
    The internal _SCHEMA_REPAIR_TIMEOUT_SECONDS cap is the safety valve;
    if it fires, we return None and fall through.
    """
    options = {"model": "openrouter/moonshotai/kimi-k2.6"}

    async def _hang(*args, **kwargs):
        await asyncio.sleep(3600)  # would hang for an hour

    # Patch the module-level timeout to something the test can wait for.
    with patch(
        "agentfield.harness._runner._SCHEMA_REPAIR_TIMEOUT_SECONDS", 0.1
    ):
        with patch("litellm.acompletion", side_effect=_hang):
            result = await _ai_schema_repair("some text", _Schema, options)

    assert result is None
