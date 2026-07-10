"""Regression test for issue #584 — OpenRouter wav audio with streaming.

Issue #584: OpenRouterProvider.generate_audio hardcoded stream=true while
requesting audio.format="wav", which OpenRouter rejects for chat-audio models.
The provider should request pcm16 over the wire and wrap the response as wav.
"""

from __future__ import annotations

import base64
import sys
import types
from pathlib import Path

import pytest

SDK_PACKAGE_DIR = Path(__file__).resolve().parents[1] / "sdk" / "python" / "agentfield"
agentfield_pkg = types.ModuleType("agentfield")
agentfield_pkg.__path__ = [str(SDK_PACKAGE_DIR)]
sys.modules.setdefault("agentfield", agentfield_pkg)

from agentfield.media_providers import OpenRouterProvider


@pytest.mark.asyncio
async def test_issue_584(monkeypatch):
    """
    OpenRouter chat-audio generation with format="wav" should not send wav
    while stream=true, because OpenRouter only accepts pcm16 for streaming.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-issue-584")

    provider = OpenRouterProvider()
    provider._model_meta_cache["openai/gpt-audio-mini"] = {
        "id": "openai/gpt-audio-mini",
        "output_modalities": ["text", "audio"],
        "input_modalities": ["text"],
    }

    captured_payload = {}

    async def fake_stream_openrouter_audio(payload, headers, **kwargs):
        captured_payload.update(payload)
        pcm16_b64 = base64.b64encode(b"\x00\x00\x01\x00").decode("ascii")
        return pcm16_b64, "Hello world"

    monkeypatch.setattr(
        provider,
        "_stream_openrouter_audio",
        fake_stream_openrouter_audio,
    )

    result = await provider.generate_audio(
        text="Hello world",
        model="openrouter/openai/gpt-audio-mini",
        format="wav",
    )

    assert captured_payload["stream"] is True
    assert captured_payload["audio"]["format"] == "pcm16"
    assert result.audio is not None
    assert result.audio.format == "wav"
    assert base64.b64decode(result.audio.data).startswith(b"RIFF")
