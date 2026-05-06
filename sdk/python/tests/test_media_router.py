"""Unit tests for MediaRouter prefix-based provider dispatch."""

import pytest
from unittest.mock import MagicMock

from agentfield.media_router import MediaRouter


def test_resolve_fal():
    mock_fal = MagicMock()
    mock_fal.supported_modalities = ["image", "audio", "video"]
    router = MediaRouter()
    router.register("fal-ai/", mock_fal)
    assert router.resolve("fal-ai/flux/dev", "image") is mock_fal


def test_resolve_openrouter():
    mock_or = MagicMock()
    mock_or.supported_modalities = ["image", "video"]
    router = MediaRouter()
    router.register("openrouter/", mock_or)
    assert router.resolve("openrouter/google/veo-3.1", "video") is mock_or


def test_resolve_fallback():
    mock_litellm = MagicMock()
    mock_litellm.supported_modalities = ["image", "audio"]
    router = MediaRouter()
    router.register("", mock_litellm)
    assert router.resolve("dall-e-3", "image") is mock_litellm


def test_resolve_longest_prefix_wins():
    mock_fal = MagicMock()
    mock_fal.supported_modalities = ["image"]
    mock_fallback = MagicMock()
    mock_fallback.supported_modalities = ["image"]
    router = MediaRouter()
    router.register("", mock_fallback)
    router.register("fal-ai/", mock_fal)
    assert router.resolve("fal-ai/flux/dev", "image") is mock_fal


def test_resolve_unsupported_raises():
    mock = MagicMock()
    mock.supported_modalities = ["image"]
    router = MediaRouter()
    router.register("fal-ai/", mock)
    with pytest.raises(ValueError, match="No provider"):
        router.resolve("fal-ai/flux/dev", "music")
