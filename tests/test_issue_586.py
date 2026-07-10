"""
Regression test for issue #586.

OpenRouter currently has no upstream provider for
google/gemini-2.5-flash-image that accepts image_config.aspect_ratio, so the
first requests can 404 with "No endpoints found". The SDK should recover by
retrying once without image_config instead of surfacing the provider-matrix
404 to consumers.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


SDK_ROOT = Path(__file__).resolve().parents[1] / "sdk" / "python"
if str(SDK_ROOT) not in sys.path:
    sys.path.insert(0, str(SDK_ROOT))


class _OKPKey:
    pass


def _install_import_stubs(monkeypatch):
    """Stub optional package imports that are unrelated to this vision test."""
    joserfc = types.ModuleType("joserfc")
    joserfc.jwe = types.SimpleNamespace()

    jwk = types.ModuleType("joserfc.jwk")
    jwk.OKPKey = _OKPKey

    monkeypatch.setitem(sys.modules, "joserfc", joserfc)
    monkeypatch.setitem(sys.modules, "joserfc.jwk", jwk)


def _mock_openrouter_response():
    """Create a minimal successful OpenRouter/litellm image response."""
    mock_image_url = MagicMock()
    mock_image_url.url = "data:image/png;base64,issue586"

    mock_image = MagicMock()
    mock_image.image_url = mock_image_url

    mock_choice = MagicMock()
    mock_choice.message.content = "Generated image"
    mock_choice.message.images = [mock_image]

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


_NO_ENDPOINTS_ERROR = (
    "litellm.NotFoundError: NotFoundError: OpenrouterException - "
    '{"error":{"message":"No endpoints found that support the requested '
    'output modalities: image, text","code":404}}'
)


@pytest.mark.asyncio
async def test_issue_586(monkeypatch):
    """
    Reproduce OpenRouter's 404 for gemini-2.5-flash-image with aspect_ratio.

    The first three attempts simulate OpenRouter returning no route for
    image_config={"aspect_ratio": "9:16"}. The final fallback succeeds only if
    the SDK strips image_config from the request.
    """
    _install_import_stubs(monkeypatch)
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    no_endpoints = Exception(_NO_ENDPOINTS_ERROR)
    success_response = _mock_openrouter_response()

    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock(
        side_effect=[no_endpoints, no_endpoints, no_endpoints, success_response]
    )

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        from agentfield.multimodal_response import MultimodalResponse
        from agentfield.vision import generate_image_openrouter

        result = await generate_image_openrouter(
            prompt="A vertical portrait",
            model="openrouter/google/gemini-2.5-flash-image",
            size="1024x1024",
            quality="standard",
            style=None,
            response_format="url",
            image_config={"aspect_ratio": "9:16"},
        )

    assert isinstance(result, MultimodalResponse)
    assert result.raw_response is success_response
    assert len(result.images) == 1
    assert result.images[0].url == "data:image/png;base64,issue586"

    assert mock_litellm.acompletion.call_count == 4

    first_call_kwargs = mock_litellm.acompletion.call_args_list[0][1]
    assert first_call_kwargs["model"] == "openrouter/google/gemini-2.5-flash-image"
    assert first_call_kwargs["modalities"] == ["image"]
    assert first_call_kwargs["image_config"] == {"aspect_ratio": "9:16"}

    retry_without_config_kwargs = mock_litellm.acompletion.call_args_list[-1][1]
    assert "image_config" not in retry_without_config_kwargs
