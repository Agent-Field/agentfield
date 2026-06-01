"""
Tests for vision.py image generation functions.
"""

import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentfield.vision import generate_image_litellm, generate_image_openrouter
from agentfield.multimodal_response import MultimodalResponse, ImageOutput


@pytest.mark.asyncio
async def test_generate_image_litellm_success():
    """Test successful LiteLLM image generation."""
    mock_response = MultimodalResponse(
        text="",
        images=[
            ImageOutput(
                url="https://example.com/image1.png",
                b64_json=None,
                revised_prompt="A beautiful sunset",
            )
        ],
    )

    mock_litellm = MagicMock()
    mock_litellm.aimage_generation = AsyncMock(return_value={"data": []})

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        with patch(
            "agentfield.multimodal_response.detect_multimodal_response"
        ) as mock_detect:
            mock_detect.return_value = mock_response

            result = await generate_image_litellm(
                prompt="A sunset",
                model="dall-e-3",
                size="1024x1024",
                quality="hd",
                style="vivid",
                response_format="url",
            )

            assert isinstance(result, MultimodalResponse)
            mock_litellm.aimage_generation.assert_called_once()
            call_kwargs = mock_litellm.aimage_generation.call_args[1]
            assert call_kwargs["prompt"] == "A sunset"
            assert call_kwargs["model"] == "dall-e-3"
            assert call_kwargs["size"] == "1024x1024"
            assert call_kwargs["quality"] == "hd"
            assert call_kwargs["style"] == "vivid"


@pytest.mark.asyncio
async def test_generate_image_litellm_without_style():
    """Test LiteLLM image generation without style parameter for non-DALL-E models."""
    mock_litellm = MagicMock()
    mock_litellm.aimage_generation = AsyncMock(return_value={"data": []})

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        with patch(
            "agentfield.multimodal_response.detect_multimodal_response"
        ) as mock_detect:
            mock_detect.return_value = MultimodalResponse(text="", images=[])

            await generate_image_litellm(
                prompt="A cat",
                model="stable-diffusion",
                size="512x512",
                quality="standard",
                style=None,
                response_format="url",
            )

            call_kwargs = mock_litellm.aimage_generation.call_args[1]
            assert "style" not in call_kwargs


@pytest.mark.asyncio
async def test_generate_image_litellm_import_error():
    """Test ImportError when litellm is not installed."""

    def import_side_effect(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("No module named 'litellm'")
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_side_effect):
        with pytest.raises(ImportError) as exc_info:
            await generate_image_litellm(
                prompt="test",
                model="dall-e-3",
                size="1024x1024",
                quality="standard",
                style=None,
                response_format="url",
            )
        assert "litellm is not installed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_image_litellm_api_error():
    """Test error handling when LiteLLM API fails."""
    mock_litellm = MagicMock()
    mock_litellm.aimage_generation = AsyncMock(side_effect=Exception("API Error"))

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        with pytest.raises(Exception) as exc_info:
            await generate_image_litellm(
                prompt="test",
                model="dall-e-3",
                size="1024x1024",
                quality="standard",
                style=None,
                response_format="url",
            )
        assert "API Error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_image_openrouter_success():
    """Test successful OpenRouter image generation."""
    mock_image_url = MagicMock()
    mock_image_url.url = "data:image/png;base64,abc123"

    mock_image = MagicMock()
    mock_image.image_url = mock_image_url

    mock_choice = MagicMock()
    mock_choice.message.content = "Generated image"
    mock_choice.message.images = [mock_image]

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock(return_value=mock_response)

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        result = await generate_image_openrouter(
            prompt="A beautiful landscape",
            model="openrouter/google/gemini-2.5-flash-image-preview",
            size="1024x1024",
            quality="hd",
            style=None,
            response_format="url",
        )

        assert isinstance(result, MultimodalResponse)
        mock_litellm.acompletion.assert_called_once()
        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert (
            call_kwargs["model"] == "openrouter/google/gemini-2.5-flash-image-preview"
        )
        assert "modalities" in call_kwargs
        assert "image" in call_kwargs["modalities"]


@pytest.mark.asyncio
async def test_generate_image_openrouter_with_dict_images():
    """Test OpenRouter image generation with dict-based image data."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Generated"
    mock_choice.message.images = [
        {"image_url": {"url": "data:image/png;base64,xyz789"}}
    ]

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock(return_value=mock_response)

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        result = await generate_image_openrouter(
            prompt="test",
            model="openrouter/test-model",
            size="1024x1024",
            quality="standard",
            style=None,
            response_format="url",
        )

        assert isinstance(result, MultimodalResponse)
        assert len(result.images) > 0


@pytest.mark.asyncio
async def test_generate_image_openrouter_no_images():
    """Test OpenRouter response with no images."""
    mock_choice = MagicMock()
    mock_choice.message.content = "Text only response"
    mock_choice.message.images = []

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock(return_value=mock_response)

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        result = await generate_image_openrouter(
            prompt="test",
            model="openrouter/test-model",
            size="1024x1024",
            quality="standard",
            style=None,
            response_format="url",
        )

        assert isinstance(result, MultimodalResponse)
        assert len(result.images) == 0
        assert result.text == "Text only response"


@pytest.mark.asyncio
async def test_generate_image_openrouter_import_error():
    """Test ImportError when litellm is not installed."""

    def import_side_effect(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("No module named 'litellm'")
        return __import__(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=import_side_effect):
        with pytest.raises(ImportError) as exc_info:
            await generate_image_openrouter(
                prompt="test",
                model="openrouter/test",
                size="1024x1024",
                quality="standard",
                style=None,
                response_format="url",
            )
        assert "litellm is not installed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_image_openrouter_api_error():
    """Test error handling when OpenRouter API fails."""
    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock(side_effect=Exception("API Error"))

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        with pytest.raises(Exception) as exc_info:
            await generate_image_openrouter(
                prompt="test",
                model="openrouter/test",
                size="1024x1024",
                quality="standard",
                style=None,
                response_format="url",
            )
        assert "API Error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_image_openrouter_with_kwargs():
    """Test OpenRouter image generation with additional kwargs."""
    mock_choice = MagicMock()
    mock_choice.message.content = ""
    mock_choice.message.images = []

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock(return_value=mock_response)

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        await generate_image_openrouter(
            prompt="test",
            model="openrouter/test",
            size="1024x1024",
            quality="standard",
            style=None,
            response_format="url",
            image_config={"aspect_ratio": "16:9"},
            temperature=0.7,
        )

        call_kwargs = mock_litellm.acompletion.call_args[1]
        assert call_kwargs["image_config"] == {"aspect_ratio": "16:9"}
        assert call_kwargs["temperature"] == 0.7


# ---------------------------------------------------------------------------
# Retry / image_config fallback (issues #3 and #5 from reel-af)
# ---------------------------------------------------------------------------


def _build_openrouter_success_response():
    """Build a minimal mock OpenRouter success response."""
    mock_image_url = MagicMock()
    mock_image_url.url = "data:image/png;base64,abc"
    mock_image = MagicMock()
    mock_image.image_url = mock_image_url
    mock_choice = MagicMock()
    mock_choice.message.content = "ok"
    mock_choice.message.images = [mock_image]
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


_NO_ENDPOINTS_MSG = (
    "litellm.NotFoundError: NotFoundError: OpenrouterException - "
    '{"error":{"message":"No endpoints found that support the requested '
    'output modalities: image, text","code":404}}'
)


@pytest.mark.asyncio
async def test_generate_image_openrouter_retries_on_no_endpoints_then_succeeds(
    monkeypatch,
):
    """First call hits 'No endpoints found' 404; second call succeeds (issue #5)."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    success_response = _build_openrouter_success_response()
    failure = Exception(_NO_ENDPOINTS_MSG)

    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock(side_effect=[failure, success_response])

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        result = await generate_image_openrouter(
            prompt="test",
            model="openrouter/google/gemini-2.5-flash-image",
            size="1024x1024",
            quality="standard",
            style=None,
            response_format="url",
        )

    assert isinstance(result, MultimodalResponse)
    assert mock_litellm.acompletion.call_count == 2
    assert result.raw_response is success_response


@pytest.mark.asyncio
async def test_generate_image_openrouter_strips_image_config_after_retries(
    monkeypatch,
):
    """All 3 in-loop attempts fail; strip-and-retry succeeds without image_config (issue #3)."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    success_response = _build_openrouter_success_response()
    failure = Exception(_NO_ENDPOINTS_MSG)

    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock(
        side_effect=[failure, failure, failure, success_response]
    )

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        result = await generate_image_openrouter(
            prompt="test",
            model="openrouter/google/gemini-2.5-flash-image",
            size="1024x1024",
            quality="standard",
            style=None,
            response_format="url",
            image_config={"aspect_ratio": "9:16"},
        )

    assert isinstance(result, MultimodalResponse)
    assert mock_litellm.acompletion.call_count == 4
    # The final (strip) call must not carry image_config.
    final_kwargs = mock_litellm.acompletion.call_args_list[-1][1]
    assert "image_config" not in final_kwargs
    # And the earlier failing calls must have carried image_config.
    first_kwargs = mock_litellm.acompletion.call_args_list[0][1]
    assert first_kwargs.get("image_config") == {"aspect_ratio": "9:16"}
    assert result.raw_response is success_response


@pytest.mark.asyncio
async def test_generate_image_openrouter_does_not_retry_on_other_errors(monkeypatch):
    """Generic exceptions propagate immediately; no retry, no strip."""
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock(side_effect=RuntimeError("boom"))

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        with pytest.raises(RuntimeError, match="boom"):
            await generate_image_openrouter(
                prompt="test",
                model="openrouter/test",
                size="1024x1024",
                quality="standard",
                style=None,
                response_format="url",
                image_config={"aspect_ratio": "9:16"},
            )

    assert mock_litellm.acompletion.call_count == 1


@pytest.mark.asyncio
async def test_generate_image_openrouter_gives_up_after_all_retries_no_image_config(
    monkeypatch,
):
    """When image_config is None, exhaust 3 in-loop attempts then re-raise.

    No strip attempt is performed (nothing to strip), so total call count is 3.
    """
    monkeypatch.setattr("asyncio.sleep", AsyncMock())

    failure = Exception(_NO_ENDPOINTS_MSG)
    mock_litellm = MagicMock()
    mock_litellm.acompletion = AsyncMock(side_effect=[failure, failure, failure])

    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        with pytest.raises(Exception) as exc_info:
            await generate_image_openrouter(
                prompt="test",
                model="openrouter/test",
                size="1024x1024",
                quality="standard",
                style=None,
                response_format="url",
            )

    assert "No endpoints found" in str(exc_info.value)
    assert mock_litellm.acompletion.call_count == 3
