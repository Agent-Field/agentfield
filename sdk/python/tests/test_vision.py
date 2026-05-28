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
