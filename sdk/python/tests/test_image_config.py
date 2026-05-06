import pytest
from unittest.mock import AsyncMock, MagicMock, patch


def _mock_openrouter_response():
    """Create a mock OpenRouter/litellm response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "test"
    mock_response.choices[0].message.images = []
    return mock_response


@pytest.mark.asyncio
async def test_openrouter_image_config_passthrough():
    """Test that image_config is properly passed to OpenRouter API."""
    mock_acompletion = AsyncMock(return_value=_mock_openrouter_response())

    with patch("litellm.acompletion", mock_acompletion):
        from agentfield.vision import generate_image_openrouter

        await generate_image_openrouter(
            prompt="A landscape",
            model="openrouter/google/gemini-3.1-flash-image-preview",
            size="1024x1024",
            quality="standard",
            style=None,
            response_format="url",
            image_config={"aspect_ratio": "16:9", "image_size": "4K"},
        )

        # Verify image_config was passed
        call_kwargs = mock_acompletion.call_args[1]
        assert call_kwargs.get("image_config") == {
            "aspect_ratio": "16:9",
            "image_size": "4K",
        }


@pytest.mark.asyncio
async def test_openrouter_image_without_config():
    """Test that image generation works without image_config."""
    mock_acompletion = AsyncMock(return_value=_mock_openrouter_response())

    with patch("litellm.acompletion", mock_acompletion):
        from agentfield.vision import generate_image_openrouter

        await generate_image_openrouter(
            prompt="A sunset",
            model="openrouter/google/gemini-2.5-flash-image-preview",
            size="1024x1024",
            quality="standard",
            style=None,
            response_format="url",
        )

        call_kwargs = mock_acompletion.call_args[1]
        assert "image_config" not in call_kwargs
