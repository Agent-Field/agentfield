"""
Image Generation Module

Handles image generation across multiple providers (LiteLLM, OpenRouter).
Keeps provider-specific implementation details separate from the main agent code.

Supported Providers:
- LiteLLM: DALL-E, Azure DALL-E, Bedrock Stable Diffusion, etc.
- OpenRouter: Gemini image generation, etc.
"""

import asyncio
import os
from typing import Any, Dict, Optional
from agentfield.logger import log_error


async def generate_image_litellm(
    prompt: str,
    model: str,
    size: str,
    quality: str,
    style: Optional[str],
    response_format: str,
    **kwargs,
) -> Any:
    """
    Generate image using LiteLLM's image generation API.

    This function uses LiteLLM's `aimage_generation()` which supports:
    - OpenAI DALL-E (dall-e-3, dall-e-2)
    - Azure DALL-E
    - AWS Bedrock Stable Diffusion
    - And other LiteLLM-supported image generation models

    Args:
        prompt: Text prompt for image generation
        model: Model to use (e.g., "dall-e-3", "azure/dall-e-3")
        size: Image size (e.g., "1024x1024", "1792x1024")
        quality: Image quality ("standard", "hd")
        style: Image style ("vivid", "natural") - DALL-E 3 only
        response_format: Response format ("url", "b64_json")
        **kwargs: Additional LiteLLM parameters

    Returns:
        MultimodalResponse with generated image(s)

    Raises:
        ImportError: If litellm is not installed
        Exception: If image generation fails
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm is not installed. Please install it with `pip install litellm`."
        )

    # Prepare image generation parameters
    image_params = {
        "prompt": prompt,
        "model": model,
        "size": size,
        "quality": quality,
        "response_format": response_format,
        **kwargs,
    }

    # Add style parameter only for DALL-E 3
    if style and "dall-e-3" in model:
        image_params["style"] = style

    try:
        # Use LiteLLM's image generation function
        response = await litellm.aimage_generation(**image_params)

        # Import multimodal response detection
        from agentfield.multimodal_response import detect_multimodal_response

        # Detect and wrap multimodal content
        return detect_multimodal_response(response)

    except Exception as e:
        log_error(f"LiteLLM image generation failed: {e}")
        raise


async def generate_image_openrouter(
    prompt: str,
    model: str,
    size: str,
    quality: str,
    style: Optional[str],
    response_format: str,
    image_config: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Any:
    """
    Generate image using OpenRouter's chat completions API.

    OpenRouter uses modalities to enable image generation through
    the standard chat completions endpoint. This is different from
    LiteLLM's dedicated image generation API.

    Supported models:
    - google/gemini-2.5-flash-image-preview
    - And other OpenRouter models with image generation capabilities

    Args:
        prompt: Text prompt for image generation
        model: OpenRouter model (must start with "openrouter/")
        size: Image size (may not be used by all OpenRouter models)
        quality: Image quality (may not be used by all OpenRouter models)
        style: Image style (may not be used by all OpenRouter models)
        response_format: Response format (may not be used by all OpenRouter models)
        image_config: Optional dict of OpenRouter image generation settings
            (e.g., {"aspect_ratio": "16:9"}). Pass an empty dict to use
            provider defaults explicitly.
        **kwargs: Additional OpenRouter-specific parameters

    Returns:
        MultimodalResponse with generated image(s)

    Raises:
        ImportError: If litellm is not installed
        Exception: If image generation fails
    """
    try:
        import litellm
    except ImportError:
        raise ImportError(
            "litellm is not installed. Please install it with `pip install litellm`."
        )

    from agentfield.multimodal_response import ImageOutput, MultimodalResponse

    # Build messages for OpenRouter chat completions
    messages = [{"role": "user", "content": prompt}]

    # Prepare parameters for OpenRouter
    # OpenRouter uses chat completions with modalities parameter
    completion_params = {
        "model": model,
        "messages": messages,
        "modalities": ["image", "text"],
        **kwargs,
    }

    # Add image_config if provided
    if image_config is not None:
        completion_params["image_config"] = image_config

    try:
        # Use LiteLLM's completion function (OpenRouter uses chat API)
        # Wrap with timeout to prevent silent hangs
        timeout = float(os.getenv("AGENTFIELD_LLM_CALL_TIMEOUT", "120.0"))
        response = await asyncio.wait_for(
            litellm.acompletion(**completion_params),
            timeout=timeout,
        )

        # Extract images from OpenRouter response
        # OpenRouter returns images in choices[0].message.images
        images = []
        text_content = ""

        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message

            # Extract text content
            if hasattr(message, "content") and message.content:
                text_content = message.content

            # Extract images
            if hasattr(message, "images") and message.images:
                for img_data in message.images:
                    # OpenRouter images have structure: {"type": "image_url", "image_url": {"url": "data:..."}}
                    if hasattr(img_data, "image_url"):
                        image_url = (
                            img_data.image_url.url
                            if hasattr(img_data.image_url, "url")
                            else None
                        )
                    elif isinstance(img_data, dict) and "image_url" in img_data:
                        image_url = img_data["image_url"].get("url")
                    else:
                        image_url = None

                    if image_url:
                        images.append(
                            ImageOutput(
                                url=image_url,
                                b64_json=None,
                                revised_prompt=None,
                            )
                        )

        # Create MultimodalResponse
        return MultimodalResponse(
            text=text_content or prompt,
            audio=None,
            images=images,
            files=[],
            raw_response=response,
        )

    except Exception as e:
        log_error(f"OpenRouter image generation failed: {e}")
        raise
