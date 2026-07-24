"""
Media Provider Abstraction for AgentField

Provides a unified interface for different media generation backends:
- Fal.ai (Flux, SDXL, Whisper, TTS, Video models)
- OpenRouter (via LiteLLM)
- OpenAI DALL-E (via LiteLLM)
- Future: ElevenLabs, Replicate, etc.

Each provider implements the same interface, making it easy to swap
backends or add new ones without changing agent code.
"""

import ipaddress
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

from agentfield.openrouter_attribution import merge_attribution_headers
from agentfield.multimodal_response import (
    AudioOutput,
    FileOutput,
    ImageOutput,
    MultimodalResponse,
    VideoOutput,
)


# Fal image size presets
FalImageSize = Literal[
    "square_hd",  # 1024x1024
    "square",  # 512x512
    "portrait_4_3",  # 768x1024
    "portrait_16_9",  # 576x1024
    "landscape_4_3",  # 1024x768
    "landscape_16_9",  # 1024x576
]


class MediaProvider(ABC):
    """
    Abstract base class for media generation providers.

    Subclass this to add support for new image/audio generation backends.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for identification."""
        pass

    @property
    @abstractmethod
    def supported_modalities(self) -> List[str]:
        """List of supported modalities: 'image', 'audio', 'video'."""
        pass

    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: str = "1024x1024",
        quality: str = "standard",
        **kwargs,
    ) -> MultimodalResponse:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the image
            model: Model to use (provider-specific)
            size: Image dimensions or preset
            quality: Quality level
            **kwargs: Provider-specific options

        Returns:
            MultimodalResponse with generated image(s)
        """
        pass

    @abstractmethod
    async def generate_audio(
        self,
        text: str,
        model: Optional[str] = None,
        voice: str = "alloy",
        format: str = "wav",
        *,
        system: Optional[str] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """
        Generate audio/speech from text.

        Args:
            text: Text to convert to speech
            model: TTS model to use
            voice: Voice identifier
            format: Audio format
            system: Optional system instructions for providers/models that
                support chat-style audio generation
            **kwargs: Provider-specific options

        Returns:
            MultimodalResponse with generated audio
        """
        pass

    async def generate_video(
        self,
        prompt: str,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """
        Generate video from text or image.

        Args:
            prompt: Text description for video
            model: Video model to use
            image_url: Optional input image for image-to-video
            **kwargs: Provider-specific options

        Returns:
            MultimodalResponse with generated video
        """
        raise NotImplementedError(f"{self.name} does not support video generation")

    async def generate_music(
        self,
        prompt: str,
        model: Optional[str] = None,
        duration: Optional[int] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """
        Generate music from a text prompt.

        Args:
            prompt: Text description of the music to generate
            model: Music generation model to use
            duration: Duration in seconds
            **kwargs: Provider-specific options

        Returns:
            MultimodalResponse with generated audio
        """
        raise NotImplementedError(f"{self.name} does not support music generation")


class FalProvider(MediaProvider):
    """
    Fal.ai provider for image, audio, and video generation.

    Image Models:
    - fal-ai/flux/dev - FLUX.1 [dev], 12B params, high quality (default)
    - fal-ai/flux/schnell - FLUX.1 [schnell], fast 1-4 step generation
    - fal-ai/flux-pro/v1.1-ultra - FLUX Pro Ultra, up to 2K resolution
    - fal-ai/fast-sdxl - Fast SDXL
    - fal-ai/recraft-v3 - SOTA text-to-image
    - fal-ai/stable-diffusion-v35-large - SD 3.5 Large

    Video Models:
    - fal-ai/minimax-video/image-to-video - Image to video
    - fal-ai/luma-dream-machine - Luma Dream Machine
    - fal-ai/kling-video/v1/standard/text-to-video - Kling 1.0 text to video

    Audio Models:
    - fal-ai/whisper - Speech to text
    - Custom TTS deployments

    Requires FAL_KEY environment variable or explicit api_key.

    Example:
        provider = FalProvider(api_key="...")

        # Generate image
        result = await provider.generate_image(
            "A sunset over mountains",
            model="fal-ai/flux/dev",
            image_size="landscape_16_9",
            num_images=2
        )
        result.images[0].save("sunset.png")

        # Generate video from image
        result = await provider.generate_video(
            "Camera slowly pans across the scene",
            model="fal-ai/minimax-video/image-to-video",
            image_url="https://example.com/image.jpg"
        )
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Fal provider.

        Args:
            api_key: Fal.ai API key. If not provided, uses FAL_KEY env var.
        """
        self._api_key = api_key
        self._client = None

    @property
    def name(self) -> str:
        return "fal"

    @property
    def supported_modalities(self) -> List[str]:
        return ["image", "audio", "video"]

    def _get_client(self):
        """Lazy initialization of fal client."""
        if self._client is None:
            try:
                import fal_client

                if self._api_key:
                    import os

                    os.environ["FAL_KEY"] = self._api_key

                self._client = fal_client
            except ImportError:
                raise ImportError(
                    "fal-client is not installed. Install it with: pip install fal-client"
                )
        return self._client

    def _parse_image_size(self, size: str) -> Union[str, Dict[str, int]]:
        """
        Parse image size into fal format.

        Args:
            size: Either a preset like "landscape_16_9" or dimensions like "1024x768"

        Returns:
            Fal-compatible image_size (string preset or dict with width/height)
        """
        # Check if it's a fal preset
        fal_presets = {
            "square_hd",
            "square",
            "portrait_4_3",
            "portrait_16_9",
            "landscape_4_3",
            "landscape_16_9",
        }
        if size in fal_presets:
            return size

        # Parse WxH format
        if "x" in size.lower():
            parts = size.lower().split("x")
            try:
                width, height = int(parts[0]), int(parts[1])
                return {"width": width, "height": height}
            except ValueError:
                pass

        # Default to square_hd
        return "square_hd"

    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: str = "square_hd",
        quality: str = "standard",
        num_images: int = 1,
        seed: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """
        Generate image using Fal.ai.

        Args:
            prompt: Text prompt for image generation
            model: Fal model ID (defaults to "fal-ai/flux/dev")
            size: Image size - preset ("square_hd", "landscape_16_9") or "WxH"
            quality: "standard" (25 steps) or "hd" (50 steps)
            num_images: Number of images to generate (1-4)
            seed: Random seed for reproducibility
            guidance_scale: Guidance scale for generation
            num_inference_steps: Override inference steps
            **kwargs: Additional fal-specific parameters

        Returns:
            MultimodalResponse with generated images

        Example:
            result = await provider.generate_image(
                "A cyberpunk cityscape at night",
                model="fal-ai/flux/dev",
                size="landscape_16_9",
                num_images=2,
                seed=42
            )
        """
        client = self._get_client()

        # Default model
        if model is None:
            model = "fal-ai/flux/dev"

        # Parse image size
        image_size = self._parse_image_size(size)

        # Determine inference steps based on quality
        if num_inference_steps is None:
            num_inference_steps = 25 if quality == "standard" else 50

        # Build request arguments
        fal_args: Dict[str, Any] = {
            "prompt": prompt,
            "image_size": image_size,
            "num_images": num_images,
            "num_inference_steps": num_inference_steps,
        }

        # Add optional parameters
        if seed is not None:
            fal_args["seed"] = seed
        if guidance_scale is not None:
            fal_args["guidance_scale"] = guidance_scale

        # Merge any additional kwargs
        fal_args.update(kwargs)

        try:
            # Use subscribe_async for queue-based reliable execution
            result = await client.subscribe_async(
                model,
                arguments=fal_args,
                with_logs=False,
            )

            # Extract images from result
            images = []
            if "images" in result:
                for img_data in result["images"]:
                    url = img_data.get("url")
                    # width, height, content_type available but not used currently
                    # _width = img_data.get("width")
                    # _height = img_data.get("height")
                    # _content_type = img_data.get("content_type", "image/png")

                    if url:
                        images.append(
                            ImageOutput(
                                url=url,
                                b64_json=None,
                                revised_prompt=prompt,
                            )
                        )

            # Also check for single image response
            if "image" in result and not images:
                img_data = result["image"]
                url = img_data.get("url") if isinstance(img_data, dict) else img_data
                if url:
                    images.append(
                        ImageOutput(url=url, b64_json=None, revised_prompt=prompt)
                    )

            return MultimodalResponse(
                text=prompt,
                audio=None,
                images=images,
                files=[],
                raw_response=result,
            )

        except Exception as e:
            from agentfield.logger import log_error

            log_error(f"Fal image generation failed: {e}")
            raise

    async def generate_audio(
        self,
        text: str,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        format: str = "wav",
        ref_audio_url: Optional[str] = None,
        speed: float = 1.0,
        system: Optional[str] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """
        Generate audio using Fal.ai TTS models.

        For voice cloning, provide a ref_audio_url with a sample of the voice.

        Args:
            text: Text to convert to speech
            model: Fal TTS model (provider-specific)
            voice: Voice identifier or preset
            format: Audio format (wav, mp3)
            ref_audio_url: URL to reference audio for voice cloning
            speed: Speech speed multiplier
            **kwargs: Additional fal-specific parameters (gen_text, ref_text, etc.)

        Returns:
            MultimodalResponse with generated audio

        Note:
            Fal has various TTS models with different APIs. Check the specific
            model documentation for available parameters.
        """
        client = self._get_client()

        # Build request arguments based on model
        fal_args: Dict[str, Any] = {}

        # Common patterns for fal TTS models
        if "gen_text" not in kwargs:
            fal_args["gen_text"] = text
        if ref_audio_url:
            fal_args["ref_audio_url"] = ref_audio_url
        if voice and voice.startswith("http"):
            fal_args["ref_audio_url"] = voice

        # Merge additional kwargs
        fal_args.update(kwargs)
        response_format = kwargs.get("output_format") or kwargs.get(
            "response_format"
        )
        output_format = response_format if isinstance(response_format, str) else format

        try:
            result = await client.subscribe_async(
                model,
                arguments=fal_args,
                with_logs=False,
            )

            # Extract audio from result - fal returns audio in various formats
            audio = None
            audio_url = None

            # Check common response patterns
            if "audio_url" in result:
                audio_url = result["audio_url"]
            elif "audio" in result:
                audio_data = result["audio"]
                if isinstance(audio_data, dict):
                    audio_url = audio_data.get("url")
                elif isinstance(audio_data, str):
                    audio_url = audio_data

            if audio_url:
                audio = AudioOutput(
                    url=audio_url,
                    data=None,
                    format=output_format,
                )

            return MultimodalResponse(
                text=text,
                audio=audio,
                images=[],
                files=[],
                raw_response=result,
            )

        except Exception as e:
            from agentfield.logger import log_error

            log_error(f"Fal audio generation failed: {e}")
            raise

    async def generate_video(
        self,
        prompt: str,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        duration: Optional[float] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """
        Generate video using Fal.ai video models.

        Args:
            prompt: Text description for the video
            model: Fal video model (defaults to "fal-ai/minimax-video/image-to-video")
            image_url: Input image URL for image-to-video models
            duration: Video duration in seconds (model-dependent)
            **kwargs: Additional fal-specific parameters

        Returns:
            MultimodalResponse with video in files list

        Example:
            # Image to video
            result = await provider.generate_video(
                "Camera slowly pans across the mountain landscape",
                model="fal-ai/minimax-video/image-to-video",
                image_url="https://example.com/mountain.jpg"
            )

            # Text to video
            result = await provider.generate_video(
                "A cat playing with yarn",
                model="fal-ai/kling-video/v1/standard/text-to-video"
            )
        """
        client = self._get_client()

        # Default model
        if model is None:
            model = "fal-ai/minimax-video/image-to-video"

        # Build request arguments
        fal_args: Dict[str, Any] = {
            "prompt": prompt,
        }

        if image_url:
            fal_args["image_url"] = image_url
        if duration:
            fal_args["duration"] = duration

        # Merge additional kwargs
        fal_args.update(kwargs)

        try:
            result = await client.subscribe_async(
                model,
                arguments=fal_args,
                with_logs=False,
            )

            # Extract video from result
            files = []
            video_url = None

            # Check common response patterns
            if "video_url" in result:
                video_url = result["video_url"]
            elif "video" in result:
                video_data = result["video"]
                if isinstance(video_data, dict):
                    video_url = video_data.get("url")
                elif isinstance(video_data, str):
                    video_url = video_data

            if video_url:
                files.append(
                    FileOutput(
                        url=video_url,
                        data=None,
                        mime_type="video/mp4",
                        filename="generated_video.mp4",
                    )
                )

            # Create VideoOutput from the file data
            videos = []
            for f in files:
                videos.append(
                    VideoOutput(
                        url=f.url,
                        data=f.data,
                        mime_type=f.mime_type or "video/mp4",
                        filename=f.filename,
                    )
                )

            return MultimodalResponse(
                text=prompt,
                audio=None,
                images=[],
                files=files,  # Keep for backward compat
                videos=videos,
                raw_response=result,
            )

        except Exception as e:
            from agentfield.logger import log_error

            log_error(f"Fal video generation failed: {e}")
            raise

    async def transcribe_audio(
        self,
        audio_url: str,
        model: str = "fal-ai/whisper",
        language: Optional[str] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """
        Transcribe audio to text using Fal's Whisper model.

        Args:
            audio_url: URL to audio file to transcribe
            model: Whisper model (defaults to "fal-ai/whisper")
            language: Optional language hint
            **kwargs: Additional parameters

        Returns:
            MultimodalResponse with transcribed text
        """
        client = self._get_client()

        fal_args: Dict[str, Any] = {
            "audio_url": audio_url,
        }
        if language:
            fal_args["language"] = language
        fal_args.update(kwargs)

        try:
            result = await client.subscribe_async(
                model,
                arguments=fal_args,
                with_logs=False,
            )

            # Extract text from result
            text = ""
            if "text" in result:
                text = result["text"]
            elif "transcription" in result:
                text = result["transcription"]

            return MultimodalResponse(
                text=text,
                audio=None,
                images=[],
                files=[],
                raw_response=result,
            )

        except Exception as e:
            from agentfield.logger import log_error

            log_error(f"Fal transcription failed: {e}")
            raise


class LiteLLMProvider(MediaProvider):
    """
    LiteLLM-based provider for OpenAI, Azure, and other LiteLLM-supported backends.

    Uses LiteLLM's image_generation and speech APIs.

    Image Models:
    - dall-e-3 - OpenAI DALL-E 3
    - dall-e-2 - OpenAI DALL-E 2
    - azure/dall-e-3 - Azure DALL-E

    Audio Models:
    - tts-1 - OpenAI TTS
    - tts-1-hd - OpenAI TTS HD
    - gpt-4o-mini-tts - GPT-4o Mini TTS
    """

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "litellm"

    @property
    def supported_modalities(self) -> List[str]:
        return ["image", "audio"]

    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: str = "1024x1024",
        quality: str = "standard",
        style: Optional[str] = None,
        response_format: str = "url",
        **kwargs,
    ) -> MultimodalResponse:
        """Generate image using LiteLLM (DALL-E, Azure DALL-E, etc.)."""
        from agentfield import vision

        model = model or "dall-e-3"

        return await vision.generate_image_litellm(
            prompt=prompt,
            model=model,
            size=size,
            quality=quality,
            style=style,
            response_format=response_format,
            **kwargs,
        )

    async def generate_audio(
        self,
        text: str,
        model: Optional[str] = None,
        voice: str = "alloy",
        format: str = "wav",
        speed: float = 1.0,
        system: Optional[str] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """Generate audio using LiteLLM TTS."""
        try:
            import litellm

            litellm.suppress_debug_info = True
        except ImportError:
            raise ImportError(
                "litellm is not installed. Install it with: pip install litellm"
            )

        model = model or "tts-1"

        try:
            response = await litellm.aspeech(
                model=model,
                input=text,
                voice=voice,
                speed=speed,
                **kwargs,
            )

            # Extract audio data
            audio_data = None
            if hasattr(response, "content"):
                import base64

                audio_data = base64.b64encode(response.content).decode("utf-8")

            audio = AudioOutput(
                data=audio_data,
                format=format,
                url=None,
            )

            return MultimodalResponse(
                text=text,
                audio=audio,
                images=[],
                files=[],
                raw_response=response,
            )

        except Exception as e:
            from agentfield.logger import log_error

            log_error(f"LiteLLM audio generation failed: {e}")
            raise


MAX_VIDEO_BYTES = 500 * 1024 * 1024  # 500 MB hard limit for video downloads
MAX_AUDIO_B64_BYTES = (
    500 * 1024 * 1024
)  # 500 MB hard limit for accumulated audio base64


def _assert_safe_download_url(url: str) -> None:
    parsed_url = urlparse(url)
    if parsed_url.scheme != "https":
        raise RuntimeError(f"Refusing to download video from non-HTTPS URL: {url}")

    hostname = parsed_url.hostname
    if not hostname:
        raise RuntimeError(f"Refusing to download video from invalid URL: {url}")

    normalized_host = hostname.lower().rstrip(".")
    if normalized_host in {"localhost", "0.0.0.0"}:
        raise RuntimeError(f"Refusing to download video from localhost: {url}")

    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        return

    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_unspecified
        or address.is_reserved
        or address.is_multicast
    ):
        raise RuntimeError(f"Refusing to download video from private IP: {url}")


def _wrap_pcm16_bytes_as_wav(pcm: bytes, *, sample_rate: int = 24000) -> bytes:
    """Wrap raw little-endian PCM16 mono bytes in a WAV (RIFF) container."""
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def _wrap_pcm16_as_wav_b64(pcm_b64: str, *, sample_rate: int = 24000) -> str:
    """Decode base64 PCM16 → wrap as WAV → re-encode base64."""
    import base64

    pcm = base64.b64decode(pcm_b64)
    wav = _wrap_pcm16_bytes_as_wav(pcm, sample_rate=sample_rate)
    return base64.b64encode(wav).decode("ascii")


class OpenRouterProvider(MediaProvider):
    """
    OpenRouter provider for image generation via chat completions.

    Uses the modalities parameter with chat completions API for image generation.

    Supports models like:
    - google/gemini-3.1-flash-image-preview
    - Other OpenRouter models with image generation capabilities
    """

    _VIDEO_ERROR_MESSAGES = {
        400: "Bad request — check model name and parameters",
        401: "Invalid API key",
        402: "Insufficient credits",
        429: "Rate limited — try again later",
        500: "OpenRouter server error",
    }

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        # Per-instance cache of model metadata (output_modalities) so we can
        # route requests to the right OpenRouter endpoint without re-fetching
        # on every call. Keyed by the stripped model id ("hexgrad/kokoro-82m").
        self._model_meta_cache: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "openrouter"

    @property
    def supported_modalities(self) -> List[str]:
        return ["image", "video", "audio", "music"]

    @staticmethod
    def _strip_or_prefix(model: str) -> str:
        return model[len("openrouter/") :] if model.startswith("openrouter/") else model

    async def _fetch_model_meta(self, model: str) -> Dict[str, Any]:
        """Fetch + cache OpenRouter model metadata (output_modalities etc.).

        On any error, returns an empty dict so callers can fall back to
        defaults rather than fail the user's call.
        """
        import os

        import aiohttp

        stripped = self._strip_or_prefix(model)
        cached = self._model_meta_cache.get(stripped)
        if cached is not None:
            return cached

        api_key = self._api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return {}

        url = f"https://openrouter.ai/api/v1/models/{stripped}/endpoints"
        headers = merge_attribution_headers({"Authorization": f"Bearer {api_key}"})
        try:
            timeout = aiohttp.ClientTimeout(total=10.0)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        return {}
                    payload = await resp.json()
        except Exception:
            return {}

        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        arch = data.get("architecture", {}) if isinstance(data, dict) else {}
        meta = {
            "id": data.get("id", stripped),
            "output_modalities": list(arch.get("output_modalities", []) or []),
            "input_modalities": list(arch.get("input_modalities", []) or []),
        }
        self._model_meta_cache[stripped] = meta
        return meta

    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: str = "1024x1024",
        quality: str = "standard",
        image_urls: Optional[List[str]] = None,
        image_config: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """Generate image using OpenRouter's chat completions API.

        Args:
            prompt: Text description for image generation.
            model: OpenRouter model (defaults to
                ``google/gemini-3.1-flash-image-preview``).
            size: Image dimensions (model-specific).
            quality: Quality hint (model-specific).
            image_urls: Optional reference / source images for image+text→image
                models (e.g. ``x-ai/grok-imagine-image-quality``). Each entry can
                be an http(s) URL or a ``data:`` URL.
            image_config: OpenRouter-specific extras — ``aspect_ratio``,
                ``image_size``, ``strength``, ``style``, ``rgb_colors``,
                ``background_rgb_color``, ``super_resolution_references``,
                ``font_inputs``.
            extra: Arbitrary passthrough fields merged into the completion
                request (e.g. model-specific switches).
        """
        import os

        import aiohttp

        api_key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key to OpenRouterProvider."
            )

        model = model or "openrouter/google/gemini-3.1-flash-image-preview"

        send_model = self._strip_or_prefix(model)

        user_content: Any = prompt
        if image_urls:
            user_content = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": url}}
                for url in image_urls
            ]

        body: Dict[str, Any] = {
            "model": send_model,
            "messages": [{"role": "user", "content": user_content}],
            "modalities": ["image"],
        }
        if size:
            body["size"] = size
        if quality:
            body["quality"] = quality
        if image_config is not None:
            body["image_config"] = image_config
        if extra:
            body.update(extra)
        if kwargs:
            body.update(kwargs)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers = merge_attribution_headers(headers)

        timeout = aiohttp.ClientTimeout(total=120.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=body,
            ) as resp:
                if resp.status >= 400:
                    detail = await resp.text()
                    raise RuntimeError(
                        f"OpenRouter image generation failed ({resp.status}): "
                        f"{detail[:500]}"
                    )
                payload = await resp.json()

        images: List[ImageOutput] = []
        text_content = ""

        def add_image_url(url: Optional[str]) -> None:
            if not url:
                return
            b64_json = None
            if url.startswith("data:image/") and "base64," in url:
                b64_json = url.split("base64,", 1)[1]
            images.append(
                ImageOutput(url=url, b64_json=b64_json, revised_prompt=None)
            )

        for choice in payload.get("choices", []) or []:
            message = choice.get("message", {}) or {}
            content = message.get("content")
            if isinstance(content, str):
                text_content += content
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        text_content += str(part.get("text") or "")
                    elif part.get("type") in ("image_url", "image"):
                        image_url = part.get("image_url") or {}
                        if isinstance(image_url, dict):
                            add_image_url(image_url.get("url"))
            for img in message.get("images", []) or []:
                if not isinstance(img, dict):
                    continue
                image_url = img.get("image_url") or {}
                if isinstance(image_url, dict):
                    add_image_url(image_url.get("url"))

        return MultimodalResponse(
            text=text_content or prompt,
            audio=None,
            images=images,
            files=[],
            raw_response=payload,
        )

    async def generate_video(
        self,
        prompt: str,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        duration: Optional[float] = None,
        resolution: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        generate_audio: Optional[bool] = None,
        seed: Optional[int] = None,
        frame_images: Optional[List[Dict]] = None,
        input_references: Optional[List[Dict]] = None,
        extra: Optional[Dict[str, Any]] = None,
        poll_interval: float = 30.0,
        timeout: float = 600.0,
        **kwargs,
    ) -> MultimodalResponse:
        """
        Generate video using OpenRouter's async video API.

        Submits a job to POST /api/v1/videos, polls until completed,
        then downloads the video content.

        Args:
            prompt: Text description for video generation
            model: OpenRouter video model name
            image_url: Optional input image for image-to-video
            duration: Video duration in seconds
            resolution: Video resolution (e.g., "1080p")
            aspect_ratio: Aspect ratio (e.g., "16:9")
            generate_audio: Whether to generate audio track
            seed: Random seed for reproducibility
            frame_images: List of frame image dicts for guided generation
            input_references: List of reference input dicts
            poll_interval: Seconds between status polls (default 30)
            timeout: Maximum wait time in seconds (default 600)
            **kwargs: Additional parameters passed to the API

        Returns:
            MultimodalResponse with video in both files[] and videos[]
        """
        import asyncio
        import os
        import time

        import aiohttp

        api_key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key to OpenRouterProvider."
            )

        base_url = "https://openrouter.ai/api/v1"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers = merge_attribution_headers(headers)

        # Strip openrouter/ prefix from model name
        video_model = model or "openrouter/google/veo-2.0-generate-001"
        if video_model.startswith("openrouter/"):
            video_model = video_model[len("openrouter/") :]

        # Build request body
        body: Dict[str, Any] = {
            "model": video_model,
            "prompt": prompt,
        }
        if duration is not None:
            body["duration"] = duration
        if resolution is not None:
            body["resolution"] = resolution
        if aspect_ratio is not None:
            body["aspect_ratio"] = aspect_ratio
        if generate_audio is not None:
            body["generate_audio"] = generate_audio
        if seed is not None:
            body["seed"] = seed
        if frame_images is not None:
            body["frame_images"] = frame_images
        if input_references is not None:
            body["input_references"] = input_references
        if image_url is not None:
            body["image_url"] = image_url
        if extra:
            body.update(extra)

        _error_messages = self._VIDEO_ERROR_MESSAGES

        async with aiohttp.ClientSession() as session:
            # Step 1: Submit video generation job
            async with session.post(
                f"{base_url}/videos", headers=headers, json=body
            ) as resp:
                if resp.status != 202:
                    error_msg = _error_messages.get(
                        resp.status, f"Unexpected status {resp.status}"
                    )
                    detail = await resp.text()
                    raise RuntimeError(
                        f"OpenRouter video submit failed: {error_msg} — {detail[:500]}"
                    )
                submit_data = await resp.json()

            job_id = submit_data.get("id")
            if not job_id:
                raise RuntimeError(
                    f"OpenRouter video submit returned no job id: {submit_data}"
                )
            if not re.match(r"^[a-zA-Z0-9_-]+$", job_id):
                raise RuntimeError(f"OpenRouter returned invalid job id: {job_id!r}")

            # Step 2: Poll for completion
            poll_url = f"{base_url}/videos/{job_id}"
            start_time = time.monotonic()
            poll_data: Dict[str, Any] = {}

            MAX_POLL_RETRIES = 3
            consecutive_errors = 0

            while True:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"OpenRouter video generation timed out after {timeout}s "
                        f"(job {job_id})"
                    )

                try:
                    async with session.get(poll_url, headers=headers) as resp:
                        if resp.status in (502, 503, 504):
                            consecutive_errors = consecutive_errors + 1
                            if consecutive_errors >= MAX_POLL_RETRIES:
                                detail = await resp.text()
                                raise RuntimeError(
                                    f"OpenRouter video poll failed after "
                                    f"{MAX_POLL_RETRIES} retries: "
                                    f"HTTP {resp.status} — {detail[:500]}"
                                )
                            await asyncio.sleep(poll_interval)
                            continue
                        if resp.status != 200:
                            error_msg = _error_messages.get(
                                resp.status, f"Unexpected status {resp.status}"
                            )
                            detail = await resp.text()
                            raise RuntimeError(
                                f"OpenRouter video poll failed: "
                                f"{error_msg} — {detail[:500]}"
                            )
                        consecutive_errors = 0
                        poll_data = await resp.json()
                except aiohttp.ClientError:
                    consecutive_errors = consecutive_errors + 1
                    if consecutive_errors >= MAX_POLL_RETRIES:
                        raise
                    await asyncio.sleep(poll_interval)
                    continue

                status = poll_data.get("status", "")
                if status == "completed":
                    break
                elif status == "failed":
                    error = poll_data.get("error", "unknown error")
                    raise RuntimeError(
                        f"OpenRouter video generation failed: {error} (job {job_id})"
                    )
                # else pending/in_progress — keep polling

                await asyncio.sleep(poll_interval)

            # Step 3: Download video from unsigned URL
            unsigned_urls = poll_data.get("unsigned_urls", [])
            if not unsigned_urls:
                raise RuntimeError(
                    f"OpenRouter video completed but no URLs returned (job {job_id})"
                )

            video_url = unsigned_urls[0]
            _assert_safe_download_url(video_url)

            # OpenRouter's "unsigned_urls" are served from openrouter.ai itself
            # and require the same Bearer auth as the API. CDN-hosted URLs
            # (other hosts) don't need auth — strip in that case.
            from urllib.parse import urlparse

            download_headers = (
                headers
                if (urlparse(video_url).hostname or "").endswith("openrouter.ai")
                else {}
            )

            video_data_bytes: Optional[bytes] = None
            async with session.get(video_url, headers=download_headers) as resp:
                if resp.status != 200:
                    raise RuntimeError(
                        f"Failed to download video from {video_url}: HTTP {resp.status}"
                    )
                content_length = resp.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_VIDEO_BYTES:
                    raise RuntimeError(
                        f"Video too large ({int(content_length)} bytes). "
                        f"Max: {MAX_VIDEO_BYTES}"
                    )
                video_data_bytes = await resp.read()
                if len(video_data_bytes) > MAX_VIDEO_BYTES:
                    raise RuntimeError(
                        f"Video download exceeded {MAX_VIDEO_BYTES} byte limit"
                    )

        # Build response objects
        import base64

        video_b64 = base64.b64encode(video_data_bytes).decode("utf-8")
        usage_data = poll_data.get("usage", {})
        cost = usage_data.get("cost")

        file_out = FileOutput(
            url=video_url,
            data=video_b64,
            mime_type="video/mp4",
            filename="generated_video.mp4",
        )
        video_out = VideoOutput(
            url=video_url,
            data=video_b64,
            mime_type="video/mp4",
            filename="generated_video.mp4",
            cost_usd=cost,
        )

        return MultimodalResponse(
            text=prompt,
            audio=None,
            images=[],
            files=[file_out],
            videos=[video_out],
            raw_response=poll_data,
            cost_usd=cost,
        )

    async def _stream_openrouter_audio(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        *,
        timeout: float = 300.0,
        label: str = "audio",
    ) -> tuple:
        """
        Shared SSE streaming helper for audio and music generation.

        Handles: SSE line-delimited parsing via readline(), chunk accumulation
        with size limit, timeout, and error truncation.

        Args:
            payload: JSON body for the chat completions request
            headers: HTTP headers including Authorization
            timeout: Total request timeout in seconds (default 300)
            label: Label for error messages ("audio" or "music")

        Returns:
            Tuple of (b64_data: str, transcript: str)
        """
        import json as json_mod

        import aiohttp

        client_timeout = aiohttp.ClientTimeout(total=timeout)

        # Music models can send very large SSE lines (>128KB of base64
        # audio data per chunk), exceeding aiohttp's default 64KB
        # readline limit.  We read raw chunks and split on newlines
        # ourselves to avoid LineTooLong errors.
        _CHUNK_SIZE = 256 * 1024  # 256 KB read chunks

        b64_chunks: list = []
        transcript_parts: list = []
        total_size = 0

        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"OpenRouter {label} request failed ({resp.status}): "
                        f"{body[:500]}"
                    )

                # Manual SSE line parsing to handle arbitrarily long lines
                buf = b""
                done = False
                async for raw_chunk in resp.content.iter_any():
                    if done:
                        break
                    buf += raw_chunk
                    while b"\n" in buf:
                        raw_line, buf = buf.split(b"\n", 1)
                        decoded = raw_line.decode(
                            "utf-8", errors="replace"
                        ).strip()
                        if not decoded.startswith("data: "):
                            continue
                        data_str = decoded[len("data: ") :]
                        if data_str == "[DONE]":
                            done = True
                            break
                        try:
                            event = json_mod.loads(data_str)
                        except json_mod.JSONDecodeError:
                            continue

                        choices = event.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        audio_delta = delta.get("audio", {})
                        if audio_delta.get("data"):
                            chunk = audio_delta["data"]
                            total_size += len(chunk)
                            if total_size > MAX_AUDIO_B64_BYTES:
                                raise RuntimeError(
                                    f"Audio base64 data exceeded "
                                    f"{MAX_AUDIO_B64_BYTES} byte limit"
                                )
                            b64_chunks.append(chunk)
                        if audio_delta.get("transcript"):
                            transcript_parts.append(
                                audio_delta["transcript"]
                            )

        b64_full = "".join(b64_chunks)
        transcript = "".join(transcript_parts)
        return b64_full, transcript

    async def generate_audio(
        self,
        text: str,
        model: Optional[str] = None,
        voice: str = "alloy",
        format: str = "wav",
        speed: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        system: Optional[str] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """
        Generate audio via OpenRouter, auto-routing to the right endpoint.

        OpenRouter exposes two API surfaces for audio output:
          - ``POST /audio/speech`` (OpenAI-compatible TTS) — used by dedicated
            TTS models like ``hexgrad/kokoro-82m`` whose ``output_modalities``
            is ``["speech"]``.
          - ``POST /chat/completions`` with ``modalities=["text","audio"]``
            SSE streaming — used by chat-audio models like the ``openai/gpt-audio``
            family whose ``output_modalities`` contains ``"audio"``.

        We fetch the model's metadata once (cached per provider instance) and
        pick the right path. On metadata failure we default to ``/audio/speech``
        because it covers the broader population of TTS models.

        Args:
            text: Text to convert to speech
            model: OpenRouter model ID (e.g., "openai/gpt-audio-mini",
                "hexgrad/kokoro-82m"). Default: ``hexgrad/kokoro-82m``.
            voice: Voice identifier (model-specific — e.g. ``alloy`` for
                OpenAI, ``af_bella`` for Kokoro)
            format: Audio format (wav, mp3, flac, opus, pcm16). ``wav`` is
                synthesized client-side when the upstream endpoint only emits
                pcm.
            speed: Optional speech speed for ``/audio/speech`` models.
            extra: Optional extra request fields for ``/audio/speech`` models.
            system: Optional system instructions for chat-completions audio
                models. Ignored for ``/audio/speech`` models.
            **kwargs: Additional parameters (timeout overrides default 300s)

        Returns:
            MultimodalResponse with generated audio
        """
        import os

        api_key = self._api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key."
            )

        send_model = self._strip_or_prefix(model or "hexgrad/kokoro-82m")
        if send_model == "hexgrad/kokoro-82m" and voice == "alloy":
            voice = "af_alloy"

        audio_format = format
        supported_formats = {"wav", "mp3", "flac", "opus", "pcm16", "pcm"}
        if audio_format not in supported_formats:
            audio_format = "wav"

        timeout = kwargs.pop("timeout", 300.0)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers = merge_attribution_headers(headers)

        meta = await self._fetch_model_meta(send_model)
        output_mods = meta.get("output_modalities") or []
        # Choose path: TTS-only models advertise "speech"; chat-audio models
        # advertise "audio". If metadata is missing, prefer /audio/speech as
        # the broader-compat default.
        use_speech_endpoint = ("speech" in output_mods) or (not output_mods)
        if "audio" in output_mods and "speech" not in output_mods:
            use_speech_endpoint = False

        if use_speech_endpoint:
            audio_b64, mime = await self._openrouter_audio_speech(
                text=text,
                model=send_model,
                voice=voice,
                requested_format=audio_format,
                headers=headers,
                timeout=timeout,
                speed=speed,
                extra=extra,
            )
            audio_output = AudioOutput(
                data=audio_b64 if audio_b64 else None,
                format=audio_format,
                url=None,
            )
            return MultimodalResponse(
                text=text,
                audio=audio_output if audio_b64 else None,
                images=[],
                files=[],
                raw_response={
                    "endpoint": "audio/speech",
                    "model": send_model,
                    "mime_type": mime,
                },
            )

        # Chat-completions audio modality path (gpt-audio family).
        # Streaming on the OpenAI provider only emits pcm16 — fall back to
        # pcm16 over the wire and re-wrap to user's requested format below.
        wire_format = "pcm16" if audio_format == "wav" else audio_format
        messages = [{"role": "user", "content": text}]
        if system is not None:
            messages.insert(0, {"role": "system", "content": system})
        payload = {
            "model": send_model,
            "messages": messages,
            "modalities": ["text", "audio"],
            "audio": {"voice": voice, "format": wire_format},
            "stream": True,
        }
        b64_full, transcript = await self._stream_openrouter_audio(
            payload, headers, timeout=timeout, label="audio"
        )

        # Re-wrap pcm16 -> wav if user asked for wav.
        if audio_format == "wav" and b64_full:
            b64_full = _wrap_pcm16_as_wav_b64(b64_full, sample_rate=24000)

        audio_output = AudioOutput(
            data=b64_full if b64_full else None,
            format=audio_format,
            url=None,
        )
        return MultimodalResponse(
            text=transcript or text,
            audio=audio_output if b64_full else None,
            images=[],
            files=[],
            raw_response={"transcript": transcript, "model": send_model},
        )

    async def _openrouter_audio_speech(
        self,
        *,
        text: str,
        model: str,
        voice: str,
        requested_format: str,
        headers: Dict[str, str],
        timeout: float,
        speed: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Call ``POST /api/v1/audio/speech`` and return ``(b64_data, mime)``.

        Handles format translation: when the caller wants ``wav`` we ask the
        upstream for ``pcm`` and wrap it in a WAV header ourselves (24 kHz
        mono int16 — the rate that current OpenRouter TTS endpoints emit).
        """
        import base64

        import aiohttp

        # Map caller's format → upstream response_format
        if requested_format in ("wav", "pcm", "pcm16"):
            wire_format = "pcm"
        else:
            wire_format = requested_format  # mp3 / flac / opus / aac

        body: Dict[str, Any] = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": wire_format,
        }
        if speed is not None:
            body["speed"] = speed
        if extra:
            body.update(extra)

        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(
                "https://openrouter.ai/api/v1/audio/speech",
                json=body,
                headers=headers,
            ) as resp:
                content_type = resp.headers.get("Content-Type", "")
                if resp.status >= 400:
                    detail = await resp.text()
                    raise RuntimeError(
                        f"OpenRouter audio/speech request failed "
                        f"({resp.status}): {detail[:500]}"
                    )
                audio_bytes = await resp.read()

        if requested_format == "wav":
            wav_bytes = _wrap_pcm16_bytes_as_wav(audio_bytes, sample_rate=24000)
            return base64.b64encode(wav_bytes).decode("ascii"), "audio/wav"

        return base64.b64encode(audio_bytes).decode("ascii"), content_type

    async def generate_music(
        self,
        prompt: str,
        model: Optional[str] = None,
        duration: Optional[int] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """
        Generate music via OpenRouter using a music-capable model.

        Uses SSE streaming chat completions with audio modality, similar to
        generate_audio but targeting music generation models.

        Args:
            prompt: Text description of the music to generate
            model: Music model (defaults to "google/lyria-3-pro")
            duration: Duration hint in seconds (must be >0 and <=600)
            **kwargs: Additional parameters (timeout overrides default 300s)

        Returns:
            MultimodalResponse with generated audio (48kHz stereo)
        """
        import os

        api_key = self._api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var or pass api_key."
            )

        send_model = model or "google/lyria-3-pro"
        if send_model.startswith("openrouter/"):
            send_model = send_model[len("openrouter/") :]

        # Validate duration
        if duration is not None:
            if duration <= 0 or duration > 600:
                raise ValueError(f"duration must be > 0 and <= 600, got {duration}")

        # Build the user message with optional duration hint
        user_content = prompt
        if duration is not None:
            user_content = f"{prompt} (duration: {duration} seconds)"

        audio_format = kwargs.pop("format", "wav")
        timeout = kwargs.pop("timeout", 300.0)

        payload = {
            "model": send_model,
            "messages": [{"role": "user", "content": user_content}],
            "modalities": ["text", "audio"],
            "audio": {"format": audio_format},
            "stream": True,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        headers = merge_attribution_headers(headers)

        b64_full, transcript = await self._stream_openrouter_audio(
            payload, headers, timeout=timeout, label="music"
        )

        audio_output = AudioOutput(
            data=b64_full if b64_full else None,
            format=audio_format,
            url=None,
        )

        return MultimodalResponse(
            text=transcript or prompt,
            audio=audio_output if b64_full else None,
            images=[],
            files=[],
            raw_response={"transcript": transcript, "model": send_model},
        )


# Provider registry for easy access
_PROVIDERS: Dict[str, type] = {
    "fal": FalProvider,
    "litellm": LiteLLMProvider,
    "openrouter": OpenRouterProvider,
}


def get_provider(name: str, **kwargs) -> MediaProvider:
    """
    Get a media provider instance by name.

    Args:
        name: Provider name ('fal', 'litellm', 'openrouter')
        **kwargs: Provider-specific initialization arguments

    Returns:
        MediaProvider instance

    Example:
        # Fal provider for Flux
        provider = get_provider("fal", api_key="...")
        result = await provider.generate_image(
            "A sunset over mountains",
            model="fal-ai/flux/dev"
        )

        # LiteLLM provider for DALL-E
        provider = get_provider("litellm")
        result = await provider.generate_image(
            "A sunset over mountains",
            model="dall-e-3"
        )
    """
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: {name}. Available: {list(_PROVIDERS.keys())}"
        )
    return _PROVIDERS[name](**kwargs)


def register_provider(name: str, provider_class: type):
    """
    Register a custom media provider.

    Args:
        name: Provider name for lookup
        provider_class: MediaProvider subclass

    Example:
        class ReplicateProvider(MediaProvider):
            ...

        register_provider("replicate", ReplicateProvider)
    """
    if not issubclass(provider_class, MediaProvider):
        raise TypeError("provider_class must be a MediaProvider subclass")
    _PROVIDERS[name] = provider_class
