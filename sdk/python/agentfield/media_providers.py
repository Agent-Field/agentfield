"""
Media Provider Abstraction for AgentField

Provides a unified interface for different media generation backends:
- Fal.ai (Flux, SDXL, Whisper, TTS, Video models)
- MiniMax (Media generation)
- OpenRouter (via LiteLLM)
- OpenAI DALL-E (via LiteLLM)
- Future: ElevenLabs, Replicate, etc.

Each provider implements the same interface, making it easy to swap
backends or add new ones without changing agent code.
"""

import ipaddress
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import quote, urlparse

from agentfield.openrouter_attribution import merge_attribution_headers
from agentfield.multimodal_response import (
    AudioOutput,
    FileOutput,
    ImageOutput,
    MultimodalResponse,
    VideoOutput,
)


MINIMAX_GLOBAL_BASE_URL = "https://api.minimax.io/v1"
MINIMAX_CN_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_H3_MODEL = "MiniMax-H3"
MINIMAX_H3_VIDEO_METADATA: Dict[str, Any] = {
    "release_date": "2026-07-31",
    "api_version": "v2",
    "input_modes": [
        "text_to_video",
        "image_to_video",
        "first_last_frame_to_video",
        "reference_to_video",
    ],
    "input_modalities": ["text", "image", "video", "audio"],
    "output_modalities": ["video", "audio"],
    "resolutions": ["768P", "2K"],
    "duration_seconds": {"min": 4, "max": 15, "integer": True},
    "pricing": {
        "global_en": {
            "currency": "USD",
            "output_video": {
                "unit": "second",
                "rates": {"768P": 0.08, "2K": 0.13},
            },
            "input_reference_video": {
                "unit": "second",
                "rates": {"768P": 0.08, "2K": 0.13},
            },
            "input_reference_audio": {"free": True},
            "input_reference_images": {
                "unit": "image",
                "free_count": 5,
                "additional_image": 0.04,
            },
        },
        "cn_zh": {
            "currency": "CNY",
            "output_video": {
                "unit": "second",
                "rates": {"768P": 0.5, "2K": 0.8},
            },
            "input_reference_video": {
                "unit": "second",
                "rates": {"768P": 0.5, "2K": 0.8},
            },
            "input_reference_audio": {"free": True},
            "input_reference_images": {
                "unit": "image",
                "free_count": 5,
                "additional_image": 0.2,
            },
        },
    },
}


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
        response_format = kwargs.get("output_format") or kwargs.get("response_format")
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


class MiniMaxProvider(MediaProvider):
    """MiniMax media generation provider."""

    DEFAULT_VIDEO_MODEL = MINIMAX_H3_MODEL
    H3_MODEL_METADATA = {MINIMAX_H3_MODEL: MINIMAX_H3_VIDEO_METADATA}
    H3_CONTENT_TYPES = {"text", "image_url", "video_url", "audio_url"}
    H3_CONTENT_ROLES = {
        "first_frame",
        "last_frame",
        "reference_image",
        "reference_video",
        "reference_audio",
    }
    H3_RATIOS = {"adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"}
    H3_TASK_STATUSES = {
        "queued",
        "running",
        "succeeded",
        "failed",
        "cancelled",
    }
    H3_TASK_TYPES = {"generation", "h3_context_ir", "regeneration"}
    H3_MAX_REQUEST_BODY_BYTES = 64 * 1024 * 1024

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        import os

        self._api_key = api_key
        configured_base_url = (
            base_url or os.environ.get("MINIMAX_BASE_URL") or MINIMAX_GLOBAL_BASE_URL
        )
        self._base_url = configured_base_url.rstrip("/")

    @property
    def name(self) -> str:
        return "minimax"

    @property
    def supported_modalities(self) -> List[str]:
        return ["video", "music", "audio"]

    @property
    def video_model_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return provider-owned metadata for supported video models."""
        return self.H3_MODEL_METADATA

    async def generate_image(
        self,
        prompt: str,
        model: Optional[str] = None,
        size: str = "1024x1024",
        quality: str = "standard",
        **kwargs,
    ) -> MultimodalResponse:
        raise NotImplementedError("minimax does not support image generation")

    async def generate_audio(
        self,
        text: str,
        model: Optional[str] = None,
        voice: str = "alloy",
        format: str = "wav",
        *,
        system: Optional[str] = None,
        stream: bool = False,
        language_boost: Optional[str] = None,
        output_format: str = "hex",
        voice_setting: Optional[Dict[str, Any]] = None,
        pronunciation_dict: Optional[Dict[str, Any]] = None,
        audio_setting: Optional[Dict[str, Any]] = None,
        voice_modify: Optional[Dict[str, Any]] = None,
        subtitle_enable: Optional[bool] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """Generate speech via the MiniMax t2a_v2 endpoint."""
        import base64
        import os

        import aiohttp

        api_key = self._api_key or os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MiniMax API key required. Set MINIMAX_API_KEY or pass api_key "
                "to MiniMaxProvider."
            )
        if stream:
            raise ValueError("MiniMax streaming TTS is not supported by generate_audio")
        if output_format not in {"hex", "url"}:
            raise ValueError("MiniMax audio output_format must be hex or url")

        audio_options = dict(audio_setting or {})
        audio_format = audio_options.get("format", format)
        if audio_format not in {"mp3", "wav", "flac", "pcm"}:
            raise ValueError("MiniMax audio format must be mp3, wav, flac, or pcm")
        audio_options["format"] = audio_format

        voice_options = dict(voice_setting or {})
        speed = kwargs.pop("speed", None)
        if voice != "alloy":
            voice_options.setdefault("voice_id", voice)
        if speed is not None and voice_options:
            voice_options.setdefault("speed", speed)
        elif speed not in (None, 1.0):
            raise ValueError(
                "MiniMax speed requires a voice or voice_setting with voice_id"
            )
        if voice_options and not voice_options.get("voice_id"):
            raise ValueError("MiniMax voice_setting requires voice_id")

        send_model = self._strip_prefix(model or "speech-2.8-hd")
        if not send_model:
            raise ValueError("MiniMax audio generation requires a model")

        body: Dict[str, Any] = {
            "model": send_model,
            "text": text,
            "stream": False,
            "output_format": output_format,
            "audio_setting": audio_options,
        }
        optional_fields = {
            "language_boost": language_boost,
            "voice_setting": voice_options or None,
            "pronunciation_dict": pronunciation_dict,
            "voice_modify": voice_modify,
            "subtitle_enable": subtitle_enable,
        }
        body.update(
            {key: value for key, value in optional_fields.items() if value is not None}
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=120.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self._base_url}/t2a_v2",
                headers=headers,
                json=body,
            ) as response:
                if response.status >= 400:
                    detail = await response.text()
                    raise RuntimeError(
                        f"MiniMax audio generation failed ({response.status}): "
                        f"{detail[:500]}"
                    )
                data = await response.json()

        if not isinstance(data, dict):
            raise RuntimeError("MiniMax audio generation returned an invalid response")
        base_resp = data.get("base_resp") or {}
        if not isinstance(base_resp, dict):
            raise RuntimeError("MiniMax audio generation returned an invalid response")
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0):
            status_msg = base_resp.get("status_msg") or "unknown error"
            raise RuntimeError(
                f"MiniMax audio generation failed ({status_code}): {status_msg}"
            )

        response_data = data.get("data") or {}
        if not isinstance(response_data, dict):
            raise RuntimeError("MiniMax audio generation returned no audio")
        status = response_data.get("status")
        if status not in (None, 2):
            raise RuntimeError("MiniMax audio generation did not complete")
        audio_value = response_data.get("audio")
        if not isinstance(audio_value, str) or not audio_value:
            raise RuntimeError("MiniMax audio generation returned no audio")

        if output_format == "url":
            output = AudioOutput(data=None, format=audio_format, url=audio_value)
        else:
            try:
                audio_bytes = bytes.fromhex(audio_value)
            except ValueError as exc:
                raise RuntimeError(
                    "MiniMax audio generation returned invalid hex audio"
                ) from exc
            output = AudioOutput(
                data=base64.b64encode(audio_bytes).decode("ascii"),
                format=audio_format,
                url=None,
            )
        return MultimodalResponse(
            text=text,
            audio=output,
            images=[],
            files=[],
            raw_response=data,
        )

    async def clone_voice(
        self,
        audio: Union[str, Path, bytes],
        voice_id: Optional[str] = None,
        model: Optional[str] = None,
        purpose: str = "voice_clone",
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """
        Clone a voice from a reference audio sample via the MiniMax voice_clone
        endpoint.

        The reference audio is uploaded to the MiniMax file store and then used
        to create a new voice that can be referenced by ``voice_id`` in later
        speech calls.

        Args:
            audio: Reference audio sample, as a file path or raw bytes.
            voice_id: ID to assign to the newly cloned voice.
            model: MiniMax speech model used for the clone (defaults to
                ``speech-2.8-hd``). A ``minimax/`` prefix is stripped.
            purpose: File upload purpose; ``voice_clone`` or ``prompt_audio``.
            filename: Name for the uploaded reference audio (inferred from a
                file path when not provided).
            content_type: MIME type for the uploaded audio (inferred from the
                file extension when not provided).
            **kwargs: Reserved for future use.

        Returns:
            MultimodalResponse whose ``text`` carries the cloned ``voice_id``
            and whose ``raw_response`` contains the upload and clone responses.

        Raises:
            ValueError: If the API key, voice ID, model, audio data, or purpose
                is invalid.
            RuntimeError: If either the upload or the clone request fails.
        """
        import mimetypes
        import os

        import aiohttp

        api_key = self._api_key or os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MiniMax API key required. Set MINIMAX_API_KEY or pass api_key "
                "to MiniMaxProvider."
            )
        if not voice_id:
            raise ValueError("MiniMax voice cloning requires a voice_id")
        if purpose not in {"voice_clone", "prompt_audio"}:
            raise ValueError(
                "MiniMax voice cloning purpose must be voice_clone or prompt_audio"
            )
        send_model = self._strip_prefix(model or "speech-2.8-hd")
        if not send_model:
            raise ValueError("MiniMax voice cloning requires a model")

        if isinstance(audio, (str, Path)):
            audio_path = Path(audio)
            audio_data = audio_path.read_bytes()
            inferred_name = audio_path.name or None
        elif isinstance(audio, bytes):
            audio_data = audio
            inferred_name = None
        else:
            raise ValueError("MiniMax voice cloning audio must be a file path or bytes")
        upload_name = filename or inferred_name
        if not upload_name:
            raise ValueError(
                "MiniMax voice cloning requires a filename for raw audio bytes"
            )
        upload_type = (
            content_type or mimetypes.guess_type(upload_name)[0] or "audio/mpeg"
        )

        headers = {"Authorization": f"Bearer {api_key}"}
        timeout = aiohttp.ClientTimeout(total=120.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            upload_form = aiohttp.FormData()
            upload_form.add_field(
                "file",
                audio_data,
                filename=upload_name,
                content_type=upload_type,
            )
            upload_form.add_field("purpose", purpose)
            async with session.post(
                f"{self._base_url}/files/upload",
                headers=headers,
                data=upload_form,
            ) as response:
                upload_data = await self._read_response(
                    response, "file upload", service="voice cloning"
                )
            file_id = self._extract_file_id(upload_data)
            if not file_id:
                raise RuntimeError("MiniMax voice cloning upload returned no file_id")

            clone_form = aiohttp.FormData()
            clone_form.add_field("file_id", file_id)
            clone_form.add_field("voice_id", voice_id)
            clone_form.add_field("model", send_model)
            async with session.post(
                f"{self._base_url}/voice_clone",
                headers=headers,
                data=clone_form,
            ) as response:
                clone_data = await self._read_response(
                    response, "clone", service="voice cloning"
                )
            cloned_voice_id = str(clone_data.get("voice_id") or "")
            if not cloned_voice_id:
                raise RuntimeError("MiniMax voice cloning returned no voice_id")

        return MultimodalResponse(
            text=cloned_voice_id,
            audio=None,
            images=[],
            files=[],
            raw_response={"upload": upload_data, "clone": clone_data},
        )

    async def generate_music(
        self,
        prompt: str,
        model: Optional[str] = None,
        duration: Optional[int] = None,
        **kwargs,
    ) -> MultimodalResponse:
        """Generate music via the MiniMax music_generation endpoint."""
        import base64
        import os

        import aiohttp

        api_key = self._api_key or os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MiniMax API key required. Set MINIMAX_API_KEY or pass api_key "
                "to MiniMaxProvider."
            )
        body = {
            "model": model or "music-3.0",
            "prompt": prompt,
            "output_format": kwargs.pop("output_format", "url"),
            "stream": kwargs.pop("stream", False),
        }
        if duration is not None:
            body["audio_setting"] = {
                "sample_rate": kwargs.pop("sample_rate", 44100),
                "bitrate": kwargs.pop("bitrate", 256000),
                "format": kwargs.pop("format", "mp3"),
                "duration": duration,
            }
        body.update(
            {
                k: v
                for k, v in kwargs.items()
                if k
                in {
                    "lyrics",
                    "lyrics_optimizer",
                    "is_instrumental",
                    "audio_setting",
                    "aigc_watermark",
                    "cover_feature_id",
                }
            }
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._base_url}/music_generation",
                json=body,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            ) as response:
                data = await response.json()
                if (
                    response.status >= 400
                    or data.get("base_resp", {}).get("status_code") != 0
                ):
                    raise RuntimeError(
                        f"MiniMax music generation failed ({response.status}): {data}"
                    )
        audio = data.get("data", {}).get("audio")
        audio_data = (
            base64.b64encode(bytes.fromhex(audio)).decode("ascii")
            if audio and body["output_format"] == "hex"
            else None
        )
        output = AudioOutput(
            data=audio_data,
            format=body.get("audio_setting", {}).get("format", "mp3"),
            url=audio if body["output_format"] == "url" else None,
        )
        return MultimodalResponse(
            text=prompt, audio=output, images=[], files=[], raw_response=data
        )

    @staticmethod
    def _strip_prefix(model: str) -> str:
        return model[len("minimax/") :] if model.startswith("minimax/") else model

    @staticmethod
    def _check_api_response(
        payload: Dict[str, Any], operation: str, service: str = "video"
    ) -> None:
        base_resp = payload.get("base_resp") or {}
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0):
            status_msg = base_resp.get("status_msg") or "unknown error"
            raise RuntimeError(
                f"MiniMax {service} {operation} failed ({status_code}): {status_msg}"
            )

    async def _read_response(
        self, response: Any, operation: str, service: str = "video"
    ) -> Dict[str, Any]:
        if response.status >= 400:
            detail = await response.text()
            raise RuntimeError(
                f"MiniMax {service} {operation} failed ({response.status}): "
                f"{detail[:500]}"
            )
        payload = await response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"MiniMax {service} {operation} returned an invalid response"
            )
        self._check_api_response(payload, operation, service=service)
        return payload

    @staticmethod
    def _extract_file_id(payload: Dict[str, Any]) -> str:
        """Extract the uploaded file id from a MiniMax file upload response."""
        file_info = payload.get("file")
        if isinstance(file_info, dict):
            file_id = str(file_info.get("file_id") or "")
            if file_id:
                return file_id
        data = payload.get("data")
        if isinstance(data, dict):
            file_id = str(data.get("file_id") or "")
            if file_id:
                return file_id
        return str(payload.get("file_id") or "")

    def _require_api_key(self) -> str:
        import os

        api_key = self._api_key or os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MiniMax API key required. Set MINIMAX_API_KEY or pass api_key "
                "to MiniMaxProvider."
            )
        return api_key

    def _base_url_for_version(self, version: str) -> str:
        if re.search(r"/v\d+$", self._base_url):
            return re.sub(r"/v\d+$", f"/{version}", self._base_url)
        return f"{self._base_url}/{version}"

    def _uses_cn_endpoint(self) -> bool:
        return urlparse(self._base_url).hostname == "api.minimaxi.com"

    @staticmethod
    def _video_client_timeout() -> Any:
        import aiohttp

        return aiohttp.ClientTimeout(total=None, connect=30.0, sock_read=120.0)

    @staticmethod
    def _normalize_task_id(task_id: str) -> str:
        normalized = str(task_id or "").strip()
        if not normalized:
            raise ValueError("MiniMax video task_id is required")
        return normalized

    @classmethod
    def _validate_h3_content(cls, content: List[Dict[str, Any]]) -> Dict[str, bool]:
        if not isinstance(content, list) or not content:
            raise ValueError("MiniMax-H3 video content must be a non-empty list")

        text_count = 0
        unroled_images = 0
        role_counts = {role: 0 for role in cls.H3_CONTENT_ROLES}
        allowed_roles = {
            "image_url": {None, "first_frame", "last_frame", "reference_image"},
            "video_url": {"reference_video"},
            "audio_url": {"reference_audio"},
        }

        for item in content:
            if not isinstance(item, dict):
                raise ValueError("MiniMax-H3 content entries must be objects")
            content_type = item.get("type")
            if content_type not in cls.H3_CONTENT_TYPES:
                raise ValueError(
                    "MiniMax-H3 content type must be text, image_url, video_url, "
                    "or audio_url"
                )

            role = item.get("role")
            if role is not None and role not in cls.H3_CONTENT_ROLES:
                raise ValueError(f"Unsupported MiniMax-H3 content role: {role}")

            if content_type == "text":
                text = item.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("MiniMax-H3 content requires non-empty text")
                if len(text) > 7000:
                    raise ValueError(
                        "MiniMax-H3 text content cannot exceed 7000 characters"
                    )
                if role is not None:
                    raise ValueError("MiniMax-H3 text content cannot define a role")
                text_count += 1
                continue

            media = item.get(content_type)
            if not isinstance(media, dict) or not isinstance(media.get("url"), str):
                raise ValueError(
                    f"MiniMax-H3 {content_type} content requires a nested url"
                )
            if not media["url"].strip():
                raise ValueError(f"MiniMax-H3 {content_type} url cannot be empty")
            if role not in allowed_roles[content_type]:
                raise ValueError(
                    f"MiniMax-H3 {content_type} content does not support role {role}"
                )
            if role is None:
                unroled_images += 1
            else:
                role_counts[role] += 1

        if text_count == 0:
            raise ValueError("MiniMax-H3 video content requires a text entry")
        if unroled_images + role_counts["first_frame"] > 1:
            raise ValueError("MiniMax-H3 content supports at most one first frame")
        if role_counts["last_frame"] > 1:
            raise ValueError("MiniMax-H3 content supports at most one last frame")
        if role_counts["reference_image"] > 9:
            raise ValueError(
                "MiniMax-H3 content supports at most nine reference images"
            )
        if role_counts["reference_video"] > 3:
            raise ValueError(
                "MiniMax-H3 content supports at most three reference videos"
            )
        if role_counts["reference_audio"] > 3:
            raise ValueError(
                "MiniMax-H3 content supports at most three reference audio clips"
            )

        has_frames = bool(
            unroled_images or role_counts["first_frame"] or role_counts["last_frame"]
        )
        has_references = bool(
            role_counts["reference_image"]
            or role_counts["reference_video"]
            or role_counts["reference_audio"]
        )
        if has_frames and has_references:
            raise ValueError(
                "MiniMax-H3 frame inputs and reference inputs are mutually exclusive"
            )
        if role_counts["last_frame"] and not (
            unroled_images or role_counts["first_frame"]
        ):
            raise ValueError("MiniMax-H3 last_frame requires a first_frame")
        if role_counts["reference_audio"] and not (
            role_counts["reference_image"] or role_counts["reference_video"]
        ):
            raise ValueError(
                "MiniMax-H3 reference_audio requires a reference image or video"
            )

        return {
            "text_only": not has_frames and not has_references,
            "has_frames": has_frames,
            "has_references": has_references,
        }

    def _build_h3_request(
        self,
        *,
        content: List[Dict[str, Any]],
        duration: float,
        model: str = MINIMAX_H3_MODEL,
        resolution: str = "2K",
        ratio: Optional[str] = None,
        callback_url: Optional[str] = None,
        aigc_watermark: Optional[bool] = None,
    ) -> Dict[str, Any]:
        send_model = self._strip_prefix(model)
        if send_model != MINIMAX_H3_MODEL:
            raise ValueError(f"MiniMax v2 video generation requires {MINIMAX_H3_MODEL}")
        if isinstance(duration, bool) or not float(duration).is_integer():
            raise ValueError("MiniMax-H3 video duration must be a whole number")
        normalized_duration = int(duration)
        if not 4 <= normalized_duration <= 15:
            raise ValueError(
                "MiniMax-H3 video duration must be between 4 and 15 seconds"
            )
        normalized_resolution = str(resolution).upper()
        if normalized_resolution not in {"768P", "2K"}:
            raise ValueError("MiniMax-H3 video resolution must be 768P or 2K")

        content_info = self._validate_h3_content(content)
        if ratio is not None and ratio not in self.H3_RATIOS:
            raise ValueError(f"Unsupported MiniMax-H3 video ratio: {ratio}")
        if content_info["text_only"]:
            if ratio is None or ratio == "adaptive":
                raise ValueError(
                    "MiniMax-H3 text-to-video requires a non-adaptive ratio"
                )
            normalized_ratio = ratio
        elif content_info["has_frames"]:
            normalized_ratio = "adaptive"
        else:
            normalized_ratio = ratio or "adaptive"

        body: Dict[str, Any] = {
            "model": send_model,
            "content": content,
            "resolution": normalized_resolution,
            "duration": normalized_duration,
            "ratio": normalized_ratio,
        }
        if callback_url is not None:
            if not callback_url.strip():
                raise ValueError("MiniMax-H3 callback_url cannot be empty")
            body["callback_url"] = callback_url
        if aigc_watermark is not None:
            if not self._uses_cn_endpoint():
                raise ValueError(
                    "aigc_watermark is only supported by the China endpoint"
                )
            body["aigc_watermark"] = aigc_watermark

        body_size = len(
            json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if body_size > self.H3_MAX_REQUEST_BODY_BYTES:
            raise ValueError("MiniMax-H3 request body cannot exceed 64 MB")
        return body

    async def _request_v2(
        self,
        session: Any,
        method: str,
        path: str,
        operation: str,
        *,
        json_body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        api_key = self._require_api_key()
        request = getattr(session, method)
        kwargs: Dict[str, Any] = {
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        }
        if json_body is not None:
            kwargs["json"] = json_body
        if params:
            kwargs["params"] = params
        url = f"{self._base_url_for_version('v2')}{path}"
        async with request(url, **kwargs) as response:
            return await self._read_response(response, operation)

    async def create_video_task(
        self,
        *,
        content: List[Dict[str, Any]],
        duration: float,
        model: str = MINIMAX_H3_MODEL,
        resolution: str = "2K",
        ratio: Optional[str] = None,
        callback_url: Optional[str] = None,
        aigc_watermark: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Create a MiniMax-H3 v2 video generation task."""
        import aiohttp

        body = self._build_h3_request(
            content=content,
            duration=duration,
            model=model,
            resolution=resolution,
            ratio=ratio,
            callback_url=callback_url,
            aigc_watermark=aigc_watermark,
        )
        async with aiohttp.ClientSession(
            timeout=self._video_client_timeout()
        ) as session:
            return await self._request_v2(
                session,
                "post",
                "/video_generation",
                "create",
                json_body=body,
            )

    async def query_video_task(self, task_id: str) -> Dict[str, Any]:
        """Query a MiniMax-H3 v2 video generation task."""
        import aiohttp

        normalized_task_id = quote(self._normalize_task_id(task_id), safe="")
        async with aiohttp.ClientSession(
            timeout=self._video_client_timeout()
        ) as session:
            return await self._request_v2(
                session,
                "get",
                f"/query/video_generation/{normalized_task_id}",
                "query",
            )

    async def list_video_tasks(
        self,
        *,
        page_num: Optional[int] = None,
        page_size: Optional[int] = None,
        status: Optional[str] = None,
        task_ids: Optional[List[str]] = None,
        model: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List MiniMax-H3 v2 video generation tasks."""
        import aiohttp

        params: Dict[str, Any] = {}
        if page_num is not None:
            if page_num < 1:
                raise ValueError("page_num must be at least 1")
            params["page_num"] = page_num
        if page_size is not None:
            if page_size < 1:
                raise ValueError("page_size must be at least 1")
            params["page_size"] = page_size
        if status is not None:
            if status not in self.H3_TASK_STATUSES:
                raise ValueError(f"Unsupported MiniMax-H3 task status: {status}")
            params["filter.status"] = status
        if task_ids:
            params["filter.task_ids"] = task_ids
        if model is not None:
            params["filter.model"] = self._strip_prefix(model)
        if task_type is not None:
            if task_type not in self.H3_TASK_TYPES:
                raise ValueError(f"Unsupported MiniMax-H3 task type: {task_type}")
            params["filter.task_type"] = task_type

        async with aiohttp.ClientSession(
            timeout=self._video_client_timeout()
        ) as session:
            return await self._request_v2(
                session,
                "get",
                "/query/video_generation",
                "list",
                params=params,
            )

    async def delete_video_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel or delete a MiniMax-H3 v2 video generation task."""
        import aiohttp

        normalized_task_id = quote(self._normalize_task_id(task_id), safe="")
        async with aiohttp.ClientSession(
            timeout=self._video_client_timeout()
        ) as session:
            return await self._request_v2(
                session,
                "delete",
                f"/video_generation/{normalized_task_id}",
                "delete",
            )

    def _estimate_h3_cost_usd(
        self, usage: Dict[str, Any], resolution: str
    ) -> Optional[float]:
        if urlparse(self._base_url).hostname != "api.minimax.io":
            return None
        pricing = MINIMAX_H3_VIDEO_METADATA["pricing"]["global_en"]
        output_seconds = float(usage.get("output_seconds") or 0)
        input_seconds = float(usage.get("input_seconds") or 0)
        image_count = int(
            usage.get("input_image_count") or usage.get("image_count") or 0
        )
        normalized_resolution = str(resolution).upper()
        output_cost = (
            output_seconds * pricing["output_video"]["rates"][normalized_resolution]
        )
        input_video_cost = (
            input_seconds
            * pricing["input_reference_video"]["rates"][normalized_resolution]
        )
        image_cost = (
            max(image_count - pricing["input_reference_images"]["free_count"], 0)
            * pricing["input_reference_images"]["additional_image"]
        )
        return round(output_cost + input_video_cost + image_cost, 6)

    async def _generate_h3_video(
        self,
        *,
        prompt: str,
        content: Optional[List[Dict[str, Any]]],
        image_url: Optional[str],
        duration: Optional[float],
        resolution: Optional[str],
        ratio: Optional[str],
        callback_url: Optional[str],
        aigc_watermark: Optional[bool],
        extra: Optional[Dict[str, Any]],
        poll_interval: float,
        timeout: float,
        kwargs: Dict[str, Any],
    ) -> MultimodalResponse:
        import asyncio
        import time

        import aiohttp

        if duration is None:
            raise ValueError("MiniMax-H3 video generation requires duration")
        unsupported = set((extra or {})) | set(kwargs)
        if unsupported:
            raise ValueError(
                f"Unsupported MiniMax-H3 video request fields: {sorted(unsupported)}"
            )

        content_items = [dict(item) for item in (content or [])]
        if not any(item.get("type") == "text" for item in content_items):
            content_items.insert(0, {"type": "text", "text": prompt})
        if image_url:
            content_items.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                    "role": "first_frame",
                }
            )

        body = self._build_h3_request(
            content=content_items,
            duration=duration,
            resolution=resolution or "2K",
            ratio=ratio,
            callback_url=callback_url,
            aigc_watermark=aigc_watermark,
        )
        client_timeout = self._video_client_timeout()
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            submit_data = await self._request_v2(
                session,
                "post",
                "/video_generation",
                "create",
                json_body=body,
            )
            task_id = str(submit_data.get("task_id") or "")
            if not task_id:
                raise RuntimeError("MiniMax-H3 video create returned no task_id")

            encoded_task_id = quote(task_id, safe="")
            start_time = time.monotonic()
            query_data: Dict[str, Any] = {}
            task: Dict[str, Any] = {}
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"MiniMax-H3 video generation timed out after {timeout}s "
                        f"(task {task_id})"
                    )
                await asyncio.sleep(min(poll_interval, timeout - elapsed))
                query_data = await self._request_v2(
                    session,
                    "get",
                    f"/query/video_generation/{encoded_task_id}",
                    "query",
                )
                task = query_data.get("task") or {}
                status = str(task.get("status") or "").lower()
                if status == "succeeded":
                    break
                if status in {"failed", "cancelled", "expired"}:
                    error = task.get("error") or {}
                    detail = error.get("message") or status
                    raise RuntimeError(f"MiniMax-H3 video generation failed: {detail}")

        video_url = (task.get("content") or {}).get("url")
        if not video_url:
            raise RuntimeError("MiniMax-H3 video task succeeded without a content URL")
        _assert_safe_download_url(video_url)

        task_duration = task.get("duration") or body["duration"]
        task_resolution = task.get("resolution") or body["resolution"]
        task_ratio = task.get("ratio") or body["ratio"]
        usage = task.get("usage") or {}
        cost_usd = self._estimate_h3_cost_usd(usage, task_resolution)
        filename = "generated_video.mp4"
        file_output = FileOutput(
            url=video_url,
            data=None,
            mime_type="video/mp4",
            filename=filename,
        )
        video_output = VideoOutput(
            url=video_url,
            data=None,
            mime_type="video/mp4",
            filename=filename,
            duration=task_duration,
            resolution=task_resolution,
            aspect_ratio=task_ratio,
            has_audio=True,
            cost_usd=cost_usd,
        )
        return MultimodalResponse(
            text=prompt,
            audio=None,
            images=[],
            files=[file_output],
            videos=[video_output],
            raw_response={"submit": submit_data, "query": query_data},
            cost_usd=cost_usd,
            usage=usage,
            cost_source="minimax_h3_metadata" if cost_usd is not None else None,
        )

    async def generate_video(
        self,
        prompt: str,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        duration: Optional[float] = None,
        resolution: Optional[str] = None,
        content: Optional[List[Dict[str, Any]]] = None,
        ratio: Optional[str] = None,
        callback_url: Optional[str] = None,
        aigc_watermark: Optional[bool] = None,
        extra: Optional[Dict[str, Any]] = None,
        poll_interval: float = 10.0,
        timeout: float = 600.0,
        **kwargs,
    ) -> MultimodalResponse:
        """Submit, poll, and retrieve a MiniMax video generation task."""
        import asyncio
        import os
        import time

        import aiohttp

        api_key = self._api_key or os.environ.get("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MiniMax API key required. Set MINIMAX_API_KEY or pass api_key "
                "to MiniMaxProvider."
            )
        if poll_interval < 0:
            raise ValueError("poll_interval must be non-negative")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        send_model = self._strip_prefix(model or MINIMAX_H3_MODEL)
        if not send_model:
            raise ValueError("MiniMax video generation requires an explicit model")
        if send_model == MINIMAX_H3_MODEL:
            return await self._generate_h3_video(
                prompt=prompt,
                content=content,
                image_url=image_url,
                duration=duration,
                resolution=resolution,
                ratio=ratio,
                callback_url=callback_url,
                aigc_watermark=aigc_watermark,
                extra=extra,
                poll_interval=poll_interval,
                timeout=timeout,
                kwargs=kwargs,
            )
        if content is not None or any(
            value is not None for value in (ratio, callback_url, aigc_watermark)
        ):
            raise ValueError("MiniMax v2 request fields require the MiniMax-H3 model")

        body: Dict[str, Any] = {"model": send_model, "prompt": prompt}
        if image_url:
            body["first_frame_image"] = image_url
        if duration is not None:
            if not float(duration).is_integer():
                raise ValueError("MiniMax video duration must be a whole number")
            body["duration"] = int(duration)
        if resolution:
            body["resolution"] = resolution.upper()
        overrides = {**(extra or {}), **kwargs}
        # Validated/normalized above — merging them from extra/kwargs would
        # bypass those checks (e.g. a fractional duration or lowercase
        # resolution).
        reserved = {
            "model",
            "prompt",
            "first_frame_image",
            "duration",
            "resolution",
        } & overrides.keys()
        if reserved:
            raise ValueError(
                "extra/kwargs may not override validated request fields: "
                f"{sorted(reserved)}"
            )
        if overrides:
            body.update(overrides)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        submit_url = f"{self._base_url}/video_generation"
        query_url = f"{self._base_url}/query/video_generation"
        retrieve_url = f"{self._base_url}/files/retrieve"
        client_timeout = aiohttp.ClientTimeout(
            total=None,
            connect=30.0,
            sock_read=120.0,
        )

        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.post(submit_url, headers=headers, json=body) as response:
                submit_data = await self._read_response(response, "submit")

            task_id = str(submit_data.get("task_id") or "")
            if not task_id:
                raise RuntimeError("MiniMax video submit returned no task_id")

            start_time = time.monotonic()
            query_data: Dict[str, Any] = {}
            while True:
                elapsed = time.monotonic() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(
                        f"MiniMax video generation timed out after {timeout}s "
                        f"(task {task_id})"
                    )
                await asyncio.sleep(min(poll_interval, timeout - elapsed))

                async with session.get(
                    query_url,
                    headers=headers,
                    params={"task_id": task_id},
                ) as response:
                    query_data = await self._read_response(response, "query")

                status = str(query_data.get("status") or "").lower()
                if status == "success":
                    break
                if status in {"fail", "failed"}:
                    detail = (
                        query_data.get("error_message")
                        or (query_data.get("base_resp") or {}).get("status_msg")
                        or "unknown error"
                    )
                    raise RuntimeError(f"MiniMax video generation failed: {detail}")

            file_id = str(query_data.get("file_id") or "")
            if not file_id:
                raise RuntimeError("MiniMax video task succeeded without a file_id")

            async with session.get(
                retrieve_url,
                headers=headers,
                params={"file_id": file_id},
            ) as response:
                file_data = await self._read_response(response, "file retrieval")

        file_info = file_data.get("file") or {}
        video_url = file_info.get("download_url")
        if not video_url:
            raise RuntimeError("MiniMax file retrieval returned no download_url")
        _assert_safe_download_url(video_url)

        filename = file_info.get("filename") or "generated_video.mp4"
        normalized_resolution = resolution.upper() if resolution else None
        file_output = FileOutput(
            url=video_url,
            data=None,
            mime_type="video/mp4",
            filename=filename,
        )
        video_output = VideoOutput(
            url=video_url,
            data=None,
            mime_type="video/mp4",
            filename=filename,
            duration=duration,
            resolution=normalized_resolution,
        )
        return MultimodalResponse(
            text=prompt,
            audio=None,
            images=[],
            files=[file_output],
            videos=[video_output],
            raw_response={
                "submit": submit_data,
                "query": query_data,
                "file": file_data,
            },
        )


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
                {"type": "image_url", "image_url": {"url": url}} for url in image_urls
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
            images.append(ImageOutput(url=url, b64_json=b64_json, revised_prompt=None))

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
                        decoded = raw_line.decode("utf-8", errors="replace").strip()
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
                            transcript_parts.append(audio_delta["transcript"])

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
    "minimax": MiniMaxProvider,
    "openrouter": OpenRouterProvider,
}


def get_provider(name: str, **kwargs) -> MediaProvider:
    """
    Get a media provider instance by name.

    Args:
        name: Provider name ('fal', 'litellm', 'minimax', 'openrouter')
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
