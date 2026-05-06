"""
Multimodal response classes for handling LiteLLM multimodal outputs.
Provides seamless integration with audio, image, and file outputs while maintaining backward compatibility.
"""

import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agentfield.logger import log_error, log_warn
from pydantic import BaseModel, Field


class AudioOutput(BaseModel):
    """Represents audio output from LLM with convenient access methods."""

    data: Optional[str] = Field(None, description="Base64-encoded audio data")
    format: str = Field("wav", description="Audio format (wav, mp3, etc.)")
    url: Optional[str] = Field(None, description="URL to audio file if available")

    def save(self, path: Union[str, Path]) -> None:
        """Save audio to file."""
        if not self.data:
            raise ValueError("No audio data available to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Decode base64 audio data
        audio_bytes = base64.b64decode(self.data)

        with open(path, "wb") as f:
            f.write(audio_bytes)

    def get_bytes(self) -> bytes:
        """Get raw audio bytes."""
        if not self.data:
            raise ValueError("No audio data available")
        return base64.b64decode(self.data)

    def play(self) -> None:
        """Play audio if possible (requires system audio support)."""
        try:
            import pygame  # type: ignore

            pygame.mixer.init()

            # Create temporary file
            with tempfile.NamedTemporaryFile(
                suffix=f".{self.format}", delete=False
            ) as tmp:
                tmp.write(self.get_bytes())
                tmp_path = tmp.name

            pygame.mixer.music.load(tmp_path)
            pygame.mixer.music.play()

            # Clean up temp file after a delay
            import threading
            import time

            def cleanup():
                time.sleep(5)  # Wait for playback
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

            threading.Thread(target=cleanup, daemon=True).start()

        except ImportError:
            log_warn("Audio playback requires pygame: pip install pygame")
        except Exception as e:
            log_error(f"Could not play audio: {e}")


class ImageOutput(BaseModel):
    """Represents image output from LLM with convenient access methods."""

    url: Optional[str] = Field(None, description="URL to image")
    b64_json: Optional[str] = Field(None, description="Base64-encoded image data")
    revised_prompt: Optional[str] = Field(
        None, description="Revised prompt used for generation"
    )

    def save(self, path: Union[str, Path]) -> None:
        """Save image to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.b64_json:
            # Save from base64 data
            image_bytes = base64.b64decode(self.b64_json)
            with open(path, "wb") as f:
                f.write(image_bytes)
        elif self.url:
            # Download from URL
            try:
                import requests

                response = requests.get(self.url)
                response.raise_for_status()
                with open(path, "wb") as f:
                    f.write(response.content)
            except ImportError:
                raise ImportError(
                    "URL download requires requests: pip install requests"
                )
        else:
            raise ValueError("No image data or URL available to save")

    def get_bytes(self) -> bytes:
        """Get raw image bytes."""
        if self.b64_json:
            return base64.b64decode(self.b64_json)
        elif self.url:
            try:
                import requests

                response = requests.get(self.url)
                response.raise_for_status()
                return response.content
            except ImportError:
                raise ImportError(
                    "URL download requires requests: pip install requests"
                )
        else:
            raise ValueError("No image data or URL available")

    def show(self) -> None:
        """Display image if possible (requires PIL/Pillow)."""
        try:
            from PIL import Image  # type: ignore
            import io

            image_bytes = self.get_bytes()
            image = Image.open(io.BytesIO(image_bytes))
            image.show()
        except ImportError:
            log_warn("Image display requires Pillow: pip install Pillow")
        except Exception as e:
            log_error(f"Could not display image: {e}")


class FileOutput(BaseModel):
    """Represents generic file output from LLM."""

    url: Optional[str] = Field(None, description="URL to file")
    data: Optional[str] = Field(None, description="Base64-encoded file data")
    mime_type: Optional[str] = Field(None, description="MIME type of file")
    filename: Optional[str] = Field(None, description="Suggested filename")

    def save(self, path: Union[str, Path]) -> None:
        """Save file to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.data:
            # Save from base64 data
            file_bytes = base64.b64decode(self.data)
            with open(path, "wb") as f:
                f.write(file_bytes)
        elif self.url:
            # Download from URL
            try:
                import requests

                response = requests.get(self.url)
                response.raise_for_status()
                with open(path, "wb") as f:
                    f.write(response.content)
            except ImportError:
                raise ImportError(
                    "URL download requires requests: pip install requests"
                )
        else:
            raise ValueError("No file data or URL available to save")

    def get_bytes(self) -> bytes:
        """Get raw file bytes."""
        if self.data:
            return base64.b64decode(self.data)
        elif self.url:
            try:
                import requests

                response = requests.get(self.url)
                response.raise_for_status()
                return response.content
            except ImportError:
                raise ImportError(
                    "URL download requires requests: pip install requests"
                )
        else:
            raise ValueError("No file data or URL available")


class VideoOutput(BaseModel):
    """Represents video output from generation models."""

    url: Optional[str] = Field(None, description="URL to video file")
    data: Optional[str] = Field(None, description="Base64-encoded video data")
    mime_type: str = Field("video/mp4", description="MIME type")
    filename: Optional[str] = Field(None, description="Suggested filename")
    duration: Optional[float] = Field(None, description="Duration in seconds")
    resolution: Optional[str] = Field(None, description="Resolution (e.g., '1080p')")
    aspect_ratio: Optional[str] = Field(None, description="Aspect ratio (e.g., '16:9')")
    has_audio: Optional[bool] = Field(None, description="Whether video has audio track")
    cost_usd: Optional[float] = Field(None, description="Generation cost in USD")

    def save(self, path: Union[str, Path]) -> None:
        """Save video to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if self.data:
            video_bytes = base64.b64decode(self.data)
            with open(path, "wb") as f:
                f.write(video_bytes)
        elif self.url:
            try:
                import requests

                response = requests.get(self.url, timeout=120)
                response.raise_for_status()
                with open(path, "wb") as f:
                    f.write(response.content)
            except ImportError:
                raise ImportError(
                    "URL download requires requests: pip install requests"
                )
        else:
            raise ValueError("No video data or URL available to save")

    def get_bytes(self) -> bytes:
        """Get raw video bytes."""
        if self.data:
            return base64.b64decode(self.data)
        elif self.url:
            try:
                import requests

                response = requests.get(self.url, timeout=120)
                response.raise_for_status()
                return response.content
            except ImportError:
                raise ImportError(
                    "URL download requires requests: pip install requests"
                )
        else:
            raise ValueError("No video data or URL available")


class MultimodalResponse:
    """
    Enhanced response object that provides seamless access to multimodal content
    while maintaining backward compatibility with string responses.
    """

    def __init__(
        self,
        text: str = "",
        audio: Optional[AudioOutput] = None,
        images: Optional[List[ImageOutput]] = None,
        files: Optional[List[FileOutput]] = None,
        videos: Optional[List["VideoOutput"]] = None,
        raw_response: Optional[Any] = None,
        cost_usd: Optional[float] = None,
        usage: Optional[Dict[str, int]] = None,
    ):
        self._text = text
        self._audio = audio
        self._images = images or []
        self._files = files or []
        self._videos = videos or []
        self._raw_response = raw_response
        self._cost_usd = cost_usd
        self._usage = usage or {}

    def __str__(self) -> str:
        """Backward compatibility: return text content when used as string."""
        return self._text

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        parts = [f"text='{self._text[:50]}{'...' if len(self._text) > 50 else ''}'"]
        if self._audio:
            parts.append(f"audio={self._audio.format}")
        if self._images:
            parts.append(f"images={len(self._images)}")
        if self._videos:
            parts.append(f"videos={len(self._videos)}")
        if self._files:
            parts.append(f"files={len(self._files)}")
        if self._videos:
            parts.append(f"videos={len(self._videos)}")
        return f"MultimodalResponse({', '.join(parts)})"

    @property
    def text(self) -> str:
        """Get text content."""
        return self._text

    @property
    def audio(self) -> Optional[AudioOutput]:
        """Get audio output if available."""
        return self._audio

    @property
    def images(self) -> List[ImageOutput]:
        """Get list of image outputs."""
        return self._images

    @property
    def files(self) -> List[FileOutput]:
        """Get list of file outputs."""
        return self._files

    @property
    def videos(self) -> List["VideoOutput"]:
        """Get list of video outputs."""
        return self._videos

    @property
    def has_audio(self) -> bool:
        """Check if response contains audio."""
        return self._audio is not None

    @property
    def has_images(self) -> bool:
        """Check if response contains images."""
        return len(self._images) > 0

    @property
    def has_files(self) -> bool:
        """Check if response contains files."""
        return len(self._files) > 0

    @property
    def has_videos(self) -> bool:
        """Check if response contains videos."""
        return len(self._videos) > 0

    @property
    def is_multimodal(self) -> bool:
        """Check if response contains any multimodal content."""
        return self.has_audio or self.has_images or self.has_files or self.has_videos

    @property
    def raw_response(self) -> Optional[Any]:
        """Get the raw LiteLLM response object."""
        return self._raw_response

    @property
    def cost_usd(self) -> Optional[float]:
        """Estimated cost of this LLM call in USD, if available."""
        return self._cost_usd

    @property
    def usage(self) -> Dict[str, int]:
        """Token usage breakdown (prompt_tokens, completion_tokens, total_tokens)."""
        return self._usage

    def save_all(
        self, directory: Union[str, Path], prefix: str = "output"
    ) -> Dict[str, str]:
        """
        Save all multimodal content to a directory.
        Returns a dict mapping content type to saved file paths.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        saved_files = {}

        # Save audio
        if self._audio:
            audio_path = directory / f"{prefix}_audio.{self._audio.format}"
            self._audio.save(audio_path)
            saved_files["audio"] = str(audio_path)

        # Save images
        for i, image in enumerate(self._images):
            # Determine extension from URL or default to png
            ext = "png"
            if image.url:
                ext = Path(image.url).suffix.lstrip(".") or "png"

            image_path = directory / f"{prefix}_image_{i}.{ext}"
            image.save(image_path)
            saved_files[f"image_{i}"] = str(image_path)

        # Save videos
        for i, video in enumerate(self._videos):
            ext = video.mime_type.split("/")[-1] if video.mime_type else "mp4"
            raw_filename = video.filename or f"{prefix}_video_{i}.{ext}"
            safe_filename = os.path.basename(raw_filename)  # Strip path components
            video_path = directory / safe_filename
            video.save(video_path)
            saved_files[f"video_{i}"] = str(video_path)

        # Save files
        for i, file in enumerate(self._files):
            # Skip video files — they're saved in the videos loop
            if file.mime_type and file.mime_type.startswith("video/"):
                continue
            raw_filename = file.filename or f"{prefix}_file_{i}"
            safe_filename = os.path.basename(raw_filename)  # Strip path components
            file_path = directory / safe_filename
            file.save(file_path)
            saved_files[f"file_{i}"] = str(file_path)

        # Save text content
        if self._text:
            text_path = directory / f"{prefix}_text.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(self._text)
            saved_files["text"] = str(text_path)

        return saved_files


def _extract_image_from_data(data: Any) -> Optional[ImageOutput]:
    """
    Extract an ImageOutput from various data structures.
    Handles multiple formats: OpenRouter, OpenAI, and generic patterns.
    """
    if data is None:
        return None

    # Direct url/b64_json attributes (standard image generation)
    if hasattr(data, "url") or hasattr(data, "b64_json"):
        url = getattr(data, "url", None)
        b64 = getattr(data, "b64_json", None)
        if url or b64:
            return ImageOutput(
                url=url,
                b64_json=b64,
                revised_prompt=getattr(data, "revised_prompt", None),
            )

    # OpenRouter/Gemini pattern: {"type": "image_url", "image_url": {"url": "..."}}
    if hasattr(data, "image_url"):
        image_url_obj = data.image_url
        url = (
            getattr(image_url_obj, "url", None)
            if hasattr(image_url_obj, "url")
            else None
        )
        if url:
            # Handle data URLs (base64 encoded)
            if url.startswith("data:image"):
                # Extract base64 from data URL
                try:
                    b64_data = url.split(",", 1)[1] if "," in url else None
                    return ImageOutput(url=url, b64_json=b64_data, revised_prompt=None)
                except Exception:
                    return ImageOutput(url=url, b64_json=None, revised_prompt=None)
            return ImageOutput(url=url, b64_json=None, revised_prompt=None)

    # Dict-based patterns
    if isinstance(data, dict):
        # Direct url/b64_json keys
        if "url" in data or "b64_json" in data:
            url = data.get("url")
            b64 = data.get("b64_json")
            if url or b64:
                return ImageOutput(
                    url=url,
                    b64_json=b64,
                    revised_prompt=data.get("revised_prompt"),
                )

        # OpenRouter dict pattern: {"image_url": {"url": "..."}}
        if "image_url" in data:
            image_url_data = data["image_url"]
            if isinstance(image_url_data, dict):
                url = image_url_data.get("url")
                if url:
                    # Handle data URLs
                    if url.startswith("data:image"):
                        try:
                            b64_data = url.split(",", 1)[1] if "," in url else None
                            return ImageOutput(
                                url=url, b64_json=b64_data, revised_prompt=None
                            )
                        except Exception:
                            return ImageOutput(
                                url=url, b64_json=None, revised_prompt=None
                            )
                    return ImageOutput(url=url, b64_json=None, revised_prompt=None)

    return None


def _find_images_recursive(obj: Any, max_depth: int = 10) -> List[ImageOutput]:
    """
    Recursively search any structure for image data.
    This is a generalized fallback that handles unknown response formats.
    """
    if max_depth <= 0:
        return []

    images = []

    # Try direct extraction first
    img = _extract_image_from_data(obj)
    if img:
        images.append(img)
        return images  # Found at this level, don't recurse deeper

    # Handle lists/tuples
    if isinstance(obj, (list, tuple)):
        for item in obj:
            images.extend(_find_images_recursive(item, max_depth - 1))

    # Handle dicts
    elif isinstance(obj, dict):
        for value in obj.values():
            images.extend(_find_images_recursive(value, max_depth - 1))

    # Handle objects with attributes
    elif hasattr(obj, "__dict__"):
        for attr_name in dir(obj):
            if attr_name.startswith("_"):
                continue
            try:
                attr_val = getattr(obj, attr_name, None)
                if attr_val is not None and not callable(attr_val):
                    images.extend(_find_images_recursive(attr_val, max_depth - 1))
            except Exception:
                continue

    return images


def detect_multimodal_response(response: Any) -> MultimodalResponse:
    """
    Automatically detect and wrap multimodal content from LiteLLM responses.

    Args:
        response: Raw response from LiteLLM (completion or image_generation)

    Returns:
        MultimodalResponse with detected content
    """
    text = ""
    audio = None
    images = []
    files = []

    # Handle completion responses (text + potential audio + potential images)
    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        message = choice.message

        # Extract text content
        if hasattr(message, "content") and message.content:
            text = message.content

        # Extract audio content (GPT-4o-audio-preview pattern)
        if hasattr(message, "audio") and message.audio:
            audio_data = getattr(message.audio, "data", None)
            if audio_data:
                audio = AudioOutput(
                    data=audio_data,
                    format="wav",  # Default format, could be detected from response
                    url=None,
                )

        # Extract images from completion responses (OpenRouter/Gemini pattern)
        if hasattr(message, "images") and message.images:
            for img_data in message.images:
                img = _extract_image_from_data(img_data)
                if img:
                    images.append(img)

    # Handle image generation responses
    elif hasattr(response, "data") and response.data:
        # This is likely an image generation response
        for item in response.data:
            if hasattr(item, "url") or hasattr(item, "b64_json"):
                image = ImageOutput(
                    url=getattr(item, "url", None),
                    b64_json=getattr(item, "b64_json", None),
                    revised_prompt=getattr(item, "revised_prompt", None),
                )
                images.append(image)

    # Handle direct string responses
    elif isinstance(response, str):
        text = response

    # Handle TTS audio responses (from our _generate_tts_audio method)
    elif hasattr(response, "audio_data") and hasattr(response, "text"):
        text = response.text
        # Create AudioOutput from TTS response
        audio = AudioOutput(
            data=response.audio_data,
            format=getattr(response, "format", "wav"),
            url=None,
        )

    # Handle schema responses (Pydantic models)
    elif hasattr(response, "model_dump") or hasattr(response, "dict"):
        # This is a Pydantic model, convert to string representation
        try:
            if hasattr(response, "model_dump"):
                text = json.dumps(response.model_dump(), indent=2)
            else:
                text = json.dumps(response.model_dump(), indent=2)
        except Exception:
            text = str(response)

    # Fallback to string conversion
    else:
        text = str(response)

    # Fallback: if no images found yet, try recursive search
    # This catches edge cases where images are in unexpected locations
    if not images:
        images = _find_images_recursive(response, max_depth=5)

    # Extract usage and estimate cost from response
    usage_dict: Dict[str, int] = {}
    cost_usd: Optional[float] = None
    if hasattr(response, "usage") and response.usage:
        u = response.usage
        usage_dict = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
        }
    if usage_dict and hasattr(response, "model") and response.model:
        try:
            import litellm as _litellm

            cost_usd = _litellm.completion_cost(completion_response=response)
        except Exception:
            pass

    return MultimodalResponse(
        text=text,
        audio=audio,
        images=images,
        files=files,
        videos=[],
        raw_response=response,
        cost_usd=cost_usd,
        usage=usage_dict,
    )
