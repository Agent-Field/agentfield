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
        raw_response: Optional[Any] = None,
    ):
        self._text = text
        self._audio = audio
        self._images = images or []
        self._files = files or []
        self._raw_response = raw_response

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
        if self._files:
            parts.append(f"files={len(self._files)}")
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
    def is_multimodal(self) -> bool:
        """Check if response contains any multimodal content."""
        return self.has_audio or self.has_images or self.has_files

    @property
    def raw_response(self) -> Optional[Any]:
        """Get the raw LiteLLM response object."""
        return self._raw_response

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

        # Save files
        for i, file in enumerate(self._files):
            filename = file.filename or f"{prefix}_file_{i}"
            file_path = directory / filename
            file.save(file_path)
            saved_files[f"file_{i}"] = str(file_path)

        # Save text content
        if self._text:
            text_path = directory / f"{prefix}_text.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(self._text)
            saved_files["text"] = str(text_path)

        return saved_files


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

    # Handle completion responses (text + potential audio)
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

    return MultimodalResponse(
        text=text, audio=audio, images=images, files=files, raw_response=response
    )
