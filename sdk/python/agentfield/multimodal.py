import base64
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class Text(BaseModel):
    """Represents text content in a multimodal prompt."""

    type: Literal["text"] = "text"
    text: str = Field(..., description="The text content.")


class Image(BaseModel):
    """Represents image content in a multimodal prompt."""

    type: Literal["image_url"] = "image_url"
    image_url: Union[str, dict] = Field(
        ...,
        description="The URL of the image, or a dictionary with 'url' and optional 'detail' (e.g., {'url': 'https://example.com/image.jpg', 'detail': 'high'}).",
    )

    @classmethod
    def from_file(cls, file_path: Union[str, Path], detail: str = "high") -> "Image":
        """Create Image from local file by converting to base64 data URL."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Read and encode image
        with open(file_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        # Determine MIME type from extension
        ext = file_path.suffix.lower()
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(ext, "image/jpeg")

        data_url = f"data:{mime_type};base64,{image_data}"
        return cls(image_url={"url": data_url, "detail": detail})

    @classmethod
    def from_url(cls, url: str, detail: str = "high") -> "Image":
        """Create Image from URL."""
        return cls(image_url={"url": url, "detail": detail})


class Audio(BaseModel):
    """Represents audio content in a multimodal prompt."""

    type: Literal["input_audio"] = "input_audio"
    input_audio: dict = Field(
        ..., description="Audio input data with 'data' (base64) and 'format' fields."
    )

    @classmethod
    def from_file(
        cls, file_path: Union[str, Path], format: Optional[str] = None
    ) -> "Audio":
        """Create Audio from local file by converting to base64."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        # Auto-detect format from extension if not provided
        if format is None:
            ext = file_path.suffix.lower().lstrip(".")
            format = ext if ext in ["wav", "mp3", "flac", "ogg"] else "wav"

        # Read and encode audio
        with open(file_path, "rb") as f:
            audio_data = base64.b64encode(f.read()).decode()

        return cls(input_audio={"data": audio_data, "format": format})

    @classmethod
    def from_url(cls, url: str, format: str = "wav") -> "Audio":
        """Create Audio from URL (downloads and converts to base64)."""
        try:
            import requests

            response = requests.get(url)
            response.raise_for_status()
            audio_data = base64.b64encode(response.content).decode()
            return cls(input_audio={"data": audio_data, "format": format})
        except ImportError:
            raise ImportError("URL download requires requests: pip install requests")


class File(BaseModel):
    """Represents a generic file content in a multimodal prompt."""

    type: Literal["file"] = "file"
    file: Union[str, dict] = Field(
        ...,
        description="The URL of the file, or a dictionary with 'url' and optional 'mime_type'.",
    )

    @classmethod
    def from_file(
        cls, file_path: Union[str, Path], mime_type: Optional[str] = None
    ) -> "File":
        """Create File from local file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect MIME type if not provided
        if mime_type is None:
            import mimetypes

            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"

        # For now, just store the file path - could be enhanced to base64 encode
        return cls(
            file={"url": f"file://{file_path.absolute()}", "mime_type": mime_type}
        )

    @classmethod
    def from_url(cls, url: str, mime_type: Optional[str] = None) -> "File":
        """Create File from URL."""
        return cls(file={"url": url, "mime_type": mime_type})


# Union type for all multimodal content types
MultimodalContent = Union[Text, Image, Audio, File]


# Convenience functions for creating multimodal content
def text(content: str) -> Text:
    """Create text content."""
    return Text(text=content)


def image_from_file(file_path: Union[str, Path], detail: str = "high") -> Image:
    """Create image content from local file."""
    return Image.from_file(file_path, detail)


def image_from_url(url: str, detail: str = "high") -> Image:
    """Create image content from URL."""
    return Image.from_url(url, detail)


def audio_from_file(file_path: Union[str, Path], format: Optional[str] = None) -> Audio:
    """Create audio content from local file."""
    return Audio.from_file(file_path, format)


def audio_from_url(url: str, format: str = "wav") -> Audio:
    """Create audio content from URL."""
    return Audio.from_url(url, format)


def file_from_path(
    file_path: Union[str, Path], mime_type: Optional[str] = None
) -> File:
    """Create file content from local file."""
    return File.from_file(file_path, mime_type)


def file_from_url(url: str, mime_type: Optional[str] = None) -> File:
    """Create file content from URL."""
    return File.from_url(url, mime_type)
