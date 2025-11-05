import os
import re
import socket
import time
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, create_model


class AgentUtils:
    """Utility functions extracted from Agent class for better code organization."""

    @staticmethod
    def detect_input_type(input_data: Any) -> str:
        """Intelligently detect input type without explicit declarations"""

        if isinstance(input_data, str):
            # Smart string detection
            if input_data.startswith(("http://", "https://")):
                return "image_url" if AgentUtils.is_image_url(input_data) else "url"
            elif input_data.startswith("data:image"):
                return "image_base64"
            elif input_data.startswith("data:audio"):
                return "audio_base64"
            elif os.path.isfile(input_data):
                ext = os.path.splitext(input_data)[1].lower()
                if ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff"]:
                    return "image_file"
                elif ext in [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"]:
                    return "audio_file"
                elif ext in [".pdf", ".doc", ".docx", ".txt", ".rtf", ".md"]:
                    return "document_file"
                elif ext in [".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm"]:
                    return "video_file"
                else:
                    return "file"
            return "text"

        elif isinstance(input_data, bytes):
            # Detect file type from bytes
            if input_data.startswith(b"\xff\xd8\xff"):  # JPEG
                return "image_bytes"
            elif input_data.startswith(b"\x89PNG"):  # PNG
                return "image_bytes"
            elif input_data.startswith(b"GIF8"):  # GIF
                return "image_bytes"
            elif input_data.startswith(b"RIFF") and b"WAVE" in input_data[:12]:  # WAV
                return "audio_bytes"
            elif input_data.startswith(b"ID3") or input_data.startswith(
                b"\xff\xfb"
            ):  # MP3
                return "audio_bytes"
            elif b"ftyp" in input_data[:20]:  # MP4/M4A
                return "audio_bytes"
            elif input_data.startswith(b"%PDF"):  # PDF
                return "document_bytes"
            return "binary_data"

        elif isinstance(input_data, dict):
            # Check for structured input patterns
            if any(
                key in input_data for key in ["system", "user", "assistant", "role"]
            ):
                return "message_dict"
            elif any(
                key in input_data
                for key in ["image", "image_url", "audio", "file", "text"]
            ):
                return "structured_input"
            return "dict"

        elif isinstance(input_data, list):
            if len(input_data) > 0:
                # Check if it's a conversation format
                if isinstance(input_data[0], dict) and "role" in input_data[0]:
                    return "conversation_list"
                # Check if it's multimodal content
                elif any(isinstance(item, (str, dict)) for item in input_data):
                    return "multimodal_list"
            return "list"

        return "unknown"

    @staticmethod
    def is_image_url(url: str) -> bool:
        """Check if URL points to an image based on extension or content type"""
        image_extensions = [
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".webp",
            ".bmp",
            ".tiff",
            ".svg",
        ]
        return any(url.lower().endswith(ext) for ext in image_extensions)

    @staticmethod
    def is_audio_url(url: str) -> bool:
        """Check if URL points to audio based on extension"""
        audio_extensions = [".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"]
        return any(url.lower().endswith(ext) for ext in audio_extensions)

    @staticmethod
    def get_mime_type(extension: str) -> str:
        """Get MIME type from file extension"""
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".svg": "image/svg+xml",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".aac": "audio/aac",
            ".pdf": "application/pdf",
            ".doc": "application/msword",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".rtf": "application/rtf",
        }
        return mime_types.get(extension.lower(), "application/octet-stream")

    @staticmethod
    def map_json_type_to_python(json_type: str) -> Type:
        """
        Map JSON Schema types to Python types.

        Args:
            json_type: JSON Schema type string

        Returns:
            Python type
        """
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": List[Any],
            "object": Dict[str, Any],
            "null": type(None),
        }

        return type_mapping.get(json_type, str)

    @staticmethod
    def generate_skill_name(server_alias: str, tool_name: str) -> str:
        """
        Generate a valid Python function name for the MCP skill.

        Args:
            server_alias: MCP server alias
            tool_name: MCP tool name

        Returns:
            Valid Python function name
        """
        # Convert to snake_case and ensure it's a valid Python identifier
        name = f"{server_alias}_{tool_name}"
        name = re.sub(
            r"[^a-zA-Z0-9_]", "_", name
        )  # Replace invalid chars with underscore
        name = re.sub(r"_+", "_", name)  # Replace multiple underscores with single
        name = name.strip("_")  # Remove leading/trailing underscores

        # Ensure it starts with a letter or underscore
        if name and name[0].isdigit():
            name = "_" + name

        # Ensure it's not empty
        if not name:
            name = f"mcp_tool_{int(time.time())}"

        return name

    @staticmethod
    def create_input_schema_from_mcp_tool(
        skill_name: str, tool: Dict[str, Any]
    ) -> Type[BaseModel]:
        """
        Create a Pydantic input schema from MCP tool definition.

        Args:
            skill_name: Name of the skill function
            tool: MCP tool definition

        Returns:
            Pydantic model class for input validation
        """
        input_schema = tool.get("input_schema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        # Create fields for Pydantic model
        fields = {}

        for prop_name, prop_def in properties.items():
            prop_type = AgentUtils.map_json_type_to_python(
                prop_def.get("type", "string")
            )
            is_required = prop_name in required

            if is_required:
                fields[prop_name] = (prop_type, ...)
            else:
                default_value = prop_def.get("default")
                if default_value is not None:
                    fields[prop_name] = (prop_type, default_value)
                else:
                    fields[prop_name] = (Optional[prop_type], None)

        # If no fields defined, create a generic schema
        if not fields:
            fields["data"] = (Optional[Dict[str, Any]], None)

        # Create the Pydantic model
        InputModel = create_model(f"{skill_name}Input", **fields)
        return InputModel

    @staticmethod
    def is_port_available(port: int) -> bool:
        """
        Check if a port is available for use.

        Args:
            port: Port number to check

        Returns:
            True if port is available, False otherwise
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return True
        except OSError:
            return False

    @staticmethod
    def serialize_result(result: Any) -> Any:
        """Convert complex objects to JSON-serializable format"""
        try:
            if hasattr(result, "model_dump"):  # Pydantic v2
                return result.model_dump()
            elif hasattr(result, "dict"):  # Pydantic v1
                return result.model_dump()
            elif hasattr(result, "__dict__"):  # Regular objects with attributes
                return {
                    k: AgentUtils.serialize_result(v)
                    for k, v in result.__dict__.items()
                }
            elif isinstance(result, (list, tuple)):
                return [AgentUtils.serialize_result(item) for item in result]
            elif isinstance(result, dict):
                return {k: AgentUtils.serialize_result(v) for k, v in result.items()}
            else:
                # Primitive types (str, int, float, bool, None) are already JSON-serializable
                return result
        except Exception:
            # Fallback: convert to string if serialization fails
            return str(result)
