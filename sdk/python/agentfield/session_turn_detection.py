"""Validated OpenAI Realtime turn detection options."""

import math
from typing import Any, Dict, Literal, Optional, TypedDict, Union, cast


class _ServerVADType(TypedDict):
    type: Literal["server_vad"]


class ServerVAD(_ServerVADType, total=False):
    threshold: float
    prefix_padding_ms: int
    silence_duration_ms: int
    create_response: bool
    interrupt_response: bool


class _SemanticVADType(TypedDict):
    type: Literal["semantic_vad"]


class SemanticVAD(_SemanticVADType, total=False):
    eagerness: Literal["auto", "low", "medium", "high"]
    create_response: bool
    interrupt_response: bool


TurnDetection = Union[ServerVAD, SemanticVAD]


def normalize_turn_detection(
    provider: str, transport: str, config: Optional[TurnDetection]
) -> Optional[TurnDetection]:
    """Default to interruptible server VAD; never infer a supplied config's type."""
    if provider != "openai" or transport not in ("webrtc", "websocket"):
        if config is not None:
            raise ValueError("turn_detection requires provider=openai and transport=webrtc or websocket")
        return None
    if config is None:
        config = {"type": "server_vad"}
    if not isinstance(config, dict):
        raise ValueError("turn_detection must be an object")
    kind = config.get("type")
    common = {"type", "create_response", "interrupt_response"}
    if kind == "server_vad":
        allowed = common | {"threshold", "prefix_padding_ms", "silence_duration_ms"}
    elif kind == "semantic_vad":
        allowed = common | {"eagerness"}
    else:
        raise ValueError("turn_detection.type must be server_vad or semantic_vad")
    for key, value in config.items():
        if key not in allowed:
            raise ValueError(f"turn_detection.{key} is unsupported for {kind}")
        if key in ("create_response", "interrupt_response") and type(value) is not bool:
            raise ValueError(f"turn_detection.{key} must be a boolean")
        if key == "threshold" and (
            type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value)
        ):
            raise ValueError("turn_detection.threshold must be a finite number between 0 and 1")
        if key in ("prefix_padding_ms", "silence_duration_ms") and (
            type(value) is not int or value < 0
        ):
            raise ValueError(f"turn_detection.{key} must be a non-negative integer")
        if key == "eagerness" and value not in ("auto", "low", "medium", "high"):
            raise ValueError("turn_detection.eagerness must be auto, low, medium, or high")
    defaults: Dict[str, Any] = {"create_response": True, "interrupt_response": True}
    if kind == "server_vad":
        defaults.update(threshold=0.5, prefix_padding_ms=300, silence_duration_ms=500)
    else:
        defaults["eagerness"] = "auto"
    return cast(TurnDetection, {**defaults, **config})
