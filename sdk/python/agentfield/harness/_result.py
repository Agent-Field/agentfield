from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FailureType(str, Enum):
    """Classifies how a harness invocation failed.

    Providers set this on RawResult so the runner can decide retry strategy:
    - ``none``: No failure.
    - ``crash``: Process killed by signal or non-zero exit with no output.
    - ``timeout``: Execution exceeded the time limit.
    - ``api_error``: Transient API-level error (rate limit, 5xx, etc.).
    - ``no_output``: Process exited OK but produced no output file.
    - ``schema``: Output file exists but fails schema validation.
    """

    NONE = "none"
    CRASH = "crash"
    TIMEOUT = "timeout"
    API_ERROR = "api_error"
    NO_OUTPUT = "no_output"
    SCHEMA = "schema"


@dataclass
class Metrics:
    duration_ms: int = 0
    duration_api_ms: int = 0
    num_turns: int = 0
    total_cost_usd: Optional[float] = None
    usage: Optional[Dict[str, Any]] = None
    session_id: str = ""


@dataclass
class RawResult:
    result: Optional[str] = None
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    is_error: bool = False
    error_message: Optional[str] = None
    failure_type: FailureType = FailureType.NONE
    returncode: Optional[int] = None


@dataclass
class HarnessResult:
    result: Optional[str] = None
    parsed: Any = None
    is_error: bool = False
    error_message: Optional[str] = None
    failure_type: FailureType = FailureType.NONE
    cost_usd: Optional[float] = None
    num_turns: int = 0
    duration_ms: int = 0
    session_id: str = ""
    messages: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        if self.result:
            return self.result
        return ""
