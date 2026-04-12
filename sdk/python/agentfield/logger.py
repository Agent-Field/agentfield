"""
AgentField SDK Logging Utility

This module provides a centralized logging system for the AgentField SDK that:
- Replaces print statements with proper logging
- Provides configurable log levels for human-readable logs
- Emits structured execution logs when execution context is available
- Preserves stdout mirroring for local developer ergonomics
"""

import asyncio
import json
import logging
import os
import sys
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional

from .execution_context import ExecutionContext, get_current_context

if TYPE_CHECKING:
    from .client import AgentFieldClient


_DEFAULT_STRUCTURED_SOURCE = "sdk.python.logger"
_LEVEL_TO_LOGGING = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class LogLevel(Enum):
    """Log levels for AgentField SDK"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    WARNING = "WARNING"
    ERROR = "ERROR"


class AgentFieldLogger:
    """
    Centralized logger for AgentField SDK with configurable verbosity and payload truncation.

    Structured execution logs bypass the human log-level filter so that execution telemetry
    remains available even when console verbosity is reduced.
    """

    _cp_client: Optional["AgentFieldClient"] = None

    def __init__(self, name: str = "agentfield"):
        self.logger = logging.getLogger(name)
        self._setup_logger()

        # Configuration from environment variables - default to WARNING (only important events)
        self.log_level = os.getenv("AGENTFIELD_LOG_LEVEL", "WARNING").upper()
        self.truncate_length = int(os.getenv("AGENTFIELD_LOG_TRUNCATE", "200"))
        self.show_payloads = (
            os.getenv("AGENTFIELD_LOG_PAYLOADS", "false").lower() == "true"
        )
        self.show_tracking = (
            os.getenv("AGENTFIELD_LOG_TRACKING", "false").lower() == "true"
        )
        self.show_fire = os.getenv("AGENTFIELD_LOG_FIRE", "false").lower() == "true"

        # Set logger level based on configuration
        self.logger.setLevel(_LEVEL_TO_LOGGING.get(self.log_level, logging.WARNING))

    def set_level(self, level: str):
        """Set log level at runtime (e.g., 'DEBUG', 'INFO', 'WARN', 'ERROR')"""

        self.logger.setLevel(_LEVEL_TO_LOGGING.get(level.upper(), logging.INFO))

    def _setup_logger(self):
        """Setup logger with console handler if not already configured"""

        if not self.logger.handlers:
            handler = logging.StreamHandler(stream=sys.stdout)
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def _truncate_message(self, message: str) -> str:
        """Truncate message if it exceeds the configured length"""

        if len(message) <= self.truncate_length:
            return message
        return message[: self.truncate_length] + "..."

    def _format_payload(self, payload: Any) -> str:
        """Format payload for logging with truncation"""

        if not self.show_payloads:
            return "[payload hidden - set AGENTFIELD_LOG_PAYLOADS=true to show]"

        try:
            if isinstance(payload, dict):
                payload_str = json.dumps(payload, indent=2, default=str)
            else:
                payload_str = str(payload)

            return self._truncate_message(payload_str)
        except Exception:
            return self._truncate_message(str(payload))

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        )

    @staticmethod
    def _normalize_level(level: str) -> str:
        return level.upper()

    @staticmethod
    def _merge_attributes(
        attributes: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if attributes:
            merged.update(attributes)
        if extra:
            merged.update(extra)
        return merged

    def _build_execution_record(
        self,
        *,
        message: str,
        level: str,
        event_type: str,
        attributes: Optional[Dict[str, Any]] = None,
        execution_context: Optional[ExecutionContext] = None,
        system_generated: bool = False,
        source: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = execution_context or get_current_context()

        identity: Dict[str, Optional[str]] = {
            "execution_id": None,
            "workflow_id": None,
            "run_id": None,
            "root_workflow_id": None,
            "parent_execution_id": None,
            "agent_node_id": None,
            "reasoner_id": None,
        }
        extra_attributes: Dict[str, Any] = {}

        if context is not None:
            identity.update(context.to_log_identity())
            extra_attributes.update(context.to_log_attributes())

        merged_attributes = self._merge_attributes(extra_attributes, attributes)

        return {
            "ts": self._now_iso(),
            **identity,
            "level": self._normalize_level(level).lower(),
            "source": source or _DEFAULT_STRUCTURED_SOURCE,
            "event_type": event_type,
            "message": message,
            "attributes": merged_attributes,
            "system_generated": system_generated,
        }

    def _emit_structured_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str)
        print(line, file=sys.stdout, flush=True)
        self._dispatch_to_cp(record)
        return record

    def _dispatch_to_cp(self, record: Dict[str, Any]) -> None:
        """Send the structured log record to the control plane (best-effort, non-blocking)."""
        client = self._cp_client
        if client is None:
            return
        execution_id = record.get("execution_id")
        if not execution_id:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(client.post_execution_logs(execution_id, record))
        except RuntimeError:
            # No running event loop — fire from a background thread
            threading.Thread(
                target=self._dispatch_sync,
                args=(client, execution_id, record),
                daemon=True,
            ).start()

    @staticmethod
    def _dispatch_sync(
        client: "AgentFieldClient",
        execution_id: str,
        record: Dict[str, Any],
    ) -> None:
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(client.post_execution_logs(execution_id, record))
            loop.close()
        except Exception:
            pass

    def _emit_plain(
        self,
        level: str,
        message: str,
        *,
        prefix: str,
        payload: Optional[Any] = None,
    ) -> None:
        level_name = self._normalize_level(level)
        if payload is not None:
            formatted_payload = self._format_payload(payload)
            text = f"{prefix} {self._truncate_message(message)}\n{formatted_payload}"
        else:
            text = f"{prefix} {self._truncate_message(message)}"
        self.logger.log(_LEVEL_TO_LOGGING.get(level_name, logging.INFO), text)

    def _emit_optional_structured(
        self,
        level: str,
        message: str,
        *,
        prefix: str,
        event_type: str,
        execution_context: Optional[ExecutionContext] = None,
        system_generated: bool = False,
        source: Optional[str] = None,
        payload: Optional[Any] = None,
        attributes: Optional[Dict[str, Any]] = None,
        force_structured: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        context = execution_context or get_current_context()
        if context is None and not force_structured:
            self._emit_plain(level, message, prefix=prefix, payload=payload)
            return None

        merged_attributes = self._merge_attributes(attributes, extra)
        if payload is not None:
            merged_attributes.setdefault("payload", payload)

        record = self._build_execution_record(
            message=message,
            level=level,
            event_type=event_type,
            attributes=merged_attributes,
            execution_context=context,
            system_generated=system_generated,
            source=source,
        )
        return self._emit_structured_record(record)

    def heartbeat(self, message: str, **kwargs):
        """Log heartbeat messages (only shown in debug mode to avoid spam)"""

        self._emit_optional_structured(
            "DEBUG",
            message,
            prefix="💓",
            event_type="runtime.heartbeat",
            source="sdk.python.runtime",
            force_structured=False,
            extra=kwargs or None,
        )

    def track(self, message: str, **kwargs):
        """Log tracking messages (controlled by AGENTFIELD_LOG_TRACKING)"""

        if self.show_tracking:
            self._emit_optional_structured(
                "DEBUG",
                message,
                prefix="🔍 TRACK:",
                event_type="runtime.track",
                source="sdk.python.runtime",
                force_structured=False,
                extra=kwargs or None,
            )

    def fire(self, message: str, payload: Optional[Any] = None, **kwargs):
        """Log fire-and-forget workflow messages (controlled by AGENTFIELD_LOG_FIRE)"""

        if self.show_fire:
            self._emit_optional_structured(
                "DEBUG",
                message,
                prefix="🔥 FIRE:",
                event_type="workflow.fire",
                source="sdk.python.workflow",
                payload=payload,
                force_structured=False,
                extra=kwargs or None,
            )

    def debug(self, message: str, payload: Optional[Any] = None, **kwargs):
        """Log debug messages"""

        self._emit_optional_structured(
            "DEBUG",
            message,
            prefix="🔍 DEBUG:",
            event_type="log.debug",
            source=_DEFAULT_STRUCTURED_SOURCE,
            payload=payload,
            force_structured=False,
            extra=kwargs or None,
        )

    def info(self, message: str, **kwargs):
        """Log info messages"""

        self._emit_optional_structured(
            "INFO",
            message,
            prefix="ℹ️",
            event_type="log.info",
            source=_DEFAULT_STRUCTURED_SOURCE,
            force_structured=False,
            extra=kwargs or None,
        )

    def warn(self, message: str, **kwargs):
        """Log warning messages"""

        self._emit_optional_structured(
            "WARNING",
            message,
            prefix="⚠️",
            event_type="log.warning",
            source=_DEFAULT_STRUCTURED_SOURCE,
            force_structured=False,
            extra=kwargs or None,
        )

    def warning(self, message: str, **kwargs):
        """Alias for warn to match logging.Logger API"""

        self.warn(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error messages"""

        self._emit_optional_structured(
            "ERROR",
            message,
            prefix="❌",
            event_type="log.error",
            source=_DEFAULT_STRUCTURED_SOURCE,
            force_structured=False,
            extra=kwargs or None,
        )

    def critical(self, message: str, **kwargs):
        """Log critical messages"""

        self._emit_optional_structured(
            "CRITICAL",
            message,
            prefix="🚨",
            event_type="log.critical",
            source=_DEFAULT_STRUCTURED_SOURCE,
            force_structured=False,
            extra=kwargs or None,
        )

    def success(self, message: str, **kwargs):
        """Log success messages"""

        self._emit_optional_structured(
            "INFO",
            message,
            prefix="✅",
            event_type="runtime.success",
            source="sdk.python.runtime",
            force_structured=False,
            extra=kwargs or None,
        )

    def setup(self, message: str, **kwargs):
        """Log setup/initialization messages"""

        self._emit_optional_structured(
            "INFO",
            message,
            prefix="🔧",
            event_type="runtime.setup",
            source="sdk.python.runtime",
            force_structured=False,
            extra=kwargs or None,
        )

    def network(self, message: str, **kwargs):
        """Log network-related messages"""

        self._emit_optional_structured(
            "INFO",
            message,
            prefix="🌐",
            event_type="runtime.network",
            source="sdk.python.runtime",
            force_structured=False,
            extra=kwargs or None,
        )

    def security(self, message: str, **kwargs):
        """Log security/DID-related messages"""

        self._emit_optional_structured(
            "INFO",
            message,
            prefix="🔐",
            event_type="runtime.security",
            source="sdk.python.runtime",
            force_structured=False,
            extra=kwargs or None,
        )

    def log_execution(
        self,
        message: str,
        *,
        event_type: str,
        level: str = "INFO",
        attributes: Optional[Dict[str, Any]] = None,
        execution_context: Optional[ExecutionContext] = None,
        system_generated: bool = False,
        source: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Emit a structured execution log entry."""

        merged_attributes = self._merge_attributes(attributes, kwargs or None)
        record = self._build_execution_record(
            message=message,
            level=level,
            event_type=event_type,
            attributes=merged_attributes,
            execution_context=execution_context,
            system_generated=system_generated,
            source=source or "sdk.python.execution",
        )
        return self._emit_structured_record(record)


# Global logger instance
_global_logger = None


def get_logger(name: str = "agentfield") -> AgentFieldLogger:
    """Get or create a AgentField SDK logger instance"""

    global _global_logger
    if _global_logger is None:
        _global_logger = AgentFieldLogger(name)
    return _global_logger


def set_log_level(level: str):
    """Set log level for the global logger at runtime (e.g., 'DEBUG', 'INFO', 'WARN', 'ERROR')"""

    get_logger().set_level(level)


def set_cp_client(client: Optional["AgentFieldClient"]) -> None:
    """Attach a control-plane client so structured logs are forwarded."""
    get_logger()._cp_client = client


# Convenience functions for common logging patterns
def log_heartbeat(message: str, **kwargs):
    """Log heartbeat message"""

    get_logger().heartbeat(message, **kwargs)


def log_track(message: str, **kwargs):
    """Log tracking message"""

    get_logger().track(message, **kwargs)


def log_fire(message: str, payload: Optional[Any] = None, **kwargs):
    """Log fire-and-forget message"""

    get_logger().fire(message, payload, **kwargs)


def log_debug(message: str, payload: Optional[Any] = None, **kwargs):
    """Log debug message"""

    get_logger().debug(message, payload, **kwargs)


def log_info(message: str, **kwargs):
    """Log info message"""

    get_logger().info(message, **kwargs)


def log_warn(message: str, **kwargs):
    """Log warning message"""

    get_logger().warn(message, **kwargs)


def log_error(message: str, **kwargs):
    """Log error message"""

    get_logger().error(message, **kwargs)


def log_success(message: str, **kwargs):
    """Log success message"""

    get_logger().success(message, **kwargs)


def log_setup(message: str, **kwargs):
    """Log setup message"""

    get_logger().setup(message, **kwargs)


def log_network(message: str, **kwargs):
    """Log network message"""

    get_logger().network(message, **kwargs)


def log_security(message: str, **kwargs):
    """Log security message"""

    get_logger().security(message, **kwargs)


def log_execution(
    message: str,
    *,
    event_type: str,
    level: str = "INFO",
    attributes: Optional[Dict[str, Any]] = None,
    execution_context: Optional[ExecutionContext] = None,
    system_generated: bool = False,
    source: Optional[str] = None,
    **kwargs,
):
    """Emit a structured execution log entry using the global logger."""

    return get_logger().log_execution(
        message,
        event_type=event_type,
        level=level,
        attributes=attributes,
        execution_context=execution_context,
        system_generated=system_generated,
        source=source,
        **kwargs,
    )
