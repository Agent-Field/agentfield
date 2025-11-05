"""
AgentField SDK Logging Utility

This module provides a centralized logging system for the AgentField SDK that:
- Replaces print statements with proper logging
- Provides configurable log levels
- Truncates long messages and payloads
- Supports environment variable configuration
- Maintains emoji-based visual indicators for different message types
"""

import json
import logging
import os
from enum import Enum
from typing import Any, Optional


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

    Supports runtime log level changes (e.g., for dev_mode).
    """

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
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "SILENT": logging.CRITICAL + 1,  # Effectively silent
        }
        self.logger.setLevel(level_map.get(self.log_level, logging.WARNING))

    def set_level(self, level: str):
        """Set log level at runtime (e.g., 'DEBUG', 'INFO', 'WARN', 'ERROR')"""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARN": logging.WARNING,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
        }
        self.logger.setLevel(level_map.get(level.upper(), logging.INFO))

    def _setup_logger(self):
        """Setup logger with console handler if not already configured"""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
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

    def heartbeat(self, message: str, **kwargs):
        """Log heartbeat messages (only shown in debug mode to avoid spam)"""
        self.logger.debug(f"💓 {message}")

    def track(self, message: str, **kwargs):
        """Log tracking messages (controlled by AGENTFIELD_LOG_TRACKING)"""
        if self.show_tracking:
            self.logger.debug(f"🔍 TRACK: {self._truncate_message(message)}")

    def fire(self, message: str, payload: Optional[Any] = None, **kwargs):
        """Log fire-and-forget workflow messages (controlled by AGENTFIELD_LOG_FIRE)"""
        if self.show_fire:
            if payload is not None:
                formatted_payload = self._format_payload(payload)
                self.logger.debug(
                    f"🔥 FIRE: {self._truncate_message(message)}\n{formatted_payload}"
                )
            else:
                self.logger.debug(f"🔥 FIRE: {self._truncate_message(message)}")

    def debug(self, message: str, payload: Optional[Any] = None, **kwargs):
        """Log debug messages"""
        if payload is not None:
            formatted_payload = self._format_payload(payload)
            self.logger.debug(
                f"🔍 DEBUG: {self._truncate_message(message)}\n{formatted_payload}"
            )
        else:
            self.logger.debug(f"🔍 DEBUG: {self._truncate_message(message)}")

    def info(self, message: str, **kwargs):
        """Log info messages"""
        self.logger.info(f"ℹ️ {self._truncate_message(message)}")

    def warn(self, message: str, **kwargs):
        """Log warning messages"""
        self.logger.warning(f"⚠️ {self._truncate_message(message)}")

    def warning(self, message: str, **kwargs):
        """Alias for warn to match logging.Logger API"""
        self.warn(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error messages"""
        self.logger.error(f"❌ {self._truncate_message(message)}")

    def critical(self, message: str, **kwargs):
        """Log critical messages"""
        self.logger.critical(f"🚨 {self._truncate_message(message)}")

    def success(self, message: str, **kwargs):
        """Log success messages"""
        self.logger.info(f"✅ {self._truncate_message(message)}")

    def setup(self, message: str, **kwargs):
        """Log setup/initialization messages"""
        self.logger.info(f"🔧 {self._truncate_message(message)}")

    def network(self, message: str, **kwargs):
        """Log network-related messages"""
        self.logger.info(f"🌐 {self._truncate_message(message)}")

    def mcp(self, message: str, **kwargs):
        """Log MCP-related messages"""
        self.logger.info(f"🔌 {self._truncate_message(message)}")

    def security(self, message: str, **kwargs):
        """Log security/DID-related messages"""
        self.logger.info(f"🔐 {self._truncate_message(message)}")


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


def log_mcp(message: str, **kwargs):
    """Log MCP message"""
    get_logger().mcp(message, **kwargs)


def log_security(message: str, **kwargs):
    """Log security message"""
    get_logger().security(message, **kwargs)
