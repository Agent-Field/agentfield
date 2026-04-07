import asyncio
import datetime
import importlib
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import requests

from .types import (
    AgentStatus,
    CompactDiscoveryResponse,
    DiscoveryResponse,
    DiscoveryResult,
    HeartbeatData,
    WebhookConfig,
)
from .async_config import AsyncConfig
from .execution_state import ExecuteError, ExecutionStatus
from .result_cache import ResultCache
from .async_execution_manager import AsyncExecutionManager
from .logger import get_logger
from .status import normalize_status
from .execution_context import generate_run_id
from .did_auth import DIDAuthenticator
from .exceptions import (
    AgentFieldError,
    AgentFieldClientError,
    ExecutionTimeoutError,
    RegistrationError,
    ValidationError,
)

httpx = None  # type: ignore


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO 8601 string with Z suffix.

    Use this for all outbound timestamps to ensure consistent UTC-aware
    formatting across the client rather than naive local datetimes.
    """
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


# ---------------------------------------------------------------------------
# Typed response models for approval helpers
# ---------------------------------------------------------------------------

@dataclass
class ApprovalRequestResponse:
    """Response from requesting approval for an execution."""
    approval_request_id: str
    approval_request_url: str


@dataclass
class ApprovalStatusResponse:
    """Response from polling approval status."""
    status: str  # pending, approved, rejected, expired
    response: Optional[Dict[str, Any]] = None
    request_url: Optional[str] = None
    requested_at: Optional[str] = None
    responded_at: Optional[str] = None


@dataclass
class ApprovalResult:
    """Outcome of a human approval request, returned by ``Agent.pause()``."""

    decision: str  # "approved", "rejected", "request_changes", "expired", "error"
    feedback: str = ""
    execution_id: str = ""
    approval_request_id: str = ""
    raw_response: Optional[Dict[str, Any]] = None

    @property
    def approved(self) -> bool:
        return self.decision == "approved"

    @property
    def changes_requested(self) -> bool:
        return self.decision == "request_changes"


# Python 3.8 compatibility: asyncio.to_thread was added in Python 3.9
if sys.version_info >= (3, 9):
    from asyncio import to_thread as _to_thread
else:
    async def _to_thread(func, *args, **kwargs):
        """Compatibility shim for asyncio.to_thread on Python 3.8."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


def _ensure_httpx(force_reload: bool = False):
    """Load httpx lazily, allowing tests to monkeypatch the module."""
    global httpx

    if not force_reload and httpx is not None:
        return httpx

    try:
        module = importlib.import_module("httpx")
    except ImportError:
        httpx = None
    else:
        httpx = module

    return httpx


if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    import httpx  # noqa: F401


# Prime optional dependency cache at import time when available
_ensure_httpx()

# Set up logger for this module
logger = get_logger(__name__)

SUCCESS_STATUSES = {ExecutionStatus.SUCCEEDED.value}
FAILURE_STATUSES = {
    ExecutionStatus.FAILED.value,
    ExecutionStatus.CANCELLED.value,
    ExecutionStatus.TIMEOUT.value,
}


@dataclass
class _Submission:
    execution_id: str
    run_id: str
    target: str
    status: str
    target_type: Optional[str] = None


class AgentFieldClient:
    # Shared session for sync requests (class-level for reuse)
    _shared_sync_session: Optional[requests.Session] = None
    _shared_sync_session_lock: Optional[asyncio.Lock] = None

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        async_config: Optional[AsyncConfig] = None,
        did: Optional[str] = None,
        private_key_jwk: Optional[str] = None,
    ):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
        self.api_key = api_key

        # DID authentication for agent-to-agent calls
        self._did_authenticator = DIDAuthenticator(did=did, private_key_jwk=private_key_jwk)

        # Async execution components
        self.async_config = async_config or AsyncConfig()
        self._async_execution_manager: Optional[AsyncExecutionManager] = None
        self._async_http_client: Optional["httpx.AsyncClient"] = None
        self._async_http_client_lock: Optional[asyncio.Lock] = None
        self._result_cache = ResultCache(self.async_config)
        self._latest_event_stream_headers: Dict[str, str] = {}
        self._current_workflow_context = None
        # Caller agent ID for cross-agent call identification (set by Agent after init)
        self.caller_agent_id: Optional[str] = None

        # Initialize shared sync session if not already created
        if AgentFieldClient._shared_sync_session is None:
            AgentFieldClient._init_shared_sync_session()

    def _generate_id(self, prefix: str) -> str:
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        random_suffix = f"{random.getrandbits(32):08x}"
        return f"{prefix}_{timestamp}_{random_suffix}"

    def _build_event_stream_headers(
        self, source_headers: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        """Return headers that should be forwarded to the SSE event stream."""

        headers = dict(source_headers or {})
        if not headers:
            return {}

        allowed = {"authorization", "cookie"}
        event_headers: Dict[str, str] = {}
        for key, value in headers.items():
            if value is None:
                continue
            lower = key.lower()
            if lower.startswith("x-") or lower in allowed:
                event_headers[key] = value
        return event_headers

    def _sanitize_header_values(
        self, headers: Dict[str, Any]
    ) -> Dict[str, str]:
        """Ensure all header values are concrete strings for requests/httpx."""

        sanitized: Dict[str, str] = {}
        for key, value in headers.items():
            if value is None:
                continue
            if isinstance(value, bytes):
                sanitized[key] = value.decode("utf-8", errors="replace")
            elif isinstance(value, str):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        return sanitized

    def _get_auth_headers(self) -> Dict[str, str]:
        """Return auth headers if configured."""
        if not self.api_key:
            return {}
        return {"X-API-Key": self.api_key}

    def set_did_credentials(self, did: str, private_key_jwk: str) -> bool:
        """
        Set DID authentication credentials for agent-to-agent calls.

        Args:
            did: The agent's DID identifier (e.g., 'did:web:example.com:agents:my-agent')
            private_key_jwk: JWK-formatted Ed25519 private key for signing

        Returns:
            True if credentials were set successfully, False otherwise
        """
        return self._did_authenticator.set_credentials(did, private_key_jwk)

    def get_did_auth_headers(self, body: bytes) -> Dict[str, str]:
        """
        Get DID authentication headers for signing a request.

        Args:
            body: Request body bytes to sign

        Returns:
            Dictionary with DID auth headers (X-Caller-DID, X-DID-Signature, X-DID-Timestamp)
            Empty dict if DID auth is not configured
        """
        return self._did_authenticator.sign_headers(body)

    @property
    def did(self) -> Optional[str]:
        """Get the configured DID identifier."""
        return self._did_authenticator.did

    @property
    def did_auth_configured(self) -> bool:
        """Check if DID authentication is configured."""
        return self._did_authenticator.is_configured

    def _get_headers_with_context(
        self, headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """Merge caller headers with the active workflow context headers."""

        merged = self._get_auth_headers()
        merged.update(headers or {})
        context = getattr(self, "_current_workflow_context", None)
        if context and hasattr(context, "to_headers"):
            try:
                context_headers = context.to_headers()
            except Exception:
                context_headers = {}
            for key, value in (context_headers or {}).items():
                merged.setdefault(key, value)
        return merged

    def _maybe_update_event_stream_headers(
        self, source_headers: Optional[Dict[str, str]]
    ) -> None:
        """Update stored SSE headers and propagate to the manager when enabled."""

        if not self.async_config.enable_event_stream:
            return

        new_headers = self._build_event_stream_headers(source_headers)

        if (
            not new_headers
            and source_headers is None
            and self._current_workflow_context
        ):
            try:
                context_headers = self._current_workflow_context.to_headers()
            except Exception:
                context_headers = {}
            new_headers = self._build_event_stream_headers(context_headers)

        if new_headers:
            self._latest_event_stream_headers = new_headers
        elif source_headers is None and not self._latest_event_stream_headers:
            # No headers from context yet; keep empty state.
            self._latest_event_stream_headers = {}

        if self._async_execution_manager is not None:
            self._async_execution_manager.set_event_stream_headers(
                self._latest_event_stream_headers
            )

    def discover_capabilities(
        self,
        *,
        agent: Optional[str] = None,
        node_id: Optional[str] = None,
        agent_ids: Optional[List[str]] = None,
        node_ids: Optional[List[str]] = None,
        reasoner: Optional[str] = None,
        skill: Optional[str] = None,
        tags: Optional[List[str]] = None,
        include_input_schema: Optional[bool] = None,
        include_output_schema: Optional[bool] = None,
        include_descriptions: Optional[bool] = None,
        include_examples: Optional[bool] = None,
        format: str = "json",
        health_status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> DiscoveryResult:
        """
        Query the control plane discovery API.
        """

        fmt = (format or "json").lower()
        params: Dict[str, str] = {"format": fmt}

        def _dedupe(values: Optional[List[str]]) -> List[str]:
            if not values:
                return []
            seen = set()
            out: List[str] = []
            for value in values:
                if not value or value in seen:
                    continue
                seen.add(value)
                out.append(value)
            return out

        combined_agent_ids = _dedupe(
            ([agent] if agent else [])
            + ([node_id] if node_id else [])
            + (agent_ids or [])
            + (node_ids or [])
        )

        if len(combined_agent_ids) == 1:
            params["agent"] = combined_agent_ids[0]
        elif len(combined_agent_ids) > 1:
            params["agent_ids"] = ",".join(combined_agent_ids)

        if reasoner:
            params["reasoner"] = reasoner
        if skill:
            params["skill"] = skill
        if tags:
            params["tags"] = ",".join(_dedupe(tags))

        if include_input_schema is not None:
            params["include_input_schema"] = str(bool(include_input_schema)).lower()
        if include_output_schema is not None:
            params["include_output_schema"] = str(bool(include_output_schema)).lower()
        if include_descriptions is not None:
            params["include_descriptions"] = str(bool(include_descriptions)).lower()
        if include_examples is not None:
            params["include_examples"] = str(bool(include_examples)).lower()
        if health_status:
            params["health_status"] = health_status.lower()
        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)

        request_headers = self._get_headers_with_context(headers)
        request_headers["Accept"] = (
            "application/xml" if fmt == "xml" else "application/json"
        )
        sanitized_headers = self._sanitize_header_values(request_headers)

        response = requests.get(
            f"{self.api_base}/discovery/capabilities",
            params=params,
            headers=sanitized_headers,
            timeout=self.async_config.polling_timeout,
        )
        response.raise_for_status()

        raw_body = response.text
        if fmt == "xml":
            return DiscoveryResult(format=fmt, raw=raw_body, xml=raw_body)

        payload = response.json()
        if fmt == "compact":
            compact = CompactDiscoveryResponse.from_dict(payload)
            return DiscoveryResult(
                format=fmt, raw=raw_body, compact=compact, json=None
            )

        json_payload = DiscoveryResponse.from_dict(payload)
        return DiscoveryResult(format="json", raw=raw_body, json=json_payload)

    async def get_async_http_client(self) -> "httpx.AsyncClient":
        """Lazily create and return a shared httpx.AsyncClient."""
        current_module = sys.modules.get("httpx")
        reload_needed = httpx is None or current_module is not httpx
        httpx_module = _ensure_httpx(force_reload=reload_needed)
        if httpx_module is None:
            raise AgentFieldClientError("httpx is required for async HTTP operations")

        if self._async_http_client and not getattr(
            self._async_http_client, "is_closed", False
        ):
            return self._async_http_client

        if self._async_http_client_lock is None:
            self._async_http_client_lock = asyncio.Lock()

        async with self._async_http_client_lock:
            if self._async_http_client and not getattr(
                self._async_http_client, "is_closed", False
            ):
                return self._async_http_client

            client_kwargs = {
                "headers": {
                    "User-Agent": "AgentFieldSDK/1.0",
                    "Accept": "application/json",
                }
            }

            limits_factory = getattr(httpx_module, "Limits", None)
            if limits_factory:
                client_kwargs["limits"] = limits_factory(
                    max_connections=self.async_config.connection_pool_size,
                    max_keepalive_connections=self.async_config.connection_pool_per_host,
                )

            timeout_factory = getattr(httpx_module, "Timeout", None)
            if timeout_factory:
                client_kwargs["timeout"] = timeout_factory(10.0, connect=5.0)
            else:
                client_kwargs["timeout"] = 10.0

            try:
                self._async_http_client = httpx_module.AsyncClient(**client_kwargs)
            except TypeError:
                # Test doubles may not accept keyword arguments
                self._async_http_client = httpx_module.AsyncClient()
                headers = client_kwargs.get("headers")
                if headers and hasattr(self._async_http_client, "headers"):
                    try:
                        self._async_http_client.headers.update(headers)
                    except Exception:
                        pass

            return self._async_http_client

    async def _async_request(self, method: str, url: str, **kwargs):
        """Perform an HTTP request using the shared async client with sync fallback."""
        # Inject API key into headers if available
        if self.api_key:
            if "headers" not in kwargs:
                kwargs["headers"] = {}
            if "X-API-Key" not in kwargs["headers"]:
                kwargs["headers"]["X-API-Key"] = self.api_key

        try:
            client = await self.get_async_http_client()
        except AgentFieldClientError:
            return await _to_thread(self._sync_request, method, url, **kwargs)

        return await client.request(method, url, **kwargs)

    @classmethod
    def _init_shared_sync_session(cls) -> None:
        """Initialize the shared sync session with proper configuration."""
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        session = requests.Session()
        # Configure adapter with retry logic and connection pooling
        adapter = HTTPAdapter(
            max_retries=Retry(total=3, backoff_factor=0.3),
            pool_connections=20,
            pool_maxsize=20,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            "User-Agent": "AgentFieldSDK/1.0",
            "Accept": "application/json",
        })
        cls._shared_sync_session = session

    @classmethod
    def _get_sync_session(cls) -> requests.Session:
        """Get the shared sync session, initializing if needed."""
        if cls._shared_sync_session is None:
            cls._init_shared_sync_session()
        return cls._shared_sync_session

    @classmethod
    def _sync_request(cls, method: str, url: str, **kwargs):
        """Blocking HTTP request helper using shared session for connection reuse."""
        # DIAGNOSTIC: Add request size logging
        if "json" in kwargs:
            import json

            json_size = len(json.dumps(kwargs["json"]).encode("utf-8"))
            logger.debug(
                f"SYNC_REQUEST: Making {method} request to {url} with JSON payload size: {json_size} bytes"
            )

        # Get shared session (reuses connections)
        session = cls._get_sync_session()

        # Set default headers if not provided
        if "headers" not in kwargs:
            kwargs["headers"] = {}

        # Ensure proper content type for JSON requests
        if "json" in kwargs and "Content-Type" not in kwargs["headers"]:
            kwargs["headers"]["Content-Type"] = "application/json"

        # DIAGNOSTIC: Log request details
        logger.debug(f"SYNC_REQUEST: Headers: {kwargs.get('headers', {})}")

        # Configure stream=False to ensure we read the full response
        # This prevents truncation issues with large JSON responses
        if "stream" not in kwargs:
            kwargs["stream"] = False

        response = session.request(method, url, **kwargs)

        # DIAGNOSTIC: Log response details
        logger.debug(
            f"SYNC_RESPONSE: Status {response.status_code}, Content-Length: {response.headers.get('Content-Length', 'unknown')}"
        )

        # Check if response might be truncated
        content_length = response.headers.get("Content-Length")
        if content_length and len(response.content) != int(content_length):
            logger.error(
                f"RESPONSE_TRUNCATION: Expected {content_length} bytes, got {len(response.content)} bytes"
            )

        # Check for exactly 4096 bytes which indicates truncation
        if len(response.content) == 4096:
            logger.error(
                "POSSIBLE_TRUNCATION: Response is exactly 4096 bytes - likely truncated!"
            )

        return response

    async def aclose(self) -> None:
        """Close shared resources such as async HTTP clients and managers."""
        if self._async_execution_manager is not None:
            try:
                await self._async_execution_manager.stop()
            finally:
                self._async_execution_manager = None

        if self._async_http_client is not None:
            try:
                await self._async_http_client.aclose()
            finally:
                self._async_http_client = None
                self._async_http_client_lock = None

    def register_node(self, node_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register agent node with AgentField server.

        Raises:
            RegistrationError: If registration fails.
        """
        try:
            response = requests.post(
                f"{self.api_base}/nodes/register",
                json=node_data,
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()
            return response.json()
        except RegistrationError:
            raise
        except Exception as exc:
            raise RegistrationError(f"Failed to register node: {exc}") from exc

    def update_health(
        self, node_id: str, health_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update node health status"""
        response = requests.put(
            f"{self.api_base}/nodes/{node_id}/health",
            json=health_data,
            headers=self._get_auth_headers(),
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()

    def get_nodes(self) -> Dict[str, Any]:
        """Get all registered nodes"""
        response = requests.get(
            f"{self.api_base}/nodes",
            headers=self._get_auth_headers(),
        )
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()

    def _apply_vc_metadata(
        self, registration_data: Dict[str, Any], vc_metadata: Optional[Dict[str, Any]]
    ) -> None:
        """Attach VC metadata to the registration payload if supplied."""
        if not vc_metadata:
            return

        metadata = registration_data.setdefault("metadata", {})
        custom_section = metadata.setdefault("custom", {})
        custom_section["vc_generation"] = vc_metadata

    async def register_agent(
        self,
        node_id: str,
        reasoners: List[dict],
        skills: List[dict],
        base_url: str,
        discovery: Optional[Dict[str, Any]] = None,
        vc_metadata: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0",
        agent_metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Register or update agent information with AgentField server."""
        try:
            custom_metadata: Dict[str, Any] = {}
            if agent_metadata:
                custom_metadata.update(agent_metadata)

            agent_tags = tags or []
            registration_data = {
                "id": node_id,
                "team_id": "default",
                "base_url": base_url,
                "version": version,
                "reasoners": reasoners,
                "skills": skills,
                "proposed_tags": agent_tags,
                "communication_config": {
                    "protocols": ["http"],
                    "websocket_endpoint": "",
                    "heartbeat_interval": "5s",
                },
                "health_status": "healthy",
                "last_heartbeat": _utc_now_iso(),
                "registered_at": _utc_now_iso(),
                "features": {
                    "ab_testing": False,
                    "advanced_metrics": False,
                    "compliance": False,
                    "audit_logging": False,
                    "role_based_access": False,
                    "experimental": {},
                },
                "metadata": {
                    "deployment": {
                        "environment": "development",
                        "platform": "python",
                        "region": "local",
                        "tags": {"sdk_version": importlib.import_module("agentfield").__version__, "language": "python"},
                    },
                    "performance": {"latency_ms": 0, "throughput_ps": 0},
                    "custom": custom_metadata,
                },
            }

            if discovery:
                registration_data["callback_discovery"] = discovery

            self._apply_vc_metadata(registration_data, vc_metadata)

            response = await self._async_request(
                "POST",
                f"{self.api_base}/nodes/register",
                json=registration_data,
                headers=self._get_auth_headers(),
                timeout=30.0,
            )
            payload: Optional[Dict[str, Any]] = None
            if hasattr(response, "json"):
                try:
                    payload = response.json()
                except Exception:
                    payload = None

            if response.status_code not in (200, 201):
                return False, payload

            return True, payload

        except Exception:
            # self.logger.error(f"Failed to register agent: {e}")
            return False, None

    async def execute(
        self,
        target: str,
        input_data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a reasoner or skill via the durable execution gateway.

        The public signature remains unchanged, but internally we now submit the
        execution, poll for completion with adaptive backoff, and return the final
        result once the worker finishes processing.

        Raises:
            AgentFieldClientError: If submission or polling fails.
            ExecutionTimeoutError: If execution does not complete in time.
        """

        execution_headers = self._prepare_execution_headers(headers)
        submission = await self._submit_execution_async(
            target, input_data, execution_headers
        )
        status_payload = await self._await_execution_async(
            submission, execution_headers
        )
        result_value, metadata = self._format_execution_result(
            submission, status_payload
        )
        return self._build_execute_response(
            submission, status_payload, result_value, metadata
        )

    def execute_sync(
        self,
        target: str,
        input_data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Blocking version of execute used by synchronous callers.

        Raises:
            AgentFieldClientError: If submission or polling fails.
            ExecutionTimeoutError: If execution does not complete in time.
        """

        execution_headers = self._prepare_execution_headers(headers)
        submission = self._submit_execution_sync(target, input_data, execution_headers)
        status_payload = self._await_execution_sync(submission, execution_headers)
        result_value, metadata = self._format_execution_result(
            submission, status_payload
        )
        return self._build_execute_response(
            submission, status_payload, result_value, metadata
        )

    def _prepare_execution_headers(
        self, headers: Optional[Dict[str, str]]
    ) -> Dict[str, str]:
        merged_headers = self._get_headers_with_context(headers)

        final_headers: Dict[str, str] = {"Content-Type": "application/json"}
        final_headers.update(merged_headers)

        run_id = final_headers.get("X-Run-ID") or final_headers.get("x-run-id")
        if not run_id:
            final_headers["X-Run-ID"] = generate_run_id()
        else:
            final_headers["X-Run-ID"] = run_id

        # Ensure parent execution header casing is consistent if provided
        parent_execution = final_headers.pop("x-parent-execution-id", None)
        if parent_execution and parent_execution.strip():
            final_headers["X-Parent-Execution-ID"] = parent_execution.strip()

        session_id = final_headers.pop("x-session-id", None)
        if session_id:
            final_headers["X-Session-ID"] = session_id

        actor_id = final_headers.pop("x-actor-id", None)
        if actor_id:
            final_headers["X-Actor-ID"] = actor_id

        # Include caller agent ID for permission middleware identification
        if self.caller_agent_id and "X-Caller-Agent-ID" not in final_headers:
            final_headers["X-Caller-Agent-ID"] = self.caller_agent_id

        sanitized_headers = self._sanitize_header_values(final_headers)
        self._maybe_update_event_stream_headers(sanitized_headers)
        return sanitized_headers

    def _submit_execution_sync(
        self,
        target: str,
        input_data: Dict[str, Any],
        headers: Dict[str, str],
    ) -> _Submission:
        import json as json_module

        payload = {"input": input_data}
        # Serialize once so the signed bytes are exactly what gets sent.
        body_bytes = json_module.dumps(payload, separators=(",", ":")).encode("utf-8")

        # Add DID authentication headers if configured
        final_headers = dict(headers)
        final_headers["Content-Type"] = "application/json"
        if self._did_authenticator.is_configured:
            did_headers = self._did_authenticator.sign_headers(body_bytes)
            final_headers.update(did_headers)

        try:
            response = requests.post(
                f"{self.api_base}/execute/async/{target}",
                data=body_bytes,
                headers=final_headers,
                timeout=self.async_config.polling_timeout,
            )
        except requests.RequestException as exc:
            raise AgentFieldClientError(f"Failed to submit execution: {exc}") from exc
        if response.status_code >= 400:
            try:
                error_body = response.json()
            except Exception:
                error_body = None
            body_msg = ""
            if isinstance(error_body, dict):
                body_msg = error_body.get("message") or error_body.get("error") or ""
            msg = f"{response.status_code}, {body_msg}" if body_msg else str(response.status_code)
            raise ExecuteError(response.status_code, msg, error_body)
        body = response.json()
        return self._parse_submission(body, final_headers, target)

    async def _submit_execution_async(
        self,
        target: str,
        input_data: Dict[str, Any],
        headers: Dict[str, str],
    ) -> _Submission:
        import json as json_module

        payload = {"input": input_data}
        # Serialize once so the signed bytes are exactly what gets sent.
        # httpx uses compact separators (',', ':') which differ from
        # json.dumps() defaults (', ', ': '), causing signature mismatch.
        body_bytes = json_module.dumps(payload, separators=(",", ":")).encode("utf-8")

        # Add DID authentication headers if configured
        final_headers = dict(headers)
        final_headers["Content-Type"] = "application/json"
        if self._did_authenticator.is_configured:
            did_headers = self._did_authenticator.sign_headers(body_bytes)
            final_headers.update(did_headers)

        response = await self._async_request(
            "POST",
            f"{self.api_base}/execute/async/{target}",
            content=body_bytes,
            headers=final_headers,
            timeout=self.async_config.polling_timeout,
        )
        if response.status_code >= 400:
            try:
                error_body = response.json()
            except Exception:
                error_body = None
            body_msg = ""
            if isinstance(error_body, dict):
                body_msg = error_body.get("message") or error_body.get("error") or ""
            msg = f"{response.status_code}, {body_msg}" if body_msg else str(response.status_code)
            raise ExecuteError(response.status_code, msg, error_body)
        body = response.json()
        return self._parse_submission(body, final_headers, target)

    def _parse_submission(
        self,
        body: Dict[str, Any],
        headers: Dict[str, str],
        target: str,
    ) -> _Submission:
        execution_id = body.get("execution_id")
        run_id = body.get("run_id") or headers.get("X-Run-ID")
        status = (body.get("status") or "pending").lower()
        target_type = body.get("type") or body.get("target_type")

        if not execution_id or not run_id:
            raise AgentFieldClientError("Execution submission missing identifiers")

        return _Submission(
            execution_id=execution_id,
            run_id=run_id,
            target=target,
            status=status,
            target_type=target_type,
        )

    def _await_execution_sync(
        self,
        submission: _Submission,
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        cached = self._result_cache.get_execution_result(submission.execution_id)
        if cached is not None:
            return {
                "result": cached,
                "status": "succeeded",
                "run_id": submission.run_id,
            }

        interval = max(self.async_config.initial_poll_interval, 0.25)
        start = time.time()

        while True:
            response = requests.get(
                f"{self.api_base}/executions/{submission.execution_id}",
                headers=headers,
                timeout=self.async_config.polling_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            normalized_status = normalize_status(payload.get("status"))
            payload["status"] = normalized_status

            if normalized_status in SUCCESS_STATUSES:
                return payload

            if normalized_status in FAILURE_STATUSES:
                if not payload.get("error_message") and payload.get("error"):
                    payload["error_message"] = payload["error"]
                return payload

            if (time.time() - start) > self.async_config.max_execution_timeout:
                raise ExecutionTimeoutError(
                    f"Execution {submission.execution_id} exceeded timeout"
                )

            time.sleep(self._next_poll_interval(interval))
            interval = min(interval * 2, self.async_config.max_poll_interval)

    async def _await_execution_async(
        self,
        submission: _Submission,
        headers: Dict[str, str],
    ) -> Dict[str, Any]:
        cached = self._result_cache.get_execution_result(submission.execution_id)
        if cached is not None:
            return {
                "result": cached,
                "status": "succeeded",
                "run_id": submission.run_id,
            }

        interval = max(self.async_config.initial_poll_interval, 0.25)
        start = time.time()

        while True:
            response = await self._async_request(
                "GET",
                f"{self.api_base}/executions/{submission.execution_id}",
                headers=headers,
                timeout=self.async_config.polling_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            normalized_status = normalize_status(payload.get("status"))
            payload["status"] = normalized_status

            if normalized_status in SUCCESS_STATUSES:
                return payload

            if normalized_status in FAILURE_STATUSES:
                if not payload.get("error_message") and payload.get("error"):
                    payload["error_message"] = payload["error"]
                return payload

            if (time.time() - start) > self.async_config.max_execution_timeout:
                raise ExecutionTimeoutError(
                    f"Execution {submission.execution_id} exceeded timeout"
                )

            await asyncio.sleep(self._next_poll_interval(interval))
            interval = min(interval * 2, self.async_config.max_poll_interval)

    def _format_execution_result(
        self,
        submission: _Submission,
        payload: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        result_value = payload.get("result")
        if result_value is None:
            result_value = payload

        normalized_status = normalize_status(payload.get("status"))
        target = payload.get("target") or submission.target
        node_id = payload.get("node_id")
        if not node_id and target and "." in target:
            node_id = target.split(".", 1)[0]

        metadata = {
            "execution_id": submission.execution_id,
            "run_id": payload.get("run_id") or submission.run_id,
            "status": normalized_status,
            "target": target,
            "type": payload.get("type") or submission.target_type,
            "duration_ms": payload.get("duration_ms") or payload.get("duration"),
            "started_at": payload.get("started_at"),
            "completed_at": payload.get("completed_at"),
            "node_id": node_id,
            "error_message": payload.get("error_message") or payload.get("error"),
            "error_details": payload.get("error_details"),
        }

        if metadata.get("completed_at"):
            metadata["timestamp"] = metadata["completed_at"]
        elif metadata.get("started_at"):
            metadata["timestamp"] = metadata["started_at"]
        else:
            metadata["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Cache successful results for reuse
        if normalized_status in SUCCESS_STATUSES:
            try:
                self._result_cache.set_execution_result(
                    submission.execution_id, result_value
                )
            except Exception:
                logger.debug("Failed to cache execution result", exc_info=True)

        return result_value, {k: v for k, v in metadata.items() if v is not None}

    def _build_execute_response(
        self,
        submission: _Submission,
        payload: Dict[str, Any],
        result_value: Any,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized_status = normalize_status(metadata.get("status"))
        error_message = metadata.get("error_message")

        if normalized_status in SUCCESS_STATUSES:
            response_result = result_value
        elif normalized_status in FAILURE_STATUSES:
            response_result = None
        else:
            response_result = result_value

        error_details = metadata.get("error_details")

        response = {
            "execution_id": metadata.get("execution_id"),
            "run_id": metadata.get("run_id"),
            "node_id": metadata.get("node_id"),
            "type": metadata.get("type"),
            "target": metadata.get("target") or submission.target,
            "status": normalized_status,
            "duration_ms": metadata.get("duration_ms"),
            "timestamp": metadata.get("timestamp")
            or datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "result": response_result,
            "error_message": error_message,
            "error_details": error_details,
            "cost": payload.get("cost"),
        }

        return response

    def _next_poll_interval(self, current: float) -> float:
        jitter = random.uniform(0.8, 1.2)
        return max(0.05, min(current * jitter, self.async_config.max_poll_interval))

    async def send_enhanced_heartbeat(
        self, node_id: str, heartbeat_data: HeartbeatData
    ) -> bool:
        """
        Send enhanced heartbeat with status information to AgentField server.

        Args:
            node_id: The agent node ID
            heartbeat_data: Enhanced heartbeat data with status info

        Returns:
            True if heartbeat was successful, False otherwise
        """
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(self._get_auth_headers())
            response = await self._async_request(
                "POST",
                f"{self.api_base}/nodes/{node_id}/heartbeat",
                json=heartbeat_data.to_dict(),
                headers=headers,
                timeout=5.0,
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    def send_enhanced_heartbeat_sync(
        self, node_id: str, heartbeat_data: HeartbeatData
    ) -> bool:
        """
        Synchronous version of enhanced heartbeat for compatibility.

        Args:
            node_id: The agent node ID
            heartbeat_data: Enhanced heartbeat data with status info

        Returns:
            True if heartbeat was successful, False otherwise
        """
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(self._get_auth_headers())
            response = requests.post(
                f"{self.api_base}/nodes/{node_id}/heartbeat",
                json=heartbeat_data.to_dict(),
                headers=headers,
                timeout=5.0,
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    async def notify_graceful_shutdown(self, node_id: str) -> bool:
        """
        Notify AgentField server that the agent is shutting down gracefully.

        Args:
            node_id: The agent node ID

        Returns:
            True if notification was successful, False otherwise
        """
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(self._get_auth_headers())
            response = await self._async_request(
                "POST",
                f"{self.api_base}/nodes/{node_id}/shutdown",
                headers=headers,
                timeout=5.0,
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    def notify_graceful_shutdown_sync(self, node_id: str) -> bool:
        """
        Synchronous version of graceful shutdown notification.

        Args:
            node_id: The agent node ID

        Returns:
            True if notification was successful, False otherwise
        """
        try:
            headers = {"Content-Type": "application/json"}
            headers.update(self._get_auth_headers())
            response = requests.post(
                f"{self.api_base}/nodes/{node_id}/shutdown",
                headers=headers,
                timeout=5.0,
            )
            response.raise_for_status()
            return True
        except Exception:
            return False

    async def register_agent_with_status(
        self,
        node_id: str,
        reasoners: List[dict],
        skills: List[dict],
        base_url: str,
        status: AgentStatus = AgentStatus.STARTING,
        discovery: Optional[Dict[str, Any]] = None,
        suppress_errors: bool = False,
        vc_metadata: Optional[Dict[str, Any]] = None,
        version: str = "1.0.0",
        agent_metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Register agent with immediate status reporting for fast lifecycle."""
        try:
            custom_metadata: Dict[str, Any] = {}
            if agent_metadata:
                custom_metadata.update(agent_metadata)

            agent_tags = tags or []
            registration_data = {
                "id": node_id,
                "team_id": "default",
                "base_url": base_url,
                "version": version,
                "reasoners": reasoners,
                "skills": skills,
                "proposed_tags": agent_tags,
                "lifecycle_status": status.value,
                "communication_config": {
                    "protocols": ["http"],
                    "websocket_endpoint": "",
                    "heartbeat_interval": "2s",
                },
                "health_status": "healthy",
                "last_heartbeat": _utc_now_iso(),
                "registered_at": _utc_now_iso(),
                "features": {
                    "ab_testing": False,
                    "advanced_metrics": False,
                    "compliance": False,
                    "audit_logging": False,
                    "role_based_access": False,
                    "experimental": {},
                },
                "metadata": {
                    "deployment": {
                        "environment": "development",
                        "platform": "python",
                        "region": "local",
                        "tags": {"sdk_version": importlib.import_module("agentfield").__version__, "language": "python"},
                    },
                    "performance": {"latency_ms": 0, "throughput_ps": 0},
                    "custom": custom_metadata,
                },
            }

            if discovery:
                registration_data["callback_discovery"] = discovery

            self._apply_vc_metadata(registration_data, vc_metadata)

            response = await self._async_request(
                "POST",
                f"{self.api_base}/nodes/register",
                json=registration_data,
                headers=self._get_auth_headers(),
                timeout=10.0,
            )

            payload: Optional[Dict[str, Any]] = None
            try:
                if getattr(response, "content", None):
                    payload = response.json()
            except Exception:
                payload = None

            if response.status_code not in (200, 201):
                if not suppress_errors:
                    logger.error(
                        "Fast lifecycle registration failed with status %s",
                        response.status_code,
                    )
                    logger.error(
                        f"Response text: {getattr(response, 'text', '<none>')}"
                    )
                else:
                    logger.debug(
                        "Fast lifecycle registration failed with status %s",
                        response.status_code,
                    )
                return False, payload

            logger.debug(f"Agent {node_id} registered successfully")
            return True, payload

        except Exception as e:
            if not suppress_errors:
                logger.error(
                    f"Agent registration failed for {node_id}: {type(e).__name__}: {e}"
                )
            else:
                logger.debug(
                    f"Agent registration failed for {node_id}: {type(e).__name__}"
                )
            return False, None

    # Async Execution Methods

    async def _get_async_execution_manager(self) -> AsyncExecutionManager:
        """
        Get or create the async execution manager instance.

        Returns:
            AsyncExecutionManager: Active async execution manager
        """
        if self._async_execution_manager is None:
            self._async_execution_manager = AsyncExecutionManager(
                base_url=self.base_url,
                config=self.async_config,
                auth_headers=self._get_auth_headers(),
                did_authenticator=self._did_authenticator,
            )
            await self._async_execution_manager.start()
            self._maybe_update_event_stream_headers(None)

        return self._async_execution_manager

    async def execute_async(
        self,
        target: str,
        input_data: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        webhook: Optional[Union[WebhookConfig, Dict[str, Any]]] = None,
    ) -> str:
        """
        Submit an async execution and return execution_id.

        Args:
            target: Target in format 'node_id.reasoner_name' or 'node_id.skill_name'
            input_data: Input data for the reasoner/skill
            headers: Optional headers to include (will be merged with context headers)
            timeout: Optional execution timeout (uses config default if None)
            webhook: Optional webhook registration (dict or WebhookConfig)

        Returns:
            str: Execution ID for tracking the execution

        Raises:
            AgentFieldClientError: If async execution is disabled or request setup fails.
            ExecutionTimeoutError: If fallback execution exceeds timeout.
        """
        if not self.async_config.enable_async_execution:
            raise AgentFieldClientError("Async execution is disabled in configuration")

        try:
            final_headers = self._prepare_execution_headers(headers)

            # Get async execution manager and submit
            manager = await self._get_async_execution_manager()
            execution_id = await manager.submit_execution(
                target=target,
                input_data=input_data,
                headers=final_headers,
                timeout=timeout,
                webhook=webhook,
            )

            logger.debug(
                f"Submitted async execution {execution_id[:8]}... for target {target}"
            )
            return execution_id

        except Exception as e:
            logger.error(f"Failed to submit async execution for target {target}: {e}")
            if isinstance(e, AgentFieldError):
                raise

            # Never fall back on authorization errors (401/403) — these are
            # permanent failures that retrying won't fix and would hit replay
            # protection (Ed25519 signatures are deterministic within the same second).
            _status = getattr(e, "status", None)
            if _status in (401, 403):
                raise

            # Fallback to sync execution if enabled
            if self.async_config.fallback_to_sync:
                logger.warn(f"Falling back to sync execution for target {target}")
                try:
                    await self.execute(target, input_data, headers)
                    # Create a synthetic execution ID for consistency
                    synthetic_id = self._generate_id("sync")
                    logger.debug(
                        f"Sync fallback completed with synthetic ID {synthetic_id[:8]}..."
                    )
                    return synthetic_id
                except Exception as sync_error:
                    logger.error(f"Sync fallback also failed: {sync_error}")
                    if isinstance(sync_error, AgentFieldError):
                        raise
                    raise AgentFieldClientError(
                        f"Async execution failed for target {target}"
                    ) from sync_error
            else:
                raise AgentFieldClientError(
                    f"Async execution failed for target {target}"
                ) from e

    async def poll_execution_status(
        self, execution_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Poll single execution status with connection reuse.

        Args:
            execution_id: Execution ID to poll

        Returns:
            Optional[Dict]: Execution status dictionary or None if not found

        Raises:
            AgentFieldClientError: If async execution is disabled.
        """
        if not self.async_config.enable_async_execution:
            raise AgentFieldClientError("Async execution is disabled in configuration")

        try:
            manager = await self._get_async_execution_manager()
            status = await manager.get_execution_status(execution_id)

            if status:
                logger.debug(
                    f"Polled status for execution {execution_id[:8]}...: {status.get('status')}"
                )
            else:
                logger.debug(f"Execution {execution_id[:8]}... not found")

            return status

        except AgentFieldError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to poll execution status for {execution_id[:8]}...: {e}"
            )
            raise AgentFieldClientError(
                f"Failed to poll execution status: {e}"
            ) from e

    async def batch_check_statuses(
        self, execution_ids: List[str]
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Check multiple execution statuses efficiently.

        Args:
            execution_ids: List of execution IDs to check

        Returns:
            Dict[str, Optional[Dict]]: Mapping of execution_id to status dict

        Raises:
            AgentFieldClientError: If async execution is disabled.
            ValidationError: If execution_ids is empty.
        """
        if not self.async_config.enable_async_execution:
            raise AgentFieldClientError("Async execution is disabled in configuration")

        if not execution_ids:
            raise ValidationError("execution_ids list cannot be empty")

        try:
            manager = await self._get_async_execution_manager()
            results = {}

            # Use batch processing if enabled and list is large enough
            if (
                self.async_config.enable_batch_polling and len(execution_ids) >= 2
            ):  # Use batch for 2+ executions
                # Process in batches
                batch_size = self.async_config.batch_size
                for i in range(0, len(execution_ids), batch_size):
                    batch_ids = execution_ids[i : i + batch_size]

                    # Get statuses for this batch
                    for exec_id in batch_ids:
                        status = await manager.get_execution_status(exec_id)
                        results[exec_id] = status

                    logger.debug(f"Batch checked {len(batch_ids)} execution statuses")
            else:
                # Process individually
                for exec_id in execution_ids:
                    status = await manager.get_execution_status(exec_id)
                    results[exec_id] = status

                logger.debug(
                    f"Individually checked {len(execution_ids)} execution statuses"
                )

            return results

        except AgentFieldError:
            raise
        except Exception as e:
            logger.error(f"Failed to batch check execution statuses: {e}")
            raise AgentFieldClientError(
                f"Failed to batch check execution statuses: {e}"
            ) from e

    async def wait_for_execution_result(
        self, execution_id: str, timeout: Optional[float] = None
    ) -> Any:
        """
        Wait for execution completion with polling.

        Args:
            execution_id: Execution ID to wait for
            timeout: Optional timeout override (uses config default if None)

        Returns:
            Any: Execution result

        Raises:
            AgentFieldClientError: If async execution is disabled.
            ExecutionTimeoutError: If execution times out.
        """
        if not self.async_config.enable_async_execution:
            raise AgentFieldClientError("Async execution is disabled in configuration")

        try:
            manager = await self._get_async_execution_manager()
            result = await manager.wait_for_result(execution_id, timeout)

            logger.debug(f"Execution {execution_id[:8]}... completed successfully")
            return result

        except TimeoutError as exc:
            logger.error(
                f"Failed to wait for execution result {execution_id[:8]}...: {exc}"
            )
            raise ExecutionTimeoutError(
                f"Execution {execution_id} exceeded timeout"
            ) from exc
        except AgentFieldError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to wait for execution result {execution_id[:8]}...: {e}"
            )
            raise AgentFieldClientError(
                f"Failed to wait for execution result: {e}"
            ) from e

    async def cancel_async_execution(
        self, execution_id: str, reason: Optional[str] = None
    ) -> bool:
        """
        Cancel an active async execution.

        Args:
            execution_id: Execution ID to cancel
            reason: Optional cancellation reason

        Returns:
            bool: True if execution was cancelled, False if not found or already terminal

        Raises:
            AgentFieldClientError: If async execution is disabled.
        """
        if not self.async_config.enable_async_execution:
            raise AgentFieldClientError("Async execution is disabled in configuration")

        try:
            manager = await self._get_async_execution_manager()
            cancelled = await manager.cancel_execution(execution_id, reason)

            if cancelled:
                logger.debug(
                    f"Cancelled execution {execution_id[:8]}... - {reason or 'No reason provided'}"
                )
            else:
                logger.debug(
                    f"Could not cancel execution {execution_id[:8]}... (not found or already terminal)"
                )

            return cancelled

        except AgentFieldError:
            raise
        except Exception as e:
            logger.error(f"Failed to cancel execution {execution_id[:8]}...: {e}")
            raise AgentFieldClientError(
                f"Failed to cancel execution: {e}"
            ) from e

    async def list_async_executions(
        self, status_filter: Optional[str] = None, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
                List async executions with optional filtering.

                Args:
        status_filter: Optional status to filter by ('pending', 'queued', 'running', 'succeeded', 'failed', etc.)
                    limit: Optional limit on number of results

                Returns:
                    List[Dict]: List of execution status dictionaries

                Raises:
                    AgentFieldClientError: If async execution is disabled.
        """
        if not self.async_config.enable_async_execution:
            raise AgentFieldClientError("Async execution is disabled in configuration")

        try:
            manager = await self._get_async_execution_manager()

            # Convert string status to ExecutionStatus enum if provided
            status_enum = None
            if status_filter:
                try:
                    status_enum = ExecutionStatus(status_filter.lower())
                except ValueError:
                    logger.warning(f"Invalid status filter: {status_filter}")
                    return []

            executions = await manager.list_executions(status_enum, limit)
            logger.debug(f"Listed {len(executions)} async executions")

            return executions

        except AgentFieldError:
            raise
        except Exception as e:
            logger.error(f"Failed to list async executions: {e}")
            raise AgentFieldClientError(
                f"Failed to list async executions: {e}"
            ) from e

    async def get_async_execution_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics for async execution manager.

        Returns:
            Dict[str, Any]: Metrics dictionary with execution statistics

        Raises:
            AgentFieldClientError: If async execution is disabled.
        """
        if not self.async_config.enable_async_execution:
            raise AgentFieldClientError("Async execution is disabled in configuration")

        try:
            if self._async_execution_manager is None:
                return {
                    "manager_started": False,
                    "message": "Async execution manager not yet initialized",
                }

            metrics = self._async_execution_manager.get_metrics()
            logger.debug("Retrieved async execution metrics")

            return metrics

        except AgentFieldError:
            raise
        except Exception as e:
            logger.error(f"Failed to get async execution metrics: {e}")
            raise AgentFieldClientError(
                f"Failed to get async execution metrics: {e}"
            ) from e

    async def cleanup_async_executions(self) -> int:
        """
        Manually trigger cleanup of completed executions.

        Returns:
            int: Number of executions cleaned up

        Raises:
            AgentFieldClientError: If async execution is disabled.
        """
        if not self.async_config.enable_async_execution:
            raise AgentFieldClientError("Async execution is disabled in configuration")

        try:
            if self._async_execution_manager is None:
                return 0

            cleanup_count = (
                await self._async_execution_manager.cleanup_completed_executions()
            )
            logger.debug(f"Cleaned up {cleanup_count} completed async executions")

            return cleanup_count

        except AgentFieldError:
            raise
        except Exception as e:
            logger.error(f"Failed to cleanup async executions: {e}")
            raise AgentFieldClientError(
                f"Failed to cleanup async executions: {e}"
            ) from e

    # ------------------------------------------------------------------ #
    # Approval helpers                                                     #
    # ------------------------------------------------------------------ #

    async def request_approval(
        self,
        execution_id: str,
        approval_request_id: str,
        approval_request_url: str = "",
        callback_url: str = "",
        expires_in_hours: int = 72,
    ) -> ApprovalRequestResponse:
        """Request human approval for an execution, transitioning it to ``waiting``.

        Calls ``POST /api/v1/agents/{node}/executions/{id}/request-approval``
        on the control plane.  The agent is responsible for creating the
        approval request on an external service (e.g. hax-sdk) first and
        passing the resulting IDs here so the CP can track it.

        Args:
            execution_id: The execution to pause for approval.
            approval_request_id: ID of the approval request on the external service.
            approval_request_url: URL where the human can review the request.
            callback_url: URL the CP should POST to when the approval resolves.
            expires_in_hours: Time before the request expires.

        Returns:
            ApprovalRequestResponse with ``approval_request_id`` and ``approval_request_url``.

        Raises:
            AgentFieldClientError: If the request fails.
        """
        node_id = self.caller_agent_id or ""
        body: Dict[str, Any] = {
            "approval_request_id": approval_request_id,
            "expires_in_hours": expires_in_hours,
        }
        if approval_request_url:
            body["approval_request_url"] = approval_request_url
        if callback_url:
            body["callback_url"] = callback_url
        url = f"{self.api_base}/agents/{node_id}/executions/{execution_id}/request-approval"

        try:
            client = await self.get_async_http_client()
            response = await client.post(
                url,
                json=body,
                headers=self._sanitize_header_values(self._get_headers_with_context(None)),
                timeout=30,
            )
        except Exception as exc:
            raise AgentFieldClientError(
                f"Failed to request approval: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise AgentFieldClientError(
                f"Approval request failed ({response.status_code}): {response.text[:500]}"
            )

        data = response.json()
        return ApprovalRequestResponse(
            approval_request_id=data.get("approval_request_id", ""),
            approval_request_url=data.get("approval_request_url", ""),
        )

    async def get_approval_status(
        self,
        execution_id: str,
    ) -> ApprovalStatusResponse:
        """Get the current approval status for an execution.

        Calls ``GET /api/v1/agents/{node}/executions/{id}/approval-status``.

        Returns:
            ApprovalStatusResponse with ``status`` (pending/approved/rejected/expired),
            ``response``, ``request_url``, ``requested_at``, ``responded_at``.

        Raises:
            AgentFieldClientError: If the request fails.
        """
        node_id = self.caller_agent_id or ""
        url = f"{self.api_base}/agents/{node_id}/executions/{execution_id}/approval-status"

        try:
            client = await self.get_async_http_client()
            response = await client.get(
                url,
                headers=self._sanitize_header_values(self._get_headers_with_context(None)),
                timeout=30,
            )
        except Exception as exc:
            raise AgentFieldClientError(
                f"Failed to get approval status: {exc}"
            ) from exc

        if response.status_code >= 400:
            raise AgentFieldClientError(
                f"Approval status request failed ({response.status_code}): {response.text[:500]}"
            )

        data = response.json()
        return ApprovalStatusResponse(
            status=data.get("status", "unknown"),
            response=data.get("response"),
            request_url=data.get("request_url"),
            requested_at=data.get("requested_at"),
            responded_at=data.get("responded_at"),
        )

    async def wait_for_approval(
        self,
        execution_id: str,
        poll_interval: float = 5.0,
        max_interval: float = 60.0,
        timeout: Optional[float] = None,
    ) -> ApprovalStatusResponse:
        """Poll approval status with exponential backoff until resolved.

        Args:
            execution_id: Execution ID to wait for.
            poll_interval: Initial polling interval in seconds.
            max_interval: Maximum polling interval in seconds.
            timeout: Total timeout in seconds (None = wait indefinitely).

        Returns:
            ApprovalStatusResponse with the final approval status (approved/rejected/expired).

        Raises:
            AgentFieldClientError: If polling encounters a non-retryable error.
            ExecutionTimeoutError: If timeout is reached.
        """
        start_time = time.time()
        interval = poll_interval
        backoff_factor = 2.0

        while True:
            if timeout is not None and (time.time() - start_time) >= timeout:
                raise ExecutionTimeoutError(
                    f"Approval for execution {execution_id} timed out after {timeout}s"
                )

            await asyncio.sleep(interval)

            try:
                result = await self.get_approval_status(execution_id)
            except AgentFieldClientError:
                # Transient failure — back off and retry
                interval = min(interval * backoff_factor, max_interval)
                continue

            if result.status != "pending":
                return result

            interval = min(interval * backoff_factor, max_interval)

    async def post_execution_logs(
        self,
        execution_id: str,
        entries: Any,
    ) -> None:
        """POST structured execution log entries to the control plane.

        Args:
            execution_id: The execution these logs belong to.
            entries: A single log entry dict or a list of entry dicts.
        """
        if not execution_id:
            return
        url = f"{self.api_base}/executions/{execution_id}/logs"
        body: Dict[str, Any]
        if isinstance(entries, list):
            body = {"entries": entries}
        else:
            body = {"entries": [entries]}
        try:
            client = await self.get_async_http_client()
            await client.post(
                url,
                json=body,
                headers=self._get_auth_headers(),
                timeout=5.0,
            )
        except Exception:
            # Best-effort: don't break execution if log dispatch fails
            pass

    async def close_async_execution_manager(self) -> None:
        """
        Close the async execution manager and cleanup resources.

        This should be called when the AgentFieldClient is no longer needed
        to ensure proper cleanup of background tasks and connections.
        """
        if self._async_execution_manager is not None:
            try:
                await self._async_execution_manager.stop()
                self._async_execution_manager = None
                logger.debug("Async execution manager closed successfully")
            except Exception as e:
                logger.error(f"Error closing async execution manager: {e}")
                raise
