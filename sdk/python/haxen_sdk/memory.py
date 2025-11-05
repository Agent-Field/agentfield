"""
Cross-Agent Persistent Memory Client for Haxen SDK.

This module provides the memory interface that enables seamless, automatic memory
sharing and synchronization across distributed agents.
"""

import asyncio
import json
from typing import Any, List, Optional, Union
from .client import HaxenClient
from .execution_context import ExecutionContext
from .memory_events import MemoryEventClient, ScopedMemoryEventClient


class MemoryClient:
    """
    Core memory client that communicates with the Haxen server's memory API.

    This client handles the low-level HTTP operations for memory management
    and automatically includes execution context headers for proper scoping.
    """

    def __init__(self, haxen_client: HaxenClient, execution_context: ExecutionContext):
        self.haxen_client = haxen_client
        self.execution_context = execution_context

    async def _async_request(self, method: str, url: str, **kwargs):
        """Internal helper to perform HTTP requests with graceful fallbacks."""
        if hasattr(self.haxen_client, "_async_request"):
            return await self.haxen_client._async_request(method, url, **kwargs)

        try:
            import httpx

            async with httpx.AsyncClient() as client:
                return await client.request(method, url, **kwargs)
        except ImportError:
            import requests

            return await asyncio.to_thread(requests.request, method, url, **kwargs)

    async def set(self, key: str, data: Any, scope: Optional[str] = None) -> None:
        """
        Set a memory value with automatic scoping.

        Args:
            key: The memory key
            data: The data to store (will be JSON serialized)
            scope: Optional explicit scope override
        """
        from haxen_sdk.logger import log_debug

        headers = self.execution_context.to_headers()

        payload = {"key": key, "data": data}

        if scope:
            payload["scope"] = scope

        # Test JSON serialization before sending
        try:
            json.dumps(payload)
            log_debug(f"Memory set operation for key: {key}")
        except Exception as json_error:
            log_debug(
                f"JSON serialization failed for memory key {key}: {type(json_error).__name__}: {json_error}"
            )
            raise

        # Use synchronous requests to avoid event loop conflicts with Haxen SDK
        url = f"{self.haxen_client.api_base}/memory/set"

        try:
            if hasattr(self.haxen_client, "_async_request"):
                response = await self.haxen_client._async_request(
                    "POST",
                    url,
                    json=payload,
                    headers=headers,
                    timeout=10.0,
                )
            else:
                import requests

                response = await asyncio.to_thread(
                    requests.post,
                    url,
                    json=payload,
                    headers=headers,
                    timeout=10.0,
                )
            response.raise_for_status()
            log_debug(f"Memory set successful for key: {key}")
        except Exception as e:
            log_debug(f"Memory set failed for key {key}: {type(e).__name__}: {e}")
            raise

    async def get(
        self, key: str, default: Any = None, scope: Optional[str] = None
    ) -> Any:
        """
        Get a memory value with hierarchical lookup.

        Args:
            key: The memory key
            default: Default value if key not found
            scope: Optional explicit scope override

        Returns:
            The stored value or default if not found
        """
        headers = self.execution_context.to_headers()

        payload = {"key": key}

        if scope:
            payload["scope"] = scope

        response = await self._async_request(
            "POST",
            f"{self.haxen_client.api_base}/memory/get",
            json=payload,
            headers=headers,
            timeout=10.0,
        )

        if response.status_code == 404:
            return default

        response.raise_for_status()
        result = response.json()

        # Extract the actual data from the memory response
        if isinstance(result, dict) and "data" in result:
            # The server returns JSON-encoded data, so we need to decode it
            data = result["data"]
            if isinstance(data, str):
                try:
                    return json.loads(data)
                except json.JSONDecodeError:
                    return data
            return data

        return result

    async def exists(self, key: str, scope: Optional[str] = None) -> bool:
        """
        Check if a memory key exists.

        Args:
            key: The memory key
            scope: Optional explicit scope override

        Returns:
            True if key exists, False otherwise
        """
        try:
            await self.get(key, scope=scope)
            return True
        except Exception:
            return False

    async def delete(self, key: str, scope: Optional[str] = None) -> None:
        """
        Delete a memory value.

        Args:
            key: The memory key
            scope: Optional explicit scope override
        """
        headers = self.execution_context.to_headers()

        payload = {"key": key}

        if scope:
            payload["scope"] = scope

        response = await self._async_request(
            "DELETE",
            f"{self.haxen_client.api_base}/memory/delete",
            json=payload,
            headers=headers,
            timeout=10.0,
        )
        response.raise_for_status()

    async def list_keys(self, scope: str) -> List[str]:
        """
        List all keys in a specific scope.

        Args:
            scope: The scope to list keys from

        Returns:
            List of memory keys in the scope
        """
        headers = self.execution_context.to_headers()

        response = await self._async_request(
            "GET",
            f"{self.haxen_client.api_base}/memory/list",
            params={"scope": scope},
            headers=headers,
            timeout=10.0,
        )
        response.raise_for_status()
        result = response.json()

        # Extract keys from the memory list response
        if isinstance(result, list):
            return [item.get("key", "") for item in result if "key" in item]

        return []


class ScopedMemoryClient:
    """
    Memory client that operates within a specific scope.

    This provides a scoped view of memory operations, automatically
    using the specified scope for all operations.
    """

    def __init__(
        self,
        memory_client: MemoryClient,
        scope: str,
        scope_id: str,
        event_client: Optional[MemoryEventClient] = None,
    ):
        self.memory_client = memory_client
        self.scope = scope
        self.scope_id = scope_id
        self.events = (
            ScopedMemoryEventClient(event_client, scope, scope_id)
            if event_client
            else None
        )

    async def set(self, key: str, data: Any) -> None:
        """Set a value in this specific scope."""
        await self.memory_client.set(key, data, scope=self.scope)

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from this specific scope."""
        return await self.memory_client.get(key, default=default, scope=self.scope)

    async def exists(self, key: str) -> bool:
        """Check if a key exists in this specific scope."""
        return await self.memory_client.exists(key, scope=self.scope)

    async def delete(self, key: str) -> None:
        """Delete a value from this specific scope."""
        await self.memory_client.delete(key, scope=self.scope)

    async def list_keys(self) -> List[str]:
        """List all keys in this specific scope."""
        return await self.memory_client.list_keys(self.scope)

    def on_change(self, patterns: Union[str, List[str]]):
        """
        Decorator for subscribing to memory change events in this scope.

        Args:
            patterns: Pattern(s) to match against memory keys

        Returns:
            Decorator function
        """
        if self.events:
            return self.events.on_change(patterns)
        else:
            # Return a no-op decorator if events are not available
            def decorator(func):
                return func

            return decorator


class GlobalMemoryClient:
    """
    Memory client for global scope operations.

    This provides access to the global memory scope that is shared
    across all agents and sessions.
    """

    def __init__(self, memory_client: MemoryClient):
        self.memory_client = memory_client

    async def set(self, key: str, data: Any) -> None:
        """Set a value in global scope."""
        await self.memory_client.set(key, data, scope="global")

    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from global scope."""
        return await self.memory_client.get(key, default=default, scope="global")

    async def exists(self, key: str) -> bool:
        """Check if a key exists in global scope."""
        return await self.memory_client.exists(key, scope="global")

    async def delete(self, key: str) -> None:
        """Delete a value from global scope."""
        await self.memory_client.delete(key, scope="global")

    async def list_keys(self) -> List[str]:
        """List all keys in global scope."""
        return await self.memory_client.list_keys("global")


class MemoryInterface:
    """
    Developer-facing memory interface that provides the intuitive app.memory API.

    This class provides the main interface that developers interact with,
    offering automatic scoping, hierarchical lookup, and explicit scope access.
    """

    def __init__(self, memory_client: MemoryClient, event_client: MemoryEventClient):
        self.memory_client = memory_client
        self.events = event_client

    async def set(self, key: str, data: Any) -> None:
        """
        Set a memory value with automatic scoping.

        The value will be stored in the most specific available scope
        based on the current execution context.

        Args:
            key: The memory key
            data: The data to store
        """
        await self.memory_client.set(key, data)

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a memory value with hierarchical lookup.

        This will search through scopes in order: workflow -> session -> actor -> global
        and return the first match found.

        Args:
            key: The memory key
            default: Default value if key not found in any scope

        Returns:
            The stored value or default if not found
        """
        return await self.memory_client.get(key, default=default)

    async def exists(self, key: str) -> bool:
        """
        Check if a memory key exists in any scope.

        Args:
            key: The memory key

        Returns:
            True if key exists in any scope, False otherwise
        """
        return await self.memory_client.exists(key)

    async def delete(self, key: str) -> None:
        """
        Delete a memory value from the current scope.

        Args:
            key: The memory key
        """
        await self.memory_client.delete(key)

    def on_change(self, patterns: Union[str, List[str]]):
        """
        Decorator for subscribing to memory change events.

        Args:
            patterns: Pattern(s) to match against memory keys

        Returns:
            Decorator function
        """
        return self.events.on_change(patterns)

    def session(self, session_id: str) -> ScopedMemoryClient:
        """
        Get a memory client scoped to a specific session.

        Args:
            session_id: The session ID to scope to

        Returns:
            ScopedMemoryClient for the specified session
        """
        return ScopedMemoryClient(
            self.memory_client, "session", session_id, self.events
        )

    def actor(self, actor_id: str) -> ScopedMemoryClient:
        """
        Get a memory client scoped to a specific actor.

        Args:
            actor_id: The actor ID to scope to

        Returns:
            ScopedMemoryClient for the specified actor
        """
        return ScopedMemoryClient(self.memory_client, "actor", actor_id, self.events)

    def workflow(self, workflow_id: str) -> ScopedMemoryClient:
        """
        Get a memory client scoped to a specific workflow.

        Args:
            workflow_id: The workflow ID to scope to

        Returns:
            ScopedMemoryClient for the specified workflow
        """
        return ScopedMemoryClient(
            self.memory_client, "workflow", workflow_id, self.events
        )

    @property
    def global_scope(self) -> GlobalMemoryClient:
        """
        Get a memory client for global scope operations.

        Returns:
            GlobalMemoryClient for global scope access
        """
        return GlobalMemoryClient(self.memory_client)
