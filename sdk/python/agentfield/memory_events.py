import asyncio
import json
import re
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

import websockets

from agentfield.logger import log_error, log_info
from .types import MemoryChangeEvent


class PatternMatcher:
    """Utility class for wildcard pattern matching."""

    @staticmethod
    def matches_pattern(pattern: str, key: str) -> bool:
        """
        Check if a key matches a wildcard pattern.

        Args:
            pattern: Pattern with wildcards (e.g., "customer_*", "user_*.preferences")
            key: Key to match against

        Returns:
            True if key matches pattern, False otherwise
        """
        # Convert wildcard pattern to regex
        regex_pattern = pattern.replace("*", ".*")
        regex_pattern = f"^{regex_pattern}$"

        try:
            return bool(re.match(regex_pattern, key))
        except re.error:
            # If regex is invalid, fall back to exact match
            return pattern == key


class EventSubscription:
    """Represents an event subscription with patterns and callback."""

    def __init__(
        self,
        patterns: List[str],
        callback: Callable,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ):
        self.patterns = patterns
        self.callback = callback
        self.scope = scope
        self.scope_id = scope_id
        self.active = True

    def matches_event(self, event: MemoryChangeEvent) -> bool:
        """Check if this subscription matches the given event."""
        if not self.active:
            return False

        # Check scope if specified
        if self.scope and event.scope != self.scope:
            return False
        if self.scope_id and event.scope_id != self.scope_id:
            return False

        # Check if any pattern matches
        for pattern in self.patterns:
            if PatternMatcher.matches_pattern(pattern, event.key):
                return True

        return False

    def unsubscribe(self):
        """Mark this subscription as inactive."""
        self.active = False


class MemoryEventClient:
    """Enhanced memory event client with pattern-based subscriptions and event history."""

    def __init__(self, base_url: str, execution_context, api_key: Optional[str] = None):
        self.base_url = base_url.replace("http", "ws")
        self.execution_context = execution_context
        self.api_key = api_key
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_listening = False
        # Lazily initialize the lock inside an active event loop to avoid
        # `RuntimeError: There is no current event loop` in synchronous contexts.
        self._connect_lock: Optional[asyncio.Lock] = None
        self.subscriptions: List[EventSubscription] = []
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 1.0

    def _is_connected(self) -> bool:
        """
        Safely determine if the WebSocket connection is open.

        Supports both legacy WebSocketClientProtocol (with `open`) and
        newer ClientConnection objects (with `closed`).
        """
        if not self.websocket:
            return False

        open_attr = getattr(self.websocket, "open", None)
        if isinstance(open_attr, bool):
            return open_attr

        closed_attr = getattr(self.websocket, "closed", None)
        if isinstance(closed_attr, bool):
            return not closed_attr

        # Fallback: assume connected if we have a websocket object
        return True

    async def connect(
        self,
        patterns: Optional[List[str]] = None,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ):
        """
        Establishes a WebSocket connection with optional filtering.

        Args:
            patterns: List of patterns to subscribe to
            scope: Scope to filter events by
            scope_id: Scope ID to filter events by
        """
        if self._connect_lock is None:
            self._connect_lock = asyncio.Lock()

        async with self._connect_lock:
            if self._is_connected():
                return

            try:
                headers = self.execution_context.to_headers()
                if self.api_key:
                    headers["X-API-Key"] = self.api_key
                ws_url = f"{self.base_url}/api/v1/memory/events/ws"

                # Add query parameters for server-side filtering
                query_params = []
                if patterns:
                    query_params.append(f"patterns={','.join(patterns)}")
                if scope:
                    query_params.append(f"scope={scope}")
                if scope_id:
                    query_params.append(f"scope_id={scope_id}")

                if query_params:
                    ws_url += "?" + "&".join(query_params)

                self.websocket = await websockets.connect(
                    ws_url, additional_headers=headers
                )
                self.is_listening = True
                self._reconnect_attempts = 0
                asyncio.create_task(self._listen())

            except Exception as e:
                log_error(f"Failed to connect to memory events: {e}")
                await self._handle_reconnect()

    async def _listen(self):
        """Listens for incoming messages and dispatches them to subscribers."""
        if not self.websocket:
            return

        while self.is_listening:
            try:
                message = await self.websocket.recv()
                event_data = json.loads(message)
                event = MemoryChangeEvent.from_dict(event_data)

                # Dispatch to matching subscriptions
                for subscription in self.subscriptions:
                    if subscription.matches_event(event):
                        try:
                            asyncio.create_task(subscription.callback(event))
                        except Exception as e:
                            log_error(f"Error in event callback: {e}")

            except websockets.exceptions.ConnectionClosed:
                # Connection closed cleanly or unexpectedly; try to reconnect
                self.is_listening = False
                self.websocket = None
                if self._reconnect_attempts < self._max_reconnect_attempts:
                    await self._handle_reconnect()
                break
            except Exception as e:
                # Any unexpected error in the listener should reset the connection
                log_error(f"Error in event listener: {e}")
                self.is_listening = False
                if self.websocket:
                    try:
                        await self.websocket.close()
                    except Exception:
                        pass
                    self.websocket = None
                if self._reconnect_attempts < self._max_reconnect_attempts:
                    await self._handle_reconnect()
                break

    async def _handle_reconnect(self):
        """Handle automatic reconnection with exponential backoff."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            log_error(
                f"Max reconnection attempts reached ({self._max_reconnect_attempts})"
            )
            return

        self._reconnect_attempts += 1
        delay = self._reconnect_delay * (2 ** (self._reconnect_attempts - 1))

        log_info(
            f"Reconnecting to memory events (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}) in {delay}s..."
        )
        await asyncio.sleep(delay)

        try:
            await self.connect()
        except Exception as e:
            log_error(f"Reconnection failed: {e}")

    def subscribe(
        self,
        patterns: Union[str, List[str]],
        callback: Callable,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> EventSubscription:
        """
        Subscribe to memory change events with pattern matching.

        Args:
            patterns: Pattern(s) to match against memory keys
            callback: Function to call when matching events occur
            scope: Optional scope to filter by
            scope_id: Optional scope ID to filter by

        Returns:
            EventSubscription object that can be used to unsubscribe
        """
        if isinstance(patterns, str):
            patterns = [patterns]

        subscription = EventSubscription(patterns, callback, scope, scope_id)
        self.subscriptions.append(subscription)

        # If not connected, establish (or re-establish) the WebSocket connection.
        # We rely on client-side pattern matching, so we don't need to send
        # pattern filters to the server.
        if not self._is_connected():
            asyncio.create_task(self.connect())

        return subscription

    def on_change(self, patterns: Union[str, List[str]]):
        """
        Decorator for subscribing to memory change events.

        Args:
            patterns: Pattern(s) to match against memory keys

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(event: MemoryChangeEvent):
                return await func(event)

            # Subscribe to the patterns
            self.subscribe(patterns, wrapper)

            # Mark the function as a memory event listener using setattr to avoid type errors
            setattr(wrapper, "_memory_event_listener", True)
            setattr(
                wrapper,
                "_memory_event_patterns",
                patterns if isinstance(patterns, list) else [patterns],
            )

            return wrapper

        return decorator

    async def history(
        self,
        patterns: Optional[Union[str, List[str]]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
        scope: Optional[str] = None,
        scope_id: Optional[str] = None,
    ) -> List[MemoryChangeEvent]:
        """
        Get historical memory change events.

        Args:
            patterns: Pattern(s) to filter events by
            since: Only return events after this timestamp
            limit: Maximum number of events to return
            scope: Scope to filter by
            scope_id: Scope ID to filter by

        Returns:
            List of historical memory change events
        """
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                headers = self.execution_context.to_headers()
                if self.api_key:
                    headers["X-API-Key"] = self.api_key

                # Build query parameters
                params: Dict[str, Any] = {"limit": limit}
                if patterns:
                    if isinstance(patterns, str):
                        patterns = [patterns]
                    params["patterns"] = ",".join(patterns)
                if since:
                    params["since"] = since.isoformat()
                if scope:
                    params["scope"] = scope
                if scope_id:
                    params["scope_id"] = scope_id

                # Make request to history endpoint
                http_url = self.base_url.replace("ws", "http")
                response = await client.get(
                    f"{http_url}/api/v1/memory/events/history",
                    params=params,
                    headers=headers,
                    timeout=10.0,
                )
                response.raise_for_status()

                # Parse response
                events_data = response.json()
                events = []

                if isinstance(events_data, list):
                    for event_data in events_data:
                        try:
                            event = MemoryChangeEvent.from_dict(event_data)
                            events.append(event)
                        except Exception as e:
                            log_error(f"Failed to parse event: {e}")

                return events

        except ImportError:
            # Fallback to synchronous requests
            import requests

            headers = self.execution_context.to_headers()
            if self.api_key:
                headers["X-API-Key"] = self.api_key

            # Build query parameters
            params = {"limit": limit}
            if patterns:
                if isinstance(patterns, str):
                    patterns = [patterns]
                params["patterns"] = ",".join(patterns)
            if since:
                params["since"] = since.isoformat()
            if scope:
                params["scope"] = scope
            if scope_id:
                params["scope_id"] = scope_id

            # Make request to history endpoint
            http_url = self.base_url.replace("ws", "http")
            response = requests.get(
                f"{http_url}/api/v1/memory/events/history",
                params=params,
                headers=headers,
                timeout=10.0,
            )
            response.raise_for_status()

            # Parse response
            events_data = response.json()
            events = []

            if isinstance(events_data, list):
                for event_data in events_data:
                    try:
                        event = MemoryChangeEvent.from_dict(event_data)
                        events.append(event)
                    except Exception as e:
                        log_error(f"Failed to parse event: {e}")

            return events

        except Exception as e:
            log_error(f"Failed to get event history: {e}")
            return []

    def unsubscribe_all(self):
        """Unsubscribe from all event subscriptions."""
        for subscription in self.subscriptions:
            subscription.unsubscribe()
        self.subscriptions.clear()

    async def close(self):
        """Closes the WebSocket connection and cleans up subscriptions."""
        self.is_listening = False
        self.unsubscribe_all()

        if self.websocket:
            await self.websocket.close()
            self.websocket = None


class ScopedMemoryEventClient:
    """Memory event client scoped to a specific context."""

    def __init__(self, event_client: MemoryEventClient, scope: str, scope_id: str):
        self.event_client = event_client
        self.scope = scope
        self.scope_id = scope_id

    def on_change(self, patterns: Union[str, List[str]]):
        """
        Decorator for subscribing to scoped memory change events.

        Args:
            patterns: Pattern(s) to match against memory keys

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(event: MemoryChangeEvent):
                return await func(event)

            # Subscribe to the patterns with scope filtering
            self.event_client.subscribe(
                patterns, wrapper, scope=self.scope, scope_id=self.scope_id
            )

            # Mark the function as a memory event listener using setattr to avoid type errors
            setattr(wrapper, "_memory_event_listener", True)
            setattr(
                wrapper,
                "_memory_event_patterns",
                patterns if isinstance(patterns, list) else [patterns],
            )
            setattr(wrapper, "_memory_event_scope", self.scope)
            setattr(wrapper, "_memory_event_scope_id", self.scope_id)

            return wrapper

        return decorator

    async def history(
        self,
        patterns: Optional[Union[str, List[str]]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[MemoryChangeEvent]:
        """
        Get historical memory change events for this scope.

        Args:
            patterns: Pattern(s) to filter events by
            since: Only return events after this timestamp
            limit: Maximum number of events to return

        Returns:
            List of historical memory change events
        """
        return await self.event_client.history(
            patterns=patterns,
            since=since,
            limit=limit,
            scope=self.scope,
            scope_id=self.scope_id,
        )
