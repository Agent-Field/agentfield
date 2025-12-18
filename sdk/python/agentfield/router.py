"""AgentRouter provides FastAPI-style organization for agent reasoners and skills."""

from __future__ import annotations

import asyncio
import functools
import inspect

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .agent import Agent


class AgentRouter:
    """Collects reasoners and skills before registering them on an Agent."""

    def __init__(self, prefix: str = "", tags: Optional[List[str]] = None):
        self.prefix = prefix.rstrip("/") if prefix else ""
        self.tags = tags or []
        self.reasoners: List[Dict[str, Any]] = []
        self.skills: List[Dict[str, Any]] = []
        self._agent: Optional["Agent"] = None
        self._tracked_functions: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Registration helpers
    def reasoner(
        self,
        path: Optional[str] = None,
        *,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Callable[[Callable], Callable]:
        """Store a reasoner definition for later registration on an Agent.

        Returns a wrapper function that delegates to the tracked version once
        the router is attached to an agent. This ensures that direct calls
        between reasoners go through workflow tracking.
        """

        direct_registration: Optional[Callable] = None
        decorator_path = path
        decorator_tags = tags
        decorator_kwargs = dict(kwargs)

        if decorator_path and (
            inspect.isfunction(decorator_path) or inspect.ismethod(decorator_path)
        ):
            direct_registration = decorator_path
            decorator_path = None

        router_ref = self

        def decorator(func: Callable) -> Callable:
            merged_tags = router_ref.tags + (decorator_tags or [])
            func_name = func.__name__

            @functools.wraps(func)
            async def wrapper(*args: Any, **kw: Any) -> Any:
                # Look up the tracked function at call time
                tracked = router_ref._tracked_functions.get(func_name)
                if tracked is not None and tracked is not wrapper:
                    # Call the tracked version for proper workflow instrumentation
                    return await tracked(*args, **kw)
                # Fallback to original if not yet registered
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kw)
                return func(*args, **kw)

            # Store metadata on the wrapper
            wrapper._is_router_reasoner = True
            wrapper._original_func = func

            router_ref.reasoners.append(
                {
                    "func": func,
                    "wrapper": wrapper,
                    "path": decorator_path,
                    "tags": merged_tags,
                    "kwargs": dict(decorator_kwargs),
                    "registered": False,
                }
            )
            return wrapper

        if direct_registration:
            return decorator(direct_registration)

        return decorator

    def skill(
        self,
        tags: Optional[List[str]] = None,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> Callable[[Callable], Callable]:
        """Store a skill definition, merging router and local tags."""

        direct_registration: Optional[Callable] = None
        decorator_tags = tags
        decorator_path = path
        decorator_kwargs = dict(kwargs)

        if decorator_tags and (
            inspect.isfunction(decorator_tags) or inspect.ismethod(decorator_tags)
        ):
            direct_registration = decorator_tags
            decorator_tags = None

        def decorator(func: Callable) -> Callable:
            merged_tags = self.tags + (decorator_tags or [])
            self.skills.append(
                {
                    "func": func,
                    "path": decorator_path,
                    "tags": merged_tags,
                    "kwargs": decorator_kwargs,
                    "registered": False,
                }
            )
            return func

        if direct_registration:
            return decorator(direct_registration)

        return decorator

    # ------------------------------------------------------------------
    # Automatic delegation via __getattr__
    def __getattr__(self, name: str) -> Any:
        """
        Automatically delegate any unknown attribute/method to the attached agent.

        This allows AgentRouter to transparently proxy all Agent methods (like ai(),
        call(), memory, note(), discover(), etc.) without explicitly defining
        delegation methods for each one.

        Args:
            name: The attribute/method name being accessed

        Returns:
            The attribute/method from the attached agent

        Raises:
            RuntimeError: If router is not attached to an agent
            AttributeError: If the agent doesn't have the requested attribute
        """
        # Avoid infinite recursion by accessing _agent through object.__getattribute__
        try:
            agent = object.__getattribute__(self, '_agent')
        except AttributeError:
            raise RuntimeError(
                "Router not attached to an agent. Call Agent.include_router(router) first."
            )

        if agent is None:
            raise RuntimeError(
                "Router not attached to an agent. Call Agent.include_router(router) first."
            )

        # Delegate to the agent - will raise AttributeError if not found
        return getattr(agent, name)

    @property
    def app(self) -> "Agent":
        """Access the underlying Agent instance."""
        if not self._agent:
            raise RuntimeError(
                "Router not attached to an agent. Call Agent.include_router(router) first."
            )
        return self._agent

    # ------------------------------------------------------------------
    # Internal helpers

    def _combine_path(
        self,
        default: Optional[str],
        custom: Optional[str],
        override_prefix: Optional[str] = None,
    ) -> Optional[str]:
        """Return a normalized API path for a registered function."""

        if custom and custom.startswith("/"):
            return custom

        segments: List[str] = []

        prefixes: List[str] = []
        for prefix in (override_prefix, self.prefix):
            if prefix:
                prefixes.append(prefix.strip("/"))

        if custom:
            segments.extend(prefixes)
            segments.append(custom.strip("/"))
        elif default:
            stripped = default.strip("/")
            if stripped.startswith("reasoners/") or stripped.startswith("skills/"):
                head, *tail = stripped.split("/")
                segments.append(head)
                segments.extend(prefixes)
                segments.extend(tail)
            else:
                segments.extend(prefixes)
                if stripped:
                    segments.append(stripped)
        else:
            segments.extend(prefixes)

        if not segments:
            return default

        combined = "/".join(segment for segment in segments if segment)
        return f"/{combined}" if combined else "/"

    def _attach_agent(self, agent: "Agent") -> None:
        self._agent = agent
