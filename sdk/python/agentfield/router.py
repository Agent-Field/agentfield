"""AgentRouter provides FastAPI-style organization for agent reasoners and skills."""

from __future__ import annotations

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

    # ------------------------------------------------------------------
    # Registration helpers
    def reasoner(
        self, path: Optional[str] = None, **kwargs: Any
    ) -> Callable[[Callable], Callable]:
        """Store a reasoner definition for later registration on an Agent."""

        def decorator(func: Callable) -> Callable:
            self.reasoners.append(
                {
                    "func": func,
                    "path": path,
                    "kwargs": kwargs,
                    "registered": False,
                }
            )
            return func

        return decorator

    def skill(
        self,
        tags: Optional[List[str]] = None,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> Callable[[Callable], Callable]:
        """Store a skill definition, merging router and local tags."""

        def decorator(func: Callable) -> Callable:
            merged_tags = self.tags + (tags or [])
            self.skills.append(
                {
                    "func": func,
                    "path": path,
                    "tags": merged_tags,
                    "kwargs": kwargs,
                    "registered": False,
                }
            )
            return func

        return decorator

    # ------------------------------------------------------------------
    # Agent delegation helpers
    async def ai(self, *args: Any, **kwargs: Any) -> Any:
        agent = self._require_agent()
        return await agent.ai(*args, **kwargs)

    async def call(self, target: str, *args: Any, **kwargs: Any) -> Any:
        agent = self._require_agent()
        return await agent.call(target, *args, **kwargs)

    @property
    def memory(self):  # type: ignore[override]
        agent = self._require_agent()
        return agent.memory

    # ------------------------------------------------------------------
    # Internal helpers
    def _require_agent(self) -> "Agent":
        if not self._agent:
            raise RuntimeError(
                "Router not attached to an agent. Call Agent.include_router(router) first."
            )
        return self._agent

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
