"""
MediaRouter — prefix-based provider dispatch for media generation.

Routes model strings to the correct MediaProvider by matching the longest
registered prefix that also supports the requested capability (modality).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentfield.media_providers import MediaProvider


class MediaRouter:
    """
    Resolve ``(model, capability)`` pairs to a :class:`MediaProvider`.

    Prefixes are matched longest-first so that ``"fal-ai/"`` beats the
    empty-string catch-all even if registered later.
    """

    def __init__(self) -> None:
        self._providers: list[tuple[str, MediaProvider]] = []

    def register(self, prefix: str, provider: MediaProvider) -> None:
        """Register *provider* for every model whose name starts with *prefix*."""
        self._providers.append((prefix, provider))
        self._providers.sort(key=lambda x: len(x[0]), reverse=True)

    def resolve(self, model: str, capability: str) -> MediaProvider:
        """Return the first provider matching *model* and *capability*.

        Raises:
            ValueError: If no registered provider matches.
        """
        for prefix, provider in self._providers:
            if model.startswith(prefix) and capability in provider.supported_modalities:
                return provider
        raise ValueError(
            f"No provider for model '{model}' with '{capability}' capability."
        )
