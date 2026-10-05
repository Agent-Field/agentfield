"""Classify the request route without retaining endpoint URLs."""

from urllib.parse import urlparse


def routing_provider(provider: str | None, endpoint: str | None = None) -> str:
    if endpoint is not None:
        try:
            parsed = urlparse(endpoint)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                return "unknown"
            return {
                "openrouter.ai": "openrouter",
                "api.openai.com": "openai",
                "api.anthropic.com": "anthropic",
                "generativelanguage.googleapis.com": "google",
            }.get(parsed.hostname.lower(), "other")
        except (TypeError, ValueError):
            return "unknown"
    normalized = (provider or "unknown").lower()
    if normalized == "gemini":
        normalized = "google"
    return (
        normalized
        if normalized
        in {"openrouter", "openai", "anthropic", "google", "bedrock", "azure", "ollama"}
        else "unknown"
    )
