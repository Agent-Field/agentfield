"""Unit tests for the Grok Build CLI harness provider."""

from __future__ import annotations

from agentfield.harness.providers.grok import (
    GrokProvider,
    _extract_json_payload,
    _usage_metrics,
)
from agentfield.harness.providers._factory import SUPPORTED_PROVIDERS, build_provider
from agentfield.types import HarnessConfig


def test_supported_providers_includes_grok() -> None:
    assert "grok" in SUPPORTED_PROVIDERS


def test_build_provider_grok_default_bin() -> None:
    provider = build_provider(HarnessConfig(provider="grok"))
    assert isinstance(provider, GrokProvider)
    assert provider._bin == "grok"


def test_build_provider_grok_custom_bin() -> None:
    provider = build_provider(HarnessConfig(provider="grok", grok_bin="/opt/grok"))
    assert isinstance(provider, GrokProvider)
    assert provider._bin == "/opt/grok"


def test_extract_json_payload_tolerates_script_noise() -> None:
    raw = "\x04" + '{"text":"{\\"action\\":\\"approve\\"}","num_turns":1}\n'
    payload = _extract_json_payload(raw)
    assert payload is not None
    assert payload["num_turns"] == 1


def test_usage_metrics_missing_tokens_are_zero_not_none() -> None:
    metrics = _usage_metrics({"usage": {}, "num_turns": 2}, "grok-4.5")
    assert metrics.input_tokens == 0
    assert metrics.output_tokens == 0
    assert metrics.cache_read_tokens == 0
    assert metrics.cache_creation_tokens == 0
    assert metrics.num_turns == 2
    assert metrics.model == "grok-4.5"
