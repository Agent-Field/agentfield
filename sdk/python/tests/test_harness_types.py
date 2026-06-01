# pyright: reportMissingImports=false

import pytest
from pydantic import ValidationError

from agentfield.harness._result import HarnessResult, Metrics, RawResult
from agentfield.harness.providers._factory import build_provider
from agentfield.types import HarnessConfig


def test_harness_config_provider_required():
    with pytest.raises(ValidationError):
        HarnessConfig()


def test_harness_config_defaults():
    cfg = HarnessConfig(provider="codex")

    assert cfg.provider == "codex"
    assert cfg.model == "sonnet"
    assert cfg.max_turns == 30
    assert cfg.max_budget_usd is None
    assert cfg.max_retries == 3
    assert cfg.initial_delay == 1.0
    assert cfg.max_delay == 30.0
    assert cfg.backoff_factor == 2.0
    assert cfg.tools == ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
    assert cfg.permission_mode is None
    assert cfg.system_prompt is None
    assert cfg.env == {}
    assert cfg.cwd is None
    assert cfg.codex_bin == "codex"
    assert cfg.gemini_bin == "gemini"
    assert cfg.opencode_bin == "opencode"


def test_build_provider_raises_for_unknown_provider():
    cfg = HarnessConfig(provider="unknown-provider")

    with pytest.raises(ValueError, match="Unknown harness provider"):
        build_provider(cfg)


def test_build_provider_returns_claude_provider_for_claude_code():
    from agentfield.harness.providers.claude import ClaudeCodeProvider

    cfg = HarnessConfig(provider="claude-code")
    provider = build_provider(cfg)
    assert isinstance(provider, ClaudeCodeProvider)


def test_harness_result_defaults_and_text_property():
    result = HarnessResult()

    assert result.result is None
    assert result.parsed is None
    assert result.is_error is False
    assert result.error_message is None
    assert result.cost_usd is None
    assert result.num_turns == 0
    assert result.duration_ms == 0
    assert result.session_id == ""
    assert result.messages == []
    assert result.text == ""


def test_harness_result_text_uses_result_and_error_flag():
    result = HarnessResult(result="done", is_error=True, error_message="boom")

    assert result.text == "done"
    assert result.is_error is True
    assert result.error_message == "boom"


def test_metrics_defaults():
    metrics = Metrics()

    assert metrics.duration_ms == 0
    assert metrics.duration_api_ms == 0
    assert metrics.num_turns == 0
    assert metrics.total_cost_usd is None
    assert metrics.usage is None
    assert metrics.session_id == ""


def test_raw_result_defaults_and_construction():
    raw = RawResult()
    assert raw.result is None
    assert raw.messages == []
    assert isinstance(raw.metrics, Metrics)
    assert raw.is_error is False
    assert raw.error_message is None

    filled = RawResult(result="ok", is_error=True, error_message="err")
    assert filled.result == "ok"
    assert filled.is_error is True
    assert filled.error_message == "err"
