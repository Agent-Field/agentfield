from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from agentfield.harness._result import RawResult
from agentfield.harness._runner import HarnessRunner
from agentfield.types import HarnessConfig


class _Provider:
    async def execute(self, prompt: str, options: dict[str, Any]) -> RawResult:
        return RawResult(result="ok")


def test_default_provider_is_aforge(monkeypatch):
    monkeypatch.delenv("AGENTFIELD_HARNESS_PROVIDER", raising=False)
    assert HarnessConfig().provider == "aforge"


def test_env_provider_is_honoured(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_HARNESS_PROVIDER", "codex")
    assert HarnessConfig().provider == "codex"


def test_explicit_provider_beats_env(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_HARNESS_PROVIDER", "codex")
    assert HarnessConfig(provider="gemini").provider == "gemini"


def test_blank_env_is_ignored(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_HARNESS_PROVIDER", "   ")
    assert HarnessConfig().provider == "aforge"


async def _run_and_capture_provider(
    tmp_path, config: HarnessConfig | None = None, **overrides: Any
) -> str:
    captured: dict[str, str] = {}

    def fake_build_provider(factory_config):
        captured["provider"] = factory_config.provider
        return _Provider()

    with patch("agentfield.harness._runner.build_provider", fake_build_provider):
        await HarnessRunner(config=config).run(
            "hello", cwd=str(tmp_path), **overrides
        )
    return captured["provider"]


@pytest.mark.asyncio
async def test_zero_setup_runner_selects_aforge(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTFIELD_HARNESS_PROVIDER", raising=False)
    assert await _run_and_capture_provider(tmp_path) == "aforge"


@pytest.mark.asyncio
async def test_runner_honours_env_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTFIELD_HARNESS_PROVIDER", "opencode")
    assert await _run_and_capture_provider(tmp_path) == "opencode"


@pytest.mark.asyncio
async def test_runner_explicit_provider_beats_env_and_config(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENTFIELD_HARNESS_PROVIDER", "gemini")
    config = HarnessConfig(provider="codex")
    assert (
        await _run_and_capture_provider(tmp_path, config, provider="opencode")
        == "opencode"
    )


def test_model_default_is_empty(monkeypatch):
    monkeypatch.delenv("AGENTFIELD_HARNESS_PROVIDER", raising=False)
    assert HarnessConfig().model is None
