from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel
from unittest.mock import patch

from agentfield.harness._result import Metrics, RawResult
from agentfield.harness._runner import (
    HarnessRunner,
    _accumulate_metrics,
    _is_transient,
    _resolve_options,
)
from agentfield.harness._schema import get_output_path


class DemoSchema(BaseModel):
    name: str
    count: int


class MockProvider:
    def __init__(self, results=None):
        self.results = results or []
        self.call_count = 0
        self.last_prompt = None
        self.last_options = None

    async def execute(self, prompt: str, options: dict) -> RawResult:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_options = options
        if self.call_count <= len(self.results):
            return self.results[self.call_count - 1]
        return RawResult(result="default result")


class FileWritingProvider(MockProvider):
    def __init__(self, payload: str, result: RawResult | None = None):
        super().__init__([result or RawResult(result="ok")])
        self.payload = payload

    async def execute(self, prompt: str, options: dict) -> RawResult:
        output_path = get_output_path(str(options.get("cwd", ".")))
        with open(output_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(self.payload)
        return await super().execute(prompt, options)


def test_resolve_options_merges_config_and_overrides_per_call_wins():
    config = SimpleNamespace(
        provider="codex",
        model="sonnet",
        max_turns=30,
        max_budget_usd=2.0,
        tools=["Read"],
        permission_mode="auto",
        system_prompt="base",
        env={"A": "1"},
        cwd="/tmp/base",
        codex_bin="codex",
        gemini_bin="gemini",
        opencode_bin="opencode",
    )

    resolved = _resolve_options(
        config,
        {
            "model": "gpt-4.1",
            "max_turns": 10,
            "env": {"B": "2"},
            "cwd": "/tmp/override",
            "max_budget_usd": None,
        },
    )

    assert resolved["provider"] == "codex"
    assert resolved["model"] == "gpt-4.1"
    assert resolved["max_turns"] == 10
    assert resolved["max_budget_usd"] == 2.0
    assert resolved["env"] == {"B": "2"}
    assert resolved["cwd"] == "/tmp/override"


def test_is_transient_matches_and_rejects_expected_messages():
    assert _is_transient("HTTP 503 service unavailable") is True
    assert _is_transient("Rate limit reached for this model") is True
    assert _is_transient("connection reset by peer") is True
    assert _is_transient("Validation failed for user input") is False
    assert _is_transient("Permission denied") is False


@pytest.mark.asyncio
async def test_run_without_schema_returns_plain_harness_result(tmp_path):
    provider = MockProvider(
        [
            RawResult(
                result="done",
                metrics=Metrics(num_turns=2, total_cost_usd=0.42, session_id="sess-1"),
            )
        ]
    )
    runner = HarnessRunner()

    with patch("agentfield.harness._runner.build_provider", return_value=provider):
        result = await runner.run("hello", provider="codex", cwd=str(tmp_path))

    assert result.is_error is False
    assert result.result == "done"
    assert result.parsed is None
    assert result.cost_usd == 0.42
    assert result.num_turns == 2
    assert result.session_id == "sess-1"


@pytest.mark.asyncio
async def test_run_with_schema_injects_prompt_suffix_and_parses_output(tmp_path):
    provider = FileWritingProvider(json.dumps({"name": "ok", "count": 1}))
    runner = HarnessRunner()

    with patch("agentfield.harness._runner.build_provider", return_value=provider):
        result = await runner.run(
            "produce json",
            provider="codex",
            schema=DemoSchema,
            cwd=str(tmp_path),
        )

    assert provider.last_prompt is not None
    assert "OUTPUT REQUIREMENTS" in provider.last_prompt
    assert get_output_path(str(tmp_path)) in provider.last_prompt
    assert result.is_error is False
    assert isinstance(result.parsed, DemoSchema)
    assert result.parsed.name == "ok"
    assert result.parsed.count == 1


@pytest.mark.asyncio
async def test_run_raises_when_no_provider_set(tmp_path):
    runner = HarnessRunner()
    with pytest.raises(ValueError, match="No harness provider specified"):
        await runner.run("hello", cwd=str(tmp_path))


@pytest.mark.asyncio
async def test_execute_with_retry_retries_on_transient_error_then_succeeds(
    tmp_path, monkeypatch
):
    provider = MockProvider(
        [
            RawResult(is_error=True, error_message="rate limit exceeded"),
            RawResult(result="ok", metrics=Metrics(num_turns=2)),
        ]
    )
    runner = HarnessRunner()
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("agentfield.harness._runner.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("agentfield.harness._runner.random.uniform", lambda a, b: 0.0)

    with patch("agentfield.harness._runner.build_provider", return_value=provider):
        result = await runner.run(
            "hello",
            provider="codex",
            cwd=str(tmp_path),
            max_retries=3,
            initial_delay=0.01,
            max_delay=1.0,
            backoff_factor=2.0,
        )

    assert result.is_error is False
    assert result.result == "ok"
    assert provider.call_count == 2
    assert sleeps == [0.01]


@pytest.mark.asyncio
async def test_execute_with_retry_does_not_retry_non_transient_error(
    tmp_path, monkeypatch
):
    provider = MockProvider(
        [
            RawResult(is_error=True, error_message="validation failed"),
            RawResult(result="should not happen"),
        ]
    )
    runner = HarnessRunner()
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("agentfield.harness._runner.asyncio.sleep", fake_sleep)

    with patch("agentfield.harness._runner.build_provider", return_value=provider):
        result = await runner.run(
            "hello",
            provider="codex",
            cwd=str(tmp_path),
            max_retries=3,
        )

    assert result.is_error is True
    assert result.error_message == "validation failed"
    assert provider.call_count == 1
    assert sleeps == []


@pytest.mark.asyncio
async def test_schema_validation_failure_returns_error_result(tmp_path):
    provider = FileWritingProvider('{"name": "ok"}')
    runner = HarnessRunner()

    with patch("agentfield.harness._runner.build_provider", return_value=provider):
        result = await runner.run(
            "produce bad json",
            provider="codex",
            schema=DemoSchema,
            cwd=str(tmp_path),
        )

    assert result.is_error is True
    assert result.parsed is None
    assert "Schema validation failed" in (result.error_message or "")


@pytest.mark.asyncio
async def test_temp_files_are_always_cleaned_even_on_error(tmp_path):
    large_schema = {
        "type": "object",
        "properties": {
            "payload": {"type": "string", "description": "x" * 20000},
        },
    }

    class RaisingProvider:
        async def execute(self, prompt: str, options: dict) -> RawResult:
            raise RuntimeError("boom")

    runner = HarnessRunner()

    with patch(
        "agentfield.harness._runner.build_provider", return_value=RaisingProvider()
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await runner.run(
                "trigger failure",
                provider="codex",
                schema=large_schema,
                cwd=str(tmp_path),
            )

    assert not (tmp_path / ".agentfield_output.json").exists()
    assert not (tmp_path / ".agentfield_schema.json").exists()


@pytest.mark.asyncio
async def test_run_resolves_harness_config_defaults_with_per_call_overrides(tmp_path):
    config = SimpleNamespace(
        provider="codex",
        model="default-model",
        max_turns=30,
        max_budget_usd=1.5,
        tools=["Read", "Write"],
        permission_mode="plan",
        system_prompt="base system",
        env={"BASE": "1"},
        cwd=str(tmp_path),
        codex_bin="codex",
        gemini_bin="gemini",
        opencode_bin="opencode",
    )
    provider = MockProvider([RawResult(result="ok")])
    runner = HarnessRunner(config=config)

    with patch("agentfield.harness._runner.build_provider", return_value=provider):
        await runner.run(
            "hello",
            model="override-model",
            max_turns=5,
            env={"OVERRIDE": "1"},
            permission_mode="auto",
        )

    assert provider.last_options is not None
    assert provider.last_options["provider"] == "codex"
    assert provider.last_options["model"] == "override-model"
    assert provider.last_options["max_turns"] == 5
    assert provider.last_options["max_budget_usd"] == 1.5
    assert provider.last_options["permission_mode"] == "auto"
    assert provider.last_options["system_prompt"] == "base system"
    assert provider.last_options["env"] == {"OVERRIDE": "1"}


def test_accumulate_metrics_sums_costs_across_retries():
    """Cost is summed across multiple RawResults (e.g. schema retries)."""
    raws = [
        RawResult(result="a", metrics=Metrics(num_turns=1, total_cost_usd=0.01)),
        RawResult(result="b", metrics=Metrics(num_turns=2, total_cost_usd=0.02)),
        RawResult(result="c", metrics=Metrics(num_turns=1, total_cost_usd=0.005)),
    ]
    cost, turns, sid, msgs = _accumulate_metrics(raws)
    assert cost == pytest.approx(0.035)
    assert turns == 4


def test_accumulate_metrics_none_cost_skipped():
    """None costs are skipped; only non-None costs are summed."""
    raws = [
        RawResult(result="a", metrics=Metrics(num_turns=1, total_cost_usd=None)),
        RawResult(result="b", metrics=Metrics(num_turns=2, total_cost_usd=0.05)),
    ]
    cost, turns, sid, msgs = _accumulate_metrics(raws)
    assert cost == pytest.approx(0.05)
    assert turns == 3


def test_accumulate_metrics_all_none_returns_none():
    """If all costs are None, total is None (unknown), not 0."""
    raws = [
        RawResult(result="a", metrics=Metrics(num_turns=1, total_cost_usd=None)),
        RawResult(result="b", metrics=Metrics(num_turns=2, total_cost_usd=None)),
    ]
    cost, turns, sid, msgs = _accumulate_metrics(raws)
    assert cost is None
    assert turns == 3


@pytest.mark.asyncio
async def test_schema_retry_accumulates_cost(tmp_path, monkeypatch):
    """Cost is accumulated across initial attempt + schema retries."""

    async def fake_sleep(delay: float) -> None:
        pass

    monkeypatch.setattr("agentfield.harness._runner.asyncio.sleep", fake_sleep)

    call_count = 0

    class RetryProvider:
        async def execute(self, prompt: str, options: dict) -> RawResult:
            nonlocal call_count
            call_count += 1
            output_path = get_output_path(str(options.get("cwd", ".")))
            if call_count == 1:
                # First attempt: write invalid JSON (missing 'count')
                with open(output_path, "w") as f:
                    f.write('{"name": "partial"}')
                return RawResult(
                    result="partial",
                    metrics=Metrics(num_turns=1, total_cost_usd=0.01),
                )
            else:
                # Retry: write valid JSON
                with open(output_path, "w") as f:
                    f.write('{"name": "ok", "count": 42}')
                return RawResult(
                    result="ok",
                    metrics=Metrics(num_turns=1, total_cost_usd=0.02),
                )

    runner = HarnessRunner()

    with patch(
        "agentfield.harness._runner.build_provider", return_value=RetryProvider()
    ):
        result = await runner.run(
            "produce json",
            provider="codex",
            schema=DemoSchema,
            cwd=str(tmp_path),
            schema_max_retries=2,
        )

    assert result.is_error is False
    assert isinstance(result.parsed, DemoSchema)
    assert result.parsed.name == "ok"
    assert result.cost_usd == pytest.approx(0.03)  # 0.01 + 0.02
    assert result.num_turns == 2
