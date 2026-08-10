from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentfield.exceptions import HarnessProviderUnavailable
from agentfield.harness._result import FailureType
from agentfield.harness.providers.aforge import AforgeProvider


@pytest.fixture(autouse=True)
def mock_aforge_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agentfield.harness._availability.shutil.which", lambda path: path
    )


def _envelope(
    text: str = "done",
    *,
    stop: str = "done",
    usage: dict[str, object] | None = None,
    turns: int = 1,
) -> str:
    return json.dumps(
        {
            "text": text,
            "stop": stop,
            "usage": usage or {},
            "artifacts": [],
            "turns": turns,
            "elapsed_ms": 12,
        }
    )


@pytest.mark.asyncio
async def test_aforge_success_maps_envelope_and_metrics(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = cmd, env, cwd, timeout, idle_seconds, input_text
        return (
            _envelope(
                " final answer ",
                usage={
                    "calls": 3,
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "cached_tokens": 20,
                    "cost": 0.0123,
                },
                turns=4,
            ),
            "",
            0,
        )

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)

    raw = await AforgeProvider().execute("hello", {"model": "openrouter/z-ai/glm-5.2"})

    assert raw.result == "final answer"
    assert raw.is_error is False
    assert raw.failure_type is FailureType.NONE
    assert raw.metrics.input_tokens == 100
    assert raw.metrics.output_tokens == 50
    assert raw.metrics.cache_read_tokens == 20
    assert raw.metrics.cache_creation_tokens == 0
    assert raw.metrics.num_turns == 4
    assert raw.metrics.total_cost_usd == 0.0123
    assert raw.metrics.model == "openrouter/z-ai/glm-5.2"
    assert raw.metrics.duration_api_ms >= 0
    assert raw.returncode == 0
    assert raw.messages[0]["text"] == " final answer "


@pytest.mark.asyncio
async def test_aforge_strips_openrouter_prefix_for_model_env(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = cmd, cwd, timeout, idle_seconds, input_text
        captured["env"] = env
        return _envelope(), "", 0

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)

    await AforgeProvider().execute("hello", {"model": "openrouter/x/y"})

    assert captured["env"]["AFORGE_MODEL"] == "x/y"


@pytest.mark.asyncio
async def test_aforge_maps_supported_variant_and_ignores_unknown_variant(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_envs: list[dict[str, str]] = []

    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = cmd, cwd, timeout, idle_seconds, input_text
        captured_envs.append(env or {})
        return _envelope(), "", 0

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)
    provider = AforgeProvider()

    await provider.execute("hello", {"model": "openrouter/x/y#high"})
    await provider.execute("hello", {"model": "openrouter/x/y#turbo"})

    assert captured_envs[0]["AFORGE_MODEL"] == "x/y"
    assert captured_envs[0]["AFORGE_EXEC_REASONING"] == "high"
    assert captured_envs[1]["AFORGE_MODEL"] == "x/y"
    assert "AFORGE_EXEC_REASONING" not in captured_envs[1]


@pytest.mark.asyncio
async def test_aforge_caller_env_overrides_derived_env(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = cmd, cwd, timeout, idle_seconds, input_text
        captured["env"] = env
        return _envelope(), "", 0

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)

    await AforgeProvider().execute(
        "hello",
        {
            "model": "openrouter/x/y",
            "env": {"AFORGE_MODEL": "override/model", "EXTRA": "1"},
        },
    )

    assert captured["env"]["AFORGE_MODEL"] == "override/model"
    assert captured["env"]["EXTRA"] == "1"


@pytest.mark.asyncio
async def test_aforge_delivers_prompt_only_via_stdin(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = env, cwd, timeout, idle_seconds
        captured["cmd"] = cmd
        captured["input_text"] = input_text
        return _envelope(), "", 0

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)
    prompt = "a prompt that must stay off argv"

    await AforgeProvider().execute(prompt, {})

    assert captured["input_text"] == prompt
    assert all(prompt not in arg for arg in captured["cmd"])


@pytest.mark.asyncio
async def test_aforge_disables_idle_watchdog_and_honors_timeout_env(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: list[tuple[object, object]] = []

    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = cmd, env, cwd, input_text
        captured.append((timeout, idle_seconds))
        return _envelope(), "", 0

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)
    monkeypatch.delenv("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", raising=False)
    provider = AforgeProvider()

    await provider.execute("hello", {})
    monkeypatch.setenv("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", "2400")
    await provider.execute("hello", {})

    assert captured == [(1800, 0), (2400, 0)]


@pytest.mark.asyncio
async def test_aforge_passes_stripped_system_prompt_flag(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, Any] = {}

    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = env, cwd, timeout, idle_seconds, input_text
        captured["cmd"] = cmd
        return _envelope(), "", 0

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)

    await AforgeProvider().execute("hello", {"system_prompt": "  be precise  "})

    assert captured["cmd"][-2:] == ["--system", "be precise"]


@pytest.mark.asyncio
async def test_aforge_project_dir_precedes_cwd_and_cwd_is_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    captured_cmds: list[list[str]] = []

    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = env, cwd, timeout, idle_seconds, input_text
        captured_cmds.append(cmd)
        return _envelope(), "", 0

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)
    provider = AforgeProvider()

    await provider.execute(
        "hello", {"project_dir": "/project", "cwd": "/project/nested"}
    )
    await provider.execute("hello", {"cwd": "/cwd-only"})

    assert captured_cmds[0][:6] == [
        "aforge",
        "exec",
        "--json",
        "-w",
        "/project",
    ]
    assert captured_cmds[1][:6] == [
        "aforge",
        "exec",
        "--json",
        "-w",
        "/cwd-only",
    ]


@pytest.mark.asyncio
async def test_aforge_timeout_returns_timeout_result(monkeypatch: pytest.MonkeyPatch):
    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = cmd, env, cwd, timeout, idle_seconds, input_text
        raise TimeoutError("aforge timed out")

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)

    raw = await AforgeProvider().execute("hello", {})

    assert raw.is_error is True
    assert raw.failure_type is FailureType.TIMEOUT
    assert raw.error_message == "aforge timed out"


@pytest.mark.asyncio
async def test_aforge_error_exit_is_crash_with_stderr(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = cmd, env, cwd, timeout, idle_seconds, input_text
        return _envelope("", stop="error"), "authentication exploded", 5

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)

    raw = await AforgeProvider().execute("hello", {})

    assert raw.is_error is True
    assert raw.failure_type is FailureType.CRASH
    assert "aforge exit code 5" in (raw.error_message or "")
    assert "authentication exploded" in (raw.error_message or "")


@pytest.mark.asyncio
async def test_aforge_budget_exit_with_text_is_success(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = cmd, env, cwd, timeout, idle_seconds, input_text
        return _envelope("usable landing", stop="budget"), "", 2

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)

    raw = await AforgeProvider().execute("hello", {})

    assert raw.result == "usable landing"
    assert raw.is_error is False
    assert raw.failure_type is FailureType.NONE


@pytest.mark.asyncio
async def test_aforge_turn_cap_without_text_is_crash(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = cmd, env, cwd, timeout, idle_seconds, input_text
        return _envelope("", stop="turn-cap"), "turn cap reached", 3

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)

    raw = await AforgeProvider().execute("hello", {})

    assert raw.is_error is True
    assert raw.failure_type is FailureType.CRASH
    assert "aforge exit code 3" in (raw.error_message or "")


@pytest.mark.asyncio
async def test_aforge_zero_exit_without_text_is_crash(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = cmd, env, cwd, timeout, idle_seconds, input_text
        return _envelope(""), "", 0

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)

    raw = await AforgeProvider().execute("hello", {})

    assert raw.is_error is True
    assert raw.failure_type is FailureType.CRASH
    assert "aforge exit code 0" in (raw.error_message or "")


@pytest.mark.asyncio
async def test_aforge_missing_binary_fails_before_spawn(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr("agentfield.harness._availability.shutil.which", lambda _: None)
    run_cli = AsyncMock()
    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", run_cli)

    with pytest.raises(HarnessProviderUnavailable, match="aforge-missing"):
        await AforgeProvider(bin_path="aforge-missing").execute("hello", {})

    run_cli.assert_not_awaited()


@pytest.mark.asyncio
async def test_aforge_parses_last_valid_envelope_after_stray_stdout(
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = cmd, env, cwd, timeout, idle_seconds, input_text
        stdout = "\n".join(
            [
                "stray diagnostic",
                '{"type":"event"}',
                _envelope("real result"),
            ]
        )
        return stdout, "", 0

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)

    raw = await AforgeProvider().execute("hello", {})

    assert raw.result == "real result"
    assert raw.is_error is False


@pytest.mark.asyncio
async def test_aforge_missing_or_zero_cost_falls_back_to_estimate(
    monkeypatch: pytest.MonkeyPatch,
):
    estimates: list[float] = []

    async def fake_run_cli(
        cmd, *, env=None, cwd=None, timeout=None, idle_seconds=None, input_text=None
    ):
        _ = cmd, env, cwd, timeout, idle_seconds, input_text
        return _envelope(usage={"cost": 0}), "", 0

    def fake_estimate_cli_cost(*, model, prompt, result_text):
        _ = model, prompt, result_text
        estimates.append(0.456)
        return 0.456

    monkeypatch.setattr("agentfield.harness.providers.aforge.run_cli", fake_run_cli)
    monkeypatch.setattr(
        "agentfield.harness.providers.aforge.estimate_cli_cost",
        fake_estimate_cli_cost,
    )

    raw = await AforgeProvider().execute("hello", {"model": "openrouter/x/y"})

    assert raw.metrics.total_cost_usd == 0.456
    assert estimates == [0.456]
