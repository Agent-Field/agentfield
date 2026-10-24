"""V1–V5 and V7–V12: CodeAF's public headless contract, using fake CLIs."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

from agentfield.agent import Agent
from agentfield.cost_tracker import CostTracker
from agentfield.exceptions import HarnessProviderUnavailable
from agentfield.harness import harness_doctor
from agentfield.harness._availability import PROVIDER_SPECS, ensure_cli_available
from agentfield.harness._result import FailureType
from agentfield.harness._runner import HarnessRunner
from agentfield.harness.providers import codeaf
from agentfield.harness.providers._factory import build_provider
from agentfield.harness.providers.aforge import AforgeProvider
from agentfield.harness.providers.codeaf import CODEAF_DEFAULT_MODEL, CodeafProvider
from agentfield.types import HarnessConfig


@pytest.fixture(autouse=True)
def isolated_codeaf(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(codeaf.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(codeaf.logger, "propagate", False)
    caplog.set_level(logging.INFO, logger="agentfield.harness.codeaf")
    for key in (
        "CODEAF_BIN",
        "AFORGE_BIN",
        "CODEAF_HOME",
        "CODEAF_NO_UPDATE_CHECK",
        "CODEAF_TELEMETRY",
        "CODEAF_EXEC_REASONING",
        "CI",
        "DO_NOT_TRACK",
        "AGENTFIELD_TELEMETRY_ENABLED",
        "AGENTFIELD_HARNESS_PROVIDER",
        "AGENTFIELD_CODEAF_COMMAND",
        "AGENTFIELD_HARNESS_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AGENTFIELD_HOME", str(tmp_path / "af-home"))
    monkeypatch.setenv("AGENTFIELD_SKIP_CODEAF", "1")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "temp"))
    for attr in (
        "_resolution_cache",
        "_resolution_lock",
        "_resolution_task",
        "_concurrency_sem",
    ):
        monkeypatch.setattr(CodeafProvider, attr, None)
    monkeypatch.setattr(codeaf, "_download_retry_at", 0.0)
    monkeypatch.setattr(codeaf, "_last_download_error", None)
    for attr in ("_alias_logged", "_fallback_logged", "_command_warning_logged"):
        monkeypatch.setattr(CodeafProvider, attr, False)


def fake_cli(
    tmp_path: Path, *, payload=None, exit_code=0, stdout=None, stderr=""
) -> str:
    if payload is None:
        payload = {"ok": True, "stop": "done", "answer": "done"}
    script = tmp_path / "fake-codeaf"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "sys.stdin.read()\n"
        f"sys.stdout.write({(json.dumps(payload) if stdout is None else stdout)!r})\n"
        f"sys.stderr.write({stderr!r})\n"
        f"sys.exit({exit_code})\n"
    )
    script.chmod(0o755)
    return str(script)


@pytest.mark.parametrize("provider", [None, "codeaf", "aforge"])
async def test_zero_config_and_alias_usage_attribution(
    tmp_path, monkeypatch, caplog, provider
):
    """V1/V3: real runner, adapter, subprocess and cost-recording path."""
    monkeypatch.setenv(
        "CODEAF_BIN",
        fake_cli(
            tmp_path,
            payload={
                "ok": True,
                "stop": "done",
                "answer": "answer",
                "spend_usd": 0.0,
                "tokens": {"in": 11, "out": 3},
                "steps": 2,
            },
        ),
    )
    caplog.set_level(logging.INFO)
    agent = object.__new__(Agent)
    agent._get_current_execution_context = lambda: None
    agent.harness_config = None
    agent.cost_tracker = CostTracker()
    for _ in range(2):
        result = await HarnessRunner().run("task", provider=provider)
        agent._record_harness_usage(result, provider=provider)
        assert result.provider == "codeaf"
        assert result.cost_usd == 0.0
        assert result.num_turns == 2
    entries = agent.cost_tracker.serialize()["entries"]
    assert all(entry["harness"] == "codeaf" for entry in entries)
    assert len(entries) == 2
    assert caplog.text.count("provider 'aforge' now runs CodeAF") == (
        1 if provider == "aforge" else 0
    )


@pytest.mark.parametrize("pin", ["config", "env"])
async def test_legacy_pin_preserves_protocol_and_attribution(
    tmp_path, monkeypatch, pin
):
    """V4: the alias escape hatch still consumes the old envelope."""
    binary = fake_cli(
        tmp_path, payload={"text": "legacy", "usage": {"cost": 0.1}, "turns": 2}
    )
    kwargs = {"aforge_bin": binary} if pin == "config" else {}
    if pin == "env":
        monkeypatch.setenv("AFORGE_BIN", binary)
    config = HarnessConfig(provider="aforge", **kwargs)
    assert isinstance(build_provider(config), AforgeProvider)
    result = await HarnessRunner(config).run("task")
    assert result.result == "legacy"
    assert result.provider == "aforge"
    agent = object.__new__(Agent)
    agent._get_current_execution_context = lambda: None
    agent.cost_tracker = CostTracker()
    agent._record_harness_usage(result, provider="codeaf")
    assert agent.cost_tracker.serialize()["entries"][0]["harness"] == "aforge"


@pytest.mark.parametrize("command", ["exec", "do", "invalid"])
async def test_argv_stdin_and_timeouts(tmp_path, monkeypatch, caplog, command):
    """V7: argv is exact, model is explicit, and prompt travels on stdin."""
    monkeypatch.setenv("AGENTFIELD_CODEAF_COMMAND", command)
    monkeypatch.setenv("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", "90")
    run = AsyncMock(return_value=('{"ok":true,"stop":"done","answer":"ok"}', "", 0))
    monkeypatch.setattr(codeaf, "run_cli", run)
    options = {
        "cwd": "/nested",
        "project_dir": "/root",
        "model": "openrouter/openrouter/model#high",
        "max_turns": 9,
        "system_prompt": " system ",
    }
    provider = CodeafProvider(sys.executable)
    await provider.execute("prompt", options)
    await provider.execute("prompt", options)
    args, kw = run.call_args
    selected = "do" if command == "do" else "exec"
    expected = [sys.executable, selected, "--json", "--dir", "/root"]
    if selected == "do":
        expected += [
            "--yes-spend",
            "--model",
            "openrouter/model",
            "--plan-model",
            "openrouter/model",
            "--check-model",
            "openrouter/model",
            "--timeout",
            "85s",
            "-",
        ]
    else:
        expected += [
            "--model",
            "openrouter/model",
            "--timeout",
            "85s",
            "--max-turns",
            "9",
            "--system",
            "system",
        ]
    timeout_index = args[0].index("--timeout") + 1
    assert float(args[0][timeout_index][:-1]) == pytest.approx(85, abs=0.1)
    expected[timeout_index] = args[0][timeout_index]
    assert args[0] == expected
    assert kw["input_text"] == (
        "system\n\nTask:\nprompt" if selected == "do" else "prompt"
    )
    assert kw["idle_seconds"] == 0
    assert 89.9 < kw["timeout"] <= 90
    assert kw["cwd"] is None
    assert kw["env"]["CODEAF_EXEC_REASONING"] == "high"
    assert not any(key.startswith("AFORGE_") for key in kw["env"])
    assert caplog.text.count("Invalid AGENTFIELD_CODEAF_COMMAND") == (
        1 if command == "invalid" else 0
    )


@pytest.mark.parametrize(
    "model,expected", [(None, CODEAF_DEFAULT_MODEL), ("openrouter/a", "a"), ("a", "a")]
)
async def test_default_model_and_short_timeout(monkeypatch, model, expected):
    monkeypatch.setenv("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", "2")
    run = AsyncMock(return_value=('{"ok":true,"stop":"done"}', "", 0))
    monkeypatch.setattr(codeaf, "run_cli", run)
    result = await CodeafProvider(sys.executable).execute("p", {"model": model})
    cmd = run.call_args.args[0]
    assert cmd[cmd.index("--model") + 1] == expected
    assert cmd[cmd.index("--timeout") + 1] == "1s"
    assert result.metrics.model == expected


@pytest.mark.parametrize(
    "key,value",
    [
        (None, None),
        ("CI", "YES"),
        ("DO_NOT_TRACK", "on"),
        ("AGENTFIELD_TELEMETRY_ENABLED", "FALSE"),
        ("AGENTFIELD_TELEMETRY_ENABLED", "0"),
    ],
)
async def test_actual_child_environment(tmp_path, monkeypatch, key, value):
    """V8: inspect the spawned child's environment, including absent panel vars."""
    if key:
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CODEAF_MODELS", "parent-panel")
    monkeypatch.setenv("AFORGE_MODELS", "parent-legacy-panel")
    binary = Path(fake_cli(tmp_path))
    await asyncio.to_thread(
        binary.write_text,
        f"#!{sys.executable}\nimport json,os\nprint(json.dumps(dict(ok=True,stop='done',answer=json.dumps(dict(os.environ)))))\n",
    )
    result = await CodeafProvider(str(binary)).execute("p", {})
    env = json.loads(result.result)
    assert env["CODEAF_HOME"] == str(tmp_path / "af-home" / "codeaf")
    assert await asyncio.to_thread(Path(env["CODEAF_HOME"]).is_dir)
    assert env["CODEAF_NO_UPDATE_CHECK"] == "1"
    assert env.get("CODEAF_TELEMETRY") == ("off" if key else None)
    assert "CODEAF_MODELS" not in env
    assert "AFORGE_MODELS" not in env


async def test_caller_env_and_overlay_defaults_win(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEAF_HOME", "/parent")
    monkeypatch.setenv("CODEAF_NO_UPDATE_CHECK", "0")
    monkeypatch.setenv("CODEAF_TELEMETRY", "on")
    monkeypatch.setenv("CI", "1")
    env = await asyncio.to_thread(codeaf._child_environment, {}, "unsupported")
    assert env == {}
    overlay = {
        "CODEAF_HOME": "/overlay",
        "CODEAF_NO_UPDATE_CHECK": "",
        "CODEAF_TELEMETRY": "",
        "CODEAF_MODELS": "panel",
        "AFORGE_MODELS": "legacy",
        "CODEAF_EXEC_REASONING": "off",
    }
    run = AsyncMock(return_value=('{"ok":true,"stop":"done"}', "", 0))
    monkeypatch.setattr(codeaf, "run_cli", run)
    await CodeafProvider(sys.executable).execute(
        "p", {"variant": "high", "env": overlay}
    )
    assert run.call_args.kwargs["env"] == overlay


@pytest.mark.parametrize(
    "stdout,expected",
    [
        ('{\n"ok":true,\n"stop":"done",\n"answer":"pretty"\n}', "pretty"),
        (
            'diagnostic\n{"ok":true,"stop":"done","answer":"first"}\nnoise\n{"ok":true,"stop":"done","answer":"last"}\n',
            "last",
        ),
        ('{"text":"old", "deliverable":"old", "usage":{}}', None),
        ('[]\nnull\n{"ok":true}\n{"stop":"done"}', None),
    ],
)
async def test_envelope_parsing(tmp_path, stdout, expected):
    """V9: canonical fields only, whole JSON or last complete line."""
    result = await CodeafProvider(fake_cli(tmp_path, stdout=stdout)).execute("p", {})
    assert result.result == expected
    assert result.is_error == (expected is None)


@pytest.mark.parametrize(
    "stop,ok,answer,exit_code,failure",
    [
        ("deadline", False, "partial", 3, FailureType.TIMEOUT),
        ("deadline", False, "partial", 124, FailureType.TIMEOUT),
        ("done", True, "answer", 0, FailureType.NONE),
        ("done", False, "answer", 0, FailureType.CRASH),
        ("unchecked", False, "answer", 2, FailureType.NONE),
        ("unchecked", False, "", 2, FailureType.CRASH),
        ("budget", False, "answer", 3, FailureType.NONE),
        ("turn-cap", False, "answer", 3, FailureType.NONE),
        ("budget", False, "", 3, FailureType.NO_OUTPUT),
        ("turn-cap", False, "", 3, FailureType.NO_OUTPUT),
        ("question", False, "partial", 4, FailureType.CRASH),
        ("incomplete", False, "partial", 2, FailureType.CRASH),
        ("error", False, "partial", 1, FailureType.CRASH),
        ("price", False, "partial", 3, FailureType.CRASH),
        ("future", False, "answer", 0, FailureType.NONE),
        ("future", False, "answer", 1, FailureType.CRASH),
        ("future", True, "", 0, FailureType.CRASH),
    ],
)
async def test_outcome_table_and_metrics(
    tmp_path, stop, ok, answer, exit_code, failure
):
    """V10: every envelope row of §7, plus canonical metrics over legacy aliases."""
    payload = {
        "ok": ok,
        "stop": stop,
        "answer": answer,
        "error": "diagnostic",
        "blocked_on": "need input",
        "steps": 7,
        "calls": 99,
        "tokens": {"in": 123, "out": 45},
        "spend_usd": 0.0,
        "model": "returned/model",
        "usage": {"cost": 99},
        "turns": 99,
        "seconds": 999,
    }
    result = await CodeafProvider(
        fake_cli(tmp_path, payload=payload, exit_code=exit_code)
    ).execute("p", {"model": "requested/model"})
    assert result.failure_type is failure
    assert result.is_error == (failure != FailureType.NONE)
    assert result.result == (answer if failure == FailureType.NONE else None)
    assert result.metrics.num_turns == 7
    assert result.metrics.input_tokens == 123
    assert result.metrics.output_tokens == 45
    assert result.metrics.total_cost_usd == 0.0
    assert result.metrics.model == "returned/model"
    assert 0 <= result.metrics.duration_api_ms < 999000
    if result.is_error:
        assert (
            f"codeaf exit code {exit_code} (stop={stop}): diagnostic"
            in result.error_message
        )
        if stop == "question":
            assert "need input" in result.error_message


@pytest.mark.parametrize(
    "stdout,exit_code,failure",
    [
        ("", 0, FailureType.NO_OUTPUT),
        ("nonsense", 0, FailureType.CRASH),
        ("", 1, FailureType.CRASH),
        ("", -9, FailureType.CRASH),
    ],
)
async def test_no_envelope_and_signals(monkeypatch, stdout, exit_code, failure):
    run = AsyncMock(return_value=(stdout, "\x1b[31m" + "x" * 1500 + "TAIL", exit_code))
    monkeypatch.setattr(codeaf, "run_cli", run)
    result = await CodeafProvider(sys.executable).execute("p", {})
    assert result.failure_type is failure
    assert "TAIL" in result.error_message
    assert "\x1b" not in result.error_message
    assert len(result.error_message) < 1100


@pytest.mark.parametrize(
    "exception", [TimeoutError("outer timeout"), FileNotFoundError("removed")]
)
async def test_outer_timeout_and_disappearing_binary(monkeypatch, exception):
    monkeypatch.setattr(codeaf, "run_cli", AsyncMock(side_effect=exception))
    if isinstance(exception, FileNotFoundError):
        with pytest.raises(HarnessProviderUnavailable, match="af codeaf ensure"):
            await CodeafProvider(sys.executable).execute("p", {})
    else:
        result = await CodeafProvider(sys.executable).execute("p", {})
        assert result.failure_type is FailureType.TIMEOUT
        assert result.result is None


async def test_resolution_overrides_are_lazy_and_keep_per_call_precedence(monkeypatch):
    """V5: overrides beat cached resolution and the environment override."""
    resolve = Mock(side_effect=AssertionError("automatic lookup must be bypassed"))
    monkeypatch.setattr(codeaf, "_resolve_binary", resolve)
    monkeypatch.setattr(
        CodeafProvider, "_resolution_cache", ("/cached/codeaf", "codeaf", None)
    )
    monkeypatch.setattr(
        codeaf, "run_cli", AsyncMock(return_value=('{"ok":true,"stop":"done"}', "", 0))
    )
    provider = build_provider(HarnessConfig(codeaf_bin=sys.executable))
    resolve.assert_not_called()
    monkeypatch.setenv("CODEAF_BIN", "missing")
    await provider.execute("p", {})
    resolve.assert_not_called()
    monkeypatch.setenv("CODEAF_BIN", sys.executable)
    await CodeafProvider().execute("p", {})
    resolve.assert_not_called()


async def test_alias_reads_legacy_pin_at_first_execute(tmp_path, monkeypatch):
    provider = CodeafProvider(alias=True)
    monkeypatch.setenv("AFORGE_BIN", fake_cli(tmp_path, payload={"text": "legacy"}))
    result = await provider.execute("p", {})
    assert result.result == "legacy"
    assert result.provider == "aforge"


async def test_explicit_binary_name_beats_managed_binary(monkeypatch):
    monkeypatch.setattr(codeaf, "find_managed_binary", lambda: "/managed/codeaf")
    monkeypatch.setattr(codeaf.shutil, "which", lambda binary: f"/path/{binary}")
    run = AsyncMock(return_value=('{"ok":true,"stop":"done"}', "", 0))
    monkeypatch.setattr(codeaf, "run_cli", run)
    await CodeafProvider("codeaf").execute("p", {})
    assert run.call_args.args[0][0] == "/path/codeaf"


@pytest.mark.parametrize(
    "level", ["managed", "download", "path", "legacy-home", "legacy-path", "missing"]
)
async def test_automatic_resolution_order(monkeypatch, tmp_path, caplog, level):
    caplog.set_level(logging.INFO)
    calls = []
    monkeypatch.setattr(
        codeaf,
        "find_managed_binary",
        lambda: "/managed/codeaf" if level == "managed" else None,
    )

    def install():
        calls.append("download")
        if level == "download":
            return "/download/codeaf"
        raise OSError("download broke")

    def which(binary):
        calls.append(binary)
        if level == "path" and binary == "codeaf":
            return "/path/codeaf"
        if level == "legacy-home" and binary.endswith("/bin/aforge"):
            return binary
        if level == "legacy-path" and binary == "aforge":
            return "/path/aforge"
        return None

    monkeypatch.setattr(codeaf, "install_codeaf", install)
    monkeypatch.setattr(codeaf.shutil, "which", which)
    run = AsyncMock(return_value=('{"ok":true,"stop":"done","answer":"answer"}', "", 0))
    monkeypatch.setattr(codeaf, "run_cli", run)
    monkeypatch.setattr(
        "agentfield.harness.providers.aforge.ensure_cli_available", lambda *_: "found"
    )
    monkeypatch.setattr(
        "agentfield.harness.providers.aforge.run_cli",
        AsyncMock(return_value=('{"text":"legacy"}', "", 0)),
    )
    if level == "missing":
        for _ in range(2):
            with pytest.raises(
                HarnessProviderUnavailable, match="af codeaf ensure.*download broke"
            ):
                await CodeafProvider().execute("p", {})
    else:
        for _ in range(2):
            result = await CodeafProvider(alias=True).execute("p", {})
        assert result.provider == ("aforge" if level.startswith("legacy") else "codeaf")
        if level.startswith("legacy"):
            assert caplog.text.count("falling back to the legacy aforge binary") == 1
            assert "provider 'aforge' now runs CodeAF" not in caplog.text
        else:
            assert caplog.text.count("provider 'aforge' now runs CodeAF") == 1
    assert calls.count("download") == (0 if level == "managed" else 1)
    if level in {"managed", "download"}:
        assert "codeaf" not in calls
    elif level == "path":
        assert calls == ["download", "codeaf"]
    else:
        expected = ["download", "codeaf", str(tmp_path / "af-home/bin/aforge")]
        if level != "legacy-home":
            expected.append("aforge")
        if level == "missing":
            expected.extend(expected[1:])
        assert calls == expected


async def test_codeaf_config_bin_threads_through_runner(tmp_path):
    result = await HarnessRunner(HarnessConfig(codeaf_bin="missing")).run(
        "p",
        codeaf_bin=fake_cli(tmp_path),
    )
    assert result.result == "done"
    assert result.provider == "codeaf"


async def test_schema_pipeline_keeps_provider_attribution(tmp_path):
    class Response(BaseModel):
        value: int

    result = await HarnessRunner(
        HarnessConfig(
            codeaf_bin=fake_cli(
                tmp_path, payload={"ok": True, "stop": "done", "answer": '{"value":42}'}
            )
        )
    ).run("p", schema=Response, cwd=str(tmp_path))
    assert result.parsed == Response(value=42)
    assert result.provider == "codeaf"
    assert not list(tmp_path.glob(".agentfield-out-*"))


async def test_managed_availability_and_doctor_version(tmp_path, monkeypatch):
    """V11: no download; home pin is usable even with no PATH entry."""
    home = tmp_path / "af-home/bin"
    home.mkdir(parents=True)
    binary = home / "codeaf"
    binary.write_text(
        f"#!{sys.executable}\nprint('codeaf v0.8.2 built 2026-10-23 11:39 · go1.26.5 linux/amd64')\n"
    )
    binary.chmod(0o755)
    (home / ".codeaf.version").write_text("v0.8.2\n")
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert ensure_cli_available("codeaf", "codeaf") == str(binary)
    [report] = await harness_doctor(
        ["codeaf"],
        env={"AGENTFIELD_HOME": str(home.parent), "OPENROUTER_API_KEY": "test"},
    )
    assert report.binary == str(binary)
    assert report.version == "v0.8.2"
    assert report.usable
    assert report.auth == "configured"
    assert report.install_command == "af codeaf ensure"
    assert PROVIDER_SPECS["aforge"].install_command == "af aforge ensure"


@pytest.mark.parametrize(
    "new,old,expected",
    [
        (None, None, 8),
        (None, "3", 3),
        ("2", "3", 2),
        ("bad", "3", 8),
        ("0", "3", 8),
        ("-1", "3", 8),
        ("", "3", 8),
        (None, "bad", 8),
    ],
)
def test_concurrency_env_never_raises(monkeypatch, new, old, expected):
    """V12: invalid values never break module import or semaphore construction."""
    for key, value in (("CODEAF_MAX_CONCURRENT", new), ("AFORGE_MAX_CONCURRENT", old)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    assert codeaf._max_concurrent() == expected
    from agentfield.harness.providers.aforge import _max_concurrent

    assert _max_concurrent() > 0


def test_factory_codeaf_protocol_and_legacy_simple_namespace():
    from agentfield.harness.providers._base import HarnessProvider

    assert isinstance(
        build_provider(SimpleNamespace(provider="codeaf")), HarnessProvider
    )
    assert isinstance(
        build_provider(SimpleNamespace(provider="aforge")), CodeafProvider
    )


@pytest.mark.parametrize("pin", ["env", "config", "call"])
@pytest.mark.parametrize("provider", [None, "codeaf", "aforge"])
async def test_F1_implicit_legacy_pin_all_runner_paths(
    monkeypatch, tmp_path, pin, provider
):
    """F1a/b/c/e: explicit CodeAF wins; omitted provider retains legacy pins."""
    binary = fake_cli(tmp_path, payload={"text": "legacy", "usage": {"cost": 0.1}})
    if pin == "env":
        monkeypatch.setenv("AFORGE_BIN", binary)
    kwargs = {"aforge_bin": binary} if pin == "config" else {}
    if provider is not None:
        kwargs["provider"] = provider
    config = HarnessConfig(**kwargs)
    call = {"aforge_bin": binary} if pin == "call" else {}
    if provider == "codeaf":
        run = AsyncMock(
            return_value=('{"ok":true,"stop":"done","answer":"codeaf"}', "", 0)
        )
        monkeypatch.setattr(codeaf, "run_cli", run)
        call["codeaf_bin"] = sys.executable
    result = await HarnessRunner(config).run("p", **call)
    assert result.provider == ("codeaf" if provider == "codeaf" else "aforge")
    if provider is None and pin != "call":
        assert isinstance(build_provider(config), AforgeProvider)


async def test_F1c_env_provider_beats_legacy_pin(monkeypatch):
    monkeypatch.setenv("AFORGE_BIN", "/legacy")
    monkeypatch.setenv("AGENTFIELD_HARNESS_PROVIDER", "codeaf")
    assert isinstance(build_provider(HarnessConfig()), CodeafProvider)


@pytest.mark.parametrize("contract", ["R1a", "R1b", "R1c"])
async def test_R1_captured_provider_all_execution_paths(monkeypatch, contract):
    from agentfield.harness._result import RawResult
    from agentfield.harness.providers.codex import CodexProvider

    if contract == "R1a":
        monkeypatch.setenv("AGENTFIELD_HARNESS_PROVIDER", "codex")
        config = HarnessConfig(aforge_bin="/legacy")
        monkeypatch.delenv("AGENTFIELD_HARNESS_PROVIDER")
        assert "provider" not in config.model_fields_set
        expected, name = CodexProvider, "codex"
    else:
        config = (
            HarnessConfig() if contract == "R1b" else HarnessConfig(provider="codeaf")
        )
        monkeypatch.setenv("AFORGE_BIN", "/legacy")
        expected, name = (
            (AforgeProvider, "aforge")
            if contract == "R1b"
            else (CodeafProvider, "codeaf")
        )
    assert isinstance(build_provider(config), expected)
    execute = AsyncMock(return_value=RawResult(result="done", provider=name))
    monkeypatch.setattr(expected, "execute", execute)
    assert (await HarnessRunner(config).run("p")).provider == name
    app = object.__new__(Agent)
    app.harness_config = config
    app._harness_runner = None
    assert (await app.harness("p")).provider == name
    assert execute.await_count == 2
    monkeypatch.setattr(
        CodeafProvider,
        "execute",
        AsyncMock(return_value=RawResult(result="override", provider="codeaf")),
    )
    assert (await HarnessRunner(config).run("p", provider="codeaf")).provider == "codeaf"


@pytest.mark.parametrize(
    "new,old,want",
    [
        ("", "do", "do"),
        ("exec", "do", "exec"),
        (" ", "do", "do"),
        ("bad", "do", "exec"),
    ],
)
async def test_F2ab_command_fallback(monkeypatch, new, old, want):
    monkeypatch.setenv("AGENTFIELD_CODEAF_COMMAND", new)
    monkeypatch.setenv("AGENTFIELD_AFORGE_COMMAND", old)
    run = AsyncMock(return_value=('{"ok":true,"stop":"done"}', "", 0))
    monkeypatch.setattr(codeaf, "run_cli", run)
    await CodeafProvider(sys.executable).execute("p", {})
    assert run.call_args.args[0][1] == want


@pytest.mark.parametrize(
    "explicit,new,old,overlay,want",
    [
        (None, "", "openrouter/old", {}, "old"),
        (None, "new", "old", {}, "new"),
        ("openrouter/explicit#high", "new", "old", {}, "explicit"),
        (
            None,
            "new",
            "old",
            {"CODEAF_MODEL": " ", "AFORGE_MODEL": "openrouter/overlay"},
            "overlay",
        ),
        (None, "", "ambient", {"AFORGE_MODEL": "overlay"}, "overlay"),
        (None, "", "", {}, CODEAF_DEFAULT_MODEL),
    ],
)
async def test_F2cdef_model_precedence(monkeypatch, explicit, new, old, overlay, want):
    monkeypatch.setenv("CODEAF_MODEL", new)
    monkeypatch.setenv("AFORGE_MODEL", old)
    monkeypatch.setenv("AGENTFIELD_AFORGE_COMMAND", "do")
    run = AsyncMock(return_value=('{"ok":true,"stop":"done"}', "", 0))
    monkeypatch.setattr(codeaf, "run_cli", run)
    result = await CodeafProvider(sys.executable).execute(
        "p", {"model": explicit, "env": overlay}
    )
    cmd = run.call_args.args[0]
    for flag in ("--model", "--plan-model", "--check-model"):
        assert cmd[cmd.index(flag) + 1] == want
    assert result.metrics.model == want


async def test_F3ab_provisioning_timeout_preserves_shared_resolution(monkeypatch):
    import threading
    import time

    release = threading.Event()
    calls = 0

    def resolve():
        nonlocal calls
        calls += 1
        release.wait(5)
        return sys.executable, "codeaf", None

    monkeypatch.setattr(codeaf, "_resolve_binary", resolve)
    monkeypatch.setenv("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", "1")
    run = AsyncMock(return_value=('{"ok":true,"stop":"done"}', "", 0))
    monkeypatch.setattr(codeaf, "run_cli", run)
    started = time.monotonic()
    try:
        results = await asyncio.gather(
            *(CodeafProvider().execute("p", {}) for _ in range(2))
        )
        assert time.monotonic() - started < 1.5
        assert all(r.failure_type == FailureType.TIMEOUT for r in results)
        assert all("provisioning CodeAF" in r.error_message for r in results)
        assert calls == 1
        # Explicit paths bypass the still-running shared resolution.
        assert not (await CodeafProvider(sys.executable).execute("p", {})).is_error
        pending = CodeafProvider._resolution_task
        assert pending is not None and not pending.cancelled()
    finally:
        release.set()
    await pending
    assert not (await CodeafProvider().execute("p", {})).is_error
    assert calls == 1


async def test_F3c_execution_uses_remaining_budget(monkeypatch):
    import time

    def resolve():
        time.sleep(0.15)
        return sys.executable, "codeaf", None

    monkeypatch.setattr(codeaf, "_resolve_binary", resolve)
    monkeypatch.setenv("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", "10")
    run = AsyncMock(return_value=('{"ok":true,"stop":"done"}', "", 0))
    monkeypatch.setattr(codeaf, "run_cli", run)
    await CodeafProvider().execute("p", {})
    cmd = run.call_args.args[0]
    remaining = run.call_args.kwargs["timeout"]
    inner = float(cmd[cmd.index("--timeout") + 1][:-1])
    # Provisioning took >= 0.15s of the 10s budget; only the rest is passed on.
    # The inner timeout is taken first, so it can only be larger than
    # remaining - 5 (by however long the machine took in between).
    assert 5 < remaining <= 9.85 + 0.005
    assert remaining - 5 <= inner + 0.001
    assert inner <= 4.85 + 0.006


@pytest.mark.parametrize("location", ["managed", "path", "legacy", "missing", "pin"])
async def test_F5_alias_doctor_read_only(monkeypatch, tmp_path, location):
    """F5a-d: CodeAF alias, legacy fallback and strict pin checks never install."""
    pathdir = tmp_path / "path"
    pathdir.mkdir()
    monkeypatch.setenv("PATH", str(pathdir))
    if location == "managed":
        target = tmp_path / "af-home/bin/codeaf"
        target.parent.mkdir(parents=True)
        (target.parent / ".codeaf.version").write_text("v0.8.2\n")
    else:
        target = pathdir / ("aforge" if location == "legacy" else "codeaf")
    if location != "missing":
        target.write_text("#!/bin/sh\nexit 0\n")
        target.chmod(0o755)
    if location == "pin":
        monkeypatch.setenv("AFORGE_BIN", str(tmp_path / "missing"))
    install = Mock(side_effect=AssertionError("doctor must not download"))
    monkeypatch.setattr(codeaf, "install_codeaf", install)
    probe = AsyncMock(
        return_value="aforge v0.1.0" if location == "legacy" else "codeaf v0.8.2"
    )
    report = (await harness_doctor(["aforge"], version_probe=probe))[0]
    assert report.usable == (location not in {"missing", "pin"})
    if location in {"managed", "path"}:
        assert report.version == "codeaf v0.8.2"
        assert probe.call_args.args[0] == [str(target), "version"]
    if not report.usable:
        assert report.install_command == "af aforge ensure"
    install.assert_not_called()


@pytest.mark.parametrize("contract", ["R4a", "R4b", "R4c"])
async def test_R4_alias_doctor_authoritative_codeaf_override(
    monkeypatch, tmp_path, contract
):
    pathdir = tmp_path / "path"
    pathdir.mkdir()
    legacy = pathdir / "aforge"
    legacy.write_text("#!/bin/sh\nexit 0\n")
    legacy.chmod(0o755)
    monkeypatch.setenv("PATH", str(pathdir))
    target = tmp_path / "override"
    if contract != "R4b":
        monkeypatch.setenv("CODEAF_BIN", str(target))
    if contract == "R4c":
        target.write_text("#!/bin/sh\nexit 0\n")
        target.chmod(0o755)
    install = Mock(side_effect=AssertionError("doctor must not download"))
    monkeypatch.setattr(codeaf, "install_codeaf", install)
    version = "aforge v0.1.0" if contract == "R4b" else "codeaf v0.8.2"
    probe = AsyncMock(return_value=version)
    report = (await harness_doctor(["aforge"], version_probe=probe))[0]
    assert report.usable == (contract != "R4a")
    assert report.installed == report.usable
    if contract == "R4a":
        assert report.binary is None
        assert report.issues == ("binary_not_found",)
        probe.assert_not_called()
    else:
        assert report.version == version
        assert report.binary == str(legacy if contract == "R4b" else target)
        probe.assert_awaited_once_with(
            ["aforge" if contract == "R4b" else str(target), "version"]
        )
    install.assert_not_called()
