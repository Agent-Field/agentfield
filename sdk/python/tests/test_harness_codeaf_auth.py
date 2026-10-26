"""K1–K7: shared model-key contract through real runners and fake CLIs."""

import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from agentfield import HarnessProviderNotConfigured, HarnessProviderUnavailable
from agentfield.agent import Agent
from agentfield.harness._runner import HarnessRunner, _is_transient
from agentfield.harness.providers import codeaf
from agentfield.harness.providers._factory import build_provider
from agentfield.types import HarnessConfig

CONTRACT = json.loads(
    (Path(__file__).resolve().parents[2] / "codeaf-auth-contract.json").read_text()
)


@pytest.fixture(autouse=True)
def isolated_auth(monkeypatch, tmp_path, caplog):
    for key in list(os.environ):
        if key.startswith(("CODEAF_", "AFORGE_", "AGENTFIELD_")) or key in (
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("AGENTFIELD_HOME", str(tmp_path / "agentfield"))
    monkeypatch.setenv("CODEAF_TELEMETRY", "off")
    monkeypatch.setattr(
        codeaf, "install_codeaf", Mock(side_effect=AssertionError("installer called"))
    )
    monkeypatch.setattr(codeaf, "_missing_key_logged", False)
    monkeypatch.setattr(codeaf, "_construction_warning_logged", False)
    monkeypatch.setattr(codeaf.logger, "handlers", [caplog.handler])
    monkeypatch.setattr(codeaf.logger, "propagate", False)
    caplog.set_level(logging.WARNING, logger="agentfield.harness.codeaf")


def fake_binary(tmp_path):
    marker = tmp_path / "ran"
    binary = tmp_path / "fake-codeaf"
    binary.write_text(
        f"#!{sys.executable}\n"
        "import sys\nfrom pathlib import Path\n"
        "sys.stdin.read()\n"
        f"Path({str(marker)!r}).touch()\n"
        'print(\'{"ok":true,"stop":"done","answer":"answer"}\')\n'
    )
    binary.chmod(0o755)
    return str(binary), marker


def warning_messages(caplog):
    return [r.message for r in caplog.records if r.levelno == logging.WARNING]


def prepare_case(case, tmp_path, monkeypatch):
    paths = {
        "$STATE": tmp_path / "agentfield" / "codeaf",
        "$PROFILE": tmp_path / "profile",
        "$HOME": tmp_path / "home",
    }

    def expand(value):
        for token, directory in paths.items():
            value = value.replace(token, str(directory))
        return value

    for key, value in case["env"].items():
        monkeypatch.setenv(key, expand(value))
    for filename, contents in case["files"].items():
        target = Path(expand(filename))
        target.parent.mkdir(parents=True, exist_ok=True)
        if contents is None:
            target.mkdir()
        else:
            target.write_text(contents)
    return {"env": {k: expand(v) for k, v in case["overlay"].items()}}


@pytest.mark.parametrize("case", CONTRACT["cases"], ids=lambda c: c["name"])
async def test_k1_k3_k4_shared_key_ladder(case, tmp_path, monkeypatch, caplog):
    options = prepare_case(case, tmp_path, monkeypatch)
    binary, marker = fake_binary(tmp_path)
    provider = codeaf.CodeafProvider(binary)
    if case["allow"]:
        result = await provider.execute("task", options)
        assert result.result == "answer"
        assert marker.exists()
        assert warning_messages(caplog) == []
    else:
        with pytest.raises(HarnessProviderNotConfigured) as caught:
            await provider.execute("task", options)
        assert isinstance(caught.value, HarnessProviderUnavailable)
        assert str(caught.value) == CONTRACT["refusal"]
        assert caught.value.provider == "codeaf"
        assert caught.value.missing_auth_env == ["OPENROUTER_API_KEY"]
        assert caught.value.binary == binary
        assert caught.value.install_command == "af codeaf ensure"
        assert not marker.exists()
        assert warning_messages(caplog) == [CONTRACT["warning"]]
    codeaf.install_codeaf.assert_not_called()


@pytest.mark.parametrize("provider", [None, "codeaf", "aforge"])
@pytest.mark.parametrize("pin", ["automatic", "env", "option"])
async def test_k1_k2_k7_refusal_before_resolution(
    provider, pin, tmp_path, monkeypatch, caplog
):
    binary, marker = fake_binary(tmp_path)
    config = {} if provider is None else {"provider": provider}
    if pin == "env":
        monkeypatch.setenv("CODEAF_BIN", binary)
    elif pin == "option":
        config["codeaf_bin"] = binary
    resolve = Mock(side_effect=AssertionError("binary resolution called"))
    monkeypatch.setattr(codeaf, "_resolve_binary", resolve)
    harness = build_provider(HarnessConfig(**config))
    for _ in range(2):
        with pytest.raises(
            HarnessProviderNotConfigured, match="needs a model key"
        ) as caught:
            await harness.execute("task", {})
        assert str(caught.value) == CONTRACT["refusal"]
    resolve.assert_not_called()
    codeaf.install_codeaf.assert_not_called()
    assert not marker.exists()
    assert warning_messages(caplog) == [CONTRACT["warning"]]


async def test_k1_runner_does_not_retry(monkeypatch):
    assert not _is_transient(CONTRACT["refusal"])
    provider = codeaf.CodeafProvider()
    execute = AsyncMock(wraps=provider.execute)
    monkeypatch.setattr(provider, "execute", execute)
    sleep = AsyncMock(side_effect=AssertionError("retried refusal"))
    monkeypatch.setattr("agentfield.harness._runner.asyncio.sleep", sleep)
    with pytest.raises(HarnessProviderNotConfigured):
        await HarnessRunner()._execute_with_retry(provider, "task", {"max_retries": 3})
    assert execute.call_count == 1
    sleep.assert_not_called()


@pytest.mark.parametrize("pin", ["env", "option"])
async def test_k5_legacy_pin_without_key(pin, tmp_path, monkeypatch, caplog):
    binary, marker = fake_binary(tmp_path)
    config = {"provider": "aforge"}
    if pin == "env":
        monkeypatch.setenv("AFORGE_BIN", binary)
    else:
        config["aforge_bin"] = binary
    await build_provider(HarnessConfig(**config)).execute("task", {})
    assert marker.exists()
    assert warning_messages(caplog) == []


@pytest.mark.parametrize(
    "provider,key,configured,legacy,expected",
    [
        (None, "", True, False, True),
        ("codeaf", "", True, False, True),
        ("aforge", "", True, False, True),
        ("codeaf", "key", True, False, False),
        ("claude-code", "", True, False, False),
        (None, "", False, False, False),
        ("aforge", "", True, True, False),
    ],
)
def test_k6_agent_construction(
    provider, key, configured, legacy, expected, monkeypatch, caplog
):
    monkeypatch.setenv("OPENROUTER_API_KEY", key)
    if legacy:
        monkeypatch.setenv("AFORGE_BIN", "/fake/legacy")
    config = (
        HarnessConfig(**({"provider": provider} if provider else {}))
        if configured
        else None
    )
    for _ in range(2):
        Agent(node_id="auth-test", harness_config=config)
    assert warning_messages(caplog) == ([CONTRACT["warning"]] if expected else [])


@pytest.mark.parametrize("name", ["codeaf", "aforge", "claude-code"])
def test_k6_agent_environment_provider(name, monkeypatch, caplog):
    monkeypatch.setenv("AGENTFIELD_HARNESS_PROVIDER", name)
    Agent(node_id="auth-test", harness_config=HarnessConfig())
    assert warning_messages(caplog) == (
        [CONTRACT["warning"]] if name != "claude-code" else []
    )


def test_k6_construction_survives_a_failing_key_check(monkeypatch, caplog):
    def no_home(*_args):
        raise RuntimeError("no home directory")

    monkeypatch.setattr(codeaf, "_child_environment", no_home)
    Agent(node_id="auth-test", harness_config=HarnessConfig())
    assert warning_messages(caplog) == []
