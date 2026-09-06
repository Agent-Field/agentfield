from __future__ import annotations

import pytest

from agentfield.harness import ProviderHealth, harness_doctor
from agentfield.harness._availability import PROVIDER_SPECS
from agentfield.harness.providers._factory import SUPPORTED_PROVIDERS


@pytest.mark.asyncio
async def test_doctor_reports_binary_version_and_env_auth(monkeypatch):
    monkeypatch.setattr(
        "agentfield.harness._doctor.shutil.which",
        lambda binary: f"/usr/local/bin/{binary}",
    )

    async def version_probe(command: list[str]) -> str:
        assert command == ["codex", "--version"]
        return "codex-cli 1.2.3"

    reports = await harness_doctor(
        providers=["codex"],
        env={"OPENAI_API_KEY": "configured"},
        version_probe=version_probe,
    )

    assert reports == [
        ProviderHealth(
            provider="codex",
            binary="/usr/local/bin/codex",
            installed=True,
            version="codex-cli 1.2.3",
            auth="configured",
            usable=True,
            install_command="npm install -g @openai/codex",
            auth_env_vars=("OPENAI_API_KEY",),
            issues=(),
        )
    ]


@pytest.mark.asyncio
async def test_doctor_marks_missing_binary_unusable_without_version_probe(monkeypatch):
    monkeypatch.setattr("agentfield.harness._doctor.shutil.which", lambda _: None)
    probed = False

    async def version_probe(command: list[str]) -> str:
        nonlocal probed
        probed = True
        return "unexpected"

    [report] = await harness_doctor(providers=["opencode"], version_probe=version_probe)

    assert report.installed is False
    assert report.usable is False
    assert report.binary is None
    assert report.version is None
    assert report.auth == "unknown"
    assert report.issues == ("binary_not_found",)
    assert probed is False


@pytest.mark.asyncio
async def test_doctor_does_not_treat_cli_login_as_missing_auth(monkeypatch):
    monkeypatch.setattr(
        "agentfield.harness._doctor.shutil.which", lambda _: "/usr/bin/gemini"
    )

    async def version_probe(command: list[str]) -> str:
        return "0.40.1"

    [report] = await harness_doctor(
        providers=["gemini"], env={}, version_probe=version_probe
    )

    assert report.auth == "unknown"
    assert report.usable is True
    assert "auth_not_configured" not in report.issues


@pytest.mark.asyncio
async def test_doctor_probes_cursor_through_its_agent_binary(monkeypatch):
    """Cursor's CLI is `agent`, so the doctor must not probe for `cursor`."""
    monkeypatch.setattr(
        "agentfield.harness._doctor.shutil.which",
        lambda binary: f"/usr/local/bin/{binary}",
    )

    async def version_probe(command: list[str]) -> str:
        assert command == ["agent", "--version"]
        return "cursor-agent 2026.02.1"

    reports = await harness_doctor(
        providers=["cursor"],
        env={"CURSOR_API_KEY": "configured"},
        version_probe=version_probe,
    )

    assert reports == [
        ProviderHealth(
            provider="cursor",
            binary="/usr/local/bin/agent",
            installed=True,
            version="cursor-agent 2026.02.1",
            auth="configured",
            usable=True,
            install_command="curl https://cursor.com/install -fsS | bash",
            auth_env_vars=("CURSOR_API_KEY",),
            issues=(),
        )
    ]


@pytest.mark.asyncio
async def test_doctor_reports_a_missing_cursor_cli_with_its_install_command(
    monkeypatch,
):
    monkeypatch.setattr("agentfield.harness._doctor.shutil.which", lambda _: None)

    [report] = await harness_doctor(providers=["cursor"], env={})

    assert report.installed is False
    assert report.usable is False
    assert report.issues == ("binary_not_found",)
    assert report.install_command == "curl https://cursor.com/install -fsS | bash"


@pytest.mark.asyncio
async def test_every_supported_provider_has_a_doctor_spec():
    """`harness_doctor()` indexes PROVIDER_SPECS directly, so a provider added
    to SUPPORTED_PROVIDERS without a spec is a KeyError at runtime, not an
    import error. claude-code is the documented exception: it is a Python
    wrapper, not a CLI, and takes the `_claude_health` branch instead.
    """
    specced = set(PROVIDER_SPECS) | {"claude-code"}
    assert SUPPORTED_PROVIDERS <= specced


@pytest.mark.asyncio
async def test_doctor_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown harness provider"):
        await harness_doctor(providers=["not-a-provider"])


@pytest.mark.asyncio
async def test_doctor_checks_all_supported_providers_with_no_args(monkeypatch):
    monkeypatch.setattr("agentfield.harness._doctor.shutil.which", lambda _: None)

    reports = await harness_doctor(env={})

    assert [report.provider for report in reports] == sorted(SUPPORTED_PROVIDERS)


@pytest.mark.asyncio
async def test_default_version_probe_times_out_and_cleans_up(monkeypatch):
    class HangingProcess:
        returncode = None

        def __init__(self):
            self.killed = False
            self.waited = False

        async def communicate(self):
            import asyncio

            await asyncio.sleep(60)

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            self.waited = True

    process = HangingProcess()

    async def create_subprocess(*_args, **_kwargs):
        return process

    monkeypatch.setattr(
        "agentfield.harness._doctor.asyncio.create_subprocess_exec",
        create_subprocess,
    )
    monkeypatch.setattr(
        "agentfield.harness._doctor.VERSION_PROBE_TIMEOUT_SECONDS", 0.01
    )
    monkeypatch.setattr(
        "agentfield.harness._doctor.shutil.which", lambda _: "/usr/bin/codex"
    )

    [report] = await harness_doctor(providers=["codex"])

    assert report.usable is False
    assert report.issues == ("version_probe_failed",)
    assert process.killed is True
    assert process.waited is True
