from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from agentfield.harness._availability import PROVIDER_SPECS
from agentfield.harness.providers._factory import SUPPORTED_PROVIDERS

VersionProbe = Callable[[list[str]], Awaitable[str]]
VERSION_PROBE_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True)
class ProviderHealth:
    provider: str
    binary: str | None
    installed: bool
    version: str | None
    auth: str
    usable: bool
    install_command: str
    auth_env_vars: tuple[str, ...]
    issues: tuple[str, ...]


async def _probe_version(command: list[str]) -> str:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=VERSION_PROBE_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError as exc:
        process.kill()
        await process.wait()
        raise RuntimeError("version probe timed out") from exc
    output = (stdout or stderr).decode(errors="replace").strip()
    if process.returncode != 0:
        raise RuntimeError(output or f"exited with status {process.returncode}")
    return output.splitlines()[0] if output else "unknown"


async def harness_doctor(
    providers: Sequence[str] | None = None,
    *,
    env: Mapping[str, str] | None = None,
    version_probe: VersionProbe | None = None,
) -> list[ProviderHealth]:
    selected = list(providers or sorted(SUPPORTED_PROVIDERS))
    unknown = sorted(set(selected) - SUPPORTED_PROVIDERS)
    if unknown:
        raise ValueError(
            f"Unknown harness provider: {unknown[0]!r}. Supported providers: "
            f"{', '.join(sorted(SUPPORTED_PROVIDERS))}"
        )

    environment = os.environ if env is None else env
    probe = version_probe or _probe_version
    reports: list[ProviderHealth] = []
    for provider in selected:
        if provider == "claude-code":
            reports.append(_claude_health(environment))
            continue

        spec = PROVIDER_SPECS[provider]
        binary = shutil.which(spec.binary)
        issues: list[str] = []
        version: str | None = None
        if binary is None:
            issues.append("binary_not_found")
        else:
            try:
                version = await probe([spec.binary, *spec.version_args])
            except (OSError, RuntimeError):
                issues.append("version_probe_failed")

        auth = (
            "configured"
            if any(environment.get(name) for name in spec.auth_env_vars)
            else "unknown"
        )
        reports.append(
            ProviderHealth(
                provider=provider,
                binary=binary,
                installed=binary is not None,
                version=version,
                auth=auth,
                usable=binary is not None and not issues,
                install_command=spec.install_command,
                auth_env_vars=spec.auth_env_vars,
                issues=tuple(issues),
            )
        )
    return reports


def _claude_health(env: Mapping[str, str]) -> ProviderHealth:
    try:
        import claude_agent_sdk  # noqa: F401

        installed = True
    except ImportError:
        installed = False
    issues = () if installed else ("wrapper_not_installed",)
    return ProviderHealth(
        provider="claude-code",
        binary=None,
        installed=installed,
        version=None,
        auth="configured" if env.get("ANTHROPIC_API_KEY") else "unknown",
        usable=installed,
        install_command="pip install 'agentfield[harness-claude]'",
        auth_env_vars=("ANTHROPIC_API_KEY",),
        issues=issues,
    )
