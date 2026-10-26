"""CodeAF headless provider and lazy binary resolution."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from http.client import HTTPException
from typing import TYPE_CHECKING, ClassVar
from urllib.parse import urlsplit

from agentfield.exceptions import HarnessProviderNotConfigured
from agentfield.harness._availability import provider_unavailable
from agentfield.harness._cli import resolve_model_and_variant, run_cli, strip_ansi
from agentfield.harness._codeaf_install import (
    agentfield_home,
    default_state_directory,
    find_managed_binary,
    install_codeaf,
    truthy,
)
from agentfield.harness._result import FailureType, Metrics, RawResult
from agentfield.harness.providers.aforge import (
    AforgeProvider,
    _numeric,
    _strip_openrouter_prefix,
    _task_input,
)

if TYPE_CHECKING:
    from agentfield.types import HarnessConfig

logger = logging.getLogger("agentfield.harness.codeaf")
CODEAF_DEFAULT_MODEL = "~deepseek/deepseek-v4-flash-latest"
_REASONING_VARIANTS = {"off", "low", "medium", "high"}
_download_retry_at = 0.0
_last_download_error: str | None = None
CODEAF_MISSING_KEY_WARNING = (
    "CodeAF is the harness provider and no model key is configured, so harness calls will fail. "
    "Set OPENROUTER_API_KEY (https://openrouter.ai/keys), or choose another "
    "harness provider with the provider option or AGENTFIELD_HARNESS_PROVIDER."
)
_missing_key_logged = False
_construction_warning_logged = False


def _max_concurrent() -> int:
    raw = os.environ.get(
        "CODEAF_MAX_CONCURRENT", os.environ.get("AFORGE_MAX_CONCURRENT", "8")
    )
    try:
        value = int(raw)
        return value if value > 0 else 8
    except ValueError:
        return 8


def _parse_envelope(stdout: str) -> dict[str, object] | None:
    try:
        value = json.loads(stdout.strip())
    except ValueError:
        pass
    else:
        if isinstance(value, dict) and "ok" in value and "stop" in value:
            return value
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and "ok" in value and "stop" in value:
            return value
    return None


def _resolve_binary() -> tuple[str | None, str, str | None]:
    """Recheck every location after failure, throttling only download retries."""
    global _download_retry_at, _last_download_error
    managed = find_managed_binary()
    if managed:
        return managed, "codeaf", None
    attempted_at = time.monotonic()
    if attempted_at >= _download_retry_at:
        try:
            installed = install_codeaf()
            if installed:
                return installed, "codeaf", None
        except (OSError, ValueError, HTTPException) as exc:
            _last_download_error = str(exc)
            _download_retry_at = attempted_at + 60
    download_error = _last_download_error
    binary = shutil.which("codeaf")
    if binary:
        return binary, "codeaf", download_error
    legacy_name = "aforge.exe" if os.name == "nt" else "aforge"
    legacy = shutil.which(str(agentfield_home() / "bin" / legacy_name)) or shutil.which(
        "aforge"
    )
    return legacy, "aforge" if legacy else "codeaf", download_error


def _child_environment(
    options: dict[str, object], variant: str | None
) -> dict[str, str]:
    value = options.get("env")
    overlay = (
        {
            key: item
            for key, item in value.items()
            if isinstance(key, str) and isinstance(item, str)
        }
        if isinstance(value, dict)
        else {}
    )
    environment = {**os.environ, **overlay}
    if "CODEAF_HOME" not in environment:
        overlay["CODEAF_HOME"] = default_state_directory(environment)
    if "CODEAF_NO_UPDATE_CHECK" not in environment:
        overlay["CODEAF_NO_UPDATE_CHECK"] = "1"
    if "CODEAF_TELEMETRY" not in environment and (
        truthy(environment.get("CI"))
        or truthy(environment.get("DO_NOT_TRACK"))
        or environment.get("AGENTFIELD_TELEMETRY_ENABLED", "").strip().lower()
        in {"0", "false", "no", "off"}
    ):
        overlay["CODEAF_TELEMETRY"] = "off"
    if variant:
        normalized = variant.strip().lower()
        if normalized in _REASONING_VARIANTS:
            overlay.setdefault("CODEAF_EXEC_REASONING", normalized)
        else:
            logger.debug("Ignoring unsupported codeaf variant %r", variant)
    return overlay


def _reject_json_constant(name: str) -> object:
    raise ValueError(f"invalid JSON constant {name}")


def _has_model_key(environment: dict[str, str]) -> bool:
    """Mirror CodeAF's credential ladder; uncertain profile state is allowed."""

    def value(*names: str) -> str:
        return next(
            (
                environment[name].strip()
                for name in names
                if environment.get(name, "").strip()
            ),
            "",
        )

    if value("OPENROUTER_API_KEY"):
        return True
    key = value("OPENAI_API_KEY")
    base = value("CODEAF_BASE_URL", "AFORGE_BASE_URL")
    if key:
        if key.startswith("sk-or-"):
            return True
        if base:
            try:
                address = (
                    base if "://" in base else "https://" + base.removeprefix("//")
                )
                parsed = urlsplit(address)
                # Accessing port also validates malformed endpoint addresses.
                _ = parsed.port
                host = (
                    (parsed.hostname or "")
                    .lower()
                    .removesuffix(".")
                    .removeprefix("www.")
                )
                if host != "openrouter.ai":
                    return True
            except ValueError:
                return True
    names = ("CODEAF_PROFILE_DIR", "AFORGE_PROFILE_DIR", "CODEAF_HOME", "AFORGE_HOME")
    # CodeAF reads a whitespace-only spelling differently from an unset one, and
    # with every spelling empty it falls back to HOME, a legacy ~/.aforge or a
    # relative directory. Neither can be seen reliably from here, so allow it.
    if any(environment.get(name) and not environment[name].strip() for name in names):
        return True
    location = value(*names)
    if not location:
        return True
    profile = Path(location)
    for filename in ("credentials.json", "codex.json"):
        try:
            if (profile / filename).exists():
                return True
        except OSError:
            return True
    try:
        config = json.loads(
            (profile / "config.json").read_text(), parse_constant=_reject_json_constant
        )
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        return True
    if not isinstance(config, dict):
        return True
    saved_key = config.get("api_key")
    sources = config.get("model_sources")
    return (isinstance(saved_key, str) and bool(saved_key.strip())) or (
        isinstance(sources, (list, dict)) and bool(sources)
    )


def warn_codeaf_configuration(config: HarnessConfig) -> None:
    """Warn once when a CodeAF harness has no visible key; never raises."""
    try:
        _warn_codeaf_configuration(config)
    except Exception as exc:  # Agent construction must not fail over a warning.
        logger.debug("CodeAF key check skipped: %s", exc)


def _warn_codeaf_configuration(config: HarnessConfig) -> None:
    from agentfield.harness.providers._factory import resolve_execution_provider

    global _construction_warning_logged
    provider = resolve_execution_provider(config)
    if provider not in {"codeaf", "aforge"}:
        return
    if provider == "aforge" and (
        config.aforge_bin not in (None, "aforge")
        or os.environ.get("AFORGE_BIN", "").strip()
    ):
        return
    environment = {**os.environ, **_child_environment({}, None)}
    if not _has_model_key(environment) and not _construction_warning_logged:
        _construction_warning_logged = True
        logger.warning(CODEAF_MISSING_KEY_WARNING)


class CodeafProvider:
    """CodeAF exec provider; ``aforge`` without a pin is a compatibility alias."""

    _MAX_CONCURRENT: ClassVar[int] = _max_concurrent()
    _concurrency_sem: ClassVar[asyncio.Semaphore | None] = None
    _resolution_lock: ClassVar[asyncio.Lock | None] = None
    _resolution_cache: ClassVar[tuple[str | None, str, str | None] | None] = None
    _resolution_task: ClassVar[
        asyncio.Task[tuple[str | None, str, str | None]] | None
    ] = None
    _alias_logged: ClassVar[bool] = False
    _fallback_logged: ClassVar[bool] = False
    _command_warning_logged: ClassVar[bool] = False

    def __init__(self, bin_path: str | None = None, *, alias: bool = False) -> None:
        self._bin = bin_path
        self._alias = alias

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._concurrency_sem is None:
            cls._concurrency_sem = asyncio.Semaphore(cls._MAX_CONCURRENT)
        return cls._concurrency_sem

    @classmethod
    async def _resolve_automatic(cls) -> tuple[str | None, str, str | None]:
        try:
            resolved = await asyncio.to_thread(_resolve_binary)
            if resolved[0] is not None:
                cls._resolution_cache = resolved
            return resolved
        finally:
            cls._resolution_task = None

    async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
        global _missing_key_logged
        started = time.monotonic()
        timeout = int(os.environ.get("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", "1800"))
        deadline = started + timeout
        cls = type(self)
        legacy_pin = os.environ.get("AFORGE_BIN", "").strip() if self._alias else None
        if legacy_pin:
            return await AforgeProvider(legacy_pin).execute(prompt, options)
        environment = {**os.environ, **_child_environment(options, None)}
        if not _has_model_key(environment):
            if not _missing_key_logged:
                _missing_key_logged = True
                logger.warning(CODEAF_MISSING_KEY_WARNING)
            raise HarnessProviderNotConfigured(
                binary=self._bin or os.environ.get("CODEAF_BIN") or "codeaf"
            )
        override = self._bin or os.environ.get("CODEAF_BIN")
        if override:
            binary = await asyncio.to_thread(shutil.which, override)
            if binary is None:
                raise provider_unavailable("codeaf", override)
            provider = "codeaf"
        else:
            if cls._resolution_lock is None:
                cls._resolution_lock = asyncio.Lock()
            async with cls._resolution_lock:
                if cls._resolution_cache is None:
                    if cls._resolution_task is None:
                        cls._resolution_task = asyncio.create_task(
                            cls._resolve_automatic()
                        )
                    task = cls._resolution_task
                else:
                    task = None
                resolved = cls._resolution_cache
            if task is not None:
                try:
                    resolved = await asyncio.wait_for(
                        asyncio.shield(task), max(0, deadline - time.monotonic())
                    )
                except asyncio.TimeoutError:
                    return self._timeout_result(
                        started, "Timed out while provisioning CodeAF"
                    )
            assert resolved is not None
            binary, provider, download_error = resolved
            if binary is None:
                raise provider_unavailable("codeaf", "codeaf", detail=download_error)
        if provider == "aforge":
            if not cls._fallback_logged:
                logger.warning(
                    "CodeAF is not installed; falling back to the legacy aforge binary at %s. Run `af codeaf ensure`.",
                    binary,
                )
                cls._fallback_logged = True
            try:
                return await asyncio.wait_for(
                    AforgeProvider(binary).execute(prompt, options),
                    max(0, deadline - time.monotonic()),
                )
            except asyncio.TimeoutError:
                return self._timeout_result(
                    started, "CodeAF fallback timed out", "aforge"
                )
        if self._alias and not cls._alias_logged:
            logger.info(
                "provider 'aforge' now runs CodeAF; set AFORGE_BIN to pin the legacy binary"
            )
            cls._alias_logged = True

        async def execute_remaining() -> RawResult:
            async with self._get_semaphore():
                return await self._execute_impl(binary, prompt, options, deadline)

        try:
            return await asyncio.wait_for(
                execute_remaining(), max(0, deadline - time.monotonic())
            )
        except asyncio.TimeoutError:
            return self._timeout_result(started, "CodeAF timed out")

    @staticmethod
    def _timeout_result(
        started: float, message: str, provider: str = "codeaf"
    ) -> RawResult:
        return RawResult(
            provider=provider,
            is_error=True,
            failure_type=FailureType.TIMEOUT,
            error_message=message,
            metrics=Metrics(duration_api_ms=int((time.monotonic() - started) * 1000)),
        )

    async def _execute_impl(
        self, binary: str, prompt: str, options: dict[str, object], deadline: float
    ) -> RawResult:
        root = str(options.get("project_dir") or options.get("cwd") or ".")
        model_value, variant = resolve_model_and_variant(options)
        overlay = options.get("env")
        environment = {**os.environ, **(overlay if isinstance(overlay, dict) else {})}
        model = _strip_openrouter_prefix(
            model_value
            or str(environment.get("CODEAF_MODEL") or "").strip()
            or str(environment.get("AFORGE_MODEL") or "").strip()
            or CODEAF_DEFAULT_MODEL
        )
        command = (
            os.environ.get("AGENTFIELD_CODEAF_COMMAND", "").strip()
            or os.environ.get("AGENTFIELD_AFORGE_COMMAND", "").strip()
            or "exec"
        ).lower()
        if command not in {"exec", "do"}:
            if not type(self)._command_warning_logged:
                logger.warning(
                    "Invalid AGENTFIELD_CODEAF_COMMAND %r; using exec", command
                )
                type(self)._command_warning_logged = True
            command = "exec"
        cmd = [binary, command, "--json", "--dir", root]
        if command == "do":
            cmd.append("--yes-spend")
        cmd.extend(["--model", model])
        if command == "do":
            cmd.extend(["--plan-model", model, "--check-model", model])
        timeout = max(0, deadline - time.monotonic())
        cmd.extend(
            ["--timeout", f"{max(1, timeout - 5):.3f}".rstrip("0").rstrip(".") + "s"]
        )
        if command == "exec":
            turns = options.get("max_turns")
            if isinstance(turns, int) and not isinstance(turns, bool) and turns > 0:
                cmd.extend(["--max-turns", str(turns)])
            system = options.get("system_prompt")
            if isinstance(system, str) and system.strip():
                cmd.extend(["--system", system.strip()])
            input_text = prompt
        else:
            cmd.append("-")
            input_text = _task_input(prompt, options.get("system_prompt"))
        env = await asyncio.to_thread(_child_environment, options, variant)
        started = time.monotonic()
        try:
            stdout, stderr, returncode = await run_cli(
                cmd,
                env=env,
                unset_env=("CODEAF_MODELS", "AFORGE_MODELS"),
                cwd=None,
                timeout=max(0, deadline - time.monotonic()),
                idle_seconds=0,
                input_text=input_text,
            )
        except FileNotFoundError as exc:
            raise provider_unavailable("codeaf", binary) from exc
        except TimeoutError as exc:
            return RawResult(
                is_error=True,
                error_message=str(exc),
                failure_type=FailureType.TIMEOUT,
                metrics=Metrics(
                    duration_api_ms=int((time.monotonic() - started) * 1000),
                    model=model,
                ),
                provider="codeaf",
            )
        duration = int((time.monotonic() - started) * 1000)
        envelope = _parse_envelope(stdout)
        data = envelope or {}
        answer = data.get("answer")
        text = answer.strip() if isinstance(answer, str) and answer.strip() else None
        stop = str(data.get("stop", ""))
        failure = FailureType.CRASH
        if returncode < 0:
            pass
        elif stop == "deadline":
            failure = FailureType.TIMEOUT
        elif stop == "done" and data.get("ok") is True:
            failure = FailureType.NONE
        elif stop in {"unchecked", "budget", "turn-cap"} and text:
            logger.warning(
                "codeaf returned %s%s",
                "partial output: " if stop != "unchecked" else "",
                stop,
            )
            failure = FailureType.NONE
        elif stop in {"budget", "turn-cap"}:
            failure = FailureType.NO_OUTPUT
        elif envelope is None and returncode == 0 and not stdout.strip():
            failure = FailureType.NO_OUTPUT
        elif (
            stop
            not in {"done", "unchecked", "question", "incomplete", "error", "price"}
            and envelope is not None
            and returncode == 0
            and text
        ):
            failure = FailureType.NONE
        error_message = None
        if failure != FailureType.NONE:
            detail = data.get("error") or text or strip_ansi(stderr.strip())[-1000:]
            error_message = (
                f"codeaf exit code {returncode} (stop={stop}): {str(detail)[:1000]}"
            )
            if stop == "question" and data.get("blocked_on"):
                error_message += f". blocked_on: {str(data['blocked_on'])[:1000]}"
        tokens = data.get("tokens")
        tokens = tokens if isinstance(tokens, dict) else {}
        cost = _numeric(data.get("spend_usd"))
        returned_model = data.get("model")
        return RawResult(
            result=text if failure == FailureType.NONE else None,
            messages=[envelope] if envelope is not None else [],
            metrics=Metrics(
                duration_api_ms=duration,
                num_turns=int(_numeric(data.get("steps")) or 0),
                input_tokens=int(_numeric(tokens.get("in")) or 0),
                output_tokens=int(_numeric(tokens.get("out")) or 0),
                total_cost_usd=float(cost) if cost is not None else None,
                model=returned_model
                if isinstance(returned_model, str) and returned_model.strip()
                else model,
            ),
            is_error=failure != FailureType.NONE,
            error_message=error_message,
            failure_type=failure,
            returncode=returncode,
            provider="codeaf",
        )
