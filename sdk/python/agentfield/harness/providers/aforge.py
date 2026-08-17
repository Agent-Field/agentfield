"""Aforge provider using CLI subprocess."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import ClassVar

from agentfield.harness._availability import ensure_cli_available, provider_unavailable
from agentfield.harness._cli import (
    estimate_cli_cost,
    resolve_model_and_variant,
    run_cli,
    strip_ansi,
)
from agentfield.harness._result import FailureType, Metrics, RawResult

logger = logging.getLogger("agentfield.harness.aforge")

_REASONING_VARIANTS = {"off", "low", "medium", "high"}


def _strip_openrouter_prefix(model: str) -> str:
    """Strip one leading ``openrouter/`` prefix from a model slug."""
    return model.removeprefix("openrouter/")


def _parse_envelope(stdout: str) -> dict[str, object] | None:
    """Return the last canonical ``do`` or ``exec`` envelope."""
    # Both surfaces print one JSON object. Parse that shape before falling back
    # to the line-oriented form tolerated for wrappers that prepend diagnostics.
    try:
        value = json.loads(stdout.strip())
    except ValueError:
        pass
    else:
        if isinstance(value, dict) and ("deliverable" in value or "text" in value):
            return value

    for line in reversed(
        [line.strip() for line in stdout.splitlines() if line.strip()]
    ):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and ("deliverable" in value or "text" in value):
            return value
    return None


def _numeric(value: object) -> int | float | None:
    """Return a JSON numeric value, excluding booleans."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _task_input(prompt: str, system_prompt: object) -> str:
    if isinstance(system_prompt, str) and system_prompt.strip():
        return f"{system_prompt.strip()}\n\nTask:\n{prompt}"
    return prompt


def _crash_message(
    returncode: int,
    blocked_on: str,
    deliverable: str | None,
    stderr: str,
) -> str:
    """Build a consistent, bounded aforge crash message."""
    clean_stderr = strip_ansi(stderr.strip())
    exit_context = f"aforge exit code {returncode}"
    if returncode < 0:
        message = f"Process killed by signal {-returncode}. {exit_context}"
    else:
        message = exit_context
    if clean_stderr:
        message += f". stderr: {clean_stderr[:1000]}"
    elif blocked_on:
        message += f". blocked_on: {blocked_on[:1000]}"
    elif deliverable:
        message += f". partial: {deliverable[:1000]}"
    return message


class AforgeProvider:
    """Aforge CLI provider.

    ``exec`` is the default direct one-shot entry point. Set
    ``AGENTFIELD_AFORGE_COMMAND=do`` to opt into Aforge's routed workflow.
    """

    _MAX_CONCURRENT: ClassVar[int] = int(os.environ.get("AFORGE_MAX_CONCURRENT", "8"))
    _concurrency_sem: ClassVar[asyncio.Semaphore | None] = None

    def __init__(self, bin_path: str = "aforge"):
        self._bin = (
            os.environ.get("AFORGE_BIN", bin_path) if bin_path == "aforge" else bin_path
        )

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._concurrency_sem is None:
            cls._concurrency_sem = asyncio.Semaphore(cls._MAX_CONCURRENT)
        return cls._concurrency_sem

    async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
        ensure_cli_available("aforge", self._bin)
        sem = self._get_semaphore()
        logger.debug(
            "Waiting for concurrency slot (%d/%d in use)",
            self._MAX_CONCURRENT - sem._value,
            self._MAX_CONCURRENT,
        )
        async with sem:
            return await self._execute_impl(prompt, options)

    async def _execute_impl(self, prompt: str, options: dict[str, object]) -> RawResult:
        # project_dir is the canonical agent root; a nested task cwd must not
        # restrict access to sibling paths under the shared project root.
        root = str(options.get("project_dir") or options.get("cwd") or ".")
        timeout_seconds = int(
            os.environ.get("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", "1800")
        )
        model_value, variant_value = resolve_model_and_variant(options)
        # Leave a small landing window so aforge can emit its honest timeout
        # envelope before the outer subprocess watchdog has to kill it.
        aforge_timeout = max(1, timeout_seconds - 5)
        command = os.environ.get("AGENTFIELD_AFORGE_COMMAND", "exec").strip().lower()
        if command not in {"do", "exec"}:
            return RawResult(
                is_error=True,
                error_message=(
                    "AGENTFIELD_AFORGE_COMMAND must be 'do' or 'exec', "
                    f"got {command!r}"
                ),
                failure_type=FailureType.CRASH,
                metrics=Metrics(),
            )
        if command == "exec":
            cmd = [
                self._bin,
                "exec",
                "--json",
                "-w",
                root,
                "--timeout",
                str(aforge_timeout),
                "--context-fill",
                "60",
                "--completion-reserve",
                "65536",
            ]
            system_prompt = options.get("system_prompt")
            if isinstance(system_prompt, str) and system_prompt.strip():
                cmd.extend(["--system", system_prompt.strip()])
            input_text = prompt
        else:
            cmd = [
                self._bin,
                "do",
                "--json",
                "--yes-spend",
                "-w",
                root,
                "--timeout",
                str(aforge_timeout),
            ]
            input_text = _task_input(prompt, options.get("system_prompt"))

        env: dict[str, str] = {"AFORGE_MODELS": ""} if command == "exec" else {}
        if model_value:
            model_slug = _strip_openrouter_prefix(model_value)
            env["AFORGE_MODEL"] = model_slug
            if command == "exec":
                cmd.extend(["--model", model_slug, "--plan-model", model_slug])

        if variant_value:
            normalized_variant = variant_value.strip().lower()
            if normalized_variant in _REASONING_VARIANTS:
                env["AFORGE_EXEC_REASONING"] = normalized_variant
            else:
                logger.debug("Ignoring unsupported aforge variant %r", variant_value)

        env_value = options.get("env")
        if isinstance(env_value, dict):
            env.update(
                {
                    str(key): str(value)
                    for key, value in env_value.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
            )

        start_api = time.monotonic()

        try:
            stdout, stderr, returncode = await run_cli(
                cmd,
                env=env,
                cwd=None,
                timeout=timeout_seconds,
                # Aforge is stdout-silent until its final envelope; disable the
                # no-progress watchdog so legitimate long runs are not killed.
                idle_seconds=0,
                input_text=input_text,
            )
        except FileNotFoundError as exc:
            raise provider_unavailable("aforge", self._bin) from exc
        except TimeoutError as exc:
            return RawResult(
                is_error=True,
                error_message=str(exc),
                failure_type=FailureType.TIMEOUT,
                metrics=Metrics(),
            )

        api_ms = int((time.monotonic() - start_api) * 1000)
        envelope = _parse_envelope(stdout)

        result_text: str | None = None
        usage: dict[object, object] = {}
        calls = 0
        blocked_on = ""
        stop = ""
        if envelope is not None:
            text_value = envelope.get("text" if command == "exec" else "deliverable")
            if isinstance(text_value, str) and text_value.strip():
                result_text = text_value.strip()
            blocked_value = envelope.get("blocked_on")
            if isinstance(blocked_value, str):
                blocked_on = blocked_value.strip()
            usage_value = envelope.get("usage")
            if isinstance(usage_value, dict):
                usage = usage_value
                calls_value = _numeric(usage.get("calls"))
                if calls_value is not None:
                    calls = int(calls_value)
            if command == "exec":
                stop_value = envelope.get("stop")
                if isinstance(stop_value, str):
                    stop = stop_value.strip()
                turns_value = _numeric(envelope.get("turns"))
                if turns_value is not None:
                    calls = int(turns_value)

        clean_stderr = strip_ansi(stderr.strip()) if stderr else ""
        logger.info(
            "aforge finished: returncode=%d stdout=%d chars elapsed=%ds",
            returncode,
            len(stdout),
            api_ms // 1000,
        )
        if not result_text and clean_stderr:
            logger.warning("aforge no text. stderr: %s", clean_stderr[:800])

        if command == "exec":
            # Budget and turn-cap exits with a usable landing are partial
            # successes under the original exec adapter contract.
            is_error = (
                returncode < 0
                or result_text is None
                or returncode not in {0, 2, 3}
            )
        else:
            is_error = returncode != 0 or result_text is None or bool(blocked_on)
        if not is_error:
            failure_type = FailureType.NONE
        elif (command == "do" and returncode == 2) or (
            command == "exec" and returncode == 4
        ):
            failure_type = FailureType.TIMEOUT
        else:
            failure_type = FailureType.CRASH
        error_message = (
            _crash_message(returncode, blocked_on or stop, result_text, stderr)
            if is_error
            else None
        )

        input_tokens_value = _numeric(usage.get("prompt_tokens"))
        output_tokens_value = _numeric(usage.get("completion_tokens"))
        cached_tokens_value = _numeric(usage.get("cached_tokens"))
        spend_value = _numeric(envelope.get("spend")) if envelope else None
        cost_value = _numeric(usage.get("cost"))
        if spend_value is not None and spend_value > 0:
            total_cost = float(spend_value)
        elif cost_value is not None and cost_value > 0:
            total_cost = float(cost_value)
        else:
            total_cost = estimate_cli_cost(
                model=model_value or "",
                prompt=prompt,
                result_text=result_text,
            )

        return RawResult(
            result=result_text,
            messages=[envelope] if envelope is not None else [],
            metrics=Metrics(
                duration_api_ms=api_ms,
                num_turns=calls,
                total_cost_usd=total_cost,
                session_id="",
                input_tokens=int(input_tokens_value or 0),
                output_tokens=int(output_tokens_value or 0),
                cache_read_tokens=int(cached_tokens_value or 0),
                cache_creation_tokens=0,
                model=model_value,
            ),
            is_error=is_error,
            error_message=error_message,
            failure_type=failure_type,
            returncode=returncode,
        )
