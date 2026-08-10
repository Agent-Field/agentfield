"""Aforge provider using CLI subprocess."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import ClassVar, Dict, Optional

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
    prefix = "openrouter/"
    return model[len(prefix) :] if model.startswith(prefix) else model


def _parse_envelope(stdout: str) -> dict[str, object] | None:
    """Return the last JSON object containing an aforge ``text`` field."""
    for line in reversed(
        [line.strip() for line in stdout.splitlines() if line.strip()]
    ):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict) and "text" in value:
            return value
    return None


def _numeric(value: object) -> int | float | None:
    """Return a JSON numeric value, excluding booleans."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _crash_message(returncode: int, stop: str, stderr: str) -> str:
    """Build a consistent, bounded aforge crash message."""
    clean_stderr = strip_ansi(stderr.strip())
    exit_context = f"aforge exit code {returncode}, stop={stop!s}"
    if returncode < 0:
        message = f"Process killed by signal {-returncode}. {exit_context}"
    else:
        message = exit_context
    if clean_stderr:
        message += f". stderr: {clean_stderr[:1000]}"
    return message


class AforgeProvider:
    """Aforge CLI provider. Invokes ``aforge exec --json`` subprocess."""

    _MAX_CONCURRENT: ClassVar[int] = int(os.environ.get("AFORGE_MAX_CONCURRENT", "8"))
    _concurrency_sem: ClassVar[Optional[asyncio.Semaphore]] = None

    def __init__(self, bin_path: str = "aforge"):
        self._bin = bin_path

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
        cmd = [self._bin, "exec", "--json", "-w", root]

        system_prompt = options.get("system_prompt")
        if isinstance(system_prompt, str) and system_prompt.strip():
            cmd.extend(["--system", system_prompt.strip()])

        model_value, variant_value = resolve_model_and_variant(options)
        env: Dict[str, str] = {}
        if model_value:
            env["AFORGE_MODEL"] = _strip_openrouter_prefix(model_value)

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

        timeout_seconds = int(
            os.environ.get("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", "1800")
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
                input_text=prompt,
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
        stop = ""
        usage: dict[object, object] = {}
        turns = 0
        if envelope is not None:
            text_value = envelope.get("text")
            if isinstance(text_value, str) and text_value.strip():
                result_text = text_value.strip()
            stop_value = envelope.get("stop")
            if isinstance(stop_value, str):
                stop = stop_value
            usage_value = envelope.get("usage")
            if isinstance(usage_value, dict):
                usage = usage_value
            turns_value = _numeric(envelope.get("turns"))
            if turns_value is not None:
                turns = int(turns_value)

        clean_stderr = strip_ansi(stderr.strip()) if stderr else ""
        logger.info(
            "aforge finished: returncode=%d stdout=%d chars elapsed=%ds",
            returncode,
            len(stdout),
            api_ms // 1000,
        )
        if not result_text and clean_stderr:
            logger.warning("aforge no text. stderr: %s", clean_stderr[:800])

        if returncode < 0:
            is_error = True
        elif returncode in (2, 3) and result_text:
            is_error = False
        elif returncode != 0:
            is_error = True
        else:
            is_error = result_text is None

        failure_type = FailureType.CRASH if is_error else FailureType.NONE
        error_message = _crash_message(returncode, stop, stderr) if is_error else None

        input_tokens_value = _numeric(usage.get("prompt_tokens"))
        output_tokens_value = _numeric(usage.get("completion_tokens"))
        cached_tokens_value = _numeric(usage.get("cached_tokens"))
        cost_value = _numeric(usage.get("cost"))
        if cost_value is not None and cost_value > 0:
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
                num_turns=turns,
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
