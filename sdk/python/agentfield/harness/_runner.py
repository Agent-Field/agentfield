from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional

from agentfield.harness._result import FailureType, HarnessResult, RawResult
from agentfield.harness._schema import (
    build_followup_prompt,
    build_prompt_suffix,
    cleanup_temp_files,
    diagnose_output_failure,
    get_output_path,
    parse_and_validate,
)
from agentfield.harness.providers._base import HarnessProvider
from agentfield.harness.providers._factory import build_provider

logger = logging.getLogger(__name__)

TRANSIENT_PATTERNS = {
    "rate limit",
    "rate_limit",
    "overloaded",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "temporarily unavailable",
    "service unavailable",
    "503",
    "502",
    "504",
    "internal server error",
    "500",
}

DEFAULT_SCHEMA_RETRIES = 2


def _is_transient(error_str: str) -> bool:
    lower = error_str.lower()
    return any(pattern in lower for pattern in TRANSIENT_PATTERNS)


def _resolve_options(
    config: Optional[Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    if config is not None:
        for field_name in [
            "provider",
            "model",
            "max_turns",
            "max_budget_usd",
            "max_retries",
            "initial_delay",
            "max_delay",
            "backoff_factor",
            "tools",
            "permission_mode",
            "system_prompt",
            "env",
            "cwd",
            "project_dir",
            "codex_bin",
            "gemini_bin",
            "opencode_bin",
            "opencode_server",
            "schema_max_retries",
        ]:
            val = getattr(config, field_name, None)
            if val is not None:
                options[field_name] = val

    for key, val in overrides.items():
        if val is not None:
            options[key] = val
    return options


def _accumulate_metrics(
    all_raws: List[RawResult],
) -> tuple[Optional[float], int, str, List[Dict[str, Any]]]:
    total_cost: Optional[float] = None
    total_turns = 0
    session_id = ""
    all_messages: List[Dict[str, Any]] = []

    for raw in all_raws:
        if raw.metrics.total_cost_usd is not None:
            total_cost = (total_cost or 0.0) + raw.metrics.total_cost_usd
        total_turns += raw.metrics.num_turns
        if raw.metrics.session_id:
            session_id = raw.metrics.session_id
        all_messages.extend(raw.messages)

    return total_cost, total_turns, session_id, all_messages


class HarnessRunner:
    def __init__(self, config: Optional[Any] = None):
        self._config = config

    async def run(
        self,
        prompt: str,
        *,
        schema: Any = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_turns: Optional[int] = None,
        max_budget_usd: Optional[float] = None,
        tools: Optional[list[str]] = None,
        permission_mode: Optional[str] = None,
        system_prompt: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        **kwargs: Any,
    ) -> HarnessResult:
        overrides = {
            "provider": provider,
            "model": model,
            "max_turns": max_turns,
            "max_budget_usd": max_budget_usd,
            "tools": tools,
            "permission_mode": permission_mode,
            "system_prompt": system_prompt,
            "env": env,
            "cwd": cwd,
            **kwargs,
        }
        options = _resolve_options(self._config, overrides)

        resolved_provider = options.get("provider")
        if not resolved_provider:
            raise ValueError(
                "No harness provider specified. Set 'provider' in HarnessConfig "
                "or pass it to .harness() call."
            )

        resolved_cwd = str(options.get("cwd", "."))
        provider_instance = self._build_provider(str(resolved_provider), options)

        # When project_dir is set (opencode provider), place the output file
        # inside project_dir so the coding agent's Write tool can reach it.
        # Use a unique subdir to avoid collisions from parallel calls.
        project_dir = options.get("project_dir")
        output_dir = resolved_cwd
        _temp_output_dir: Optional[str] = None
        if isinstance(project_dir, str) and project_dir:
            import tempfile as _tempfile

            _temp_output_dir = _tempfile.mkdtemp(prefix=".secaf-out-", dir=project_dir)
            output_dir = _temp_output_dir

        effective_prompt = prompt
        if schema is not None:
            effective_prompt = prompt + build_prompt_suffix(schema, output_dir)
        options["_original_prompt"] = effective_prompt

        start_time = time.monotonic()
        try:
            raw = await self._execute_with_retry(
                provider_instance, effective_prompt, options
            )

            if schema is not None:
                return await self._handle_schema_with_retry(
                    raw,
                    schema,
                    output_dir,
                    start_time,
                    provider_instance,
                    options,
                )

            elapsed = int((time.monotonic() - start_time) * 1000)
            return HarnessResult(
                result=raw.result,
                parsed=None,
                is_error=raw.is_error,
                error_message=raw.error_message,
                failure_type=raw.failure_type,
                cost_usd=raw.metrics.total_cost_usd,
                num_turns=raw.metrics.num_turns,
                duration_ms=elapsed,
                session_id=raw.metrics.session_id,
                messages=raw.messages,
            )
        finally:
            if schema is not None:
                cleanup_temp_files(output_dir)
            if _temp_output_dir:
                import shutil as _shutil

                _shutil.rmtree(_temp_output_dir, ignore_errors=True)

    def _build_provider(
        self, provider_name: str, options: Dict[str, Any]
    ) -> HarnessProvider:
        from types import SimpleNamespace

        provider_options = dict(options)
        provider_options["provider"] = provider_name
        config_ns = SimpleNamespace(**provider_options)
        config_for_factory: Any = config_ns
        return build_provider(config_for_factory)

    async def _execute_with_retry(
        self,
        provider: HarnessProvider,
        prompt: str,
        options: Dict[str, Any],
    ) -> RawResult:
        max_retries = int(options.get("max_retries", 3))
        initial_delay = float(options.get("initial_delay", 1.0))
        max_delay = float(options.get("max_delay", 30.0))
        backoff_factor = float(options.get("backoff_factor", 2.0))

        last_error: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                result = await provider.execute(prompt, options)
                if not result.is_error:
                    return result

                error_msg = result.error_message or ""
                if _is_transient(error_msg) and attempt < max_retries:
                    delay = min(initial_delay * (backoff_factor**attempt), max_delay)
                    delay += random.uniform(-delay * 0.25, delay * 0.25)
                    await asyncio.sleep(delay)
                    continue
                return result
            except Exception as exc:
                last_error = exc
                if _is_transient(str(exc)) and attempt < max_retries:
                    delay = min(initial_delay * (backoff_factor**attempt), max_delay)
                    delay += random.uniform(-delay * 0.25, delay * 0.25)
                    await asyncio.sleep(delay)
                    continue
                raise

        if last_error is not None:
            raise last_error
        return RawResult(is_error=True, error_message="Max retries exceeded")

    async def _handle_schema_with_retry(
        self,
        initial_raw: RawResult,
        schema: Any,
        cwd: str,
        start_time: float,
        provider: HarnessProvider,
        options: Dict[str, Any],
    ) -> HarnessResult:
        output_path = get_output_path(cwd)
        schema_max_retries = int(
            options.get("schema_max_retries", DEFAULT_SCHEMA_RETRIES)
        )

        all_raws: List[RawResult] = [initial_raw]

        validated = parse_and_validate(output_path, schema)
        if validated is not None:
            elapsed = int((time.monotonic() - start_time) * 1000)
            cost, turns, sid, msgs = _accumulate_metrics(all_raws)
            return HarnessResult(
                result=initial_raw.result,
                parsed=validated,
                is_error=False,
                cost_usd=cost,
                num_turns=turns,
                duration_ms=elapsed,
                session_id=sid,
                messages=msgs,
            )

        _retryable = {FailureType.CRASH, FailureType.NO_OUTPUT, FailureType.NONE}
        if (
            initial_raw.is_error
            and not os.path.exists(output_path)
            and initial_raw.failure_type not in _retryable
        ) or (
            schema_max_retries == 0
            and initial_raw.is_error
            and not os.path.exists(output_path)
        ):
            elapsed = int((time.monotonic() - start_time) * 1000)
            cost, turns, sid, msgs = _accumulate_metrics(all_raws)
            provider_error = initial_raw.error_message or "Provider execution failed."
            return HarnessResult(
                result=initial_raw.result,
                parsed=None,
                is_error=True,
                error_message=(
                    f"{provider_error} Output file was not created at {output_path}."
                ),
                failure_type=initial_raw.failure_type,
                cost_usd=cost,
                num_turns=turns,
                duration_ms=elapsed,
                session_id=sid,
                messages=msgs,
            )

        last_session_id = initial_raw.metrics.session_id

        for retry_num in range(schema_max_retries):
            if retry_num > 0:
                await asyncio.sleep(min(0.5 * (2 ** (retry_num - 1)), 5.0))

            is_crash = all_raws[
                -1
            ].failure_type == FailureType.CRASH and not os.path.exists(output_path)
            if is_crash:
                original_prompt = options.get("_original_prompt", "")
                retry_prompt = (
                    original_prompt
                    if original_prompt
                    else build_followup_prompt(
                        diagnose_output_failure(output_path, schema), cwd, schema
                    )
                )
            else:
                error_detail = diagnose_output_failure(output_path, schema)
                retry_prompt = build_followup_prompt(error_detail, cwd, schema)

            detail_for_log = diagnose_output_failure(output_path, schema)

            logger.info(
                "Schema validation retry %d/%d: %s",
                retry_num + 1,
                schema_max_retries,
                detail_for_log[:200],
            )

            retry_options = dict(options)
            if last_session_id and not is_crash:
                retry_options["resume_session_id"] = last_session_id

            retry_raw = await self._execute_with_retry(
                provider, retry_prompt, retry_options
            )
            all_raws.append(retry_raw)

            if retry_raw.metrics.session_id:
                last_session_id = retry_raw.metrics.session_id

            if retry_raw.is_error:
                logger.warning(
                    "Schema retry %d provider error: %s",
                    retry_num + 1,
                    retry_raw.error_message,
                )
                continue

            validated = parse_and_validate(output_path, schema)
            if validated is not None:
                elapsed = int((time.monotonic() - start_time) * 1000)
                cost, turns, sid, msgs = _accumulate_metrics(all_raws)
                logger.info("Schema validation succeeded on retry %d", retry_num + 1)
                return HarnessResult(
                    result=retry_raw.result,
                    parsed=validated,
                    is_error=False,
                    cost_usd=cost,
                    num_turns=turns,
                    duration_ms=elapsed,
                    session_id=sid,
                    messages=msgs,
                )

        elapsed = int((time.monotonic() - start_time) * 1000)
        cost, turns, sid, msgs = _accumulate_metrics(all_raws)
        final_diagnosis = diagnose_output_failure(output_path, schema)
        return HarnessResult(
            result=all_raws[-1].result,
            parsed=None,
            is_error=True,
            error_message=(
                f"Schema validation failed after {schema_max_retries} "
                f"retry attempt(s). Last error: {final_diagnosis}"
            ),
            failure_type=FailureType.SCHEMA,
            cost_usd=cost,
            num_turns=turns,
            duration_ms=elapsed,
            session_id=sid,
            messages=msgs,
        )
