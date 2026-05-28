from __future__ import annotations

import asyncio
import json
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
    try_parse_from_text,
)
from agentfield.harness.providers._base import HarnessProvider
from agentfield.harness.providers._factory import build_provider

logger = logging.getLogger(__name__)

TRANSIENT_PATTERNS = {
    "rate limit",
    "rate_limit",
    "overloaded",
    # Note: "timeout" / "timed out" are intentionally NOT here. The opencode
    # provider's wall-clock timeout (default 30 min) is the cap on a *single*
    # subprocess; if that hits, the prompt or model is wrong and re-running
    # the entire 30-min subprocess won't fix it. Real network blips that
    # surface as "connection reset" / "503" still trigger retry.
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

# Wall-clock cap on a single AI-based JSON repair call. The repair is one
# LLM call with no tools, no repo access — it should finish in seconds.
# If it doesn't, fall through to the existing harness retry path.
_SCHEMA_REPAIR_TIMEOUT_SECONDS = 90


def _is_transient(error_str: str) -> bool:
    lower = error_str.lower()
    return any(pattern in lower for pattern in TRANSIENT_PATTERNS)


async def _ai_schema_repair(
    text: str,
    schema: Any,
    options: Dict[str, Any],
) -> Optional[Any]:
    """Try to coerce raw text into a valid pydantic schema instance via ONE LLM call.

    The harness's older fallback for schema-validation failures was to re-run the
    *entire* opencode subprocess (30-minute wall-clock, full repo browsing) hoping
    the model re-emitted something parseable. That's wildly expensive when the
    actual problem is just "the JSON closes with an extra trailing comma" or
    similar surface-level malformation — the model already produced the correct
    information, it just didn't structure it cleanly.

    This helper attempts cheap repair instead: take the existing text + the
    pydantic schema, ask a fresh LLM call to re-emit valid JSON conforming to
    the schema, and parse. No tools, no exploration, just reformatting.

    Returns the parsed instance on success, None on any failure (caller falls
    through to the existing retry-with-harness path). Skipped entirely when
    `text` is empty — there's nothing to repair from in that case, and a real
    rerun is the only option.
    """
    if not text or not text.strip():
        # Nothing to repair from. The user explicitly called out this case:
        # "if it actually failed and there was no output, then we can't do
        # an AI-based JSON repair." Fall through to existing retry path.
        return None

    try:
        import litellm  # type: ignore[import-not-found]
    except ImportError:
        return None

    schema_json: Optional[Dict[str, Any]] = None
    try:
        if hasattr(schema, "model_json_schema"):
            schema_json = schema.model_json_schema()
    except Exception:
        schema_json = None

    schema_section = (
        f"Target JSON schema:\n```json\n{json.dumps(schema_json, indent=2)}\n```\n\n"
        if schema_json is not None
        else ""
    )

    repair_prompt = (
        "The text below was produced as the answer to a structured-output task, "
        "but it did not parse as valid JSON conforming to the expected schema. "
        "Re-emit the SAME content as valid JSON matching the schema. Preserve "
        "every fact and finding from the original text — do NOT invent new "
        "information or drop fields. Output ONLY the JSON object, no surrounding "
        "prose, no markdown code fences, no commentary.\n\n"
        f"{schema_section}"
        f"Original text:\n{text}"
    )

    model = options.get("model")
    if not model:
        # No model configured — caller's harness call should have set one.
        # Bail rather than guess; the existing retry path will handle this.
        return None

    try:
        response = await asyncio.wait_for(
            litellm.acompletion(
                model=str(model),
                messages=[{"role": "user", "content": repair_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            ),
            timeout=_SCHEMA_REPAIR_TIMEOUT_SECONDS,
        )
        content = response.choices[0].message.content  # type: ignore[union-attr]
    except Exception as exc:
        logger.info("AI schema repair: LLM call failed (%s); falling through to harness retry", exc)
        return None

    if not content:
        return None

    return try_parse_from_text(content, schema)


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

        output_dir = resolved_cwd

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

        if validated is None and initial_raw.result:
            logger.info(
                "Output file missing/invalid at %s — trying stdout fallback",
                output_path,
            )
            validated = try_parse_from_text(initial_raw.result, schema)
            if validated is not None:
                logger.info("Stdout fallback succeeded")

        # Cheap-repair tier: if we have non-empty text but still no valid
        # parse, try one focused LLM call that just reformats the text into
        # valid JSON. This is dramatically cheaper than the retry loop below
        # (which spawns a whole new opencode subprocess), so we prefer it
        # whenever there's text to work with.
        if validated is None and initial_raw.result and initial_raw.result.strip():
            logger.info("Attempting AI-based JSON repair before harness retry")
            validated = await _ai_schema_repair(initial_raw.result, schema, options)
            if validated is not None:
                logger.info("AI schema repair succeeded — skipped harness retry")

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

            if validated is None and retry_raw.result:
                validated = try_parse_from_text(retry_raw.result, schema)
                if validated is not None:
                    logger.info(
                        "Schema retry %d succeeded via stdout fallback",
                        retry_num + 1,
                    )

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
