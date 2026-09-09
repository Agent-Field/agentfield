"""OpenCode provider using CLI subprocess."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from typing import ClassVar, Dict, Optional

from agentfield.harness._cli import (
    estimate_cli_cost,
    extract_final_text,
    resolve_model_and_variant,
    extract_token_usage,
    parse_jsonl,
    run_cli,
    strip_ansi,
)
from agentfield.harness._availability import ensure_cli_available, provider_unavailable
from agentfield.harness._result import FailureType, Metrics, RawResult

logger = logging.getLogger("agentfield.harness.opencode")

# opencode CLI sometimes prints a hard error to stderr but exits 0
# (notably "Model not found", auth errors, schema validation failures).
# These patterns mark stderr as containing a real failure, not just noise
# like the one-time SQLite migration prelude.
_OPENCODE_STDERR_ERROR_PATTERNS = (
    re.compile(r"^Error:", re.MULTILINE),
    re.compile(r"\bModel not found\b"),
    re.compile(r"\bAuthenticationError\b"),
    re.compile(r"\bUnauthorized\b"),
    re.compile(r"\bAPIError\b"),
)

_OPENCODE_CONFIG_SCHEMA = "https://opencode.ai/config.json"
_OPENCODE_AGENT_NAME = "agentfield-harness"
_AGENTFIELD_WORKER_INSTRUCTION = (
    "You are an AgentField-launched worker. Complete the assigned prompt directly. "
    "Do not invoke AgentField orchestration, the `af` CLI, `swe-planner.plan`, "
    "or delegate work back to AgentField."
)
_OPENCODE_INLINE_SYSTEM_PROMPT_ENV = "AGENTFIELD_OPENCODE_INLINE_SYSTEM_PROMPT"
_TRUE_ENV_VALUES = frozenset(("1", "true", "yes", "on"))


def _opencode_permissions() -> dict[str, object]:
    """Build the fixed headless permission baseline for the harness agent.

    The wildcard allow preserves the existing compatibility behavior. The
    AgentField skill rule is narrower than disabling skills entirely: it
    prevents an AgentField-launched worker from loading AgentField
    orchestration skills while leaving unrelated user and project skills
    available. OpenCode uses the last matching rule, so this rule follows the
    wildcard allow. HarnessConfig.tools and permission_mode are intentionally
    not translated until OpenCode has a strict per-tool authorization mode.
    """
    # Headless runs must never wait for an interactive response. Keep this
    # baseline independent of permission_mode: question and task are always
    # denied, while the wildcard keeps the current permissive tool behavior.
    return {
        "*": "allow",
        "skill": {"agentfield*": "deny"},
        "question": "deny",
        "task": "deny",
    }


def _agent_system_prompt(options: dict[str, object]) -> str:
    system_prompt = options.get("system_prompt")
    caller_prompt = system_prompt.strip() if isinstance(system_prompt, str) else ""
    return (
        f"{caller_prompt}\n\n{_AGENTFIELD_WORKER_INSTRUCTION}"
        if caller_prompt
        else _AGENTFIELD_WORKER_INSTRUCTION
    )


def _inline_system_prompt_enabled(options: dict[str, object]) -> bool:
    env_value = options.get("env")
    if isinstance(env_value, dict) and _OPENCODE_INLINE_SYSTEM_PROMPT_ENV in env_value:
        value = env_value[_OPENCODE_INLINE_SYSTEM_PROMPT_ENV]
    else:
        value = os.environ.get(_OPENCODE_INLINE_SYSTEM_PROMPT_ENV, "")
    return isinstance(value, str) and value.strip().lower() in _TRUE_ENV_VALUES


def _deep_merge_config(
    base: dict[str, object], overlay: dict[str, object]
) -> dict[str, object]:
    """Merge nested OpenCode objects, with the generated overlay taking precedence."""
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_config(existing, value)
        else:
            merged[key] = value
    return merged


def _normalize_agentfield_harness_config(
    config: dict[str, object], *, strip_system_prompt: bool
) -> None:
    agents = config.get("agent")
    selected_agent = (
        agents.get(_OPENCODE_AGENT_NAME) if isinstance(agents, dict) else None
    )
    if not isinstance(selected_agent, dict):
        return

    if strip_system_prompt:
        selected_agent.pop("prompt", None)

    permission = selected_agent.get("permission")
    if not isinstance(permission, dict):
        return

    # OpenCode evaluates the last matching rule. Keep the generated wildcard
    # first, caller-specific rules in the middle, and AgentField's targeted
    # denials last even when the caller already supplied permission entries.
    ordered: dict[str, object] = {}
    if "*" in permission:
        ordered["*"] = permission["*"]
    for key, value in permission.items():
        if key not in {"*", "skill", "question", "task"}:
            ordered[key] = value

    if "skill" in permission:
        skill = permission["skill"]
        if isinstance(skill, dict):
            ordered_skill = {
                key: value for key, value in skill.items() if key != "agentfield*"
            }
            if "agentfield*" in skill:
                ordered_skill["agentfield*"] = skill["agentfield*"]
            ordered["skill"] = ordered_skill
        else:
            ordered["skill"] = skill
    if "question" in permission:
        ordered["question"] = permission["question"]
    if "task" in permission:
        ordered["task"] = permission["task"]
    selected_agent["permission"] = ordered


def _merge_opencode_config_content(
    existing_content: Optional[str],
    overlay_content: str,
    *,
    strip_system_prompt: bool = False,
) -> str:
    """Merge the AgentField overlay into caller-provided OpenCode JSON."""
    if not existing_content or not existing_content.strip():
        return overlay_content

    try:
        existing = json.loads(existing_content)
        overlay = json.loads(overlay_content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "OPENCODE_CONFIG_CONTENT must be valid JSON when supplied to the "
            "AgentField OpenCode provider"
        ) from exc

    if not isinstance(existing, dict):
        raise ValueError(
            "OPENCODE_CONFIG_CONTENT must contain a JSON object when supplied "
            "to the AgentField OpenCode provider"
        )
    if not isinstance(overlay, dict):  # pragma: no cover - internal invariant
        raise ValueError("AgentField OpenCode overlay must contain a JSON object")

    merged = _deep_merge_config(existing, overlay)
    _normalize_agentfield_harness_config(
        merged, strip_system_prompt=strip_system_prompt
    )
    return json.dumps(merged, ensure_ascii=False)


def _inline_opencode_prompt(prompt: str, options: dict[str, object]) -> str:
    return (
        "SYSTEM INSTRUCTIONS:\n"
        f"{_agent_system_prompt(options)}\n\n"
        "---\n\n"
        f"USER REQUEST:\n{prompt}"
    )


def _build_opencode_config_content(
    options: dict[str, object],
    model_value: Optional[str],
    variant_value: Optional[str],
    *,
    include_system_prompt: bool = True,
) -> str:
    """Serialize the per-run OpenCode agent overlay."""
    agent: dict[str, object] = {
        "mode": "primary",
        "steps": 500,
        "permission": _opencode_permissions(),
    }
    if include_system_prompt:
        agent["prompt"] = _agent_system_prompt(options)
    if model_value:
        agent["model"] = model_value
    if variant_value:
        agent["reasoningEffort"] = variant_value

    return json.dumps(
        {
            "$schema": _OPENCODE_CONFIG_SCHEMA,
            "default_agent": _OPENCODE_AGENT_NAME,
            "agent": {_OPENCODE_AGENT_NAME: agent},
        },
        ensure_ascii=False,
    )


def _prompt_via_stdin() -> bool:
    """Whether to hand the prompt to opencode over stdin instead of argv.

    On Windows the CLI on PATH is usually an npm .cmd shim that runs via
    cmd.exe, whose ~8k command-line cap real prompts blow straight through
    ("The command line is too long."). opencode reads the prompt from stdin
    when the positional arg is absent, so feed it that way there. POSIX keeps
    the battle-tested positional-arg path.
    """
    return os.name == "nt"


def _count_turns_from_events(events: list[dict[str, object]]) -> int:
    """Count opencode turns from JSON events.

    Preferred definition is one turn per ``step_start`` event. If a stream
    has no step markers, fall back to counting ``tool_use`` events.
    """
    step_starts = sum(1 for event in events if event.get("type") == "step_start")
    if step_starts > 0:
        return step_starts

    tool_uses = sum(1 for event in events if event.get("type") == "tool_use")
    return tool_uses


def _cost_from_events(events: list[dict[str, object]]) -> float | None:
    """Sum opencode per-step costs when present in the JSON stream."""
    total_cost = 0.0
    found_cost = False

    for event in events:
        if event.get("type") != "step_finish":
            continue
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        cost = part.get("cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool):
            total_cost += float(cost)
            found_cost = True

    return total_cost if found_cost else None


def _tokens_from_events(
    events: list[dict[str, object]],
) -> tuple[dict[str, int], bool]:
    """Sum token counts from opencode ``step_finish.part.tokens`` objects."""
    total = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_creation_tokens": 0,
    }
    found_tokens = False

    def _int(value: object) -> int:
        if isinstance(value, bool):
            return 0
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0

    for event in events:
        if event.get("type") != "step_finish":
            continue
        part = event.get("part")
        if not isinstance(part, dict):
            continue
        tokens = part.get("tokens")
        if not isinstance(tokens, dict):
            continue
        found_tokens = True
        total["input_tokens"] += _int(tokens.get("input"))
        total["output_tokens"] += _int(tokens.get("output")) + _int(
            tokens.get("reasoning")
        )
        cache = tokens.get("cache")
        if isinstance(cache, dict):
            total["cache_read_tokens"] += _int(cache.get("read"))
            total["cache_creation_tokens"] += _int(cache.get("write"))

    return total, found_tokens


def _extract_opencode_event_error(events: list[dict[str, object]]) -> str | None:
    """Pull a meaningful failure message from an in-band JSON error event."""
    for event in events:
        if event.get("type") != "error":
            continue

        for key in ("message", "error", "text"):
            value = event.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:1000]

        part = event.get("part")
        if isinstance(part, dict):
            for key in ("message", "error", "text"):
                value = part.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:1000]

        return str(event)[:1000]

    return None


def _extract_opencode_error(stderr: str) -> str:
    """Pull the meaningful failure line(s) out of opencode stderr.

    opencode's stderr typically opens with the SQLite migration prelude
    ("Performing one time database migration…") followed by the real error.
    Naively truncating the first 800 chars hides the part that matters,
    so prefer the line carrying the error marker plus a small window of
    context around it.
    """
    lines = stderr.splitlines()
    for i, line in enumerate(lines):
        for pat in _OPENCODE_STDERR_ERROR_PATTERNS:
            if pat.search(line):
                window = lines[max(0, i - 1) : i + 5]
                return "\n".join(window).strip()[:1000]
    return stderr[:1000]


class OpenCodeProvider:
    """OpenCode CLI provider. Invokes ``opencode run`` subprocess (v1.4+)."""

    # Global concurrency limiter: prevents too many simultaneous opencode
    # processes from overwhelming the LLM API with concurrent requests.
    # Each opencode run spawns a full subprocess (pyright, DB migration, etc.)
    # so unbounded concurrency causes rate-limiting and transient failures.
    # Default raised 3 → 10 to match the typical pr-af review fan-out
    # (~6–8 review_dimension phases + 3 meta-lenses); OpenRouter handles
    # this comfortably on Kimi K2.6. Lower via OPENCODE_MAX_CONCURRENT if
    # your provider has tighter per-key rate limits.
    _MAX_CONCURRENT: ClassVar[int] = int(
        os.environ.get("OPENCODE_MAX_CONCURRENT", "10")
    )
    _concurrency_sem: ClassVar[Optional[asyncio.Semaphore]] = None

    # Shared XDG_DATA_HOME across calls when opt-in is enabled. SQLite
    # migrations only run once per process instead of per-call. None means
    # "fresh tempdir per call" (current default).
    _shared_data_dir: ClassVar[Optional[str]] = None

    def __init__(self, bin_path: str = "opencode"):
        self._bin = bin_path

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        if cls._concurrency_sem is None:
            cls._concurrency_sem = asyncio.Semaphore(cls._MAX_CONCURRENT)
        return cls._concurrency_sem

    async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
        ensure_cli_available("opencode", self._bin)
        sem = self._get_semaphore()
        logger.debug(
            "Waiting for concurrency slot (%d/%d in use)",
            self._MAX_CONCURRENT - sem._value,
            self._MAX_CONCURRENT,
        )
        async with sem:
            return await self._execute_impl(prompt, options)

    async def _execute_impl(self, prompt: str, options: dict[str, object]) -> RawResult:
        # opencode v1.4+ uses the `run` subcommand (replaces deprecated -p/-c flags)
        cmd = [self._bin, "run"]
        cmd.extend(["--format", "json"])

        # Select a deterministic per-run agent. Its configuration is supplied
        # through OPENCODE_CONFIG_CONTENT below rather than a shared file. The
        # inline opt-out changes only the system-prompt transport so the same
        # targeted permissions and model settings remain active.
        inline_system_prompt = _inline_system_prompt_enabled(options)
        cmd.extend(["--agent", _OPENCODE_AGENT_NAME])

        # --dir is the project root the agent may read and write. project_dir is
        # the canonical field; fall back to cwd when it is unset. Previously cwd
        # took precedence, so a nested cwd under a shared project_dir made
        # opencode treat the rest of the root as an external directory and
        # auto-reject reads/writes of sibling paths — see agentfield#684. This
        # now matches the Go SDK's opencode provider precedence.
        dir_value = options.get("project_dir") or options.get("cwd")
        if isinstance(dir_value, str):
            cmd.extend(["--dir", dir_value])

        # Pass model via -m. A "#variant" suffix on the model (or an explicit
        # options["variant"]) maps to --variant — opencode's provider-specific
        # reasoning effort (e.g. high, max, minimal).
        model_value, variant_value = resolve_model_and_variant(options)
        if model_value:
            cmd.extend(["-m", model_value])
        if variant_value:
            cmd.extend(["--variant", variant_value])

        # opencode v1.14 does not accept --dangerously-skip-permissions on the
        # `run` subcommand — passing it makes yargs print the run-help screen
        # to stdout and exit 0, which the SDK then captures as the LLM
        # response. opencode in non-TTY mode proceeds without permission
        # prompting, so no flag is needed. See agentfield#582.

        env: Dict[str, str] = {}
        env_value = options.get("env")
        if isinstance(env_value, dict):
            env = {
                str(key): str(value)
                for key, value in env_value.items()
                if isinstance(key, str) and isinstance(value, str)
            }

        # Keep caller credentials and configuration in place. The per-call
        # value follows normal subprocess precedence over the ambient one; the
        # generated harness overlay is then merged into that value. Inline mode
        # removes the selected agent's configured prompt after merging so the
        # system instructions travel through the legacy prompt path instead.
        existing_config = env.get("OPENCODE_CONFIG_CONTENT")
        if existing_config is None:
            existing_config = os.environ.get("OPENCODE_CONFIG_CONTENT")
        env["OPENCODE_CONFIG_CONTENT"] = _merge_opencode_config_content(
            existing_config,
            _build_opencode_config_content(
                options,
                model_value,
                variant_value,
                include_system_prompt=not inline_system_prompt,
            ),
            strip_system_prompt=inline_system_prompt,
        )

        if inline_system_prompt:
            # The fixed AgentField instruction remains present in this
            # compatibility and rollback path.
            effective_prompt = _inline_opencode_prompt(prompt, options)
        else:
            # The system prompt belongs to the selected agent, so the task
            # remains the only user-facing prompt sent to OpenCode.
            effective_prompt = prompt

        prompt_via_stdin = _prompt_via_stdin()
        if not prompt_via_stdin:
            cmd.append(effective_prompt)

        cwd: Optional[str] = None

        # Per-call XDG_DATA_HOME by default — guarantees session isolation.
        # AGENTFIELD_OPENCODE_REUSE_DATA_DIR=true reuses one dir across calls
        # in this process so SQLite migrations only run once. Opt-in because
        # the implications vary by container layout (read-only /tmp, multi-
        # tenant deployments, etc.) — default behavior is unchanged.
        reuse_data_dir = os.environ.get(
            "AGENTFIELD_OPENCODE_REUSE_DATA_DIR", "false"
        ).strip().lower() in ("1", "true", "yes")

        temp_data_dir: Optional[str] = None
        if reuse_data_dir:
            if type(self)._shared_data_dir is None or not os.path.isdir(
                type(self)._shared_data_dir or ""
            ):
                type(self)._shared_data_dir = tempfile.mkdtemp(
                    prefix=".secaf-opencode-data-shared-"
                )
            data_dir = type(self)._shared_data_dir
        else:
            temp_data_dir = tempfile.mkdtemp(prefix=".secaf-opencode-data-")
            data_dir = temp_data_dir
        env["XDG_DATA_HOME"] = data_dir

        # Wall-clock cap for ONE opencode subprocess. Default 1800s (30min):
        # Kimi K2.6 on a complex review can need 20+ minutes; cutting at 600s
        # (the previous default) was killing slow-but-progressing calls and
        # then re-running them from scratch via the runner's transient retry
        # path. If a call doesn't finish in 30 min, the prompt or the model
        # is wrong — re-running won't help — so we let it fail cleanly.
        # Override via AGENTFIELD_HARNESS_TIMEOUT_SECONDS for tighter caps.
        timeout_seconds = int(
            os.environ.get("AGENTFIELD_HARNESS_TIMEOUT_SECONDS", "1800")
        )

        start_api = time.monotonic()

        try:
            try:
                stdout, stderr, returncode = await run_cli(
                    cmd,
                    env=env,
                    cwd=cwd,
                    timeout=timeout_seconds,
                    input_text=effective_prompt if prompt_via_stdin else None,
                )
            except FileNotFoundError as exc:
                raise provider_unavailable("opencode", self._bin) from exc
            except TimeoutError as exc:
                return RawResult(
                    is_error=True,
                    error_message=str(exc),
                    failure_type=FailureType.TIMEOUT,
                    metrics=Metrics(),
                )
        finally:
            # Only clean up the per-call tempdir; the shared one outlives the
            # call by design (skip the SQLite migration on the next call).
            if temp_data_dir is not None:
                shutil.rmtree(temp_data_dir, ignore_errors=True)

        api_ms = int((time.monotonic() - start_api) * 1000)
        events = parse_jsonl(stdout)
        if events:
            result_text = extract_final_text(events)
        else:
            result_text = stdout.strip() if stdout.strip() else None
        event_error = _extract_opencode_event_error(events)
        clean_stderr = strip_ansi(stderr.strip()) if stderr else ""

        logger.info(
            "opencode finished: returncode=%d stdout=%d chars elapsed=%ds",
            returncode,
            len(stdout),
            api_ms // 1000,
        )
        if not result_text and clean_stderr:
            logger.warning("opencode no stdout. stderr: %s", clean_stderr[:800])

        if returncode < 0:
            failure_type = FailureType.CRASH
            is_error = True
            error_message: str | None = (
                f"Process killed by signal {-returncode}. stderr: {clean_stderr[:500]}"
                if clean_stderr
                else f"Process killed by signal {-returncode}."
            )
        elif returncode != 0 and result_text is None:
            failure_type = FailureType.CRASH
            is_error = True
            error_message = (
                _extract_opencode_error(clean_stderr)
                if clean_stderr
                else (f"Process exited with code {returncode} and produced no output.")
            )
        elif event_error is not None and result_text is None:
            failure_type = FailureType.CRASH
            is_error = True
            error_message = event_error
        elif (
            result_text is None
            and clean_stderr
            and any(pat.search(clean_stderr) for pat in _OPENCODE_STDERR_ERROR_PATTERNS)
        ):
            # opencode sometimes exits 0 even on hard failures like
            # "Model not found" — surface the real error from stderr instead
            # of silently returning empty output that downstream callers
            # interpret as "agent failed to produce a valid result".
            failure_type = FailureType.CRASH
            is_error = True
            error_message = _extract_opencode_error(clean_stderr)
        else:
            failure_type = FailureType.NONE
            is_error = False
            error_message = None

        stream_cost = _cost_from_events(events)
        if stream_cost is not None:
            estimated_cost = stream_cost
        else:
            estimated_cost = estimate_cli_cost(
                model=model_value or "",
                prompt=effective_prompt,
                result_text=result_text,
            )

        num_turns = _count_turns_from_events(events)
        if num_turns == 0 and result_text:
            num_turns = 1

        tokens, found_tokens = _tokens_from_events(events)
        if not found_tokens:
            tokens = extract_token_usage(events)

        return RawResult(
            result=result_text,
            messages=events,
            metrics=Metrics(
                duration_api_ms=api_ms,
                num_turns=num_turns,
                total_cost_usd=estimated_cost,
                session_id="",
                input_tokens=tokens["input_tokens"],
                output_tokens=tokens["output_tokens"],
                cache_read_tokens=tokens["cache_read_tokens"],
                cache_creation_tokens=tokens["cache_creation_tokens"],
                model=model_value,
            ),
            is_error=is_error,
            error_message=error_message,
            failure_type=failure_type,
            returncode=returncode,
        )
