"""OpenCode provider using CLI subprocess."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import tempfile
import time
from typing import ClassVar, Dict, Optional

from agentfield.harness._cli import estimate_cli_cost, run_cli, strip_ansi
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
    _MAX_CONCURRENT: ClassVar[int] = int(os.environ.get("OPENCODE_MAX_CONCURRENT", "10"))
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

        # Use --dir for project directory (replaces deprecated -c which now means --continue)
        cwd_value = options.get("cwd")
        if isinstance(cwd_value, str):
            cmd.extend(["--dir", cwd_value])
        elif isinstance(options.get("project_dir"), str):
            cmd.extend(["--dir", str(options["project_dir"])])

        # Pass model via -m flag on the run subcommand
        if options.get("model"):
            cmd.extend(["-m", str(options["model"])])

        # Skip interactive permission prompts for headless execution
        cmd.append("--dangerously-skip-permissions")

        # Handle system prompt - prepend to user prompt since OpenCode
        # has no native --system-prompt flag
        effective_prompt = prompt
        system_prompt = options.get("system_prompt")
        if isinstance(system_prompt, str) and system_prompt.strip():
            effective_prompt = (
                f"SYSTEM INSTRUCTIONS:\n{system_prompt.strip()}\n\n"
                f"---\n\nUSER REQUEST:\n{prompt}"
            )

        # Prompt is a positional arg to `opencode run` (not -p)
        cmd.append(effective_prompt)

        env: Dict[str, str] = {}
        env_value = options.get("env")
        if isinstance(env_value, dict):
            env = {
                str(key): str(value)
                for key, value in env_value.items()
                if isinstance(key, str) and isinstance(value, str)
            }

        # Model is passed via -m flag on the run subcommand (see above)

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
                    cmd, env=env, cwd=cwd, timeout=timeout_seconds
                )
            except FileNotFoundError:
                return RawResult(
                    is_error=True,
                    error_message=(
                        f"OpenCode binary not found at '{self._bin}'. "
                        "Install OpenCode: https://opencode.ai"
                    ),
                    failure_type=FailureType.CRASH,
                    metrics=Metrics(),
                )
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
        result_text = stdout.strip() if stdout.strip() else None
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

        estimated_cost = estimate_cli_cost(
            model=str(options.get("model", "")),
            prompt=effective_prompt,
            result_text=result_text,
        )

        return RawResult(
            result=result_text,
            messages=[],
            metrics=Metrics(
                duration_api_ms=api_ms,
                num_turns=1 if result_text else 0,
                total_cost_usd=estimated_cost,
                session_id="",
            ),
            is_error=is_error,
            error_message=error_message,
            failure_type=failure_type,
            returncode=returncode,
        )
