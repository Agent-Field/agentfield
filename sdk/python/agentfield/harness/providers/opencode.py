"""OpenCode provider using CLI subprocess."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import time
from typing import ClassVar, Dict, Optional

from agentfield.harness._cli import estimate_cli_cost, run_cli, strip_ansi
from agentfield.harness._result import FailureType, Metrics, RawResult

logger = logging.getLogger("agentfield.harness.opencode")


class OpenCodeProvider:
    """OpenCode CLI provider. Invokes ``opencode run`` subprocess (v1.4+)."""

    # Global concurrency limiter: prevents too many simultaneous opencode
    # processes from overwhelming the LLM API with concurrent requests.
    # Each opencode run spawns a full subprocess (pyright, DB migration, etc.)
    # so unbounded concurrency causes rate-limiting and transient failures.
    _MAX_CONCURRENT: ClassVar[int] = int(os.environ.get("OPENCODE_MAX_CONCURRENT", "3"))
    _concurrency_sem: ClassVar[Optional[asyncio.Semaphore]] = None

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

        temp_data_dir = tempfile.mkdtemp(prefix=".secaf-opencode-data-")
        env["XDG_DATA_HOME"] = temp_data_dir

        start_api = time.monotonic()

        try:
            try:
                stdout, stderr, returncode = await run_cli(
                    cmd, env=env, cwd=cwd, timeout=600
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
                clean_stderr[:1000]
                if clean_stderr
                else (f"Process exited with code {returncode} and produced no output.")
            )
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
