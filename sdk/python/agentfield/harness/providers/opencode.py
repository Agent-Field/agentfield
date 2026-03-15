"""OpenCode provider using CLI subprocess."""

from __future__ import annotations

import asyncio
import json
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
    """OpenCode CLI provider. Invokes ``opencode run`` subprocess."""

    # Global concurrency limiter: prevents too many simultaneous opencode
    # processes from overwhelming the LLM API with concurrent requests.
    # Each opencode run spawns a full subprocess (pyright, DB migration, etc.)
    # so unbounded concurrency causes rate-limiting and transient failures.
    _MAX_CONCURRENT: ClassVar[int] = int(os.environ.get("OPENCODE_MAX_CONCURRENT", "3"))
    _concurrency_sem: ClassVar[Optional[asyncio.Semaphore]] = None

    def __init__(
        self,
        bin_path: str = "opencode",
        server_url: Optional[str] = None,
    ):
        self._bin = bin_path
        self._explicit_server = server_url or os.environ.get("OPENCODE_SERVER")

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
        cmd = [self._bin, "run"]

        if options.get("model"):
            cmd.extend(["--model", str(options["model"])])

        # --dir sets the project root the coding agent explores.
        # Use project_dir (the actual target repo) if available, otherwise
        # fall back to cwd (which may be a temp dir for output).
        project_dir = options.get("project_dir")
        if isinstance(project_dir, str) and project_dir:
            cmd.extend(["--dir", project_dir])

        cwd: Optional[str] = None
        cwd_value = options.get("cwd")
        if isinstance(cwd_value, str):
            cwd = cwd_value

        # Prepend system prompt to the user prompt if provided.
        system_prompt = options.get("system_prompt")
        effective_prompt = prompt
        if isinstance(system_prompt, str) and system_prompt.strip():
            effective_prompt = (
                f"SYSTEM INSTRUCTIONS:\n{system_prompt.strip()}\n\n"
                f"---\n\nUSER REQUEST:\n{prompt}"
            )

        cmd.extend(["-f", "json"])
        cmd.append(effective_prompt)

        env: Dict[str, str] = {}
        env_value = options.get("env")
        if isinstance(env_value, dict):
            env = {
                str(key): str(value)
                for key, value in env_value.items()
                if isinstance(key, str) and isinstance(value, str)
            }

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

        parsed_cost: float | None = None
        parsed_prompt_tokens: int | None = None
        parsed_completion_tokens: int | None = None

        try:
            json_output = json.loads(stdout.strip())
            result_text = json_output.get("response", "").strip() or None
            raw_cost = json_output.get("cost")
            if raw_cost is not None and float(raw_cost) > 0:
                parsed_cost = float(raw_cost)
            raw_pt = json_output.get("prompt_tokens")
            if raw_pt is not None:
                parsed_prompt_tokens = int(raw_pt)
            raw_ct = json_output.get("completion_tokens")
            if raw_ct is not None:
                parsed_completion_tokens = int(raw_ct)
        except (json.JSONDecodeError, ValueError, TypeError):
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

        if parsed_cost is not None:
            final_cost = parsed_cost
        else:
            final_cost = estimate_cli_cost(
                model=str(options.get("model", "")),
                prompt=effective_prompt,
                result_text=result_text,
            )

        usage_data = None
        if parsed_prompt_tokens is not None or parsed_completion_tokens is not None:
            usage_data = {
                "prompt_tokens": parsed_prompt_tokens or 0,
                "completion_tokens": parsed_completion_tokens or 0,
            }

        return RawResult(
            result=result_text,
            messages=[],
            metrics=Metrics(
                duration_api_ms=api_ms,
                num_turns=1 if result_text else 0,
                total_cost_usd=final_cost,
                usage=usage_data,
                session_id="",
            ),
            is_error=is_error,
            error_message=error_message,
            failure_type=failure_type,
            returncode=returncode,
        )
