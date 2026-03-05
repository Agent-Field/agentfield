"""Gemini CLI provider using subprocess."""

from __future__ import annotations

import time
from typing import Dict, Optional

from agentfield.harness._cli import run_cli, strip_ansi
from agentfield.harness._result import FailureType, Metrics, RawResult


class GeminiProvider:
    """Gemini CLI provider. Invokes `gemini` CLI subprocess."""

    def __init__(self, bin_path: str = "gemini"):
        self._bin = bin_path

    async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
        cmd = [self._bin]

        if options.get("cwd"):
            cmd.extend(["-C", str(options["cwd"])])
        if options.get("permission_mode") == "auto":
            cmd.extend(["--sandbox"])
        if options.get("model"):
            cmd.extend(["-m", str(options["model"])])
        cmd.extend(["-p", prompt])

        env: Dict[str, str] = {}
        env_value = options.get("env")
        if isinstance(env_value, dict):
            env = {
                str(key): str(value)
                for key, value in env_value.items()
                if isinstance(key, str) and isinstance(value, str)
            }

        cwd: Optional[str] = None
        cwd_value = options.get("cwd")
        if isinstance(cwd_value, str):
            cwd = cwd_value

        start_api = time.monotonic()

        try:
            stdout, stderr, returncode = await run_cli(cmd, env=env, cwd=cwd)
        except FileNotFoundError:
            return RawResult(
                is_error=True,
                error_message=(
                    f"Gemini binary not found at '{self._bin}'. "
                    "Install Gemini CLI: https://github.com/google-gemini/gemini-cli"
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

        api_ms = int((time.monotonic() - start_api) * 1000)
        result_text = stdout.strip() if stdout.strip() else None
        clean_stderr = strip_ansi(stderr.strip()) if stderr else ""

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

        return RawResult(
            result=result_text,
            messages=[],
            metrics=Metrics(
                duration_api_ms=api_ms,
                num_turns=1 if result_text else 0,
                session_id="",
            ),
            is_error=is_error,
            error_message=error_message,
            failure_type=failure_type,
            returncode=returncode,
        )
