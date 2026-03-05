"""OpenCode provider using CLI subprocess."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from typing import Dict, Optional

from agentfield.harness._cli import run_cli, strip_ansi
from agentfield.harness._result import FailureType, Metrics, RawResult


class OpenCodeProvider:
    """OpenCode CLI provider. Invokes ``opencode run`` subprocess."""

    def __init__(
        self,
        bin_path: str = "opencode",
        server_url: Optional[str] = None,
    ):
        self._bin = bin_path
        self._explicit_server = server_url or os.environ.get("OPENCODE_SERVER")

    async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
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
                stdout, stderr, returncode = await run_cli(cmd, env=env, cwd=cwd)
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
