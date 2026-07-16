"""Gemini CLI provider using subprocess."""

from __future__ import annotations

import time
from typing import Dict, Optional

from agentfield.harness._cli import estimate_cli_cost, run_cli, strip_ansi
from agentfield.harness._availability import ensure_cli_available, provider_unavailable
from agentfield.harness._result import FailureType, Metrics, RawResult


class GeminiProvider:
    """Gemini CLI provider. Invokes `gemini` CLI subprocess."""

    def __init__(self, bin_path: str = "gemini"):
        self._bin = bin_path

    async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
        ensure_cli_available("gemini", self._bin)
        cmd = [self._bin]

        # permission_mode → approval mode (agentfield#687). NOTE: gemini's
        # --sandbox RESTRICTS execution (runs tools in a container); it does NOT
        # grant access. The old code used --sandbox for "auto", which was
        # backwards. --yolo auto-approves all actions.
        permission_mode = options.get("permission_mode")
        if permission_mode == "auto":
            cmd.append("--yolo")
        elif permission_mode == "plan":
            cmd.extend(["--approval-mode", "plan"])

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

        # Gemini has NO -C/--cd flag — it rejects unknown args outright
        # ("Unknown argument: C"), so the previous `-C <cwd>` crashed every call.
        # The agent root is the process working directory; project_dir is the
        # canonical field, cwd is the fallback (agentfield#686).
        cwd: Optional[str] = None
        root = options.get("project_dir") or options.get("cwd")
        if isinstance(root, str):
            cwd = root

        start_api = time.monotonic()

        try:
            stdout, stderr, returncode = await run_cli(cmd, env=env, cwd=cwd)
        except FileNotFoundError as exc:
            raise provider_unavailable("gemini", self._bin) from exc
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

        estimated_cost = estimate_cli_cost(
            model=str(options.get("model", "")),
            prompt=prompt,
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
