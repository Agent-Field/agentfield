"""Cursor provider using CLI subprocess (agent -p --output-format stream-json)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agentfield.harness._availability import ensure_cli_available, provider_unavailable
from agentfield.harness._cli import (
    estimate_cli_cost,
    extract_final_text,
    extract_token_usage,
    parse_jsonl,
    resolve_model_and_variant,
    run_cli,
    strip_ansi,
)
from agentfield.harness._result import FailureType, Metrics, RawResult


class CursorProvider:
    """Cursor CLI provider. Invokes `agent -p --output-format stream-json`.

    The binary is named ``agent``, not ``cursor``: Cursor's CLI ships its
    headless agent under that name, which is why the default here does not
    match the provider name the way it does for codex or gemini.
    """

    def __init__(self, bin_path: str = "agent"):
        self._bin = bin_path

    async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
        ensure_cli_available("cursor", self._bin)
        # -p is headless (print) mode. --trust accepts the workspace without
        # an interactive prompt, which a subprocess can never answer.
        cmd = [self._bin, "-p", "--trust", "--output-format", "stream-json"]

        # Agent root: project_dir is the canonical field, fall back to cwd,
        # matching the codex provider. Cursor spells it --workspace.
        root = options.get("project_dir") or options.get("cwd")
        if isinstance(root, str):
            cmd.extend(["--workspace", root])

        # permission_mode -> Cursor's --force / --mode. Unlike codex there is
        # no sandbox dimension: --force is "act without asking", `--mode plan`
        # plans without editing, and `--mode ask` would block forever in a
        # subprocess, so an unset mode maps to plan rather than ask.
        permission_mode = options.get("permission_mode")
        if permission_mode == "plan":
            cmd.extend(["--mode", "plan"])
        elif permission_mode == "auto":
            cmd.append("--force")
        else:
            cmd.extend(["--mode", "plan"])

        model_value, _variant_value = resolve_model_and_variant(options)
        if model_value:
            cmd.extend(["--model", model_value])

        # Session resume: the chat id comes back as session_id on the result
        # event, and goes out as --resume on the next call.
        resume_session_id = options.get("resume_session_id")
        if isinstance(resume_session_id, str) and resume_session_id:
            cmd.extend(["--resume", resume_session_id])

        # Prompt last, as a positional argument.
        cmd.append(prompt)

        env: Dict[str, str] = {}
        env_value = options.get("env")
        if isinstance(env_value, dict):
            env = {
                str(key): str(value)
                for key, value in env_value.items()
                if isinstance(key, str) and isinstance(value, str)
            }

        # The CLI reads its credential from the environment. Only set it when
        # the caller supplied one, so an inherited CURSOR_API_KEY is not
        # clobbered with an empty string.
        api_key = options.get("api_key")
        if isinstance(api_key, str) and api_key:
            env["CURSOR_API_KEY"] = api_key

        cwd: Optional[str] = root if isinstance(root, str) else None
        start_api = time.monotonic()

        try:
            stdout, stderr, returncode = await run_cli(cmd, env=env, cwd=cwd)
        except FileNotFoundError as exc:
            raise provider_unavailable("cursor", self._bin) from exc
        except TimeoutError as exc:
            return RawResult(
                is_error=True,
                error_message=str(exc),
                failure_type=FailureType.TIMEOUT,
                metrics=Metrics(),
            )

        api_ms = int((time.monotonic() - start_api) * 1000)
        events = parse_jsonl(stdout)
        result_text = extract_final_text(events)

        num_turns = 0
        total_cost: Optional[float] = estimate_cli_cost(
            model=model_value or "",
            prompt=prompt,
            result_text=result_text,
        )
        session_id = ""
        messages: List[Dict[str, Any]] = events

        for event in events:
            event_type = event.get("type")
            if event_type == "assistant":
                num_turns += 1
            elif event_type == "result":
                # The result event carries the chat id to resume with. Read it
                # from any event that has one, since a stream may carry the id
                # on an earlier system event too.
                session_id = str(event.get("session_id", "")) or session_id
            elif not session_id and event.get("session_id"):
                session_id = str(event.get("session_id", ""))

        tokens = extract_token_usage(events)

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
            messages=messages,
            metrics=Metrics(
                duration_api_ms=api_ms,
                num_turns=num_turns,
                total_cost_usd=total_cost,
                session_id=session_id,
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
