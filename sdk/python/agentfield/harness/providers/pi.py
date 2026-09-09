"""Pi-family harness providers using their JSON event-stream CLIs."""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional

from agentfield.harness._availability import ensure_cli_available, provider_unavailable
from agentfield.harness._cli import (
    parse_jsonl,
    resolve_model_and_variant,
    run_cli,
    strip_ansi,
)
from agentfield.harness._result import FailureType, Metrics, RawResult


_READ_ONLY_TOOLS = {"read", "grep", "find", "glob", "ls", "lsp"}


def _normalise_tools(tools: Iterable[object], *, omp: bool) -> list[str]:
    """Translate AgentField's provider-neutral tool names to Pi CLI names."""
    aliases = {"glob": "glob" if omp else "find"}
    normalised: list[str] = []
    for tool in tools:
        name = str(tool).strip().lower()
        if not name:
            continue
        name = aliases.get(name, name)
        if name not in normalised:
            normalised.append(name)
    return normalised


def _assistant_messages(events: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for event in events:
        if event.get("type") != "message_end":
            continue
        message = event.get("message")
        if isinstance(message, dict) and message.get("role") == "assistant":
            yield message


def _text_content(message: dict[str, Any]) -> Optional[str]:
    content = message.get("content")
    if isinstance(content, str):
        return content or None
    if not isinstance(content, list):
        return None
    parts = [
        part.get("text", "")
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
    ]
    text = "".join(parts)
    return text or None


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _float(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _parse_pi_events(
    events: list[dict[str, Any]], *, configured_model: Optional[str]
) -> tuple[Optional[str], Metrics, Optional[str]]:
    result_text: Optional[str] = None
    session_id = ""
    num_turns = sum(1 for event in events if event.get("type") == "turn_end")
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    cache_creation_tokens = 0
    total_cost: Optional[float] = None
    reported_model: Optional[str] = None
    provider_error: Optional[str] = None

    for event in events:
        if event.get("type") == "session" and isinstance(event.get("id"), str):
            session_id = event["id"]

    for message in _assistant_messages(events):
        text = _text_content(message)
        if text:
            result_text = text

        if isinstance(message.get("model"), str):
            reported_model = message["model"]

        usage = message.get("usage")
        if isinstance(usage, dict):
            input_tokens += _int(usage.get("input"))
            output_tokens += _int(usage.get("output"))
            cache_read_tokens += _int(usage.get("cacheRead"))
            cache_creation_tokens += _int(usage.get("cacheWrite"))
            cost = usage.get("cost")
            if isinstance(cost, dict):
                native_cost = _float(cost.get("total"))
                if native_cost is not None:
                    total_cost = (total_cost or 0.0) + native_cost

        stop_reason = message.get("stopReason")
        if stop_reason in {"error", "aborted"}:
            detail = message.get("errorMessage") or message.get("error")
            provider_error = str(detail or f"Pi stopped with reason {stop_reason!r}.")
        else:
            provider_error = None

    if num_turns == 0 and result_text:
        num_turns = 1

    return (
        result_text,
        Metrics(
            num_turns=num_turns,
            total_cost_usd=total_cost,
            session_id=session_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            model=configured_model or reported_model,
        ),
        provider_error,
    )


class _PiFamilyProvider:
    def __init__(self, *, provider: str, bin_path: str, omp: bool):
        self._provider = provider
        self._bin = bin_path
        self._omp = omp

    async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
        ensure_cli_available(self._provider, self._bin)
        cmd = [self._bin, "--print", "--mode", "json"]

        root = options.get("project_dir") or options.get("cwd")
        cwd = root if isinstance(root, str) else None
        if self._omp and cwd:
            cmd.extend(["--cwd", cwd])

        model_value, variant_value = resolve_model_and_variant(options)
        if model_value:
            cmd.extend(["--model", model_value])
        if variant_value:
            cmd.extend(["--thinking", variant_value])

        system_prompt = options.get("system_prompt")
        if isinstance(system_prompt, str) and system_prompt.strip():
            cmd.extend(["--system-prompt", system_prompt.strip()])

        resume_session_id = options.get("resume_session_id")
        if isinstance(resume_session_id, str) and resume_session_id:
            cmd.extend(["--resume" if self._omp else "--session", resume_session_id])

        permission_mode = options.get("permission_mode")
        # --tools is the enforced, vendor-documented read-only allowlist. Pi has
        # no approval flag (unknown options fail); OMP read-only tiers are
        # auto-approved even under always-ask.
        if permission_mode == "auto":
            if self._omp:
                cmd.append("--auto-approve")

        tools_value = options.get("tools")
        tools = (
            _normalise_tools(tools_value, omp=self._omp)
            if isinstance(tools_value, (list, tuple, set))
            else []
        )
        if permission_mode == "plan":
            tools = [tool for tool in tools if tool in _READ_ONLY_TOOLS]
            if not tools:
                tools = ["read", "grep", "glob" if self._omp else "find"]
        if isinstance(tools_value, (list, tuple, set)) or permission_mode == "plan":
            if tools:
                cmd.extend(["--tools", ",".join(tools)])
            else:
                cmd.append("--no-tools")

        env: Dict[str, str] = {}
        env_value = options.get("env")
        if isinstance(env_value, dict):
            env = {
                str(key): str(value)
                for key, value in env_value.items()
                if isinstance(key, str) and isinstance(value, str)
            }

        timeout: Optional[float] = None
        timeout_value = options.get("timeout")
        if isinstance(timeout_value, (int, float)) and not isinstance(
            timeout_value, bool
        ):
            timeout = float(timeout_value)

        start_api = time.monotonic()
        try:
            stdout, stderr, returncode = await run_cli(
                cmd,
                env=env,
                cwd=cwd,
                timeout=timeout,
                input_text=prompt,
            )
        except FileNotFoundError as exc:
            raise provider_unavailable(self._provider, self._bin) from exc
        except TimeoutError as exc:
            return RawResult(
                is_error=True,
                error_message=str(exc),
                failure_type=FailureType.TIMEOUT,
                metrics=Metrics(),
            )

        api_ms = int((time.monotonic() - start_api) * 1000)
        events = parse_jsonl(stdout)
        result_text, metrics, provider_error = _parse_pi_events(
            events, configured_model=model_value
        )
        metrics.duration_api_ms = api_ms
        clean_stderr = strip_ansi(stderr.strip()) if stderr else ""

        if returncode < 0:
            error_message = f"Process killed by signal {-returncode}."
            failure_type = FailureType.CRASH
        elif returncode != 0:
            error_message = (
                clean_stderr[:1000]
                or provider_error
                or (f"Process exited with code {returncode}.")
            )
            failure_type = FailureType.CRASH
        elif provider_error:
            error_message = provider_error
            failure_type = FailureType.API_ERROR
        elif result_text is None:
            error_message = clean_stderr[:1000] or (
                f"{self._provider} exited successfully without an assistant response."
            )
            failure_type = FailureType.NO_OUTPUT
        else:
            error_message = None
            failure_type = FailureType.NONE

        return RawResult(
            result=result_text,
            messages=events,
            metrics=metrics,
            is_error=failure_type != FailureType.NONE,
            error_message=error_message,
            failure_type=failure_type,
            returncode=returncode,
        )


class PiProvider(_PiFamilyProvider):
    """Pi coding-agent CLI provider."""

    def __init__(self, bin_path: str = "pi"):
        super().__init__(provider="pi", bin_path=bin_path, omp=False)


class OMPProvider(_PiFamilyProvider):
    """Oh My Pi (OMP) coding-agent CLI provider."""

    def __init__(self, bin_path: str = "omp"):
        super().__init__(provider="omp", bin_path=bin_path, omp=True)
