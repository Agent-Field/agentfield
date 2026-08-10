"""Grok Build CLI provider (local `grok` binary).

Invokes headless multi-turn Grok agent sessions under a PTY allocated by
``script(1)`` — the Grok CLI errors with ``Device not configured (os error
6)`` when started with plain pipes. The exact ``script`` invocation is
flavor-specific; see :func:`_pty_command`.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from agentfield.harness._availability import ensure_cli_available, provider_unavailable
from agentfield.harness._cli import resolve_model_and_variant, run_cli, strip_ansi
from agentfield.harness._result import FailureType, Metrics, RawResult

_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}\s*$")


def _pty_command(cmd: List[str]) -> List[str]:
    """Wrap ``cmd`` in ``script(1)`` so the child gets a real PTY.

    The two ``script`` flavors take the command to run in mutually
    incompatible ways, and getting it wrong fails *silently* rather than
    loudly:

    * BSD/macOS — ``script [options] [file [command ...]]``. The command is
      trailing argv, already correctly escaped by the OS.
    * util-linux (Linux) — the only command form is ``-c "<string>"``. Extra
      positional arguments after the typescript file are **not** treated as a
      command: ``script`` falls back to spawning ``$SHELL`` interactively, so
      the wrapped CLI never runs and the process hangs on a shell prompt until
      the harness idle watchdog kills it. Verified on util-linux 2.37.2 (the
      Ubuntu 22.04 runner image) for both ``script -q /dev/null grok …`` and
      the ``--``-separated ``script -q /dev/null -- grok …``.

    ``-e`` is required alongside ``-c``: without it util-linux ``script``
    exits 0 regardless of the child's status, masking every non-zero grok
    exit from the returncode handling below.

    The util-linux form runs the command string through a shell, so every
    argument is passed through :func:`shlex.quote`. grok argv carries
    caller-controlled text (``--system-prompt-override``, project paths,
    model ids) that must reach the CLI verbatim and must never be
    re-interpreted as shell syntax.

    Windows has no ``script(1)`` and no PTY story here, so ``cmd`` is
    returned unchanged — same as when ``script`` is missing from PATH.
    """
    if os.name == "nt" or shutil.which("script") is None:
        return list(cmd)
    if sys.platform.startswith("linux"):
        # -q quiet, -e propagate child exit status, -c command, /dev/null
        # typescript file; stdout still carries the agent JSON.
        return ["script", "-q", "-e", "-c", shlex.join(cmd), "/dev/null"]
    return ["script", "-q", "/dev/null", *cmd]


def _permission_mode(options: dict[str, object]) -> str:
    mode = options.get("permission_mode")
    if mode == "plan":
        return "plan"
    if mode in {"auto", "acceptEdits", "bypassPermissions", "dontAsk", "default"}:
        # Harness runs are unattended; prefer non-interactive approvals.
        if mode == "plan":
            return "plan"
        if mode in {"auto", "acceptEdits", "bypassPermissions"}:
            return "bypassPermissions" if mode in {"auto", "bypassPermissions"} else "acceptEdits"
        return str(mode)
    return "bypassPermissions"


def _extract_json_payload(stdout: str) -> Optional[dict[str, Any]]:
    """Parse Grok ``--output-format json`` payload, tolerating script PTY noise."""
    cleaned = strip_ansi(stdout or "")
    # script(1) may prefix with control characters (e.g. ^D).
    cleaned = cleaned.replace("\x04", "").strip()
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(cleaned)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _result_text(payload: dict[str, Any]) -> Optional[str]:
    for key in ("text", "result", "message", "content", "output"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _usage_metrics(payload: dict[str, Any], model: Optional[str]) -> Metrics:
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    model_usage = payload.get("modelUsage") if isinstance(payload.get("modelUsage"), dict) else {}
    # Prefer first modelUsage entry for model id / token totals when present.
    resolved_model = model
    first_model_usage: dict[str, Any] = {}
    if model_usage:
        first_key = next(iter(model_usage.keys()), None)
        if not resolved_model and first_key is not None:
            resolved_model = first_key
        if first_key is not None and isinstance(model_usage.get(first_key), dict):
            first_model_usage = model_usage[first_key]

    def _int(value: object, default: int = 0) -> int:
        # Token fields on Metrics are non-optional ints. Missing usage must be
        # 0 — returning None breaks _accumulate_metrics (int += None).
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return int(value)
        return default

    def _token(*keys: str) -> int:
        for key in keys:
            if key in usage:
                return _int(usage.get(key))
            if key in first_model_usage:
                return _int(first_model_usage.get(key))
        return 0

    cost = payload.get("total_cost_usd")
    if not isinstance(cost, (int, float)) or isinstance(cost, bool):
        cost = None
    else:
        cost = float(cost)

    turns = payload.get("num_turns")
    if not isinstance(turns, int):
        turns = 0

    session_id = payload.get("sessionId") or payload.get("session_id") or ""
    return Metrics(
        num_turns=turns,
        total_cost_usd=cost,
        session_id=str(session_id) if session_id else "",
        input_tokens=_token("input_tokens", "prompt_tokens"),
        output_tokens=_token("output_tokens", "completion_tokens"),
        cache_read_tokens=_token(
            "cache_read_input_tokens", "cache_read_tokens", "cached_input_tokens"
        ),
        cache_creation_tokens=_token(
            "cache_creation_input_tokens", "cache_creation_tokens"
        ),
        model=str(resolved_model) if resolved_model else model,
    )


class GrokProvider:
    """Grok Build CLI provider. Invokes local ``grok`` headless sessions."""

    def __init__(self, bin_path: str = "grok"):
        self._bin = bin_path

    async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
        ensure_cli_available("grok", self._bin)

        root = options.get("project_dir") or options.get("cwd")
        if not isinstance(root, str) or not root.strip():
            root = os.getcwd()

        model_value, variant_value = resolve_model_and_variant(options)
        max_turns = options.get("max_turns")
        if not isinstance(max_turns, int) or max_turns <= 0:
            max_turns = 30

        permission = _permission_mode(options)

        # Long prompts exceed argv limits; always feed via --prompt-file.
        prompt_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                prefix="agentfield-grok-prompt-",
                suffix=".txt",
                delete=False,
            ) as handle:
                handle.write(prompt)
                prompt_path = handle.name

            # Grok requires a PTY; wrap with script(1) when available.
            grok_cmd = [
                self._bin,
                "--cwd",
                root,
                "--permission-mode",
                permission,
                "--always-approve",
                "--output-format",
                "json",
                "--max-turns",
                str(max_turns),
                "--no-alt-screen",
                "--prompt-file",
                prompt_path,
            ]
            if model_value:
                grok_cmd.extend(["-m", model_value])
            if variant_value:
                grok_cmd.extend(["--reasoning-effort", variant_value])

            system_prompt = options.get("system_prompt")
            if isinstance(system_prompt, str) and system_prompt.strip():
                grok_cmd.extend(["--system-prompt-override", system_prompt])

            cmd = _pty_command(grok_cmd)

            env: Dict[str, str] = {}
            env_value = options.get("env")
            if isinstance(env_value, dict):
                env = {
                    str(key): str(value)
                    for key, value in env_value.items()
                    if isinstance(key, str) and isinstance(value, str)
                }
            # Ensure localhost control-plane traffic is not proxy-hijacked.
            env.setdefault("NO_PROXY", "127.0.0.1,localhost,::1")
            env.setdefault("no_proxy", "127.0.0.1,localhost,::1")

            start_api = time.monotonic()
            try:
                stdout, stderr, returncode = await run_cli(cmd, env=env, cwd=root)
            except FileNotFoundError as exc:
                raise provider_unavailable("grok", self._bin) from exc
            except TimeoutError as exc:
                return RawResult(
                    is_error=True,
                    error_message=str(exc),
                    failure_type=FailureType.TIMEOUT,
                    metrics=Metrics(model=model_value),
                )

            api_ms = int((time.monotonic() - start_api) * 1000)
            payload = _extract_json_payload(stdout)
            # Grok sometimes emits {"type":"error","message":"..."} on stdout with rc!=0.
            if payload and payload.get("type") == "error":
                err_msg = payload.get("message") or payload.get("error") or str(payload)
                return RawResult(
                    result=None,
                    messages=[payload],
                    metrics=Metrics(model=model_value),
                    is_error=True,
                    error_message=str(err_msg)[:1000],
                    failure_type=FailureType.CRASH,
                    returncode=returncode,
                )

            result_text = _result_text(payload) if payload else None
            metrics = (
                _usage_metrics(payload, model_value)
                if payload
                else Metrics(model=model_value)
            )
            metrics.duration_api_ms = api_ms

            messages: List[Dict[str, Any]] = [payload] if payload else []
            clean_stderr = strip_ansi(stderr.strip()) if stderr else ""

            if returncode < 0:
                return RawResult(
                    result=result_text,
                    messages=messages,
                    metrics=metrics,
                    is_error=True,
                    error_message=(
                        f"Process killed by signal {-returncode}. stderr: {clean_stderr[:500]}"
                        if clean_stderr
                        else f"Process killed by signal {-returncode}."
                    ),
                    failure_type=FailureType.CRASH,
                    returncode=returncode,
                )

            if returncode != 0 and result_text is None:
                return RawResult(
                    result=None,
                    messages=messages,
                    metrics=metrics,
                    is_error=True,
                    error_message=(
                        clean_stderr[:1000]
                        if clean_stderr
                        else f"Process exited with code {returncode} and produced no output."
                    ),
                    failure_type=FailureType.CRASH,
                    returncode=returncode,
                )

            if result_text is None and payload is None:
                return RawResult(
                    result=stdout.strip() or None,
                    messages=messages,
                    metrics=metrics,
                    is_error=True,
                    error_message=(
                        "Grok produced no parseable JSON output. "
                        + (clean_stderr[:500] if clean_stderr else f"stdout[:500]={stdout[:500]!r}")
                    ),
                    failure_type=FailureType.CRASH,
                    returncode=returncode,
                )

            return RawResult(
                result=result_text,
                messages=messages,
                metrics=metrics,
                is_error=False,
                error_message=None,
                failure_type=FailureType.NONE,
                returncode=returncode,
            )
        finally:
            if prompt_path:
                try:
                    os.unlink(prompt_path)
                except OSError:
                    pass
