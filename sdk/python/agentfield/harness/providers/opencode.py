"""OpenCode provider using CLI subprocess.

Uses ``opencode serve`` + ``opencode run --attach`` to work around the
"Session not found" bug in ``opencode run`` standalone (v1.2.10–v1.2.16).

The provider auto-manages the serve process: it starts one lazily on the
first call and reuses it for all subsequent calls.  The server is cleaned
up on process exit.

To skip auto-management, set ``OPENCODE_SERVER`` (or ``opencode_server``
in HarnessConfig) to point to an externally managed server.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import signal
import socket
import subprocess
import time
from typing import ClassVar, Dict, Optional

from agentfield.harness._cli import run_cli
from agentfield.harness._result import Metrics, RawResult


def _find_free_port() -> int:
    """Find a free TCP port by briefly binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class OpenCodeProvider:
    """OpenCode CLI provider with auto-managed ``opencode serve``.

    On first invocation the provider starts ``opencode serve`` on a random
    port and connects all subsequent ``opencode run --attach`` calls to it.
    The serve process is terminated when the Python process exits.
    """

    # Class-level singleton: one serve process shared across all provider
    # instances (parallel harness calls all hit the same server).
    _serve_proc: ClassVar[Optional[subprocess.Popen[bytes]]] = None
    _serve_url: ClassVar[Optional[str]] = None
    _serve_lock: ClassVar[Optional[asyncio.Lock]] = None

    def __init__(
        self,
        bin_path: str = "opencode",
        server_url: Optional[str] = None,
    ):
        self._bin = bin_path
        # Explicit URL takes precedence over auto-managed server.
        self._explicit_server = server_url or os.environ.get("OPENCODE_SERVER")

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._serve_lock is None:
            cls._serve_lock = asyncio.Lock()
        return cls._serve_lock

    @classmethod
    def _cleanup_serve(cls) -> None:
        """Kill the managed serve process (called via atexit)."""
        proc = cls._serve_proc
        if proc is not None and proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            cls._serve_proc = None
            cls._serve_url = None

    async def _ensure_serve(self) -> str:
        """Return the URL of a running ``opencode serve`` instance."""
        if self._explicit_server:
            return self._explicit_server

        # Fast path — server already running.
        if self.__class__._serve_url and self.__class__._serve_proc:
            if self.__class__._serve_proc.poll() is None:
                return self.__class__._serve_url

        async with self._get_lock():
            # Double-check after acquiring lock.
            if self.__class__._serve_url and self.__class__._serve_proc:
                if self.__class__._serve_proc.poll() is None:
                    return self.__class__._serve_url

            port = _find_free_port()
            url = f"http://127.0.0.1:{port}"

            proc = subprocess.Popen(
                [self._bin, "serve", "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env={**os.environ},
            )

            # Wait for the server to be ready (up to 15s).
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"opencode serve exited immediately (code {proc.returncode}). "
                        "Ensure opencode is installed and OPENROUTER_API_KEY is set."
                    )
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                        break
                except OSError:
                    await asyncio.sleep(0.3)
            else:
                proc.kill()
                raise RuntimeError(
                    f"opencode serve did not start within 15s on port {port}."
                )

            self.__class__._serve_proc = proc
            self.__class__._serve_url = url
            atexit.register(self.__class__._cleanup_serve)
            return url

    async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
        server_url = await self._ensure_serve()

        cmd = [self._bin, "run", "--attach", server_url]

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

        start_api = time.monotonic()

        try:
            stdout, stderr, returncode = await run_cli(cmd, env=env, cwd=cwd)
        except FileNotFoundError:
            return RawResult(
                is_error=True,
                error_message=(
                    f"OpenCode binary not found at '{self._bin}'. "
                    "Install OpenCode: https://opencode.ai"
                ),
                metrics=Metrics(),
            )
        except TimeoutError as exc:
            return RawResult(
                is_error=True,
                error_message=str(exc),
                metrics=Metrics(),
            )

        api_ms = int((time.monotonic() - start_api) * 1000)
        result_text = stdout.strip() if stdout.strip() else None
        is_error = returncode != 0 and result_text is None

        return RawResult(
            result=result_text,
            messages=[],
            metrics=Metrics(
                duration_api_ms=api_ms,
                num_turns=1 if result_text else 0,
                session_id="",
            ),
            is_error=is_error,
            error_message=stderr.strip() if is_error else None,
        )
