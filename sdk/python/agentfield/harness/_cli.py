"""Shared async subprocess utilities for CLI-based harness providers."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from agentfield.openrouter_attribution import apply_subprocess_env

_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# 300 rather than 120: provider CLIs in JSON mode (e.g. `opencode run
# --format json`) emit events only at completion boundaries - never
# token-by-token - so one long reasoning completion over a large context is
# minutes of legitimate stdout silence. At 120s the watchdog routinely
# killed healthy runs on slower models; 300s tolerates a long single
# completion while still catching genuine hangs.
_DEFAULT_IDLE_SECONDS = 300.0


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _resolve_idle_seconds(idle_seconds: Optional[float]) -> Optional[float]:
    """Resolve the no-progress watchdog window.

    Precedence: explicit ``idle_seconds`` arg, then env
    ``AGENTFIELD_HARNESS_IDLE_SECONDS``, then ``_DEFAULT_IDLE_SECONDS`` (300s).
    A value <= 0 disables the watchdog.
    """
    if idle_seconds is None:
        raw = os.environ.get("AGENTFIELD_HARNESS_IDLE_SECONDS")
        if raw is not None:
            try:
                idle_seconds = float(raw)
            except ValueError:
                idle_seconds = _DEFAULT_IDLE_SECONDS
        else:
            idle_seconds = _DEFAULT_IDLE_SECONDS
    return idle_seconds if idle_seconds and idle_seconds > 0 else None


async def _drain(
    stream: Optional[asyncio.StreamReader],
    chunks: List[bytes],
    last_activity: List[float],
) -> None:
    """Read a stream incrementally, recording each chunk and its arrival time."""
    if stream is None:
        return
    while True:
        chunk = await stream.read(65536)
        if not chunk:
            break
        chunks.append(chunk)
        last_activity[0] = asyncio.get_event_loop().time()


# Grace period for the child's exit notification after its pipes hit EOF (or
# after it was killed). Post-EOF a child is dead or exiting, so this only
# delays the pathological cases handled in _wait_process_exit.
_PROC_EXIT_GRACE_SECONDS = 10.0


async def _wait_process_exit(
    proc: asyncio.subprocess.Process, kill_group
) -> Optional[int]:
    """Await the child's exit without trusting the exit notification.

    A bare ``await proc.wait()`` can park forever on Windows' proactor loop:
    the RegisterWaitWithQueue completion that resolves it is occasionally lost
    even though the child already exited (CPython gh-81562 / gh-111604 —
    observed in production as a swe-planner reasoner silently wedged for 26
    minutes with no child process and a perfectly healthy event loop; the
    coroutine sat here, after both pipes had hit EOF, waiting on an exit
    notification that never came). It also parks when an orphaned grandchild
    outlives a killed parent while holding the output pipes, since asyncio
    resolves ``wait()`` only once every pipe disconnects. Bound the wait; on
    timeout, kill whatever may remain and recover the real exit status by
    polling the underlying Popen object, which needs no event delivery.
    Returns the exit code, or None if it cannot be determined.
    """
    try:
        return await asyncio.wait_for(proc.wait(), timeout=_PROC_EXIT_GRACE_SECONDS)
    except (asyncio.TimeoutError, TimeoutError):
        pass
    # The notification may have landed while we grace-waited (in which case
    # the transport finalizer may already have dropped its Popen reference —
    # but ``proc.returncode`` stays available), so check it at every step.
    if proc.returncode is not None:
        return proc.returncode
    kill_group()
    popen = getattr(getattr(proc, "_transport", None), "_proc", None)
    for _ in range(100):  # <= 10s; TerminateProcess/SIGKILL act promptly
        if proc.returncode is not None:
            return proc.returncode
        if popen is None:
            break
        returncode = popen.poll()
        if returncode is not None:
            return returncode
        await asyncio.sleep(0.1)
    return proc.returncode


async def run_cli(
    cmd: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
    idle_seconds: Optional[float] = None,
) -> Tuple[str, str, int]:
    """Run a CLI command async. Returns (stdout, stderr, returncode).

    Streams stdout and stderr concurrently so a no-progress (idle) watchdog can
    abort a stalled child early. If no output arrives for ``idle_seconds`` (env
    ``AGENTFIELD_HARNESS_IDLE_SECONDS``, default 120s; <= 0 disables), the process
    group is killed and ``TimeoutError`` is raised. ``timeout`` remains the outer
    wall-clock bound.
    """
    merged_env = {**os.environ}
    if env:
        merged_env.update(env)
    apply_subprocess_env(merged_env)

    idle = _resolve_idle_seconds(idle_seconds)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=merged_env,
        cwd=cwd,
        start_new_session=True,
    )

    stdout_chunks: List[bytes] = []
    stderr_chunks: List[bytes] = []
    last_activity = [asyncio.get_event_loop().time()]

    # Drain both pipes concurrently to avoid a pipe-buffer deadlock.
    drain = asyncio.gather(
        _drain(proc.stdout, stdout_chunks, last_activity),
        _drain(proc.stderr, stderr_chunks, last_activity),
    )

    def _kill_group() -> None:
        pid = proc.pid
        # killpg only exists on POSIX; on Windows use taskkill /T instead.
        if hasattr(os, "killpg") and isinstance(pid, int) and pid > 0:
            try:
                os.killpg(pid, signal.SIGKILL)
                return
            except (ProcessLookupError, PermissionError, OSError):
                pass
        if os.name == "nt" and isinstance(pid, int) and pid > 0:
            # proc.kill() terminates only the direct child. CLI shims
            # (.cmd -> cmd.exe -> node/bun) do their real work in
            # grandchildren that inherit the output pipes; if they outlive
            # the parent they hold the pipes open and asyncio's proc.wait()
            # — which resolves only after every pipe disconnects — blocks
            # indefinitely. taskkill /T is the closest analog to killpg.
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            proc.kill()
        except ProcessLookupError:
            pass

    timed_out = False
    idle_timed_out = False
    completed = False
    deadline = asyncio.get_event_loop().time() + timeout if timeout else None

    try:
        while True:
            now = asyncio.get_event_loop().time()
            waits: List[float] = []
            if idle is not None:
                waits.append(idle - (now - last_activity[0]))
            if deadline is not None:
                waits.append(deadline - now)
            wait_for = min(waits) if waits else None
            if wait_for is not None and wait_for <= 0:
                wait_for = 0.0

            try:
                await asyncio.wait_for(asyncio.shield(drain), timeout=wait_for)
                completed = True
                break  # both pipes hit EOF: child is done
            except asyncio.TimeoutError:
                now = asyncio.get_event_loop().time()
                if deadline is not None and now >= deadline:
                    timed_out = True
                    break
                if idle is not None and (now - last_activity[0]) >= idle:
                    idle_timed_out = True
                    break
                # Spurious wakeup (progress reset the idle window): loop again.
    finally:
        if not completed:
            # Timeout, cancellation, or an internal error: stop the child so
            # nothing keeps running and the exit wait below can conclude.
            # (Previously only the timeout paths killed; a cancelled run_cli
            # left the child running and awaited its natural exit.)
            _kill_group()
        drain.cancel()
        try:
            await drain
        except BaseException:
            pass
        fallback_returncode = await _wait_process_exit(proc, _kill_group)

    if idle_timed_out:
        raise TimeoutError(
            f"CLI command made no progress for {idle}s: {' '.join(cmd)}"
        )
    if timed_out:
        raise TimeoutError(f"CLI command timed out after {timeout}s: {' '.join(cmd)}")

    returncode = proc.returncode
    if returncode is None:
        returncode = fallback_returncode
    return (
        b"".join(stdout_chunks).decode("utf-8", errors="replace"),
        b"".join(stderr_chunks).decode("utf-8", errors="replace"),
        returncode if returncode is not None else -1,
    )


def parse_jsonl(text: str) -> List[Dict[str, Any]]:
    """Parse JSONL (newline-delimited JSON) output. Skips invalid lines."""
    events = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def extract_final_text(events: List[Dict[str, Any]]) -> Optional[str]:
    """Extract the final result text from a list of JSONL events.

    Looks for common patterns across different CLI tools:
    - type: "result" with text/result field
    - type: "item.completed" with item.text field (Codex)
    - type: "text" with part.text field (OpenCode JSON stream)
    - Last assistant message text
    """
    result_text = None
    current_text_parts: List[str] = []

    for event in events:
        event_type = event.get("type", "")

        if event_type == "step_start":
            current_text_parts = []
        elif event_type == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                text = item.get("text", "")
                if text:
                    result_text = text
        elif event_type == "result":
            result_text = event.get("result", event.get("text", result_text))
        elif event_type == "turn.completed":
            text = event.get("text", "")
            if text:
                result_text = text
        elif event_type in ("message", "assistant"):
            content = event.get("content", event.get("text", ""))
            if isinstance(content, str) and content:
                result_text = content
        elif event_type == "text":
            content = event.get("text", event.get("content", ""))
            part = event.get("part")
            if not content and isinstance(part, dict):
                content = part.get("text", "")
            if isinstance(content, str) and content:
                current_text_parts.append(content)
                result_text = "".join(current_text_parts)

    return result_text


def estimate_cli_cost(
    model: str,
    prompt: str,
    result_text: str | None,
) -> float | None:
    """Estimate LLM cost from prompt/completion text using litellm.

    Returns None if the model isn't in litellm's pricing DB or litellm
    is not available — callers should treat None as "unknown", not "free".
    """
    if not model:
        return None
    try:
        import litellm

        cost = litellm.completion_cost(
            model=model,
            prompt=prompt,
            completion=result_text or "",
        )
        return cost if cost and cost > 0 else None
    except Exception:
        return None
