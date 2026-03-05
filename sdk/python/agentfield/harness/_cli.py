"""Shared async subprocess utilities for CLI-based harness providers."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

_ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


async def run_cli(
    cmd: List[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
    timeout: Optional[float] = None,
) -> Tuple[str, str, int]:
    """Run a CLI command async. Returns (stdout, stderr, returncode)."""
    merged_env = {**os.environ}
    if env:
        merged_env.update(env)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=merged_env,
        cwd=cwd,
    )

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"CLI command timed out after {timeout}s: {' '.join(cmd)}")

    return (
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
        proc.returncode if proc.returncode is not None else -1,
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
    - Last assistant message text
    """
    result_text = None

    for event in events:
        event_type = event.get("type", "")

        if event_type == "item.completed":
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

    return result_text
