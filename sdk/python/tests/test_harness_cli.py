from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agentfield.harness._cli import (
    extract_final_text,
    parse_jsonl,
    run_cli,
    strip_ansi,
)


def test_strip_ansi_removes_color_sequences() -> None:
    assert strip_ansi("\x1b[31mError\x1b[0m") == "Error"


@pytest.mark.asyncio
async def test_run_cli_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"OK", b""

    captured: dict[str, Any] = {}

    async def fake_create_subprocess_exec(*args: str, **kwargs: Any) -> FakeProc:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    stdout, stderr, code = await run_cli(
        ["agentfield", "--version"],
        env={"AGENTFIELD_TEST": "1"},
        cwd="/tmp/work",
        timeout=1,
    )

    assert stdout == "OK"
    assert stderr == ""
    assert code == 0
    assert captured["args"] == ("agentfield", "--version")
    assert captured["kwargs"]["cwd"] == "/tmp/work"
    assert captured["kwargs"]["env"]["AGENTFIELD_TEST"] == "1"


@pytest.mark.asyncio
async def test_run_cli_timeout_kills_process(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeProc:
        returncode = None

        def __init__(self) -> None:
            self.killed = False
            self.waited = False

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(0.05)
            return b"", b""

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> None:
            self.waited = True

    proc = FakeProc()

    async def fake_create_subprocess_exec(*_args: str, **_kwargs: Any) -> FakeProc:
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    with pytest.raises(TimeoutError, match="timed out after 0.001s"):
        await run_cli(["agentfield"], timeout=0.001)

    assert proc.killed is True
    assert proc.waited is True


def test_parse_jsonl_skips_invalid_lines() -> None:
    text = '{"type":"a"}\nnot-json\n\n{"type":"b"}'

    assert parse_jsonl(text) == [{"type": "a"}, {"type": "b"}]


def test_extract_final_text_codex_style_event() -> None:
    events = [
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "final answer"},
        }
    ]

    assert extract_final_text(events) == "final answer"


def test_extract_final_text_accumulates_text_parts() -> None:
    events = [
        {"type": "step_start"},
        {"type": "text", "part": {"text": "hello "}},
        {"type": "text", "part": {"text": "world"}},
    ]

    assert extract_final_text(events) == "hello world"
