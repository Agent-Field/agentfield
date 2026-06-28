"""Tests for shared subprocess helpers used by CLI harness providers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentfield.harness._cli import (
    estimate_cli_cost,
    extract_final_text,
    parse_jsonl,
    run_cli,
    strip_ansi,
)


def test_strip_ansi_removes_colors():
    assert strip_ansi("\x1b[31mError\x1b[0m") == "Error"


def _stream_reader(chunks: list[bytes]) -> MagicMock:
    """Build a fake asyncio StreamReader yielding ``chunks`` then EOF (b"")."""
    queued = list(chunks) + [b""]
    reader = MagicMock()
    reader.read = AsyncMock(side_effect=queued)
    return reader


@pytest.mark.asyncio
async def test_run_cli_success():
    process = MagicMock()
    process.stdout = _stream_reader([b"OK"])
    process.stderr = _stream_reader([])
    process.returncode = 0
    process.wait = AsyncMock(return_value=0)

    create_process = AsyncMock(return_value=process)

    with patch("asyncio.create_subprocess_exec", create_process):
        stdout, stderr, returncode = await run_cli(
            ["agentfield", "status"],
            env={"AGENTFIELD_TEST": "1"},
            cwd=".",
            timeout=1,
        )

    assert stdout == "OK"
    assert stderr == ""
    assert returncode == 0
    create_process.assert_awaited_once()
    _, kwargs = create_process.call_args
    assert kwargs["env"]["AGENTFIELD_TEST"] == "1"
    assert kwargs["cwd"] == "."
    assert kwargs["stdin"] is asyncio.subprocess.DEVNULL
    assert kwargs["stdout"] is asyncio.subprocess.PIPE
    assert kwargs["stderr"] is asyncio.subprocess.PIPE


@pytest.mark.asyncio
async def test_run_cli_timeout():
    async def never_ready(_n):
        # Streams that never reach EOF: the watchdog must abort the run.
        await asyncio.sleep(10)
        return b""

    process = MagicMock()
    process.pid = 2147483647  # nonexistent pid: killpg falls back to kill()
    process.returncode = None
    process.stdout = MagicMock(read=AsyncMock(side_effect=never_ready))
    process.stderr = MagicMock(read=AsyncMock(side_effect=never_ready))
    process.kill = MagicMock()
    process.wait = AsyncMock(return_value=None)

    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
        with pytest.raises(TimeoutError, match="CLI command timed out"):
            await run_cli(["agentfield", "hang"], timeout=0.01, idle_seconds=0)

    process.wait.assert_awaited()


def test_parse_jsonl_skips_invalid():
    events = parse_jsonl('{"type":"a"}\nnot-json\n{"type":"b"}')

    assert events == [{"type": "a"}, {"type": "b"}]


def test_extract_final_text_codex_style():
    events = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "first"}},
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "final answer"},
        },
    ]

    assert extract_final_text(events) == "final answer"


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        ([{"type": "result", "result": "result answer"}], "result answer"),
        ([{"type": "result", "text": "text answer"}], "text answer"),
        ([{"type": "turn.completed", "text": "turn answer"}], "turn answer"),
        ([{"type": "message", "content": "message answer"}], "message answer"),
        ([{"type": "assistant", "text": "assistant answer"}], "assistant answer"),
    ],
)
def test_extract_final_text_event_variants(events, expected):
    assert extract_final_text(events) == expected


def test_extract_final_text_empty_events():
    assert extract_final_text([]) is None


def test_estimate_cli_cost_calls_litellm():
    mock_litellm = MagicMock()
    mock_litellm.completion_cost.return_value = 0.05

    with patch.dict("sys.modules", {"litellm": mock_litellm}):
        cost = estimate_cli_cost(
            model="openai/gpt-4o",
            prompt="Summarize this run",
            result_text="Done",
        )

    assert cost == 0.05
    mock_litellm.completion_cost.assert_called_once_with(
        model="openai/gpt-4o",
        prompt="Summarize this run",
        completion="Done",
    )


def test_estimate_cli_cost_returns_none_without_model():
    assert estimate_cli_cost(model="", prompt="prompt", result_text="Done") is None


def test_estimate_cli_cost_returns_none_when_litellm_missing():
    with patch.dict("sys.modules", {"litellm": None}):
        cost = estimate_cli_cost(
            model="openai/gpt-4o",
            prompt="Summarize this run",
            result_text="Done",
        )

    assert cost is None


@pytest.mark.parametrize("raw_cost", [0, None])
def test_estimate_cli_cost_returns_none_for_non_positive_cost(raw_cost):
    mock_litellm = MagicMock()
    mock_litellm.completion_cost.return_value = raw_cost

    with patch.dict("sys.modules", {"litellm": mock_litellm}):
        cost = estimate_cli_cost(
            model="openai/gpt-4o",
            prompt="Summarize this run",
            result_text="Done",
        )

    assert cost is None


def test_estimate_cli_cost_returns_none_when_litellm_raises():
    mock_litellm = MagicMock()
    mock_litellm.completion_cost.side_effect = RuntimeError("pricing unavailable")

    with patch.dict("sys.modules", {"litellm": mock_litellm}):
        cost = estimate_cli_cost(
            model="openai/gpt-4o",
            prompt="Summarize this run",
            result_text="Done",
        )

    assert cost is None
