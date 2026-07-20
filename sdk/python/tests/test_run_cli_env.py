"""Regression test: ``run_cli`` must propagate parent env to subprocesses.

This is the contract that lets reasoners running through CLI-based providers
(opencode, codex, gemini) pick up deployment-level env vars — most importantly
provider tool gates like ``OPENCODE_ENABLE_EXA`` / ``EXA_API_KEY`` that
opencode reads to enable its built-in ``websearch`` tool.

If a refactor of ``run_cli`` ever drops the ``merged_env = {**os.environ}``
line, every CLI provider stops seeing parent env vars and tools that depend
on them silently break across every consumer (SWE-AF, PR-AF, ...). This
test catches that regression early.
"""

from __future__ import annotations

import asyncio
import sys
import time

import pytest

from agentfield.harness._cli import run_cli

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_run_cli_inherits_parent_env(monkeypatch):
    """Vars set in the parent process must reach the subprocess."""
    monkeypatch.setenv("AGENTFIELD_TEST_PARENT_ONLY", "from_parent_value_xyz")

    stdout, _stderr, returncode = await run_cli(
        ["bash", "-c", "echo $AGENTFIELD_TEST_PARENT_ONLY"],
        timeout=10.0,
    )

    assert returncode == 0
    assert stdout.strip() == "from_parent_value_xyz"


@pytest.mark.asyncio
async def test_run_cli_caller_env_overrides_parent(monkeypatch):
    """Explicit env passed to run_cli must win over parent env on conflict."""
    monkeypatch.setenv("AGENTFIELD_TEST_CONFLICT", "parent_loses")

    stdout, _stderr, returncode = await run_cli(
        ["bash", "-c", "echo $AGENTFIELD_TEST_CONFLICT"],
        env={"AGENTFIELD_TEST_CONFLICT": "explicit_wins"},
        timeout=10.0,
    )

    assert returncode == 0
    assert stdout.strip() == "explicit_wins"


@pytest.mark.asyncio
async def test_run_cli_layers_explicit_on_top_of_parent(monkeypatch):
    """Vars only in parent and vars only in explicit env should both reach
    the subprocess — the merge is additive, not replacement."""
    monkeypatch.setenv("AGENTFIELD_TEST_PARENT_KEY", "p_value")

    stdout, _stderr, returncode = await run_cli(
        [
            "bash",
            "-c",
            "echo $AGENTFIELD_TEST_PARENT_KEY:$AGENTFIELD_TEST_EXPLICIT_KEY",
        ],
        env={"AGENTFIELD_TEST_EXPLICIT_KEY": "e_value"},
        timeout=10.0,
    )

    assert returncode == 0
    assert stdout.strip() == "p_value:e_value"


@pytest.mark.asyncio
async def test_run_cli_works_without_explicit_env(monkeypatch):
    """env=None (the default) must still propagate parent env."""
    monkeypatch.setenv("AGENTFIELD_TEST_NO_ENV_ARG", "still_there")

    stdout, _stderr, returncode = await run_cli(
        ["bash", "-c", "echo $AGENTFIELD_TEST_NO_ENV_ARG"],
        timeout=10.0,
    )

    assert returncode == 0
    assert stdout.strip() == "still_there"


def test_run_cli_merges_openrouter_attribution_defaults(monkeypatch):
    monkeypatch.delenv("AGENTFIELD_OPENROUTER_SITE_URL", raising=False)
    monkeypatch.delenv("AGENTFIELD_OPENROUTER_APP_NAME", raising=False)
    monkeypatch.delenv("OR_SITE_URL", raising=False)
    monkeypatch.delenv("OR_APP_NAME", raising=False)

    stdout, _stderr, returncode = asyncio.run(
        run_cli(
            [
                "bash",
                "-c",
                "echo $AGENTFIELD_OPENROUTER_SITE_URL:$AGENTFIELD_OPENROUTER_APP_NAME:$OR_SITE_URL:$OR_APP_NAME",
            ],
            timeout=10.0,
        )
    )

    assert returncode == 0
    assert stdout.strip() == (
        "https://agentfield.ai:AgentField AI:https://agentfield.ai:AgentField AI"
    )


def test_run_cli_openrouter_attribution_caller_env_wins(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_OPENROUTER_SITE_URL", "https://parent.example")

    stdout, _stderr, returncode = asyncio.run(
        run_cli(
            [
                "bash",
                "-c",
                "echo $AGENTFIELD_OPENROUTER_SITE_URL:$OR_SITE_URL",
            ],
            env={
                "AGENTFIELD_OPENROUTER_SITE_URL": "https://caller.example",
                "OR_SITE_URL": "https://or-caller.example",
            },
            timeout=10.0,
        )
    )

    assert returncode == 0
    assert stdout.strip() == "https://caller.example:https://or-caller.example"


@pytest.mark.asyncio
async def test_run_cli_idle_watchdog_aborts_stalled_child():
    """A child that emits one line then sleeps must be killed by the idle
    watchdog well before the wall-clock timeout, not after."""
    start = time.monotonic()
    with pytest.raises(TimeoutError) as exc:
        await run_cli(
            ["bash", "-c", "echo started; sleep 600"],
            # Generous wall-clock bound; the idle watchdog should fire first.
            timeout=60.0,
            idle_seconds=1.0,
        )
    elapsed = time.monotonic() - start
    # Killed by the idle watchdog (~1s), not the 60s wall-clock bound.
    assert elapsed < 10.0, f"idle watchdog did not fire promptly ({elapsed:.1f}s)"
    assert "no progress" in str(exc.value).lower()


@pytest.mark.asyncio
async def test_run_cli_fast_command_returns_full_output():
    """A normal fast command still returns its complete stdout and exit code
    even with a short idle window configured."""
    stdout, _stderr, returncode = await run_cli(
        ["bash", "-c", "printf 'line1\\nline2\\nline3\\n'"],
        timeout=10.0,
        idle_seconds=1.0,
    )
    assert returncode == 0
    assert stdout == "line1\nline2\nline3\n"


@pytest.mark.asyncio
async def test_run_cli_idle_seconds_from_env(monkeypatch):
    """The idle window is read from AGENTFIELD_HARNESS_IDLE_SECONDS when the
    explicit arg is not passed."""
    monkeypatch.setenv("AGENTFIELD_HARNESS_IDLE_SECONDS", "1")
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await run_cli(
            ["bash", "-c", "echo started; sleep 600"],
            timeout=60.0,
        )
    elapsed = time.monotonic() - start
    assert elapsed < 10.0, f"env idle watchdog did not fire promptly ({elapsed:.1f}s)"


@pytest.mark.asyncio
async def test_run_cli_bounded_when_grandchild_holds_pipes():
    """Functional regression for the silent-wedge class (real subprocesses).

    A grandchild that inherits the output pipes and outlives its parent used
    to park run_cli forever at the exit wait: asyncio resolves ``proc.wait()``
    only once every pipe disconnects, and on Windows the group kill cannot
    reach an already-orphaned grandchild. Observed in production as a
    swe-planner reasoner stuck ``running`` for 26 minutes with no harness
    process and a healthy event loop. run_cli must now conclude in bounded
    time on every platform.
    """
    parent = (
        "import subprocess, sys; "
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        "print('parent-exiting', flush=True)"
    )
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        await run_cli(
            [sys.executable, "-c", parent],
            timeout=60.0,
            idle_seconds=2.0,
        )
    elapsed = time.monotonic() - start
    # POSIX: killpg reaps the whole group at the idle deadline (~2s).
    # Windows: worst case idle (2s) + exit-notification grace (10s) + poll.
    assert elapsed < 45.0, f"exit wait not bounded ({elapsed:.1f}s)"
