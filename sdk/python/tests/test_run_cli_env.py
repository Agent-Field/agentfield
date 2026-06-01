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
