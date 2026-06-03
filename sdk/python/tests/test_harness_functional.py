"""Live functional tests for harness providers.

These tests invoke REAL coding agents and make real API calls.
Excluded from default ``pytest`` runs via the ``harness_live`` marker.

Prerequisites
~~~~~~~~~~~~~
- Coding agent CLIs installed (claude, codex, opencode)
- Valid API keys / auth configured for each provider
- Internet access for API calls

Run all harness live tests::

    pytest tests/test_harness_functional.py -m harness_live -v --timeout=300

Run a single provider::

    pytest tests/test_harness_functional.py -m harness_live -v -k codex --timeout=300
    pytest tests/test_harness_functional.py -m harness_live -v -k claude --timeout=300
    pytest tests/test_harness_functional.py -m harness_live -v -k opencode --timeout=300
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from typing import List

import pytest
from pydantic import BaseModel

from agentfield.harness._runner import HarnessRunner
from agentfield.harness._result import HarnessResult  # noqa: F401
from agentfield.types import HarnessConfig

# ────────────────────────────────────────────────────────────────────────
# Schemas
# ────────────────────────────────────────────────────────────────────────


class SimpleResponse(BaseModel):
    """Minimal response schema — a greeting string and an integer."""

    greeting: str
    number: int


class CodeReviewResponse(BaseModel):
    """Slightly richer schema with a list field."""

    summary: str
    score: int
    suggestions: List[str]


# ────────────────────────────────────────────────────────────────────────
# Provider availability
# ────────────────────────────────────────────────────────────────────────

HAS_CODEX = shutil.which("codex") is not None
HAS_OPENCODE = shutil.which("opencode") is not None

try:
    import claude_agent_sdk  # noqa: F401

    HAS_CLAUDE_SDK = True
except ImportError:
    HAS_CLAUDE_SDK = False


# ────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def work_dir():
    """Create a temp directory with a git repo (required by Codex etc.)."""
    tmpdir = tempfile.mkdtemp(prefix="agentfield_harness_test_")
    subprocess.run(
        ["git", "init", tmpdir],
        capture_output=True,
        check=True,
    )
    # Create a dummy file + initial commit so git doesn't complain about empty repo
    dummy = os.path.join(tmpdir, ".gitkeep")
    with open(dummy, "w") as f:
        f.write("")
    subprocess.run(
        ["git", "-C", tmpdir, "add", "."],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "-C", tmpdir, "commit", "-m", "init", "--allow-empty"],
        capture_output=True,
        check=True,
    )
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


# Module‐level markers — every test in this file is harness_live + asyncio
pytestmark = [pytest.mark.harness_live, pytest.mark.asyncio]


# ════════════════════════════════════════════════════════════════════════
# CODEX
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_CODEX, reason="codex CLI not installed")
class TestCodexLive:
    """Live tests against the Codex CLI provider (``codex exec --json``)."""

    async def test_basic_prompt(self, work_dir: str) -> None:
        """Provider can execute a trivial prompt and return text."""
        from agentfield.harness.providers.codex import CodexProvider

        provider = CodexProvider()
        result = await provider.execute(
            "What is 2+2? Reply with ONLY the number, nothing else.",
            {"cwd": work_dir, "permission_mode": "auto"},
        )

        assert not result.is_error, f"Codex returned error: {result.error_message}"
        assert result.result is not None, "Codex returned empty result"
        assert "4" in result.result, f"Expected '4' in result, got: {result.result!r}"

    async def test_schema_pipeline(self, work_dir: str) -> None:
        """Full schema pipeline: prompt → Codex writes JSON file → parsed to Pydantic model."""
        runner = HarnessRunner()
        result = await runner.run(
            'Return exactly: greeting="Hello from Codex" and number=42. '
            "Follow the OUTPUT REQUIREMENTS below precisely.",
            provider="codex",
            schema=SimpleResponse,
            cwd=work_dir,
            permission_mode="auto",
            max_retries=1,
        )

        assert not result.is_error, (
            f"Schema pipeline failed: {result.error_message}\n"
            f"Raw result: {result.result!r}"
        )
        assert result.parsed is not None, (
            f"Parsed is None — agent may not have written the output file.\n"
            f"Raw result: {result.result!r}"
        )
        assert isinstance(result.parsed, SimpleResponse)
        assert (
            isinstance(result.parsed.greeting, str) and len(result.parsed.greeting) > 0
        )
        assert isinstance(result.parsed.number, int)

    async def test_cleanup_after_schema_run(self, work_dir: str) -> None:
        """Temp files are cleaned up after a schema run completes."""
        runner = HarnessRunner()
        await runner.run(
            'Return greeting="cleanup" and number=0. Follow the OUTPUT REQUIREMENTS below.',
            provider="codex",
            schema=SimpleResponse,
            cwd=work_dir,
            permission_mode="auto",
            max_retries=1,
        )

        assert not os.path.exists(os.path.join(work_dir, ".agentfield_output.json")), (
            "Output file was not cleaned up"
        )

    async def test_agent_class_e2e(self, work_dir: str) -> None:
        """Agent.harness() E2E integration through Codex provider."""
        from agentfield.agent import Agent

        config = HarnessConfig(provider="codex", permission_mode="auto")
        agent = Agent(
            node_id="functional-test-codex",
            harness_config=config,
            auto_register=False,
        )

        result = await agent.harness(
            'Return greeting="Hi from Agent" and number=7. '
            "Follow the OUTPUT REQUIREMENTS below.",
            schema=SimpleResponse,
            cwd=work_dir,
        )

        assert not result.is_error, f"Agent.harness() error: {result.error_message}"
        assert result.parsed is not None, "Agent.harness() returned None parsed"
        assert isinstance(result.parsed, SimpleResponse)


# ════════════════════════════════════════════════════════════════════════
# CLAUDE CODE
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_CLAUDE_SDK, reason="claude-agent-sdk not installed")
class TestClaudeCodeLive:
    """Live tests against Claude Code (``claude_agent_sdk``)."""

    async def test_basic_prompt(self, work_dir: str) -> None:
        """Provider can execute a trivial prompt without erroring."""
        from agentfield.harness.providers.claude import ClaudeCodeProvider

        provider = ClaudeCodeProvider()
        result = await provider.execute(
            "What is 2+2? Reply with ONLY the number, nothing else.",
            {"cwd": work_dir, "permission_mode": "auto"},
        )

        assert not result.is_error, f"Claude returned error: {result.error_message}"
        assert len(result.messages) > 0, "Claude returned no messages"

    async def test_schema_pipeline(self, work_dir: str) -> None:
        """Full schema pipeline: prompt → Claude Code writes JSON file → parsed to Pydantic model."""
        runner = HarnessRunner()
        result = await runner.run(
            'Return exactly: greeting="Hello from Claude" and number=42. '
            "Follow the OUTPUT REQUIREMENTS below precisely.",
            provider="claude-code",
            schema=SimpleResponse,
            cwd=work_dir,
            permission_mode="auto",
            max_retries=1,
        )

        assert not result.is_error, (
            f"Schema pipeline failed: {result.error_message}\n"
            f"Raw result: {result.result!r}"
        )
        assert result.parsed is not None, (
            f"Parsed is None — agent may not have written the output file.\n"
            f"Raw result: {result.result!r}"
        )
        assert isinstance(result.parsed, SimpleResponse)
        assert (
            isinstance(result.parsed.greeting, str) and len(result.parsed.greeting) > 0
        )
        assert isinstance(result.parsed.number, int)

    async def test_complex_schema(self, work_dir: str) -> None:
        """Validates a richer schema with a list[str] field."""
        runner = HarnessRunner()
        result = await runner.run(
            "Review this code: `x = 1 + 1`. "
            "Provide summary, score (1-10), and a list of improvement suggestions. "
            "Follow the OUTPUT REQUIREMENTS below precisely.",
            provider="claude-code",
            schema=CodeReviewResponse,
            cwd=work_dir,
            permission_mode="auto",
            max_retries=1,
        )

        assert not result.is_error, f"Complex schema failed: {result.error_message}"
        assert result.parsed is not None
        assert isinstance(result.parsed, CodeReviewResponse)
        assert isinstance(result.parsed.suggestions, list)
        assert 1 <= result.parsed.score <= 10

    async def test_agent_class_e2e(self, work_dir: str) -> None:
        """Agent.harness() E2E integration through Claude Code provider."""
        from agentfield.agent import Agent

        config = HarnessConfig(provider="claude-code", permission_mode="auto")
        agent = Agent(
            node_id="functional-test-claude",
            harness_config=config,
            auto_register=False,
        )

        result = await agent.harness(
            'Return greeting="Hi from Agent" and number=7. '
            "Follow the OUTPUT REQUIREMENTS below.",
            schema=SimpleResponse,
            cwd=work_dir,
        )

        assert not result.is_error, f"Agent.harness() error: {result.error_message}"
        assert result.parsed is not None, "Agent.harness() returned None parsed"
        assert isinstance(result.parsed, SimpleResponse)


# ════════════════════════════════════════════════════════════════════════
# OPENCODE
# ════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_OPENCODE, reason="opencode CLI not installed")
class TestOpenCodeLive:
    """Live tests against OpenCode CLI (``opencode run``).

    .. note::
        All tests are marked ``xfail`` because OpenCode v1.2.10 has a known
        upstream bug in headless mode — ``opencode run`` returns
        "Session not found" regardless of context.
        See: https://github.com/anomalyco/opencode/issues/13851
    """

    @pytest.mark.xfail(
        reason="OpenCode v1.2.10 headless 'Session not found' bug (upstream)",
        strict=False,
    )
    async def test_basic_prompt(self, work_dir: str) -> None:
        """Provider can execute a trivial prompt and return text."""
        from agentfield.harness.providers.opencode import OpenCodeProvider

        provider = OpenCodeProvider()
        result = await provider.execute(
            "What is 2+2? Reply with ONLY the number, nothing else.",
            {"cwd": work_dir},
        )

        assert not result.is_error, f"OpenCode returned error: {result.error_message}"
        assert result.result is not None, "OpenCode returned empty result"

    @pytest.mark.xfail(
        reason="OpenCode v1.2.10 headless 'Session not found' bug (upstream)",
        strict=False,
    )
    async def test_schema_pipeline(self, work_dir: str) -> None:
        """Full schema pipeline: prompt → OpenCode writes JSON file → parsed to Pydantic model."""
        runner = HarnessRunner()
        result = await runner.run(
            'Return exactly: greeting="Hello from OpenCode" and number=42. '
            "Follow the OUTPUT REQUIREMENTS below precisely.",
            provider="opencode",
            schema=SimpleResponse,
            cwd=work_dir,
            max_retries=1,
        )

        assert not result.is_error, (
            f"Schema pipeline failed: {result.error_message}\n"
            f"Raw result: {result.result!r}"
        )
        assert result.parsed is not None, (
            f"Parsed is None — agent may not have written the output file.\n"
            f"Raw result: {result.result!r}"
        )
        assert isinstance(result.parsed, SimpleResponse)
        assert (
            isinstance(result.parsed.greeting, str) and len(result.parsed.greeting) > 0
        )
        assert isinstance(result.parsed.number, int)

    @pytest.mark.xfail(
        reason="OpenCode v1.2.10 headless 'Session not found' bug (upstream)",
        strict=False,
    )
    async def test_agent_class_e2e(self, work_dir: str) -> None:
        """Agent.harness() E2E integration through OpenCode provider."""
        from agentfield.agent import Agent

        config = HarnessConfig(provider="opencode")
        agent = Agent(
            node_id="functional-test-opencode",
            harness_config=config,
            auto_register=False,
        )

        result = await agent.harness(
            'Return greeting="Hi from Agent" and number=7. '
            "Follow the OUTPUT REQUIREMENTS below.",
            schema=SimpleResponse,
            cwd=work_dir,
        )

        assert not result.is_error, f"Agent.harness() error: {result.error_message}"
        assert result.parsed is not None, "Agent.harness() returned None parsed"
        assert isinstance(result.parsed, SimpleResponse)


# ════════════════════════════════════════════════════════════════════════
# CROSS-PROVIDER CONSISTENCY
# ════════════════════════════════════════════════════════════════════════


class TestCrossProvider:
    """Verify consistent behaviour across multiple providers."""

    @pytest.mark.skipif(
        sum([HAS_CODEX, HAS_CLAUDE_SDK, HAS_OPENCODE]) < 2,
        reason="Need at least 2 providers to run cross-provider test",
    )
    async def test_same_schema_multiple_providers(self, work_dir: str) -> None:
        """Multiple providers return data conforming to the same Pydantic schema."""
        providers: list[tuple[str, dict]] = []
        if HAS_CODEX:
            providers.append(("codex", {"permission_mode": "auto"}))
        if HAS_CLAUDE_SDK:
            providers.append(("claude-code", {"permission_mode": "auto"}))
        if HAS_OPENCODE:
            providers.append(("opencode", {}))

        runner = HarnessRunner()
        results: dict[str, object] = {}

        for name, extra_opts in providers[:2]:  # test first 2 available
            subdir = os.path.join(work_dir, name.replace("-", "_"))
            os.makedirs(subdir, exist_ok=True)
            result = await runner.run(
                'Return greeting="hello" and number=42. '
                "Follow the OUTPUT REQUIREMENTS below.",
                provider=name,
                schema=SimpleResponse,
                cwd=subdir,
                max_retries=1,
                **extra_opts,
            )
            results[name] = result

        for name, result in results.items():
            assert not result.is_error, f"{name} failed: {result.error_message}"
            assert result.parsed is not None, f"{name} returned None parsed"
            assert isinstance(result.parsed, SimpleResponse), (
                f"{name} parsed is {type(result.parsed)}, expected SimpleResponse"
            )
