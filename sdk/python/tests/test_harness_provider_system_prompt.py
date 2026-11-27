from __future__ import annotations

from typing import Any

import pytest

from agentfield.harness.providers.codex import CodexProvider
from agentfield.harness.providers.gemini import GeminiProvider


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("module", "provider_cls"),
    [("codex", CodexProvider), ("gemini", GeminiProvider)],
)
async def test_system_prompt_is_prepended_to_prompt(
    monkeypatch: pytest.MonkeyPatch, module: str, provider_cls: Any
):
    captured: list[str] = []

    async def fake_run_cli(cmd, *, env=None, cwd=None, timeout=None):
        captured[:] = cmd
        return "", "", 0

    monkeypatch.setattr(
        "agentfield.harness._availability.shutil.which", lambda path: path
    )
    monkeypatch.setattr(f"agentfield.harness.providers.{module}.run_cli", fake_run_cli)

    await provider_cls().execute("hello", {"system_prompt": "Be terse."})
    assert captured[-1] == "Be terse.\n\nTask:\nhello"

    await provider_cls().execute("hello", {"system_prompt": None})
    assert captured[-1] == "hello"
