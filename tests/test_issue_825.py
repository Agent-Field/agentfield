"""Regression coverage for issue #825: ``Agent.run`` server arguments."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk" / "python"))

from agentfield import Agent


def test_issue_825(monkeypatch: pytest.MonkeyPatch):
    """``app.run()`` must honor documented server-mode command-line flags."""
    served_with = {}
    app = SimpleNamespace(
        cli_handler=SimpleNamespace(run_cli=lambda: pytest.fail("CLI mode ran")),
        serve=lambda **kwargs: served_with.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["agent.py", "--port", "8765", "--host", "127.0.0.1", "--dev"],
    )

    Agent.run(app)

    assert served_with == {"port": 8765, "host": "127.0.0.1", "dev": True}
