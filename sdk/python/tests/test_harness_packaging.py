from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


def test_harness_optional_dependencies_match_provider_wrappers():
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )
    extras = pyproject["project"]["optional-dependencies"]

    assert extras["harness-claude"] == ["claude-agent-sdk>=0.1"]
    assert extras["harness-codex"] == [
        "openai-codex>=0.1.0b3",
        "openai-codex-cli-bin==0.137.0a4",
    ]
    assert extras["harness-opencode"] == ["opencode-ai>=0.1.0a36"]
    assert extras["harness-all"] == [
        "claude-agent-sdk>=0.1",
        "openai-codex>=0.1.0b3",
        "openai-codex-cli-bin==0.137.0a4",
        "opencode-ai>=0.1.0a36",
    ]


def test_legacy_harness_extra_remains_backward_compatible():
    pyproject = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text()
    )
    extras = pyproject["project"]["optional-dependencies"]

    assert extras["harness"] == extras["harness-claude"]
