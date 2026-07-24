"""Per-reasoner/skill descriptions in registration metadata.

Validation contract:
  - C1: @app.reasoner(description=...) puts that text in the reasoner's
    registration metadata dict (the payload the control plane stores and
    discovery/`af ls` surface).
  - C2: without an explicit description, the first paragraph of the function
    docstring (whitespace-collapsed) is used; no docstring -> no description
    key at all (older control planes see an unchanged payload).
  - C3: @router.reasoner(description=...) reaches the agent metadata through
    include_router unchanged.
  - C4: skills follow the same contract as reasoners.
"""

import pytest
from agentfield import Agent
from agentfield.router import AgentRouter


def _metadata_by_id(entries):
    return {entry["id"]: entry for entry in entries}


@pytest.mark.unit
def test_reasoner_explicit_description_wins_over_docstring():
    app = Agent(node_id="test_agent", auto_register=False)

    @app.reasoner(description="Implement one scoped issue on a branch")
    async def implement_issue(issue: dict) -> dict:
        """Docstring that should NOT be used."""
        return {}

    meta = _metadata_by_id(app.reasoners)["implement_issue"]
    assert meta["description"] == "Implement one scoped issue on a branch"


@pytest.mark.unit
def test_reasoner_description_defaults_to_docstring_first_paragraph():
    app = Agent(node_id="test_agent", auto_register=False)

    @app.reasoner()
    async def build(goal: str) -> dict:
        """End-to-end: plan, execute, verify
        across multiple lines.

        Args:
            goal: everything after the blank line stays local.
        """
        return {}

    meta = _metadata_by_id(app.reasoners)["build"]
    assert meta["description"] == "End-to-end: plan, execute, verify across multiple lines."


@pytest.mark.unit
def test_reasoner_without_docstring_omits_description_key():
    app = Agent(node_id="test_agent", auto_register=False)

    @app.reasoner()
    async def bare(x: str) -> dict:
        return {"x": x}

    meta = _metadata_by_id(app.reasoners)["bare"]
    assert "description" not in meta


@pytest.mark.unit
def test_router_reasoner_description_passes_through_include_router():
    app = Agent(node_id="test_agent", auto_register=False)
    router = AgentRouter(tags=["swe-issue"])

    @router.reasoner(description="Sub-harness entry point", tags=["entrypoint"])
    async def implement_issue(issue: dict) -> dict:
        return {}

    app.include_router(router)

    meta = _metadata_by_id(app.reasoners)["implement_issue"]
    assert meta["description"] == "Sub-harness entry point"
    assert "entrypoint" in meta["tags"]


@pytest.mark.unit
def test_skill_description_explicit_and_docstring_default():
    app = Agent(node_id="test_agent", auto_register=False)

    @app.skill(description="Deterministic helper")
    def helper(x: int) -> int:
        """Docstring that should NOT be used."""
        return x

    @app.skill()
    def documented(x: int) -> int:
        """Adds one to x."""
        return x + 1

    skills = _metadata_by_id(app.skills)
    assert skills["helper"]["description"] == "Deterministic helper"
    assert skills["documented"]["description"] == "Adds one to x."
