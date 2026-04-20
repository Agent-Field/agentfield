"""Unit tests for the four reasoners, exercised via the registration helper."""

from __future__ import annotations

import pytest

pytest.importorskip("copilot")

from copilot.generated.session_events import SessionEventType  # noqa: E402

from conftest import make_event  # noqa: E402


def _register(fake_app):
    import reasoners

    reasoners.register(fake_app)
    return reasoners


@pytest.mark.asyncio
async def test_ask_has_no_tools(fake_app, stub_copilot_client):
    r = _register(fake_app)
    stub_copilot_client["events"] = [
        make_event(SessionEventType.ASSISTANT_MESSAGE, content="42"),
    ]
    stub_copilot_client["final_event"] = make_event(SessionEventType.SESSION_IDLE)

    out = await r.register.ask(prompt="what is the answer?")
    assert out["answer"] == "42"
    kwargs = stub_copilot_client["captured_session_kwargs"]
    assert kwargs["available_tools"] == []


@pytest.mark.asyncio
async def test_plan_has_no_tools(fake_app, stub_copilot_client):
    r = _register(fake_app)
    stub_copilot_client["events"] = [
        make_event(SessionEventType.ASSISTANT_MESSAGE, content="step 1; step 2"),
    ]
    stub_copilot_client["final_event"] = make_event(SessionEventType.SESSION_IDLE)

    out = await r.register.plan(task="bake a cake")
    assert "step 1" in out["answer"]
    kwargs = stub_copilot_client["captured_session_kwargs"]
    assert kwargs["available_tools"] == []
    # Prompt wraps the task with a no-execute instruction.
    assert "step-by-step plan" in stub_copilot_client["captured_prompt"]
    assert "bake a cake" in stub_copilot_client["captured_prompt"]


@pytest.mark.asyncio
async def test_review_with_diff_disables_tools(fake_app, stub_copilot_client):
    r = _register(fake_app)
    stub_copilot_client["events"] = [
        make_event(SessionEventType.ASSISTANT_MESSAGE, content="1. nit: rename foo"),
    ]
    stub_copilot_client["final_event"] = make_event(SessionEventType.SESSION_IDLE)

    out = await r.register.review(diff="- foo\n+ bar\n")
    assert "nit" in out["answer"]
    kwargs = stub_copilot_client["captured_session_kwargs"]
    assert kwargs["available_tools"] == []


@pytest.mark.asyncio
async def test_review_with_files_uses_readonly_allowlist(fake_app, stub_copilot_client):
    r = _register(fake_app)
    stub_copilot_client["events"] = [
        make_event(SessionEventType.ASSISTANT_MESSAGE, content="ok"),
    ]
    stub_copilot_client["final_event"] = make_event(SessionEventType.SESSION_IDLE)

    await r.register.review(files=["a.py", "b.py"], cwd="/tmp")
    kwargs = stub_copilot_client["captured_session_kwargs"]
    assert set(kwargs["available_tools"]) == {
        "read_file",
        "list_directory",
        "grep",
        "git_diff",
    }


@pytest.mark.asyncio
async def test_review_requires_diff_or_files(fake_app, stub_copilot_client):
    r = _register(fake_app)
    out = await r.register.review()
    assert out["finished_reason"] == "error"
    assert "requires" in out["error"]
    # No session should have been created.
    assert stub_copilot_client["captured_session_kwargs"] is None


@pytest.mark.asyncio
async def test_run_task_denies_without_allow_tools(fake_app, stub_copilot_client):
    r = _register(fake_app)

    out = await r.register.run_task(task="delete all files")
    assert out["finished_reason"] == "error"
    assert "allow_tools=True" in out["error"]
    assert stub_copilot_client["captured_session_kwargs"] is None


@pytest.mark.asyncio
async def test_run_task_allow_list_beats_deny_list(fake_app, stub_copilot_client):
    r = _register(fake_app)
    stub_copilot_client["events"] = []
    stub_copilot_client["final_event"] = make_event(SessionEventType.SESSION_IDLE)

    await r.register.run_task(
        task="do it",
        allow_tools=True,
        allow_list=["read_file"],
        deny_list=["execute_shell_command"],
    )
    kwargs = stub_copilot_client["captured_session_kwargs"]
    assert kwargs["available_tools"] == ["read_file"]
    # excluded_tools must NOT be set when allow_list is provided.
    assert "excluded_tools" not in kwargs


@pytest.mark.asyncio
async def test_run_task_deny_list_only(fake_app, stub_copilot_client):
    r = _register(fake_app)
    stub_copilot_client["events"] = []
    stub_copilot_client["final_event"] = make_event(SessionEventType.SESSION_IDLE)

    await r.register.run_task(
        task="do it",
        allow_tools=True,
        deny_list=["execute_shell_command"],
    )
    kwargs = stub_copilot_client["captured_session_kwargs"]
    assert "available_tools" not in kwargs
    assert kwargs["excluded_tools"] == ["execute_shell_command"]
