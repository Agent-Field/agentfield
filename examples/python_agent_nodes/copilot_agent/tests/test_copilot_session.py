"""Unit tests for the :mod:`copilot_session` wrapper using a stubbed SDK."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("copilot")

from copilot.generated.session_events import SessionEventType  # noqa: E402

from conftest import make_event  # noqa: E402


@pytest.mark.asyncio
async def test_ask_happy_path(fake_app, stub_copilot_client):
    import copilot_session as cs

    stub_copilot_client["events"] = [
        make_event(SessionEventType.ASSISTANT_MESSAGE, content="Hello, world."),
        make_event(SessionEventType.ASSISTANT_USAGE, input_tokens=10, output_tokens=3),
    ]
    stub_copilot_client["final_event"] = make_event(SessionEventType.SESSION_IDLE)

    result = await cs.run_copilot(
        app=fake_app,
        prompt="say hi",
        node_id="copilot-test",
        af_session_id="s-1",
        model="gpt-5",
        available_tools=[],
    )

    assert result.finished_reason == "idle"
    assert result.answer == "Hello, world."
    assert result.usage == {"input_tokens": 10, "output_tokens": 3}
    assert result.copilot_session_id
    # Permission handler defaults to the deny-all handler defined in the module.
    kwargs = stub_copilot_client["captured_session_kwargs"]
    assert kwargs["on_permission_request"] is cs.deny_all_handler
    assert kwargs["available_tools"] == []
    assert stub_copilot_client["captured_prompt"] == "say hi"


@pytest.mark.asyncio
async def test_timeout_path(fake_app, stub_copilot_client):
    import copilot_session as cs

    stub_copilot_client["raise_timeout"] = True

    result = await cs.run_copilot(
        app=fake_app,
        prompt="slow",
        node_id="copilot-test",
        af_session_id="s-1",
        timeout=0.01,
    )

    assert result.finished_reason == "timeout"
    assert "fake timeout" in (result.error or "")
    assert result.answer == ""


@pytest.mark.asyncio
async def test_tool_calls_collected(fake_app, stub_copilot_client):
    import copilot_session as cs

    stub_copilot_client["events"] = [
        make_event(
            SessionEventType.TOOL_EXECUTION_START,
            tool_call_id="tc-1",
            tool_name="read_file",
        ),
        make_event(
            SessionEventType.TOOL_EXECUTION_COMPLETE,
            tool_call_id="tc-1",
            tool_name="read_file",
        ),
        make_event(SessionEventType.ASSISTANT_MESSAGE, content="done"),
    ]
    stub_copilot_client["final_event"] = make_event(SessionEventType.SESSION_IDLE)

    result = await cs.run_copilot(
        app=fake_app,
        prompt="do it",
        node_id="copilot-test",
        af_session_id="s-1",
    )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["status"] == "complete"
    assert result.tool_calls[0]["tool_name"] == "read_file"
    assert result.answer == "done"


@pytest.mark.asyncio
async def test_session_error_sets_error_finish(fake_app, stub_copilot_client):
    import copilot_session as cs

    stub_copilot_client["events"] = [
        make_event(SessionEventType.SESSION_ERROR, message="boom"),
    ]
    stub_copilot_client["final_event"] = make_event(SessionEventType.SESSION_IDLE)

    result = await cs.run_copilot(
        app=fake_app,
        prompt="bad",
        node_id="copilot-test",
        af_session_id="s-1",
    )

    assert result.finished_reason == "error"
    assert result.error == "boom"


@pytest.mark.asyncio
async def test_auth_env_forwarded_to_subprocess_config(
    fake_app, stub_copilot_client, monkeypatch
):
    import copilot_session as cs

    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghp-fake-token")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    stub_copilot_client["events"] = []
    stub_copilot_client["final_event"] = make_event(SessionEventType.SESSION_IDLE)

    # Capture the CopilotClient kwargs by monkeypatching the __init__.
    captured: dict = {}
    real_client = cs.CopilotClient

    class _Client(real_client):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **kw):  # noqa: D401
            captured["args"] = a
            captured["kwargs"] = kw

    monkeypatch.setattr(cs, "CopilotClient", _Client)

    await cs.run_copilot(
        app=fake_app,
        prompt="ok",
        node_id="copilot-test",
        af_session_id="s-1",
    )

    cfg = captured["kwargs"].get("subprocess_config")
    assert cfg is not None
    assert cfg.github_token == "ghp-fake-token"


@pytest.mark.asyncio
async def test_isolation_opt_in_creates_per_node_config_dir(
    fake_app, stub_copilot_client, monkeypatch, tmp_path
):
    import copilot_session as cs

    monkeypatch.setenv("AGENTFIELD_HOME", str(tmp_path))
    monkeypatch.delenv("AGENTFIELD_COPILOT_ISOLATE", raising=False)

    stub_copilot_client["events"] = []
    stub_copilot_client["final_event"] = make_event(SessionEventType.SESSION_IDLE)

    # Default: no config_dir is passed (shared ~/.copilot).
    await cs.run_copilot(
        app=fake_app,
        prompt="ok",
        node_id="copilot-nodeA",
        af_session_id="s-1",
    )
    kwargs = stub_copilot_client["captured_session_kwargs"]
    assert "config_dir" not in kwargs

    # Explicit isolate=True creates a per-node sandbox.
    await cs.run_copilot(
        app=fake_app,
        prompt="ok",
        node_id="copilot-nodeA",
        af_session_id="s-2",
        isolate=True,
    )
    kwargs = stub_copilot_client["captured_session_kwargs"]
    assert kwargs["config_dir"] == str(tmp_path / "copilot-home" / "copilot-nodeA")
    assert os.path.isdir(kwargs["config_dir"])


@pytest.mark.asyncio
async def test_session_id_mapping_in_memory(fake_app, stub_copilot_client):
    import copilot_session as cs

    stub_copilot_client["events"] = []
    stub_copilot_client["final_event"] = make_event(SessionEventType.SESSION_IDLE)

    # First call with continue_session=False always gets a fresh id.
    r1 = await cs.run_copilot(
        app=fake_app,
        prompt="ok",
        node_id="copilot-nodeA",
        af_session_id="af-1",
        continue_session=False,
    )
    # Second call with continue_session=True but nothing stored → fresh id,
    # then persisted.
    r2 = await cs.run_copilot(
        app=fake_app,
        prompt="ok",
        node_id="copilot-nodeA",
        af_session_id="af-1",
        continue_session=True,
    )
    # Third call with continue_session=True reuses r2's id.
    r3 = await cs.run_copilot(
        app=fake_app,
        prompt="ok",
        node_id="copilot-nodeA",
        af_session_id="af-1",
        continue_session=True,
    )

    assert r1.copilot_session_id != r2.copilot_session_id
    assert r2.copilot_session_id == r3.copilot_session_id
    # Different AF session id → different mapping.
    r4 = await cs.run_copilot(
        app=fake_app,
        prompt="ok",
        node_id="copilot-nodeA",
        af_session_id="af-2",
        continue_session=True,
    )
    assert r4.copilot_session_id != r2.copilot_session_id


@pytest.mark.asyncio
async def test_to_dict_round_trip(fake_app, stub_copilot_client):
    import copilot_session as cs

    stub_copilot_client["events"] = [
        make_event(SessionEventType.ASSISTANT_MESSAGE, content="hi"),
    ]
    stub_copilot_client["final_event"] = make_event(SessionEventType.SESSION_IDLE)

    result = await cs.run_copilot(
        app=fake_app,
        prompt="p",
        node_id="n",
        af_session_id="s",
    )
    d = result.to_dict()
    assert set(d.keys()) == {
        "af_session_id",
        "copilot_session_id",
        "model",
        "answer",
        "transcript",
        "tool_calls",
        "usage",
        "finished_reason",
        "error",
    }
