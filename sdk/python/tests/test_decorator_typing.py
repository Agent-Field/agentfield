"""Tests for improved type annotations on core decorators."""

import asyncio
import inspect

import pytest

from agentfield.decorators import (
    legacy_reasoner,
    on_change,
    on_event,
    on_schedule,
    reasoner,
)
from agentfield.triggers import EventTrigger, ScheduleTrigger


def test_reasoner_overload_no_parentheses_preserves_signature():
    """@reasoner (no parens) should preserve the original function signature."""

    @reasoner
    def add(a: int, b: int) -> int:
        return a + b

    sig = inspect.signature(add)
    params = list(sig.parameters.values())
    assert len(params) == 2
    assert params[0].name == "a"
    assert params[0].annotation is int
    assert params[1].name == "b"
    assert params[1].annotation is int
    assert sig.return_annotation is int
    assert asyncio.run(add(2, 3)) == 5


def test_reasoner_overload_with_parens_preserves_signature():
    """@reasoner(...) with arguments should preserve the original function signature."""

    @reasoner(tags=["math"], track_workflow=False)
    def multiply(x: float, y: float) -> float:
        return x * y

    sig = inspect.signature(multiply)
    params = list(sig.parameters.values())
    assert len(params) == 2
    assert params[0].name == "x"
    assert params[0].annotation is float
    assert params[1].name == "y"
    assert params[1].annotation is float
    assert sig.return_annotation is float
    assert asyncio.run(multiply(3.0, 4.0)) == 12.0


def test_reasoner_overload_empty_parens_preserves_signature():
    """@reasoner() with empty parens should preserve the original function signature."""

    @reasoner()
    def greet(name: str) -> str:
        return f"Hello {name}"

    sig = inspect.signature(greet)
    params = list(sig.parameters.values())
    assert len(params) == 1
    assert params[0].name == "name"
    assert params[0].annotation is str
    assert sig.return_annotation is str
    assert asyncio.run(greet("world")) == "Hello world"


def test_reasoner_metadata_types():
    """Metadata attributes on decorated functions should be properly typed."""

    @reasoner(tags=["test"], description="A test reasoner")
    def fn(a: int) -> int:
        return a

    assert fn._is_reasoner is True
    assert fn._track_workflow is True
    assert fn._reasoner_tags == ["test"]
    assert fn._reasoner_description == "A test reasoner"
    assert fn._original_func is fn.__wrapped__


def test_on_event_preserves_function():
    """@on_event should return the same function, preserving signature."""

    @on_event(source="stripe", types=["payment_intent.succeeded"])
    def handler(payload: dict) -> str:
        return "ok"

    sig = inspect.signature(handler)
    assert list(sig.parameters.keys()) == ["payload"]
    assert sig.parameters["payload"].annotation is dict
    pending = getattr(handler, "_pending_triggers", [])
    assert len(pending) == 1
    assert isinstance(pending[0], EventTrigger)
    assert pending[0].source == "stripe"


def test_on_schedule_preserves_function():
    """@on_schedule should return the same function, preserving signature."""

    @on_schedule("*/5 * * * *")
    def poll(data: str) -> None:
        return None

    sig = inspect.signature(poll)
    assert list(sig.parameters.keys()) == ["data"]
    pending = getattr(poll, "_pending_triggers", [])
    assert len(pending) == 1
    assert isinstance(pending[0], ScheduleTrigger)
    assert pending[0].cron == "*/5 * * * *"


def test_on_change_preserves_signature():
    """@on_change should preserve the original function signature."""

    @on_change("memory:*")
    async def listener(key: str, value: object) -> None:
        pass

    sig = inspect.signature(listener)
    params = list(sig.parameters.values())
    assert len(params) == 2
    assert params[0].name == "key"
    assert params[0].annotation is str
    assert params[1].name == "value"
    assert params[1].annotation is object
    assert hasattr(listener, "_memory_event_listener")


def test_on_change_with_list_pattern_preserves_signature():
    """@on_change with a list of patterns should preserve the signature."""

    @on_change(["users:*", "orders:*"])
    async def watcher(event: str) -> int:
        return 1

    sig = inspect.signature(watcher)
    assert list(sig.parameters.keys()) == ["event"]
    assert watcher._memory_event_patterns == ["users:*", "orders:*"]


def test_legacy_reasoner_preserves_signature():
    """legacy_reasoner should preserve the original function signature."""

    @legacy_reasoner("old-fn", {"x": "int"}, {"y": "int"})
    def old_fn(x: int) -> int:
        return x * 2

    sig = inspect.signature(old_fn)
    assert list(sig.parameters.keys()) == ["x"]
    assert old_fn(5) == 10
    assert old_fn._reasoner_def.id == "old-fn"


@pytest.mark.asyncio
async def test_execute_with_tracking_type_preservation(monkeypatch):
    """_execute_with_tracking should accept typed functions and return correct types."""

    monkeypatch.setattr(
        "agentfield.decorators._send_workflow_start",
        lambda *a, **k: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        "agentfield.decorators._send_workflow_completion",
        lambda *a, **k: asyncio.sleep(0),
    )

    from agentfield.decorators import _execute_with_tracking

    async def typed_fn(x: int, label: str) -> str:
        return f"{label}: {x * 2}"

    sig = inspect.signature(typed_fn)
    params = list(sig.parameters.values())
    assert params[0].name == "x"
    assert params[0].annotation is int
    assert params[1].name == "label"
    assert params[1].annotation is str

    # Call without tracking context (no agent)
    result = await _execute_with_tracking(typed_fn, 5, label="test")
    assert result == "test: 10"


def test_reasoner_accepts_webhook_types():
    """accepts_webhook parameter should accept bool or str."""

    @reasoner(accepts_webhook=True)
    def with_t_a(x: int) -> int:
        return x

    assert with_t_a._accepts_webhook is True

    @reasoner(accepts_webhook=False)
    def with_f_a(x: int) -> int:
        return x

    assert with_f_a._accepts_webhook is False

    @reasoner(accepts_webhook="warn")
    def with_warn(x: int) -> int:
        return x

    assert with_warn._accepts_webhook == "warn"


def test_reasoner_tags_default():
    """tags should default to empty list."""

    @reasoner
    def no_tags(x: int) -> int:
        return x

    assert no_tags._reasoner_tags == []
