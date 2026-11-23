"""Direct unit tests for ``agentfield.decorator_metadata`` helpers."""

from __future__ import annotations

import functools
import inspect
from types import SimpleNamespace

import pytest

from agentfield.decorator_metadata import (
    PENDING_TRIGGERS_ATTR,
    code_origin,
    resolve_reasoner_metadata,
    split_direct_registration_arg,
    stage_trigger,
)


def _sample_fn() -> None:
    return None


class _SampleReceiver:
    def method(self) -> None:
        return None


def test_code_origin_plain_and_wrapped_functions() -> None:
    _, expected_line = inspect.getsourcelines(_sample_fn)
    origin = code_origin(_sample_fn)
    assert origin is not None
    src, line = origin.rsplit(":", 1)
    assert src.endswith("test_decorator_metadata.py")
    assert int(line) == expected_line

    @functools.wraps(_sample_fn)
    def wrapper() -> None:
        return _sample_fn()

    assert code_origin(wrapper) == origin


def test_code_origin_returns_none_for_builtin_or_missing_source() -> None:
    assert code_origin(len) is None
    callable_obj = SimpleNamespace(__call__=lambda: None)
    # Functions without a readable file source (e.g. builtins) return None.
    assert code_origin(callable_obj) is None  # type: ignore[arg-type]


def test_split_direct_registration_arg() -> None:
    receiver = _SampleReceiver()
    bound = receiver.method

    assert split_direct_registration_arg(_sample_fn) == (_sample_fn, None)
    assert split_direct_registration_arg(bound) == (bound, None)

    for non_callable_opt in ("custom-name", ["tag-a", "tag-b"], None, "", 0, False):
        assert split_direct_registration_arg(non_callable_opt) == (
            None,
            non_callable_opt,
        )


def test_stage_trigger_appends_across_repeated_calls() -> None:
    def target() -> None:
        return None

    first = SimpleNamespace(name="first")
    second = SimpleNamespace(name="second")

    stage_trigger(target, first)
    stage_trigger(target, second)

    assert getattr(target, PENDING_TRIGGERS_ATTR) == [first, second]


def test_resolve_reasoner_metadata_merges_cleans_pending_and_preserves_origin() -> None:
    def handler() -> None:
        return None

    existing = SimpleNamespace(source="existing")
    explicit = SimpleNamespace(source="explicit")
    staged_unstamped = SimpleNamespace(source="staged", code_origin=None)
    staged_preset = SimpleNamespace(source="preset", code_origin="custom/file.py:42")

    handler._reasoner_triggers = [existing]  # type: ignore[attr-defined]
    stage_trigger(handler, staged_unstamped)
    stage_trigger(handler, staged_preset)

    merged, accepts = resolve_reasoner_metadata(handler, triggers=[explicit])

    assert merged == [existing, explicit, staged_unstamped, staged_preset]
    assert accepts is True
    assert not hasattr(handler, PENDING_TRIGGERS_ATTR)

    expected_origin = code_origin(handler)
    assert expected_origin is not None
    assert existing.code_origin == expected_origin
    assert explicit.code_origin == expected_origin
    assert staged_unstamped.code_origin == expected_origin
    assert staged_preset.code_origin == "custom/file.py:42"


def test_resolve_reasoner_metadata_handles_non_deletable_pending_and_no_origin() -> (
    None
):
    class ReadOnlyPendingCallable:
        @property
        def _pending_triggers(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(source="class-attr")]

        def __call__(self) -> None:
            return None

    instance = ReadOnlyPendingCallable()
    merged, accepts = resolve_reasoner_metadata(instance)

    assert len(merged) == 1
    assert merged[0].source == "class-attr"
    assert not hasattr(merged[0], "code_origin")
    assert accepts is True


@pytest.mark.parametrize(
    (
        "existing_webhook",
        "explicit_webhook",
        "explicit_triggers",
        "staged",
        "pre_existing_triggers",
        "want",
    ),
    [
        # Explicit accepts_webhook always wins over triggers and existing attribute.
        ("warn", False, [SimpleNamespace()], [], [], False),
        (True, "warn", [], [SimpleNamespace()], [], "warn"),
        # Existing "warn" upgrades when explicit or staged triggers are passed.
        ("warn", None, [SimpleNamespace()], [], [], True),
        ("warn", None, [], [SimpleNamespace()], [], True),
        # Existing non-None accepts_webhook is preserved when no new triggers arrive.
        (False, None, [], [], [SimpleNamespace()], False),
        # No existing accepts_webhook: triggers present -> True, empty -> "warn".
        (None, None, [SimpleNamespace()], [], [], True),
        (None, None, [], [], [SimpleNamespace()], True),
        (None, None, [], [], [], "warn"),
    ],
)
def test_resolve_reasoner_metadata_accepts_webhook_branches(
    existing_webhook: bool | str | None,
    explicit_webhook: bool | str | None,
    explicit_triggers: list[SimpleNamespace],
    staged: list[SimpleNamespace],
    pre_existing_triggers: list[SimpleNamespace],
    want: bool | str,
) -> None:
    def fn() -> None:
        return None

    if existing_webhook is not None:
        fn._accepts_webhook = existing_webhook  # type: ignore[attr-defined]
    if pre_existing_triggers:
        fn._reasoner_triggers = list(pre_existing_triggers)  # type: ignore[attr-defined]
    for trg in staged:
        stage_trigger(fn, trg)

    merged, got = resolve_reasoner_metadata(
        fn,
        triggers=explicit_triggers,
        accepts_webhook=explicit_webhook,
    )

    assert got is want if isinstance(want, bool) else got == want
    assert len(merged) == len(pre_existing_triggers) + len(explicit_triggers) + len(
        staged
    )
