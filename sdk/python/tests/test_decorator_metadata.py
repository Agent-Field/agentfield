"""Direct unit tests for the shared decorator metadata helpers."""

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


def _plain_function():
    return None


class TestCodeOrigin:
    def test_plain_function_returns_path_and_line(self):
        origin = code_origin(_plain_function)

        assert origin == (
            f"{inspect.getsourcefile(_plain_function)}:"
            f"{_plain_function.__code__.co_firstlineno}"
        )

    def test_builtin_without_source_returns_none(self):
        assert code_origin(len) is None


class TestSplitDirectRegistrationArg:
    def test_function_is_direct_registration(self):
        assert split_direct_registration_arg(_plain_function) == (
            _plain_function,
            None,
        )

    @pytest.mark.parametrize("value", ["custom_name", ["tag-a", "tag-b"], None])
    def test_non_function_is_returned_as_option(self, value):
        assert split_direct_registration_arg(value) == (None, value)


class TestStageTrigger:
    def test_repeated_calls_append_to_pending_triggers(self):
        def fn():
            return None

        first = SimpleNamespace()
        second = SimpleNamespace()

        stage_trigger(fn, first)
        stage_trigger(fn, second)

        assert getattr(fn, PENDING_TRIGGERS_ATTR) == [first, second]


class TestResolveReasonerMetadataAcceptsWebhook:
    def test_explicit_value_wins(self):
        def fn():
            return None

        fn._accepts_webhook = "warn"

        _, accepts_webhook = resolve_reasoner_metadata(
            fn, triggers=[SimpleNamespace()], accepts_webhook=False
        )

        assert accepts_webhook is False

    def test_warn_is_upgraded_when_triggers_are_added(self):
        def fn():
            return None

        fn._accepts_webhook = "warn"

        _, accepts_webhook = resolve_reasoner_metadata(
            fn, triggers=[SimpleNamespace()]
        )

        assert accepts_webhook is True

    def test_warn_is_upgraded_when_triggers_are_staged(self):
        def fn():
            return None

        fn._accepts_webhook = "warn"
        stage_trigger(fn, SimpleNamespace())

        _, accepts_webhook = resolve_reasoner_metadata(fn)

        assert accepts_webhook is True

    def test_existing_value_is_kept(self):
        def fn():
            return None

        fn._accepts_webhook = False

        _, accepts_webhook = resolve_reasoner_metadata(
            fn, triggers=[SimpleNamespace()]
        )

        assert accepts_webhook is False

    def test_triggers_present_gives_true(self):
        def fn():
            return None

        fn._reasoner_triggers = [SimpleNamespace()]

        _, accepts_webhook = resolve_reasoner_metadata(fn)

        assert accepts_webhook is True

    def test_nothing_gives_warn(self):
        def fn():
            return None

        triggers, accepts_webhook = resolve_reasoner_metadata(fn)

        assert triggers == []
        assert accepts_webhook == "warn"


class TestResolveReasonerMetadataTriggers:
    def test_staged_triggers_are_merged_and_pending_attr_removed(self):
        def fn():
            return None

        existing = SimpleNamespace()
        explicit = SimpleNamespace()
        staged = SimpleNamespace()
        fn._reasoner_triggers = [existing]
        stage_trigger(fn, staged)

        triggers, _ = resolve_reasoner_metadata(fn, triggers=[explicit])

        assert triggers == [existing, explicit, staged]
        assert not hasattr(fn, PENDING_TRIGGERS_ATTR)

    def test_staged_triggers_on_bound_method_are_merged(self):
        class Handlers:
            def handle(self):
                return None

        staged = SimpleNamespace()
        stage_trigger(Handlers.handle, staged)

        # The bound method exposes the function's attributes but cannot
        # delete them, so the AttributeError from delattr is swallowed.
        triggers, accepts_webhook = resolve_reasoner_metadata(Handlers().handle)

        assert triggers == [staged]
        assert accepts_webhook is True

    def test_code_origin_is_stamped_unless_already_set(self):
        def fn():
            return None

        preset = SimpleNamespace(code_origin="handlers.py:7")
        fresh = SimpleNamespace()

        triggers, _ = resolve_reasoner_metadata(fn, triggers=[preset, fresh])

        assert triggers == [preset, fresh]
        assert preset.code_origin == "handlers.py:7"
        assert fresh.code_origin == code_origin(fn)
