from dataclasses import dataclass

import pytest

from agentfield.async_config import AsyncConfig
from agentfield.client import AgentFieldClient


def test_async_config_validate_defaults_ok():
    cfg = AsyncConfig()
    # Should not raise
    cfg.validate()


def test_async_config_validate_bad_intervals():
    cfg = AsyncConfig(
        initial_poll_interval=1.0,
        fast_poll_interval=0.5,  # out of order
    )
    try:
        cfg.validate()
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_get_poll_interval_for_age():
    cfg = AsyncConfig(
        fast_execution_threshold=10.0,
        medium_execution_threshold=60.0,
        fast_poll_interval=0.1,
        medium_poll_interval=0.5,
        slow_poll_interval=2.0,
    )
    assert cfg.get_poll_interval_for_age(5) == 0.1
    assert cfg.get_poll_interval_for_age(20) == 0.5
    assert cfg.get_poll_interval_for_age(120) == 2.0


def test_from_environment_overrides(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_ASYNC_MAX_EXECUTION_TIMEOUT", "123")
    monkeypatch.setenv("AGENTFIELD_ASYNC_BATCH_SIZE", "7")
    monkeypatch.setenv("AGENTFIELD_ASYNC_ENABLE_RESULT_CACHING", "false")
    monkeypatch.setenv("AGENTFIELD_ASYNC_ENABLE_EVENT_STREAM", "true")
    monkeypatch.setenv("AGENTFIELD_ASYNC_EVENT_STREAM_PATH", "/stream")
    monkeypatch.setenv("AGENTFIELD_ASYNC_EVENT_STREAM_RETRY_BACKOFF", "4.5")

    cfg = AsyncConfig.from_environment()
    assert cfg.max_execution_timeout == 123
    assert cfg.batch_size == 7
    assert cfg.enable_result_caching is False
    assert cfg.enable_event_stream is True
    assert cfg.event_stream_path == "/stream"
    assert cfg.event_stream_retry_backoff == 4.5


def test_client_default_async_config_uses_environment(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_ASYNC_MAX_EXECUTION_TIMEOUT", "321")
    monkeypatch.setenv("AGENTFIELD_ASYNC_ENABLE_EVENT_STREAM", "true")
    monkeypatch.setenv("AGENTFIELD_ASYNC_EVENT_STREAM_PATH", "/client-events")

    client = AgentFieldClient()

    assert client.async_config.max_execution_timeout == 321
    assert client.async_config.enable_event_stream is True
    assert client.async_config.event_stream_path == "/client-events"


def test_client_keeps_explicit_async_config(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_ASYNC_MAX_EXECUTION_TIMEOUT", "321")
    explicit_config = AsyncConfig(max_execution_timeout=456)

    client = AgentFieldClient(async_config=explicit_config)

    assert client.async_config is explicit_config


# The four flags below default to True, so an environment variable has to be able
# to turn them *off*. The package already settles this shape for default-on env
# flags: log_writer._queue_enabled, logger._stdout_mirror_enabled,
# node_logs.logs_enabled and openrouter_attribution.attribution_enabled all treat
# ("0", "false", "no", "off") as the opt-out vocabulary and everything else as
# "keep the default".
DEFAULT_ON_FLAGS = [
    ("AGENTFIELD_ASYNC_ENABLE_ASYNC_EXECUTION", "enable_async_execution"),
    ("AGENTFIELD_ASYNC_ENABLE_BATCH_POLLING", "enable_batch_polling"),
    ("AGENTFIELD_ASYNC_ENABLE_RESULT_CACHING", "enable_result_caching"),
    ("AGENTFIELD_ASYNC_FALLBACK_TO_SYNC", "fallback_to_sync"),
]

TRUTHY_VALUES = ["1", "true", "TRUE", "yes", "on", " true "]
FALSEY_VALUES = ["0", "false", "FALSE", "no", "off", " false "]
UNPARSEABLE_VALUES = ["maybe", "", "2"]

EVENT_STREAM_ENV = "AGENTFIELD_ASYNC_ENABLE_EVENT_STREAM"


@pytest.mark.parametrize("env_name,field", DEFAULT_ON_FLAGS)
@pytest.mark.parametrize("value", TRUTHY_VALUES)
def test_default_on_flag_accepts_conventional_truthy_values(
    monkeypatch, env_name, field, value
):
    """`=1`/`=yes`/`=on` must not silently disable a default-on feature."""
    monkeypatch.setenv(env_name, value)

    assert getattr(AsyncConfig.from_environment(), field) is True


@pytest.mark.parametrize("env_name,field", DEFAULT_ON_FLAGS)
@pytest.mark.parametrize("value", FALSEY_VALUES)
def test_default_on_flag_still_opts_out(monkeypatch, env_name, field, value):
    """The opt-out vocabulary that already works keeps working."""
    monkeypatch.setenv(env_name, value)

    assert getattr(AsyncConfig.from_environment(), field) is False


@pytest.mark.parametrize("env_name,field", DEFAULT_ON_FLAGS)
@pytest.mark.parametrize("value", UNPARSEABLE_VALUES)
def test_default_on_flag_falls_back_to_default_when_unparseable(
    monkeypatch, env_name, field, value
):
    """A value that means nothing must leave the field at its default.

    This is the contract PR #714 states for from_environment(): it "only
    overrides fields when the corresponding env var is set (falling back to the
    default on unparseable values)". The float and int converters honour that
    through get_env_var's except clause; a boolean converter that never raises
    has to honour it through its vocabulary instead.
    """
    monkeypatch.setenv(env_name, value)

    expected = getattr(AsyncConfig(), field)
    assert getattr(AsyncConfig.from_environment(), field) is expected


@pytest.mark.parametrize("value", TRUTHY_VALUES)
def test_event_stream_opts_in_with_truthy_values(monkeypatch, value):
    """enable_event_stream defaults to False, so it needs the opt-in vocabulary."""
    monkeypatch.setenv(EVENT_STREAM_ENV, value)

    assert AsyncConfig.from_environment().enable_event_stream is True


@pytest.mark.parametrize("value", FALSEY_VALUES + UNPARSEABLE_VALUES)
def test_event_stream_stays_off_for_falsey_or_unparseable(monkeypatch, value):
    monkeypatch.setenv(EVENT_STREAM_ENV, value)

    assert AsyncConfig.from_environment().enable_event_stream is False


def test_boolean_env_flags_reach_the_client_default(monkeypatch):
    """The public path: AgentFieldClient() with no explicit async_config."""
    monkeypatch.setenv("AGENTFIELD_ASYNC_ENABLE_RESULT_CACHING", "1")
    monkeypatch.setenv("AGENTFIELD_ASYNC_ENABLE_EVENT_STREAM", "yes")

    client = AgentFieldClient()

    assert client.async_config.enable_result_caching is True
    assert client.async_config.enable_event_stream is True


# Issue #1089: each polarity wrapper bakes in the field's built-in default, so a
# subclass that changes a default gets the wrapper's answer for an unrecognised
# string instead of its own. One parser that rejects unknown values lets
# get_env_var's existing fallback supply the class's real default.


@dataclass
class CachingOffConfig(AsyncConfig):
    """Flips a default-on flag's built-in default."""

    enable_result_caching: bool = False


@dataclass
class StreamOnConfig(AsyncConfig):
    """Flips the default-off flag's built-in default."""

    enable_event_stream: bool = True


def test_unrecognised_value_keeps_a_subclass_default_on_flag_off(monkeypatch):
    """The fallback must come from the class, not from the parser's polarity."""
    monkeypatch.setenv("AGENTFIELD_ASYNC_ENABLE_RESULT_CACHING", "maybe")

    assert CachingOffConfig.from_environment().enable_result_caching is False


def test_unrecognised_value_keeps_a_subclass_default_off_flag_on(monkeypatch):
    monkeypatch.setenv("AGENTFIELD_ASYNC_ENABLE_EVENT_STREAM", "nonsense")

    assert StreamOnConfig.from_environment().enable_event_stream is True


@pytest.mark.parametrize(
    "field,env_suffix",
    [
        ("enable_async_execution", "ENABLE_ASYNC_EXECUTION"),
        ("enable_batch_polling", "ENABLE_BATCH_POLLING"),
        ("enable_result_caching", "ENABLE_RESULT_CACHING"),
        ("fallback_to_sync", "FALLBACK_TO_SYNC"),
        ("enable_event_stream", "ENABLE_EVENT_STREAM"),
    ],
)
@pytest.mark.parametrize("value", UNPARSEABLE_VALUES)
def test_every_flag_falls_back_to_its_own_class_default(
    monkeypatch, field, env_suffix, value
):
    """All five flags, both classes: an unknown value never invents a boolean."""
    monkeypatch.setenv(f"AGENTFIELD_ASYNC_{env_suffix}", value)

    for cls in (AsyncConfig, CachingOffConfig, StreamOnConfig):
        expected = getattr(cls(), field)
        assert getattr(cls.from_environment(), field) is expected


def test_env_flag_rejects_values_outside_the_vocabulary():
    """One parser, no baked-in polarity: the caller decides what unknown means."""
    from agentfield.async_config import _env_flag

    assert _env_flag(" ON ") is True
    assert _env_flag("Off") is False

    for value in ("maybe", "", "2", "truthy"):
        with pytest.raises(ValueError):
            _env_flag(value)


def test_polarity_wrappers_keep_their_contract_for_external_callers():
    """logger.py imports _env_flag_default_off (PR #1090); both wrappers must hold."""
    from agentfield.async_config import _env_flag_default_off, _env_flag_default_on

    assert _env_flag_default_off("1") is True
    assert _env_flag_default_off("off") is False
    assert _env_flag_default_off("maybe") is False

    assert _env_flag_default_on("0") is False
    assert _env_flag_default_on("yes") is True
    assert _env_flag_default_on("maybe") is True
