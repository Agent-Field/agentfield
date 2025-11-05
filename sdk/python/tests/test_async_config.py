from haxen_sdk.async_config import AsyncConfig


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
    monkeypatch.setenv("HAXEN_ASYNC_MAX_EXECUTION_TIMEOUT", "123")
    monkeypatch.setenv("HAXEN_ASYNC_BATCH_SIZE", "7")
    monkeypatch.setenv("HAXEN_ASYNC_ENABLE_RESULT_CACHING", "false")
    monkeypatch.setenv("HAXEN_ASYNC_ENABLE_EVENT_STREAM", "true")
    monkeypatch.setenv("HAXEN_ASYNC_EVENT_STREAM_PATH", "/stream")
    monkeypatch.setenv("HAXEN_ASYNC_EVENT_STREAM_RETRY_BACKOFF", "4.5")

    cfg = AsyncConfig.from_environment()
    assert cfg.max_execution_timeout == 123
    assert cfg.batch_size == 7
    assert cfg.enable_result_caching is False
    assert cfg.enable_event_stream is True
    assert cfg.event_stream_path == "/stream"
    assert cfg.event_stream_retry_backoff == 4.5
