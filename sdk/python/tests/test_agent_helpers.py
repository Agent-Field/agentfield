from haxen_sdk.agent import _resolve_callback_url, _build_callback_candidates


def test_resolve_callback_url_prefers_explicit_url():
    url = _resolve_callback_url("https://example.com:9000", port=8000)
    assert url == "https://example.com:9000"

    url = _resolve_callback_url("example.org", port=7000)
    assert url == "http://example.org:7000"


def test_resolve_callback_url_uses_env(monkeypatch):
    monkeypatch.setenv("AGENT_CALLBACK_URL", "https://env.example.com")
    try:
        url = _resolve_callback_url(None, port=5000)
        assert url == "https://env.example.com:5000"
    finally:
        monkeypatch.delenv("AGENT_CALLBACK_URL", raising=False)


def test_resolve_callback_url_handles_container_overrides(monkeypatch):
    monkeypatch.delenv("AGENT_CALLBACK_URL", raising=False)
    monkeypatch.setenv("RAILWAY_SERVICE_NAME", "my-service")
    monkeypatch.setenv("RAILWAY_ENVIRONMENT", "prod")

    monkeypatch.setattr("haxen_sdk.agent._is_running_in_container", lambda: True)
    monkeypatch.setattr("haxen_sdk.agent._detect_container_ip", lambda: None)
    monkeypatch.setattr("haxen_sdk.agent._detect_local_ip", lambda: "10.0.0.5")

    url = _resolve_callback_url(None, port=4500)
    assert url == "http://my-service.railway.internal:4500"


def test_resolve_callback_url_fallback_to_detected_ips(monkeypatch):
    monkeypatch.delenv("AGENT_CALLBACK_URL", raising=False)
    monkeypatch.delenv("RAILWAY_SERVICE_NAME", raising=False)
    monkeypatch.delenv("RAILWAY_ENVIRONMENT", raising=False)

    monkeypatch.setattr("haxen_sdk.agent._is_running_in_container", lambda: True)
    monkeypatch.setattr("haxen_sdk.agent._detect_container_ip", lambda: "203.0.113.10")

    url = _resolve_callback_url(None, port=3200)
    assert url == "http://203.0.113.10:3200"

    monkeypatch.setattr("haxen_sdk.agent._detect_container_ip", lambda: None)
    monkeypatch.setattr("haxen_sdk.agent._detect_local_ip", lambda: "192.168.1.2")

    url = _resolve_callback_url(None, port=3201)
    assert url == "http://192.168.1.2:3201"


def test_resolve_callback_url_final_fallback(monkeypatch):
    monkeypatch.delenv("AGENT_CALLBACK_URL", raising=False)
    monkeypatch.setattr("haxen_sdk.agent._is_running_in_container", lambda: False)
    monkeypatch.setattr("haxen_sdk.agent._detect_local_ip", lambda: None)
    monkeypatch.setattr("haxen_sdk.agent.socket.gethostname", lambda: "")

    url = _resolve_callback_url(None, port=8080)
    assert url in {"http://localhost:8080", "http://host.docker.internal:8080"}


def test_build_callback_candidates_includes_defaults(monkeypatch):
    monkeypatch.setattr("haxen_sdk.agent._is_running_in_container", lambda: False)
    monkeypatch.setattr("haxen_sdk.agent._detect_local_ip", lambda: "192.168.1.50")
    monkeypatch.setattr("haxen_sdk.agent.socket.gethostname", lambda: "my-host")

    candidates = _build_callback_candidates(None, 9000)

    assert "http://192.168.1.50:9000" in candidates
    assert "http://localhost:9000" in candidates
    assert "http://127.0.0.1:9000" in candidates
