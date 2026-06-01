from agentfield.rate_limiter import StatelessRateLimiter
from agentfield.types import AIConfig


def test_rate_limiter_fail_fast_defaults():
    """Verify rate limiter defaults are tuned for fail-fast behavior."""
    limiter = StatelessRateLimiter()
    assert limiter.max_retries == 5
    assert limiter.base_delay == 0.5
    assert limiter.max_delay == 30.0
    assert limiter.jitter_factor == 0.25
    assert limiter.circuit_breaker_threshold == 5
    assert limiter.circuit_breaker_timeout == 30


def test_ai_config_rate_limit_defaults_match_limiter():
    """AIConfig defaults must stay in sync with StatelessRateLimiter defaults."""
    cfg = AIConfig()
    assert cfg.rate_limit_max_retries == 5
    assert cfg.rate_limit_base_delay == 0.5
    assert cfg.rate_limit_max_delay == 30.0
    assert cfg.rate_limit_jitter_factor == 0.25
    assert cfg.rate_limit_circuit_breaker_threshold == 5
    assert cfg.rate_limit_circuit_breaker_timeout == 30


def test_rate_limiter_max_theoretical_wait():
    """Max theoretical wait with new defaults should be under 3 minutes."""
    limiter = StatelessRateLimiter()
    max_wait = limiter.max_retries * limiter.max_delay
    assert max_wait <= 180, f"Max theoretical wait {max_wait}s exceeds 3 minutes"


def test_ai_config_defaults_and_to_dict():
    cfg = AIConfig()
    d = cfg.to_dict()
    assert d["model"] == "gpt-4o"
    assert d["response_format"] == "auto"
    assert d["audio_model"] in ("tts-1", "tts-1-hd", "gpt-4o-mini-tts")


def test_ai_config_trim_by_chars():
    cfg = AIConfig()
    text = "A" * 50 + "B" * 50
    trimmed = cfg.trim_by_chars(text, limit=60, head_ratio=0.5)
    assert "…TRIMMED…" in trimmed
    assert len(trimmed) <= 80  # rough upper bound


def test_ai_config_get_litellm_params_uses_overrides_and_prunes_none():
    cfg = AIConfig(max_tokens=None, api_key=None)
    params = cfg.get_litellm_params(max_tokens=123, temperature=0.4)
    assert params["max_tokens"] == 123
    assert params["temperature"] == 0.4
    # ensure None-valued fields are removed
    assert "api_key" not in params


def test_ai_config_safe_prompt_chars_uses_cache():
    cfg = AIConfig()
    cfg.model_limits_cache["gpt-4o"] = {
        "context_length": 10000,
        "max_output_tokens": 1000,
    }
    safe = cfg.get_safe_prompt_chars()
    assert safe > 0
