"""
Tests for litellm_adapters.py provider-specific patches.
"""

from agentfield.litellm_adapters import (
    get_provider_from_model,
    apply_openai_patches,
    apply_provider_patches,
    filter_none_values,
)


def test_get_provider_from_model():
    """Test provider extraction from model strings."""
    tests = [
        ("openai/gpt-4o", "openai"),
        ("anthropic/claude-3-opus", "anthropic"),
        ("cohere/command", "cohere"),
        ("gpt-4o", "unknown"),  # No provider prefix
        ("azure/gpt-4o", "azure"),
    ]

    for model, expected in tests:
        assert get_provider_from_model(model) == expected


def test_apply_openai_patches():
    """Test OpenAI-specific parameter patches."""
    params = {
        "model": "openai/gpt-4o",
        "max_tokens": 1000,
        "temperature": 0.7,
    }

    result = apply_openai_patches(params)

    # max_tokens should be converted to max_completion_tokens
    assert "max_completion_tokens" in result
    assert result["max_completion_tokens"] == 1000
    assert "max_tokens" not in result

    # Other params should remain unchanged
    assert result["model"] == "openai/gpt-4o"
    assert result["temperature"] == 0.7


def test_apply_openai_patches_no_max_tokens():
    """Test OpenAI patches when max_tokens is not present."""
    params = {
        "model": "openai/gpt-4o",
        "temperature": 0.7,
    }

    result = apply_openai_patches(params)

    # Should not add max_completion_tokens if max_tokens wasn't present
    assert "max_completion_tokens" not in result
    assert "max_tokens" not in result


def test_apply_openai_patches_immutable():
    """Test that apply_openai_patches doesn't mutate the original dict."""
    params = {
        "model": "openai/gpt-4o",
        "max_tokens": 1000,
    }

    original_params = params.copy()
    result = apply_openai_patches(params)

    # Original should be unchanged
    assert params == original_params
    assert "max_tokens" in params
    assert "max_tokens" not in result


def test_apply_provider_patches_openai():
    """Test provider patches routing for OpenAI."""
    params = {
        "model": "openai/gpt-4o",
        "max_tokens": 1000,
    }

    result = apply_provider_patches(params, "openai/gpt-4o")

    assert "max_completion_tokens" in result
    assert result["max_completion_tokens"] == 1000


def test_apply_provider_patches_unknown():
    """Test provider patches for unknown provider."""
    params = {
        "model": "unknown/model",
        "max_tokens": 1000,
    }

    result = apply_provider_patches(params, "unknown/model")

    # Should return unchanged for unknown providers
    assert result == params
    assert "max_tokens" in result


def test_apply_provider_patches_no_provider():
    """Test provider patches for model without provider prefix."""
    params = {
        "model": "gpt-4o",
        "max_tokens": 1000,
    }

    result = apply_provider_patches(params, "gpt-4o")

    # Should return unchanged
    assert result == params


def test_filter_none_values():
    """Test filtering None values from parameter dictionary."""
    params = {
        "model": "gpt-4o",
        "temperature": 0.7,
        "max_tokens": None,
        "top_p": None,
        "frequency_penalty": 0.0,
    }

    result = filter_none_values(params)

    assert "model" in result
    assert "temperature" in result
    assert "frequency_penalty" in result
    assert "max_tokens" not in result
    assert "top_p" not in result


def test_filter_none_values_empty():
    """Test filtering None values from empty dict."""
    params = {}
    result = filter_none_values(params)
    assert result == {}


def test_filter_none_values_all_none():
    """Test filtering when all values are None."""
    params = {
        "a": None,
        "b": None,
        "c": None,
    }

    result = filter_none_values(params)
    assert result == {}


def test_filter_none_values_no_none():
    """Test filtering when no values are None."""
    params = {
        "a": 1,
        "b": "test",
        "c": 0.5,
    }

    result = filter_none_values(params)
    assert result == params


def test_apply_provider_patches_chain():
    """Test chaining provider patches with other operations."""
    params = {
        "model": "openai/gpt-4o",
        "max_tokens": 1000,
        "temperature": None,
        "top_p": 0.9,
    }

    # Apply provider patches
    result = apply_provider_patches(params, "openai/gpt-4o")

    # Then filter None values
    result = filter_none_values(result)

    assert "max_completion_tokens" in result
    assert "max_tokens" not in result
    assert "temperature" not in result
    assert "top_p" in result
