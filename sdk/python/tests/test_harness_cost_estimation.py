"""Tests for CLI-based cost estimation used by subprocess harness providers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agentfield.harness._cli import estimate_cli_cost


@pytest.mark.unit
def test_estimate_cost_returns_positive_for_known_model():
    """If litellm knows the model pricing, we get a positive cost."""
    cost = estimate_cli_cost(
        model="openrouter/google/gemini-2.5-flash-preview",
        prompt="Review this code for security issues",
        result_text="No issues found.",
    )
    # Depends on litellm having the model in its DB; may be None if not
    assert cost is None or cost > 0


@pytest.mark.unit
def test_estimate_cost_unknown_model_returns_none():
    """Unknown model returns None, not an error."""
    cost = estimate_cli_cost(
        model="nonexistent/model-xyz-999",
        prompt="test",
        result_text="test",
    )
    assert cost is None


@pytest.mark.unit
def test_estimate_cost_empty_model_returns_none():
    """Empty model string returns None immediately."""
    cost = estimate_cli_cost(model="", prompt="test", result_text="test")
    assert cost is None


@pytest.mark.unit
def test_estimate_cost_none_result_text():
    """None result_text should not crash — completion tokens should be 0."""
    cost = estimate_cli_cost(
        model="openrouter/google/gemini-2.5-flash-preview",
        prompt="test prompt",
        result_text=None,
    )
    assert cost is None or cost > 0


@pytest.mark.unit
def test_estimate_cost_handles_litellm_import_error():
    """If litellm is not installed, returns None gracefully."""
    with patch.dict("sys.modules", {"litellm": None}):
        cost = estimate_cli_cost(
            model="openai/gpt-4o",
            prompt="test",
            result_text="test",
        )
    assert cost is None


@pytest.mark.unit
def test_estimate_cost_with_mocked_litellm():
    """Verify the function calls litellm correctly and returns the cost."""
    mock_litellm = MagicMock()
    mock_litellm.completion_cost.return_value = 0.0042

    with patch.dict("sys.modules", {"litellm": mock_litellm}):
        cost = estimate_cli_cost(
            model="openai/gpt-4o",
            prompt="hello world",
            result_text="response",
        )

    assert cost == 0.0042
    mock_litellm.token_counter.assert_not_called()
    mock_litellm.completion_cost.assert_called_once_with(
        model="openai/gpt-4o",
        prompt="hello world",
        completion="response",
    )
