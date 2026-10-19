import pytest
from agentfield.cost_tracker import derive_provider
from agentfield.usage_routing import routing_provider


@pytest.mark.parametrize(
    "model,endpoint,want",
    [
        ("openai/gpt-4o", "https://openrouter.ai/api/v1", "openrouter"),
        ("openrouter/anthropic/claude", None, "openrouter"),
        ("gemini/gemini-pro", None, "google"),
        ("openai/gpt-4o", None, "openai"),
        ("openrouter/claude", "https://private.example/openrouter.ai", "other"),
        ("openai/gpt", "https://fooapi.openai.com/v1", "other"),
        ("openai/gpt", "not-a-url", "unknown"),
    ],
)
def test_request_endpoint_overrides_adapter_and_model_vendor(model, endpoint, want):
    assert routing_provider(derive_provider(model), endpoint) == want
