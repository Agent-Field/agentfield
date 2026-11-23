import pytest

from agentfield.types import AIConfig


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5",
        "openai/gpt-5-mini",
        "o3-mini",
        "openai/o4-mini",
        "gpt-5.5",
        "anthropic/claude-opus-5",
        "claude-sonnet-5-20260101",
    ],
)
def test_temperature_omitted_for_models_that_reject_it(model):
    cfg = AIConfig(model=model, temperature=0.3)
    assert "temperature" not in cfg.get_litellm_params()
    assert "temperature" not in cfg.get_litellm_params(temperature=0.9)


@pytest.mark.parametrize(
    "model",
    ["gpt-4o", "openai/gpt-5.2", "gpt-5.4", "omni-moderation", "claude-sonnet-4-5"],
)
def test_temperature_kept_for_models_that_accept_it(model):
    cfg = AIConfig(model=model, temperature=0.3)
    assert cfg.get_litellm_params()["temperature"] == 0.3
    assert cfg.get_litellm_params(temperature=0.9)["temperature"] == 0.9
