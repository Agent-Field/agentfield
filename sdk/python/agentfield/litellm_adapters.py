"""
LiteLLM Provider Adapters

This module centralizes provider-specific parameter transformations and patches
required to ensure compatibility across different LLM providers.

Each patch should be:
1. Well-documented with the reason for its existence
2. Tied to specific providers/models that require it
3. Transparent about what transformation is being applied

This abstraction allows the core SDK to remain clean while handling necessary
provider-specific quirks in one maintainable location.
"""

from typing import Dict, Any


def get_provider_from_model(model: str) -> str:
    """
    Extract provider name from model string.

    LiteLLM uses the format "provider/model-name" (e.g., "openai/gpt-4o").
    This function extracts the provider prefix.

    Args:
        model: Model string in LiteLLM format

    Returns:
        Provider name (e.g., "openai", "anthropic", "cohere")
        Returns "unknown" if format doesn't match

    Examples:
        >>> get_provider_from_model("openai/gpt-4o")
        'openai'
        >>> get_provider_from_model("anthropic/claude-3-opus")
        'anthropic'
        >>> get_provider_from_model("gpt-4o")
        'unknown'
    """
    if "/" in model:
        return model.split("/")[0]
    return "unknown"


def apply_openai_patches(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply OpenAI-specific parameter patches.

    **Patch 1: max_tokens → max_completion_tokens**

    Reason: OpenAI's newer models (gpt-4o, gpt-4o-mini, etc.) use
    `max_completion_tokens` instead of `max_tokens` to disambiguate between
    input tokens and output tokens. LiteLLM may not always handle this
    transformation automatically for all OpenAI models.

    This patch ensures compatibility by renaming the parameter when targeting
    OpenAI models.

    Reference: https://platform.openai.com/docs/api-reference/chat/create

    Args:
        params: Parameter dictionary to transform

    Returns:
        Transformed parameter dictionary
    """
    # Create a copy to avoid mutating the original
    patched = params.copy()

    # Patch: max_tokens → max_completion_tokens for OpenAI
    if "max_tokens" in patched:
        patched["max_completion_tokens"] = patched.pop("max_tokens")

    return patched


def apply_provider_patches(params: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Apply provider-specific parameter transformations.

    This is the main entry point for all provider-specific patches. It detects
    the provider from the model string and applies appropriate transformations.

    **When to add a new patch:**
    1. A specific provider requires a different parameter name
    2. A provider has parameter constraints that differ from LiteLLM defaults
    3. There's a known incompatibility that needs a workaround

    **How to add a new patch:**
    1. Create a new function: `apply_{provider}_patches(params)`
    2. Document the patch reason and affected models
    3. Add a new elif branch in this function

    Args:
        params: Parameter dictionary from AIConfig.get_litellm_params()
        model: Model string (e.g., "openai/gpt-4o")

    Returns:
        Transformed parameter dictionary with provider-specific patches applied

    Examples:
        >>> params = {"model": "openai/gpt-4o", "max_tokens": 1000}
        >>> apply_provider_patches(params, "openai/gpt-4o")
        {'model': 'openai/gpt-4o', 'max_completion_tokens': 1000}
    """
    provider = get_provider_from_model(model)

    # Apply provider-specific patches
    if provider == "openai":
        return apply_openai_patches(params)

    # Add more providers here as needed:
    # elif provider == "anthropic":
    #     return apply_anthropic_patches(params)
    # elif provider == "cohere":
    #     return apply_cohere_patches(params)

    # No patches needed for this provider
    return params


def filter_none_values(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove None values from parameter dictionary.

    This ensures we only pass explicitly set parameters to LiteLLM,
    allowing it to use its own defaults for unset values.

    Args:
        params: Parameter dictionary potentially containing None values

    Returns:
        Dictionary with None values removed

    Examples:
        >>> filter_none_values({"a": 1, "b": None, "c": "test"})
        {'a': 1, 'c': 'test'}
    """
    return {k: v for k, v in params.items() if v is not None}
