"""
Shared retry policy for OpenRouter's transient "no endpoints" 404.

OpenRouter occasionally routes an image-generation request to an upstream
replica that does not expose the image output modality, and answers with a 404
whose body carries the ``No endpoints found that support the requested output
modalities`` message (issues #586 and #588). The same 404 is returned
deterministically when ``image_config`` is set and no upstream provider in the
model's matrix accepts that parameter.

Both the LiteLLM path (:mod:`agentfield.vision`) and the direct-HTTP provider
path (:class:`agentfield.media_providers.OpenRouterProvider`) share the marker
and the backoff schedule defined here.
"""

import asyncio

# Substring identifying OpenRouter's "no upstream provider" 404. Matched against
# the exception text on the LiteLLM path and against the response body on the
# direct-HTTP provider path.
NO_ENDPOINTS_MARKER = "No endpoints found that support the requested output modalities"

# Sleeps between the 3 in-loop attempts.
# Sequence: attempt 1 -> sleep 1s -> attempt 2 -> sleep 2s -> attempt 3 -> (sleep
# 4s -> strip-and-retry) if image_config was set, else give up.
NO_ENDPOINTS_TOTAL_ATTEMPTS = 3
NO_ENDPOINTS_INTER_SLEEPS = (1.0, 2.0)
NO_ENDPOINTS_STRIP_SLEEP = 4.0


def is_no_endpoints_error(text: str) -> bool:
    """Return True if *text* carries OpenRouter's "no endpoints found" marker."""
    return NO_ENDPOINTS_MARKER in (text or "")


async def sleep(seconds: float) -> None:
    """Back off between retries.

    Indirection kept at module level so tests can patch the backoff
    (``agentfield.openrouter_retry.sleep``) without patching ``asyncio.sleep``
    process-wide.
    """
    await asyncio.sleep(seconds)
