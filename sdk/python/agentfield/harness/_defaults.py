"""Default harness provider selection.

`aforge` is AgentField's native harness. Provider precedence is
explicit value > AGENTFIELD_HARNESS_PROVIDER env var > DEFAULT_HARNESS_PROVIDER.
"""

from __future__ import annotations

import os
from typing import Optional

DEFAULT_HARNESS_PROVIDER = "aforge"
HARNESS_PROVIDER_ENV_VAR = "AGENTFIELD_HARNESS_PROVIDER"


def resolve_harness_provider(explicit: Optional[str] = None) -> str:
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    env_provider = os.environ.get(HARNESS_PROVIDER_ENV_VAR)
    if isinstance(env_provider, str) and env_provider.strip():
        return env_provider.strip()

    return DEFAULT_HARNESS_PROVIDER
