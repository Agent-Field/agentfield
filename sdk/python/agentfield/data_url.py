"""
Helpers for ``data:`` URLs.

Image / video / file models reached through OpenRouter (Gemini image models in
particular) return their output inline as ``data:<mime>;base64,<payload>``
rather than as an http(s) URL. Any code path that treats a media URL as
downloadable has to decode those locally instead of handing them to
``requests`` — which raises ``InvalidSchema`` for a ``data:`` URL (issue #587).
"""

import base64
from typing import Any


def is_data_url(url: Any) -> bool:
    """Return True if *url* is a ``data:`` URL."""
    return isinstance(url, str) and url.startswith("data:")


def decode_data_url(url: str) -> bytes:
    """Return the decoded bytes carried inline by a base64 ``data:`` URL."""
    _, _, payload = url.partition(",")
    return base64.b64decode(payload)
