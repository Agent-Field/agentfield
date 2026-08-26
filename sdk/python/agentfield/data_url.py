"""
Helpers for ``data:`` URLs.

Image / video / file models reached through OpenRouter (Gemini image models in
particular) return their output inline as ``data:<mime>;base64,<payload>``
rather than as an http(s) URL. Any code path that treats a media URL as
downloadable has to decode those locally instead of handing them to
``requests`` — which raises ``InvalidSchema`` for a ``data:`` URL (issue #587).

Only the base64 form is supported. A ``data:`` URL with a percent-encoded (or
missing) payload is rejected loudly by :func:`decode_data_url` rather than
decoded into wrong-but-plausible bytes: silently returning ``b""`` for
``data:image/png`` would be worse than the ``InvalidSchema`` these helpers
replaced.
"""

import base64
from typing import Any

# Cap on how much of a rejected URL is echoed back in the error. Only the
# metadata head (everything before the first comma) is ever reported, so a
# payload never reaches the message or the logs.
_MAX_REPORTED_HEAD = 64


def is_data_url(url: Any) -> bool:
    """Return True if *url* is a ``data:`` URL.

    URL schemes are case-insensitive (RFC 3986 §3.1), so ``DATA:`` and
    ``data:`` are both matched.
    """
    return isinstance(url, str) and url.lower().startswith("data:")


def decode_data_url(url: str) -> bytes:
    """Return the decoded bytes carried inline by a base64 ``data:`` URL.

    Raises:
        ValueError: if *url* has no ``,`` separator or its metadata head does
            not declare ``;base64`` — i.e. it is not a form this helper can
            decode.
        binascii.Error: if the payload is declared base64 but is corrupt.
    """
    head, sep, payload = url.partition(",")
    if not sep or ";base64" not in head.lower():
        reported = head[:_MAX_REPORTED_HEAD] + (
            "..." if len(head) > _MAX_REPORTED_HEAD else ""
        )
        raise ValueError(
            f"Unsupported data: URL (expected base64 payload): {reported!r}"
        )
    return base64.b64decode(payload, validate=True)
