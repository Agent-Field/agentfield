"""
Helpers for ``data:`` URLs.

Image / video / file models reached through OpenRouter (Gemini image models in
particular) return their output inline as ``data:<mime>;base64,<payload>``
rather than as an http(s) URL. Any code path that treats a media URL as
downloadable has to decode those locally instead of handing them to
``requests`` — which raises ``InvalidSchema`` for a ``data:`` URL (issue #587).

Only the base64 form is supported. A ``data:`` URL that declares no base64
payload, or whose payload is empty, is rejected loudly by
:func:`decode_data_url` rather than decoded into wrong-but-plausible bytes:
silently returning ``b""`` for ``data:image/png`` or ``data:image/png;base64,``
would be worse than the ``InvalidSchema`` these helpers replaced.

Decoding is strict about the base64 alphabet but *not* about layout: ASCII
whitespace is stripped from the payload first, so an RFC 2045 line-wrapped
payload, a trailing newline, or a space after the comma still decode — those
are all shapes real producers emit, and the permissive decoder these helpers
replaced accepted them.
"""

import base64
from typing import Any

# Cap on how much of a rejected URL is echoed back in the error. Only the
# metadata head (everything before the first comma) is ever reported, so a
# payload never reaches the message or the logs.
_MAX_REPORTED_HEAD = 64

# ASCII whitespace is not part of the base64 alphabet, so ``validate=True``
# rejects it — but RFC 2045 encoders wrap payloads at 76 columns and some
# producers pad the comma with a space. Deleting it keeps those decodable
# while a genuinely corrupt payload still raises.
_ASCII_WHITESPACE = {ord(c): None for c in " \t\n\r\v\f"}


def is_data_url(url: Any) -> bool:
    """Return True if *url* is a ``data:`` URL.

    URL schemes are case-insensitive (RFC 3986 §3.1), so ``DATA:`` and
    ``data:`` are both matched. Only the scheme is lower-cased — this runs on
    every ``get_bytes()`` / ``save()`` call and a media URL can carry a
    multi-megabyte inline payload, so the whole string is never copied.
    """
    return isinstance(url, str) and url[:5].lower() == "data:"


def data_url_mime_type(url: str) -> str:
    """Return the MIME type declared by a ``data:`` URL, lower-cased.

    Returns ``""`` when the URL declares none (``data:;base64,…``), which per
    RFC 2397 means ``text/plain;charset=US-ASCII``. Callers that care about the
    distinction supply their own default.
    """
    head = url.partition(",")[0]
    return head[len("data:") :].partition(";")[0].strip().lower()


def decode_data_url(url: str) -> bytes:
    """Return the decoded bytes carried inline by a base64 ``data:`` URL.

    ASCII whitespace in the payload (line wrapping, a trailing newline, a space
    after the comma) is ignored.

    Raises:
        ValueError: if *url* has no ``,`` separator, its metadata head does not
            declare ``;base64``, or the payload is empty — i.e. it is not a
            form this helper can decode into real bytes.
        binascii.Error: if the payload is declared base64 but is corrupt.
            (``binascii.Error`` subclasses ``ValueError``.)
    """
    head, sep, payload = url.partition(",")
    reported = head[:_MAX_REPORTED_HEAD] + (
        "..." if len(head) > _MAX_REPORTED_HEAD else ""
    )
    if not sep or ";base64" not in head.lower():
        raise ValueError(
            f"Unsupported data: URL (expected base64 payload): {reported!r}"
        )
    payload = payload.translate(_ASCII_WHITESPACE)
    if not payload:
        raise ValueError(f"Empty data: URL payload: {reported!r}")
    return base64.b64decode(payload, validate=True)
