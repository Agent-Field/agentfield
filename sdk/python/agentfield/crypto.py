"""DID-based payload encryption (JWE over X25519 keyAgreement keys).

This module lets one party encrypt a payload *to* an agent's DID such that only
that agent — the holder of the matching X25519 private key — can decrypt it. It
underpins the discuss/aggregator split: hax-sdk (or any caller) encrypts a scoped
payload to the aggregator's DID; the untrusted discuss agent forwards the
ciphertext but cannot read it; only the aggregator decrypts.

Wire format is standard **JWE compact, ``ECDH-ES`` + ``A256GCM``** over an X25519
key (RFC 7518 / RFC 8037). This is interoperable with the TypeScript SDK's
``encryptForDid`` (which uses ``jose``) — a ciphertext produced there decrypts
here and vice-versa.

Key-ownership model: the agent **owns** its keypair. The X25519 private key lives
in the agent's environment (see :func:`load_private_key`); the public key is
published in the agent's DID document as a ``keyAgreement`` verification method of
type ``X25519KeyAgreementKey2020``. The control plane never holds the private key.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Callable, Dict, Optional, Union

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from joserfc import jwe
from joserfc.jwk import OKPKey

__all__ = [
    "JWE_ALG",
    "JWE_ENC",
    "JWS_ALG",
    "KEY_AGREEMENT_TYPE",
    "DEFAULT_PRIVATE_KEY_ENV",
    "generate_x25519_keypair",
    "generate_ed25519_keypair",
    "load_private_key",
    "extract_key_agreement_jwk",
    "encrypt_to_jwk",
    "encrypt_for_did",
    "decrypt",
    "sign",
    "verify",
    "PayloadEncryptionError",
]

# JOSE algorithm parameters. Must match the TypeScript SDK exactly for interop.
JWE_ALG = "ECDH-ES"
JWE_ENC = "A256GCM"

# JWS signature algorithm for payload *authenticity* (EdDSA over an Ed25519 key).
# Encryption (JWE) gives confidentiality only — anyone with the recipient's public
# key can encrypt. Signing lets the recipient verify the payload's ORIGIN: only the
# holder of the signing private key (e.g. hax-sdk) could have produced it. Sign then
# encrypt (JWS-in-JWE). Pinned to EdDSA so a forged header cannot downgrade to
# "none" or another algorithm.
JWS_ALG = "EdDSA"

# W3C verification-method type published in the DID document for the X25519 key.
KEY_AGREEMENT_TYPE = "X25519KeyAgreementKey2020"

# Default env var holding the agent's X25519 private key (a JWK JSON string).
DEFAULT_PRIVATE_KEY_ENV = "AGENTFIELD_X25519_PRIVATE_KEY"


class PayloadEncryptionError(Exception):
    """Raised when encrypting to / decrypting from a DID fails."""


# A resolver fetches a DID document (or resolve-response) for a given DID string.
DIDResolver = Callable[[str], Optional[Dict[str, Any]]]
Payload = Union[bytes, str, Dict[str, Any]]


def generate_x25519_keypair() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate a fresh X25519 keypair for an agent.

    Returns ``(private_jwk, public_jwk)`` as plain dicts. Persist ``private_jwk``
    into the agent's environment (e.g. ``AGENTFIELD_X25519_PRIVATE_KEY``) and
    publish ``public_jwk`` as the agent's DID ``keyAgreement`` key.
    """
    key = OKPKey.generate_key("X25519", private=True)
    return key.as_dict(private=True), key.as_dict(private=False)


def load_private_key(
    env_var: str = DEFAULT_PRIVATE_KEY_ENV,
    *,
    value: Optional[str] = None,
) -> OKPKey:
    """Load the agent's X25519 private key from the environment (or an explicit value).

    Accepts a full JWK JSON object (``{"kty":"OKP","crv":"X25519","x":...,"d":...}``).
    """
    raw = value if value is not None else os.environ.get(env_var)
    if not raw:
        raise PayloadEncryptionError(
            f"no X25519 private key found (env var {env_var!r} is unset)"
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PayloadEncryptionError(
            f"{env_var} must contain a JWK JSON object: {exc}"
        ) from exc
    if data.get("crv") != "X25519" or "d" not in data:
        raise PayloadEncryptionError(
            "private key JWK must be an X25519 OKP key with a 'd' (private) component"
        )
    try:
        return OKPKey.import_key(data)
    except Exception as exc:  # noqa: BLE001 - normalize to our error type
        raise PayloadEncryptionError(f"invalid X25519 private key: {exc}") from exc


def extract_key_agreement_jwk(did_document: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the X25519 ``keyAgreement`` public JWK out of a resolved DID document.

    Handles both a bare W3C DID document and a control-plane resolve response that
    wraps it under ``did_document``. A ``keyAgreement`` entry may be an inline
    verification method (with ``publicKeyJwk``) or a string reference into
    ``verificationMethod`` — both are supported.
    """
    if not isinstance(did_document, dict):
        raise PayloadEncryptionError("DID document must be a JSON object")
    doc = did_document.get("did_document", did_document)

    # Control-plane resolve responses for did:key return the X25519 public key as a
    # flat `key_agreement` JWK rather than a full W3C keyAgreement array.
    flat = doc.get("key_agreement")
    if isinstance(flat, dict) and flat.get("crv") == "X25519":
        return flat

    key_agreement = doc.get("keyAgreement")
    if not key_agreement:
        raise PayloadEncryptionError(
            "DID document has no keyAgreement key; the agent has not published an "
            "X25519 encryption key"
        )

    verification_methods = {
        vm.get("id"): vm
        for vm in doc.get("verificationMethod", [])
        if isinstance(vm, dict)
    }

    entry = key_agreement[0] if isinstance(key_agreement, list) else key_agreement
    if isinstance(entry, str):
        vm = verification_methods.get(entry)
        if vm is None:
            raise PayloadEncryptionError(
                f"keyAgreement references unknown verification method {entry!r}"
            )
    elif isinstance(entry, dict):
        vm = entry
    else:
        raise PayloadEncryptionError("unsupported keyAgreement entry shape")

    jwk = vm.get("publicKeyJwk")
    if not isinstance(jwk, dict) or jwk.get("crv") != "X25519":
        raise PayloadEncryptionError(
            "keyAgreement verification method does not carry an X25519 publicKeyJwk"
        )
    return jwk


def _payload_bytes(payload: Payload) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def encrypt_to_jwk(public_jwk: Dict[str, Any], payload: Payload) -> str:
    """Encrypt ``payload`` to an X25519 public JWK, returning a compact JWE string."""
    if not isinstance(public_jwk, dict) or public_jwk.get("crv") != "X25519":
        raise PayloadEncryptionError("public_jwk must be an X25519 OKP JWK")
    try:
        key = OKPKey.import_key(public_jwk)
        protected = {"alg": JWE_ALG, "enc": JWE_ENC}
        return jwe.encrypt_compact(protected, _payload_bytes(payload), key)
    except PayloadEncryptionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise PayloadEncryptionError(f"encryption failed: {exc}") from exc


def encrypt_for_did(did: str, payload: Payload, resolver: DIDResolver) -> str:
    """Resolve ``did``, read its X25519 keyAgreement key, and encrypt ``payload`` to it.

    ``resolver`` maps a DID string to its document — e.g.
    ``DIDManager.resolve_did``. Returns a compact JWE that only the holder of the
    matching private key (the DID's owner) can decrypt.
    """
    document = resolver(did)
    if document is None:
        raise PayloadEncryptionError(f"could not resolve DID {did!r}")
    public_jwk = extract_key_agreement_jwk(document)
    return encrypt_to_jwk(public_jwk, payload)


def decrypt(token: str, private_key: Union[OKPKey, Dict[str, Any], str]) -> bytes:
    """Decrypt a compact JWE produced by :func:`encrypt_for_did` / the TS SDK.

    ``private_key`` may be a loaded :class:`OKPKey`, a JWK dict, or a JWK JSON
    string. Returns the raw plaintext bytes (JSON callers can ``json.loads``).
    """
    if isinstance(private_key, OKPKey):
        key = private_key
    elif isinstance(private_key, dict):
        key = OKPKey.import_key(private_key)
    else:
        key = load_private_key(value=private_key)
    try:
        return jwe.decrypt_compact(token, key).plaintext
    except Exception as exc:  # noqa: BLE001
        raise PayloadEncryptionError(f"decryption failed: {exc}") from exc


# --- JWS signing: payload authenticity (Ed25519 / EdDSA) --------------------


def generate_ed25519_keypair() -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Generate a fresh Ed25519 signing keypair.

    Returns ``(private_jwk, public_jwk)`` as plain dicts. Hold ``private_jwk`` in
    the signing party's environment (e.g. hax-sdk's ``HAX_SDK_SIGNING_PRIVATE_KEY``)
    and pin ``public_jwk`` as the verifier's trust anchor (e.g. the aggregator's
    ``HAX_SDK_SIGNING_PUBLIC_KEY``).
    """
    key = OKPKey.generate_key("Ed25519", private=True)
    return key.as_dict(private=True), key.as_dict(private=False)


def _b64u_decode(segment: str) -> bytes:
    """Decode a base64url segment, tolerating absent padding (per RFC 7515)."""
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def _b64u_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _ed25519_jwk(key: Union[OKPKey, Dict[str, Any], str]) -> Dict[str, Any]:
    """Normalise an Ed25519 key (OKPKey / JWK dict / JWK JSON string) to a dict."""
    if isinstance(key, OKPKey):
        return key.as_dict(private=key.is_private)
    if isinstance(key, dict):
        return key
    try:
        return json.loads(key)
    except (TypeError, json.JSONDecodeError) as exc:
        raise PayloadEncryptionError(
            "Ed25519 key must be an OKPKey, JWK dict, or JWK JSON string"
        ) from exc


def sign(payload: Payload, private_key: Union[OKPKey, Dict[str, Any], str]) -> str:
    """Sign ``payload`` with an Ed25519 private key, returning a compact JWS.

    Wire-compatible with the TypeScript SDK's compact JWS (``jose`` ``CompactSign``
    with ``{alg: "EdDSA"}``, RFC 7515 + RFC 8037): a token signed there verifies
    here and vice-versa. Implemented directly over ``cryptography`` rather than via
    ``joserfc`` so the EdDSA wire name stays stable for cross-language interop.
    """
    jwk = _ed25519_jwk(private_key)
    if jwk.get("crv") != "Ed25519" or "d" not in jwk:
        raise PayloadEncryptionError(
            "signing requires an Ed25519 private key (JWK with a 'd' component)"
        )
    try:
        priv = Ed25519PrivateKey.from_private_bytes(_b64u_decode(jwk["d"]))
    except Exception as exc:  # noqa: BLE001
        raise PayloadEncryptionError(f"invalid Ed25519 private key: {exc}") from exc
    protected = _b64u_encode(
        json.dumps({"alg": JWS_ALG}, separators=(",", ":")).encode("utf-8")
    )
    body = _b64u_encode(_payload_bytes(payload))
    signing_input = f"{protected}.{body}".encode("ascii")
    return f"{protected}.{body}.{_b64u_encode(priv.sign(signing_input))}"


def verify(token: str, public_key: Union[OKPKey, Dict[str, Any], str]) -> bytes:
    """Verify a compact JWS against an Ed25519 public key; return the payload bytes.

    Raises :class:`PayloadEncryptionError` on a missing, malformed, or invalid
    signature, or a non-EdDSA algorithm. The ``alg`` header is pinned to ``EdDSA``
    so a forged header (e.g. ``none``) cannot bypass verification.
    """
    jwk = _ed25519_jwk(public_key)
    if jwk.get("crv") != "Ed25519" or "x" not in jwk:
        raise PayloadEncryptionError("verification requires an Ed25519 public key")
    parts = token.split(".") if isinstance(token, str) else []
    if len(parts) != 3 or not all(parts):
        raise PayloadEncryptionError("malformed compact JWS (expected 3 parts)")
    protected_b64, body_b64, signature_b64 = parts
    try:
        header = json.loads(_b64u_decode(protected_b64))
    except Exception as exc:  # noqa: BLE001
        raise PayloadEncryptionError(f"invalid JWS header: {exc}") from exc
    if not isinstance(header, dict) or header.get("alg") != JWS_ALG:
        raise PayloadEncryptionError(
            f"unexpected JWS alg {header.get('alg') if isinstance(header, dict) else None!r}; "
            f"expected {JWS_ALG}"
        )
    try:
        pub = Ed25519PublicKey.from_public_bytes(_b64u_decode(jwk["x"]))
    except Exception as exc:  # noqa: BLE001
        raise PayloadEncryptionError(f"invalid Ed25519 public key: {exc}") from exc
    signing_input = f"{protected_b64}.{body_b64}".encode("ascii")
    try:
        pub.verify(_b64u_decode(signature_b64), signing_input)
    except InvalidSignature as exc:
        raise PayloadEncryptionError("signature verification failed") from exc
    except Exception as exc:  # noqa: BLE001
        raise PayloadEncryptionError(f"signature verification failed: {exc}") from exc
    return _b64u_decode(body_b64)
