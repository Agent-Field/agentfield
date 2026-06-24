"""Unit tests for DID-based payload encryption (agentfield.crypto)."""

import base64
import json

import pytest

from agentfield import crypto
from agentfield.crypto import PayloadEncryptionError


pytestmark = pytest.mark.unit


def _did_document(public_jwk: dict, did: str = "did:key:zAgg") -> dict:
    """A minimal W3C DID document publishing an X25519 keyAgreement key."""
    return {
        "id": did,
        "verificationMethod": [
            {
                "id": f"{did}#key-agreement-1",
                "type": crypto.KEY_AGREEMENT_TYPE,
                "controller": did,
                "publicKeyJwk": public_jwk,
            }
        ],
        "keyAgreement": [f"{did}#key-agreement-1"],
    }


def test_roundtrip_dict_payload():
    priv, pub = crypto.generate_x25519_keypair()
    payload = {"scope": {"tier": "project", "workspaceId": "ws_1", "projectId": "p_2"}}

    token = crypto.encrypt_to_jwk(pub, payload)
    plaintext = crypto.decrypt(token, priv)

    assert json.loads(plaintext) == payload


def test_roundtrip_str_and_bytes():
    priv, pub = crypto.generate_x25519_keypair()
    assert crypto.decrypt(crypto.encrypt_to_jwk(pub, "hello"), priv) == b"hello"
    assert crypto.decrypt(crypto.encrypt_to_jwk(pub, b"raw"), priv) == b"raw"


def test_encrypt_for_did_resolves_key_agreement():
    priv, pub = crypto.generate_x25519_keypair()
    doc = _did_document(pub)

    token = crypto.encrypt_for_did("did:key:zAgg", {"x": 1}, resolver=lambda _d: doc)

    assert json.loads(crypto.decrypt(token, priv)) == {"x": 1}


def test_encrypt_for_did_accepts_inline_key_agreement():
    priv, pub = crypto.generate_x25519_keypair()
    did = "did:key:zAgg"
    doc = {
        "id": did,
        "keyAgreement": [
            {
                "id": f"{did}#k",
                "type": crypto.KEY_AGREEMENT_TYPE,
                "publicKeyJwk": pub,
            }
        ],
    }
    token = crypto.encrypt_for_did(did, "inline", resolver=lambda _d: doc)
    assert crypto.decrypt(token, priv) == b"inline"


def test_encrypt_for_did_unwraps_resolve_response():
    """The control-plane resolve endpoint wraps the document under did_document."""
    priv, pub = crypto.generate_x25519_keypair()
    resolve_response = {"did": "did:key:zAgg", "did_document": _did_document(pub)}
    token = crypto.encrypt_for_did("did:key:zAgg", "wrapped", resolver=lambda _d: resolve_response)
    assert crypto.decrypt(token, priv) == b"wrapped"


def test_encrypt_for_did_accepts_flat_key_agreement():
    """The CP did:key resolve response returns a flat `key_agreement` JWK."""
    priv, pub = crypto.generate_x25519_keypair()
    resolve_response = {"did": "did:key:zAgg", "key_agreement": pub}
    token = crypto.encrypt_for_did("did:key:zAgg", "flat", resolver=lambda _d: resolve_response)
    assert crypto.decrypt(token, priv) == b"flat"


def test_wrong_key_cannot_decrypt():
    _priv, pub = crypto.generate_x25519_keypair()
    other_priv, _ = crypto.generate_x25519_keypair()
    token = crypto.encrypt_to_jwk(pub, "secret")

    with pytest.raises(PayloadEncryptionError):
        crypto.decrypt(token, other_priv)


def test_missing_key_agreement_raises():
    with pytest.raises(PayloadEncryptionError, match="no keyAgreement"):
        crypto.extract_key_agreement_jwk({"id": "did:key:z", "verificationMethod": []})


def test_unresolvable_did_raises():
    with pytest.raises(PayloadEncryptionError, match="could not resolve"):
        crypto.encrypt_for_did("did:key:zMissing", "x", resolver=lambda _d: None)


def test_non_x25519_key_rejected():
    with pytest.raises(PayloadEncryptionError, match="X25519"):
        crypto.encrypt_to_jwk({"kty": "OKP", "crv": "Ed25519", "x": "AAAA"}, "x")


def test_load_private_key_from_value():
    priv, pub = crypto.generate_x25519_keypair()
    key = crypto.load_private_key(value=json.dumps(priv))
    token = crypto.encrypt_to_jwk(pub, "viaenv")
    assert crypto.decrypt(token, key) == b"viaenv"


def test_load_private_key_missing_env(monkeypatch):
    monkeypatch.delenv(crypto.DEFAULT_PRIVATE_KEY_ENV, raising=False)
    with pytest.raises(PayloadEncryptionError, match="no X25519 private key"):
        crypto.load_private_key()


# --- JWS signing: payload authenticity (Ed25519 / EdDSA) --------------------

# A compact JWS produced by the TypeScript SDK (jose CompactSign, alg=EdDSA) over
# {"scope":{"tier":"workspace","workspace_id":"ws_demo"}}, signed with the Ed25519
# key whose public half is _ED25519_PUB. Frozen here so cross-language interop
# (TS sign -> Python verify) is asserted without a node toolchain at test time.
_ED25519_PUB = {
    "crv": "Ed25519",
    "x": "LZG6OR6azeoR7_cVXjZOTY1wBIpFp-loce90N1_XOiY",
    "kty": "OKP",
}
_TS_JWS_VECTOR = (
    "eyJhbGciOiJFZERTQSJ9"
    ".eyJzY29wZSI6eyJ0aWVyIjoid29ya3NwYWNlIiwid29ya3NwYWNlX2lkIjoid3NfZGVtbyJ9fQ"
    ".xNm_F_9qbMP4QFbFEHD5WRj-7CcIXPz5uNz3JoXddktyYWPXCC5Y0xgfms41h50aJem4fiPjqv-zaUeoFhPyDw"
)


def test_sign_verify_roundtrip_dict():
    priv, pub = crypto.generate_ed25519_keypair()
    payload = {"scope": {"tier": "project", "workspace_id": "ws_1", "project_id": "p_2"}}
    assert json.loads(crypto.verify(crypto.sign(payload, priv), pub)) == payload


def test_sign_verify_str_and_bytes():
    priv, pub = crypto.generate_ed25519_keypair()
    assert crypto.verify(crypto.sign("hello", priv), pub) == b"hello"
    assert crypto.verify(crypto.sign(b"raw", priv), pub) == b"raw"


def test_ts_signed_jws_verifies_in_python():
    """A jose (TypeScript) compact JWS must verify here — cross-language interop."""
    assert json.loads(crypto.verify(_TS_JWS_VECTOR, _ED25519_PUB)) == {
        "scope": {"tier": "workspace", "workspace_id": "ws_demo"}
    }


def test_verify_rejects_wrong_key():
    _priv, other_pub = crypto.generate_ed25519_keypair()
    with pytest.raises(PayloadEncryptionError, match="signature verification failed"):
        crypto.verify(_TS_JWS_VECTOR, other_pub)


def test_verify_rejects_tampered_payload():
    head, body, sig = _TS_JWS_VECTOR.split(".")
    forged = f"{head}.{body[:-2]}AA.{sig}"
    with pytest.raises(PayloadEncryptionError):
        crypto.verify(forged, _ED25519_PUB)


def test_verify_rejects_alg_none():
    """An unsigned 'alg: none' token (empty signature) must not bypass verification."""
    _head, body, _sig = _TS_JWS_VECTOR.split(".")
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    # Real 'none' tokens carry an empty signature segment -> rejected as malformed;
    # a 'none' header kept with a non-empty signature is rejected at the alg gate
    # (see test_verify_rejects_non_eddsa_alg). Either way it never verifies.
    with pytest.raises(PayloadEncryptionError):
        crypto.verify(f"{header}.{body}.", _ED25519_PUB)


def test_verify_rejects_non_eddsa_alg():
    _head, body, sig = _TS_JWS_VECTOR.split(".")
    header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').rstrip(b"=").decode()
    with pytest.raises(PayloadEncryptionError, match="alg"):
        crypto.verify(f"{header}.{body}.{sig}", _ED25519_PUB)


def test_verify_rejects_malformed():
    with pytest.raises(PayloadEncryptionError, match="3 parts"):
        crypto.verify("only.two", _ED25519_PUB)


def test_sign_requires_private_key():
    _priv, pub = crypto.generate_ed25519_keypair()
    with pytest.raises(PayloadEncryptionError, match="private key"):
        crypto.sign("x", pub)  # a public-only JWK cannot sign
