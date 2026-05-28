"""Tests for DID authentication module (agentfield/did_auth.py)."""

import base64
import hashlib
import json
import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from agentfield.did_auth import (
    HEADER_CALLER_DID,
    HEADER_DID_SIGNATURE,
    HEADER_DID_TIMESTAMP,
    HEADER_DID_NONCE,
    DIDAuthenticator,
    _load_ed25519_private_key,
    create_did_auth_headers,
    sign_request,
)


# ---------------------------------------------------------------------------
# Helpers: generate a real Ed25519 key pair in JWK format
# ---------------------------------------------------------------------------

def _generate_ed25519_jwk():
    """Generate a fresh Ed25519 key pair and return (private_key_jwk_str, public_key_obj, private_key_obj)."""
    private_key = Ed25519PrivateKey.generate()

    # Extract raw 32-byte private key seed
    raw_private = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )

    # Extract raw 32-byte public key
    raw_public = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    # Build JWK (base64url without padding)
    d_b64 = base64.urlsafe_b64encode(raw_private).rstrip(b"=").decode("ascii")
    x_b64 = base64.urlsafe_b64encode(raw_public).rstrip(b"=").decode("ascii")

    jwk = {
        "kty": "OKP",
        "crv": "Ed25519",
        "d": d_b64,
        "x": x_b64,
    }
    return json.dumps(jwk), private_key.public_key(), private_key


@pytest.fixture
def ed25519_jwk():
    """Fixture providing (jwk_str, public_key, private_key)."""
    return _generate_ed25519_jwk()


@pytest.fixture
def did_string():
    return "did:web:example.com:agents:test-agent"


# ===========================================================================
# Tests for _load_ed25519_private_key
# ===========================================================================

class TestLoadEd25519PrivateKey:
    """Tests for _load_ed25519_private_key."""

    def test_valid_jwk_loads_successfully(self, ed25519_jwk):
        jwk_str, _public_key, _private_key = ed25519_jwk
        loaded = _load_ed25519_private_key(jwk_str)
        assert loaded is not None
        # Verify the loaded key can sign
        sig = loaded.sign(b"test payload")
        assert len(sig) == 64  # Ed25519 signatures are 64 bytes

    def test_valid_jwk_as_dict(self, ed25519_jwk):
        """The function should also accept a dict, not just a string."""
        jwk_str, _, _ = ed25519_jwk
        jwk_dict = json.loads(jwk_str)
        loaded = _load_ed25519_private_key(jwk_dict)
        assert loaded is not None

    def test_loaded_key_matches_original(self, ed25519_jwk):
        """Signing with loaded key should be verifiable with the original public key."""
        jwk_str, public_key, _ = ed25519_jwk
        loaded = _load_ed25519_private_key(jwk_str)
        message = b"verify me"
        sig = loaded.sign(message)
        # If verify fails, it raises InvalidSignature
        public_key.verify(sig, message)

    def test_invalid_kty_raises_valueerror(self, ed25519_jwk):
        jwk_str, _, _ = ed25519_jwk
        jwk = json.loads(jwk_str)
        jwk["kty"] = "RSA"
        with pytest.raises(ValueError, match="Invalid key type"):
            _load_ed25519_private_key(json.dumps(jwk))

    def test_invalid_crv_raises_valueerror(self, ed25519_jwk):
        jwk_str, _, _ = ed25519_jwk
        jwk = json.loads(jwk_str)
        jwk["crv"] = "P-256"
        with pytest.raises(ValueError, match="Invalid key type"):
            _load_ed25519_private_key(json.dumps(jwk))

    def test_missing_kty_raises_valueerror(self):
        jwk = {"crv": "Ed25519", "d": "AAAA"}
        with pytest.raises(ValueError, match="Invalid key type"):
            _load_ed25519_private_key(json.dumps(jwk))

    def test_missing_crv_raises_valueerror(self):
        jwk = {"kty": "OKP", "d": "AAAA"}
        with pytest.raises(ValueError, match="Invalid key type"):
            _load_ed25519_private_key(json.dumps(jwk))

    def test_missing_d_field_raises_valueerror(self, ed25519_jwk):
        jwk_str, _, _ = ed25519_jwk
        jwk = json.loads(jwk_str)
        del jwk["d"]
        with pytest.raises(ValueError, match="Missing 'd'"):
            _load_ed25519_private_key(json.dumps(jwk))

    def test_empty_d_field_raises_valueerror(self):
        jwk = {"kty": "OKP", "crv": "Ed25519", "d": ""}
        with pytest.raises(ValueError, match="Missing 'd'"):
            _load_ed25519_private_key(json.dumps(jwk))

    def test_invalid_base64_d_field_raises(self):
        """An invalid base64 value for 'd' that decodes to the wrong number of bytes."""
        jwk = {"kty": "OKP", "crv": "Ed25519", "d": "!!!not-base64!!!"}
        with pytest.raises(Exception):
            _load_ed25519_private_key(json.dumps(jwk))

    def test_wrong_length_d_field_raises(self):
        """A valid base64 string that decodes to the wrong byte length for Ed25519."""
        # 16 bytes instead of 32
        bad_d = base64.urlsafe_b64encode(b"\x00" * 16).rstrip(b"=").decode()
        jwk = {"kty": "OKP", "crv": "Ed25519", "d": bad_d}
        with pytest.raises(Exception):
            _load_ed25519_private_key(json.dumps(jwk))

    def test_invalid_json_raises_valueerror(self):
        with pytest.raises(ValueError, match="Invalid JWK format"):
            _load_ed25519_private_key("{not valid json")

    def test_non_json_string_raises_valueerror(self):
        with pytest.raises(ValueError, match="Invalid JWK format"):
            _load_ed25519_private_key("just a plain string")


# ===========================================================================
# Tests for sign_request
# ===========================================================================

class TestSignRequest:
    """Tests for the sign_request function."""

    def test_returns_four_element_tuple(self, ed25519_jwk, did_string):
        jwk_str, _, _ = ed25519_jwk
        result = sign_request(b"hello", jwk_str, did_string)
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_signature_is_valid_base64(self, ed25519_jwk, did_string):
        jwk_str, _, _ = ed25519_jwk
        sig_b64, _, _, _ = sign_request(b"body", jwk_str, did_string)
        # Should decode without error
        decoded = base64.b64decode(sig_b64)
        assert len(decoded) == 64  # Ed25519 signature

    def test_timestamp_is_recent(self, ed25519_jwk, did_string):
        jwk_str, _, _ = ed25519_jwk
        before = int(time.time())
        _, timestamp_str, _, _ = sign_request(b"body", jwk_str, did_string)
        after = int(time.time())
        ts = int(timestamp_str)
        assert before <= ts <= after

    def test_nonce_is_hex_string(self, ed25519_jwk, did_string):
        jwk_str, _, _ = ed25519_jwk
        _, _, nonce, _ = sign_request(b"body", jwk_str, did_string)
        assert len(nonce) == 32  # 16 bytes = 32 hex chars
        int(nonce, 16)  # Should not raise

    def test_did_is_returned_unchanged(self, ed25519_jwk, did_string):
        jwk_str, _, _ = ed25519_jwk
        _, _, _, returned_did = sign_request(b"body", jwk_str, did_string)
        assert returned_did == did_string

    def test_signature_verifies_with_public_key(self, ed25519_jwk, did_string):
        jwk_str, public_key, _ = ed25519_jwk
        body = b"some request body"
        sig_b64, timestamp_str, nonce, _ = sign_request(body, jwk_str, did_string)

        # Reconstruct the payload the same way sign_request does
        body_hash = hashlib.sha256(body).hexdigest()
        payload = f"{timestamp_str}:{nonce}:{body_hash}".encode("utf-8")

        sig_bytes = base64.b64decode(sig_b64)
        # verify raises InvalidSignature on failure
        public_key.verify(sig_bytes, payload)

    def test_different_bodies_produce_different_signatures(self, ed25519_jwk, did_string):
        jwk_str, _, _ = ed25519_jwk
        sig1, _, _, _ = sign_request(b"body_a", jwk_str, did_string)
        sig2, _, _, _ = sign_request(b"body_b", jwk_str, did_string)
        # Signatures should differ (different body hash)
        assert sig1 != sig2

    def test_same_body_produces_different_signatures_via_nonce(self, ed25519_jwk, did_string):
        """Two calls with the same body should produce different signatures due to nonce."""
        jwk_str, _, _ = ed25519_jwk
        sig1, _, nonce1, _ = sign_request(b"same body", jwk_str, did_string)
        sig2, _, nonce2, _ = sign_request(b"same body", jwk_str, did_string)
        assert nonce1 != nonce2
        assert sig1 != sig2

    def test_invalid_key_raises(self, did_string):
        bad_jwk = json.dumps({"kty": "OKP", "crv": "Ed25519"})
        with pytest.raises(ValueError):
            sign_request(b"body", bad_jwk, did_string)


# ===========================================================================
# Tests for create_did_auth_headers
# ===========================================================================

class TestCreateDIDAuthHeaders:
    """Tests for the create_did_auth_headers convenience function."""

    def test_returns_all_four_headers(self, ed25519_jwk, did_string):
        jwk_str, _, _ = ed25519_jwk
        headers = create_did_auth_headers(b"body", jwk_str, did_string)
        assert HEADER_CALLER_DID in headers
        assert HEADER_DID_SIGNATURE in headers
        assert HEADER_DID_TIMESTAMP in headers
        assert HEADER_DID_NONCE in headers

    def test_caller_did_matches_input(self, ed25519_jwk, did_string):
        jwk_str, _, _ = ed25519_jwk
        headers = create_did_auth_headers(b"body", jwk_str, did_string)
        assert headers[HEADER_CALLER_DID] == did_string

    def test_timestamp_header_is_numeric_string(self, ed25519_jwk, did_string):
        jwk_str, _, _ = ed25519_jwk
        headers = create_did_auth_headers(b"body", jwk_str, did_string)
        assert headers[HEADER_DID_TIMESTAMP].isdigit()

    def test_nonce_header_is_hex(self, ed25519_jwk, did_string):
        jwk_str, _, _ = ed25519_jwk
        headers = create_did_auth_headers(b"body", jwk_str, did_string)
        nonce = headers[HEADER_DID_NONCE]
        assert len(nonce) == 32
        int(nonce, 16)  # Should not raise


# ===========================================================================
# Tests for DIDAuthenticator
# ===========================================================================

class TestDIDAuthenticator:
    """Tests for the DIDAuthenticator class."""

    # --- Unconfigured state ---

    def test_default_not_configured(self):
        auth = DIDAuthenticator()
        assert auth.is_configured is False

    def test_default_did_is_none(self):
        auth = DIDAuthenticator()
        assert auth.did is None

    def test_unconfigured_sign_headers_returns_empty(self):
        auth = DIDAuthenticator()
        assert auth.sign_headers(b"body") == {}

    def test_did_only_not_configured(self):
        """Providing DID without key should NOT be configured."""
        auth = DIDAuthenticator(did="did:web:example")
        assert auth.is_configured is False

    def test_key_only_not_configured(self, ed25519_jwk):
        """Providing key without DID should NOT be configured."""
        jwk_str, _, _ = ed25519_jwk
        auth = DIDAuthenticator(private_key_jwk=jwk_str)
        assert auth.is_configured is False

    def test_invalid_key_not_configured(self):
        """Invalid key should leave authenticator unconfigured (logged warning)."""
        auth = DIDAuthenticator(
            did="did:web:example",
            private_key_jwk='{"kty":"RSA"}',
        )
        assert auth.is_configured is False

    # --- Configured state ---

    def test_configured_with_valid_credentials(self, ed25519_jwk, did_string):
        jwk_str, _, _ = ed25519_jwk
        auth = DIDAuthenticator(did=did_string, private_key_jwk=jwk_str)
        assert auth.is_configured is True
        assert auth.did == did_string

    def test_sign_headers_returns_all_headers(self, ed25519_jwk, did_string):
        jwk_str, _, _ = ed25519_jwk
        auth = DIDAuthenticator(did=did_string, private_key_jwk=jwk_str)
        headers = auth.sign_headers(b"body content")
        assert HEADER_CALLER_DID in headers
        assert HEADER_DID_SIGNATURE in headers
        assert HEADER_DID_TIMESTAMP in headers
        assert HEADER_DID_NONCE in headers
        assert headers[HEADER_CALLER_DID] == did_string

    def test_sign_headers_signature_is_verifiable(self, ed25519_jwk, did_string):
        jwk_str, public_key, _ = ed25519_jwk
        auth = DIDAuthenticator(did=did_string, private_key_jwk=jwk_str)
        body = b"test body"
        headers = auth.sign_headers(body)

        sig_bytes = base64.b64decode(headers[HEADER_DID_SIGNATURE])
        ts = headers[HEADER_DID_TIMESTAMP]
        nonce = headers[HEADER_DID_NONCE]
        body_hash = hashlib.sha256(body).hexdigest()
        payload = f"{ts}:{nonce}:{body_hash}".encode("utf-8")
        public_key.verify(sig_bytes, payload)

    # --- set_credentials ---

    def test_set_credentials_configures_authenticator(self, ed25519_jwk, did_string):
        auth = DIDAuthenticator()
        jwk_str, _, _ = ed25519_jwk
        result = auth.set_credentials(did_string, jwk_str)
        assert result is True
        assert auth.is_configured is True
        assert auth.did == did_string

    def test_set_credentials_with_invalid_key_returns_false(self):
        auth = DIDAuthenticator()
        result = auth.set_credentials("did:web:x", '{"kty":"RSA"}')
        assert result is False
        assert auth.is_configured is False

    def test_set_credentials_replaces_previous(self, did_string):
        jwk1, pub1, _ = _generate_ed25519_jwk()
        jwk2, pub2, _ = _generate_ed25519_jwk()

        auth = DIDAuthenticator(did="did:old", private_key_jwk=jwk1)
        assert auth.did == "did:old"

        auth.set_credentials(did_string, jwk2)
        assert auth.did == did_string
        # Verify signing uses the new key
        body = b"check"
        headers = auth.sign_headers(body)
        sig_bytes = base64.b64decode(headers[HEADER_DID_SIGNATURE])
        ts = headers[HEADER_DID_TIMESTAMP]
        nonce = headers[HEADER_DID_NONCE]
        body_hash = hashlib.sha256(body).hexdigest()
        payload = f"{ts}:{nonce}:{body_hash}".encode("utf-8")
        pub2.verify(sig_bytes, payload)

    # --- get_auth_info ---

    def test_get_auth_info_unconfigured(self):
        auth = DIDAuthenticator()
        info = auth.get_auth_info()
        assert info["configured"] is False
        assert info["did"] is None

    def test_get_auth_info_configured(self, ed25519_jwk, did_string):
        jwk_str, _, _ = ed25519_jwk
        auth = DIDAuthenticator(did=did_string, private_key_jwk=jwk_str)
        info = auth.get_auth_info()
        assert info["configured"] is True
        assert info["did"] == did_string


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Edge-case tests for body signing."""

    def test_empty_body(self, ed25519_jwk, did_string):
        jwk_str, public_key, _ = ed25519_jwk
        body = b""
        sig_b64, ts, nonce, _ = sign_request(body, jwk_str, did_string)
        body_hash = hashlib.sha256(body).hexdigest()
        payload = f"{ts}:{nonce}:{body_hash}".encode("utf-8")
        sig_bytes = base64.b64decode(sig_b64)
        public_key.verify(sig_bytes, payload)

    def test_large_body(self, ed25519_jwk, did_string):
        jwk_str, public_key, _ = ed25519_jwk
        body = b"x" * (1024 * 1024)  # 1 MB
        sig_b64, ts, nonce, _ = sign_request(body, jwk_str, did_string)
        body_hash = hashlib.sha256(body).hexdigest()
        payload = f"{ts}:{nonce}:{body_hash}".encode("utf-8")
        sig_bytes = base64.b64decode(sig_b64)
        public_key.verify(sig_bytes, payload)

    def test_non_ascii_body(self, ed25519_jwk, did_string):
        jwk_str, public_key, _ = ed25519_jwk
        body = "Unicode payload: \u00e9\u00e8\u00ea \u4e16\u754c \U0001f680".encode("utf-8")
        sig_b64, ts, nonce, _ = sign_request(body, jwk_str, did_string)
        body_hash = hashlib.sha256(body).hexdigest()
        payload = f"{ts}:{nonce}:{body_hash}".encode("utf-8")
        sig_bytes = base64.b64decode(sig_b64)
        public_key.verify(sig_bytes, payload)

    def test_binary_body(self, ed25519_jwk, did_string):
        jwk_str, public_key, _ = ed25519_jwk
        body = bytes(range(256))
        sig_b64, ts, nonce, _ = sign_request(body, jwk_str, did_string)
        body_hash = hashlib.sha256(body).hexdigest()
        payload = f"{ts}:{nonce}:{body_hash}".encode("utf-8")
        sig_bytes = base64.b64decode(sig_b64)
        public_key.verify(sig_bytes, payload)

    def test_sign_headers_empty_body_via_authenticator(self, ed25519_jwk, did_string):
        jwk_str, public_key, _ = ed25519_jwk
        auth = DIDAuthenticator(did=did_string, private_key_jwk=jwk_str)
        headers = auth.sign_headers(b"")
        assert HEADER_DID_SIGNATURE in headers
        sig_bytes = base64.b64decode(headers[HEADER_DID_SIGNATURE])
        ts = headers[HEADER_DID_TIMESTAMP]
        nonce = headers[HEADER_DID_NONCE]
        body_hash = hashlib.sha256(b"").hexdigest()
        payload = f"{ts}:{nonce}:{body_hash}".encode("utf-8")
        public_key.verify(sig_bytes, payload)
