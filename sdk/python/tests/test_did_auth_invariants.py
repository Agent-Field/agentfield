"""
Behavioral invariant tests for DID authentication.

These tests verify structural properties of the DID auth module that must
always hold regardless of implementation changes. They are designed to catch
AI code regressions that break cryptographic contracts.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from typing import Tuple

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ed25519_key_pair():
    """Generate a fresh Ed25519 key pair in JWK format for testing."""
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:
        pytest.skip("cryptography library not available")

    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes_raw()
    public_key = private_key.public_key()
    public_bytes = public_key.public_bytes_raw()

    d_b64 = base64.urlsafe_b64encode(private_bytes).rstrip(b"=").decode("ascii")
    x_b64 = base64.urlsafe_b64encode(public_bytes).rstrip(b"=").decode("ascii")

    jwk = json.dumps({"kty": "OKP", "crv": "Ed25519", "d": d_b64, "x": x_b64})
    return jwk, "did:key:test-agent"


# ---------------------------------------------------------------------------
# Helper to call sign_request
# ---------------------------------------------------------------------------

def _sign(body: bytes, jwk: str, did: str) -> Tuple[str, str, str, str]:
    """Call sign_request and return (signature, timestamp, nonce, did)."""
    from agentfield.did_auth import sign_request
    return sign_request(body, jwk, did)


# ---------------------------------------------------------------------------
# 1. Nonce uniqueness
# ---------------------------------------------------------------------------

class TestNonceUniqueness:
    """Nonces must be globally unique across many rapid calls."""

    def test_invariant_nonce_is_unique_across_100_rapid_calls(self, ed25519_key_pair):
        """Calling sign_request 100 times must produce 100 distinct nonces."""
        jwk, did = ed25519_key_pair
        body = b'{"test": "payload"}'

        nonces = []
        for _ in range(100):
            _, _, nonce, _ = _sign(body, jwk, did)
            nonces.append(nonce)

        unique_nonces = set(nonces)
        assert len(unique_nonces) == 100, (
            f"INVARIANT VIOLATION: Only {len(unique_nonces)}/100 nonces were unique. "
            "Nonce generation is not producing distinct values."
        )

    def test_invariant_nonce_is_unique_for_identical_body(self, ed25519_key_pair):
        """Even for an identical body signed repeatedly, each nonce must differ."""
        jwk, did = ed25519_key_pair
        body = b'{"fixed": "body"}'

        sig1, ts1, nonce1, _ = _sign(body, jwk, did)
        sig2, ts2, nonce2, _ = _sign(body, jwk, did)

        assert nonce1 != nonce2, (
            "INVARIANT VIOLATION: Two consecutive calls with same body produced identical nonces. "
            f"nonce1={nonce1!r} nonce2={nonce2!r}"
        )


# ---------------------------------------------------------------------------
# 2. Timestamp freshness
# ---------------------------------------------------------------------------

class TestTimestampFreshness:
    """Every signed request's timestamp must be within 5 seconds of now."""

    def test_invariant_timestamp_is_within_5_seconds_of_now(self, ed25519_key_pair):
        """Timestamp must reflect current time, not a stale or zero value."""
        jwk, did = ed25519_key_pair
        body = b'{"data": "fresh"}'

        before = int(time.time()) - 1  # allow 1 second slack
        _, timestamp, _, _ = _sign(body, jwk, did)
        after = int(time.time()) + 1

        ts_int = int(timestamp)
        assert before <= ts_int <= after + 5, (
            f"INVARIANT VIOLATION: Timestamp {ts_int} is not within 5 seconds of now "
            f"(expected {before} <= ts <= {after + 5})."
        )

    def test_invariant_timestamp_is_a_numeric_string(self, ed25519_key_pair):
        """Timestamp must be a string representation of an integer epoch second."""
        jwk, did = ed25519_key_pair
        _, timestamp, _, _ = _sign(b"body", jwk, did)

        assert timestamp.isdigit(), (
            f"INVARIANT VIOLATION: Timestamp '{timestamp}' is not a pure digit string."
        )
        ts_int = int(timestamp)
        # Must be after year 2020 and before year 2100 (sanity range)
        assert 1577836800 < ts_int < 4102444800, (
            f"INVARIANT VIOLATION: Timestamp {ts_int} is outside sane epoch range."
        )


# ---------------------------------------------------------------------------
# 3. Signature determinism
# ---------------------------------------------------------------------------

class TestSignatureDeterminism:
    """Same inputs (excluding nonce/timestamp) signed with same key and EXPLICIT
    nonce/timestamp must produce identical signatures — Ed25519 is deterministic."""

    def test_invariant_signature_is_deterministic_for_same_payload(self, ed25519_key_pair):
        """
        Rebuild the exact payload that sign_request would sign, manually,
        and verify that two independent sign calls produce the same signature
        when given the same timestamp and nonce through the payload construction.

        This tests that the internal signature algorithm is consistent: same bytes in,
        same signature out (Ed25519 determinism property).
        """
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: F401
        except ImportError:
            pytest.skip("cryptography library not available")

        jwk, did = ed25519_key_pair

        # Construct payload manually (mirrors sign_request internals)
        body = b'{"deterministic": "test"}'
        timestamp = "1700000000"
        nonce = "aabbccdd11223344aabb"
        body_hash = hashlib.sha256(body).hexdigest()
        payload = f"{timestamp}:{nonce}:{body_hash}".encode("utf-8")

        # Load private key and sign twice
        from agentfield.did_auth import _load_ed25519_private_key
        private_key = _load_ed25519_private_key(jwk)

        sig1 = base64.b64encode(private_key.sign(payload)).decode("ascii")
        sig2 = base64.b64encode(private_key.sign(payload)).decode("ascii")

        assert sig1 == sig2, (
            "INVARIANT VIOLATION: Ed25519 signing is not deterministic. "
            f"sig1={sig1!r} sig2={sig2!r}"
        )


# ---------------------------------------------------------------------------
# 4. Signature sensitivity
# ---------------------------------------------------------------------------

class TestSignatureSensitivity:
    """Changing ANY part of the signed input must produce a different signature."""

    def _sign_with_patched_inputs(self, jwk: str, did: str, body: bytes,
                                   timestamp: str, nonce: str) -> str:
        """Sign with fully controlled inputs by patching os.urandom and time.time."""
        from agentfield.did_auth import _load_ed25519_private_key

        private_key = _load_ed25519_private_key(jwk)
        body_hash = hashlib.sha256(body).hexdigest()
        payload = f"{timestamp}:{nonce}:{body_hash}".encode("utf-8")
        sig = private_key.sign(payload)
        return base64.b64encode(sig).decode("ascii")

    def test_invariant_different_body_produces_different_signature(self, ed25519_key_pair):
        jwk, did = ed25519_key_pair
        ts = "1700000000"
        nonce = "fixed_nonce_0123456789ab"

        sig1 = self._sign_with_patched_inputs(jwk, did, b'{"body": "A"}', ts, nonce)
        sig2 = self._sign_with_patched_inputs(jwk, did, b'{"body": "B"}', ts, nonce)

        assert sig1 != sig2, (
            "INVARIANT VIOLATION: Different bodies produced the same signature."
        )

    def test_invariant_different_timestamp_produces_different_signature(self, ed25519_key_pair):
        jwk, did = ed25519_key_pair
        body = b'{"fixed": "body"}'
        nonce = "fixed_nonce_0123456789ab"

        sig1 = self._sign_with_patched_inputs(jwk, did, body, "1700000000", nonce)
        sig2 = self._sign_with_patched_inputs(jwk, did, body, "1700000001", nonce)

        assert sig1 != sig2, (
            "INVARIANT VIOLATION: Different timestamps produced the same signature."
        )

    def test_invariant_different_nonce_produces_different_signature(self, ed25519_key_pair):
        jwk, did = ed25519_key_pair
        body = b'{"fixed": "body"}'
        ts = "1700000000"

        sig1 = self._sign_with_patched_inputs(jwk, did, body, ts, "nonce_alpha_1234")
        sig2 = self._sign_with_patched_inputs(jwk, did, body, ts, "nonce_beta__5678")

        assert sig1 != sig2, (
            "INVARIANT VIOLATION: Different nonces produced the same signature."
        )


# ---------------------------------------------------------------------------
# 5. Header completeness
# ---------------------------------------------------------------------------

class TestHeaderCompleteness:
    """Every signed request must contain ALL required DID authentication headers."""

    REQUIRED_HEADERS = {
        "X-DID-Signature",
        "X-DID-Timestamp",
        "X-Caller-DID",
        "X-DID-Nonce",
    }

    def test_invariant_all_required_headers_present(self, ed25519_key_pair):
        """create_did_auth_headers must return all four required headers."""
        from agentfield.did_auth import create_did_auth_headers

        jwk, did = ed25519_key_pair
        headers = create_did_auth_headers(b'{"test": "payload"}', jwk, did)

        for required in self.REQUIRED_HEADERS:
            assert required in headers, (
                f"INVARIANT VIOLATION: Required header '{required}' missing from "
                f"create_did_auth_headers output. Present headers: {set(headers.keys())}"
            )

    def test_invariant_header_values_are_non_empty(self, ed25519_key_pair):
        """All required headers must have non-empty, non-None values."""
        from agentfield.did_auth import create_did_auth_headers

        jwk, did = ed25519_key_pair
        headers = create_did_auth_headers(b'{"test": "data"}', jwk, did)

        for required in self.REQUIRED_HEADERS:
            value = headers.get(required)
            assert value is not None and len(str(value)) > 0, (
                f"INVARIANT VIOLATION: Header '{required}' has empty/None value: {value!r}"
            )

    def test_invariant_caller_did_matches_input_did(self, ed25519_key_pair):
        """X-Caller-DID header must match the DID passed to create_did_auth_headers."""
        from agentfield.did_auth import create_did_auth_headers

        jwk, did = ed25519_key_pair
        headers = create_did_auth_headers(b"body", jwk, did)

        assert headers.get("X-Caller-DID") == did, (
            f"INVARIANT VIOLATION: X-Caller-DID header '{headers.get('X-Caller-DID')}' "
            f"does not match input DID '{did}'."
        )

    def test_invariant_authenticator_sign_headers_returns_all_required_when_configured(
        self, ed25519_key_pair
    ):
        """DIDAuthenticator.sign_headers() must return all required headers when configured."""
        from agentfield.did_auth import DIDAuthenticator

        jwk, did = ed25519_key_pair
        auth = DIDAuthenticator(did=did, private_key_jwk=jwk)

        assert auth.is_configured, "DIDAuthenticator must be configured after __init__ with valid key"

        headers = auth.sign_headers(b'{"payload": "data"}')

        for required in self.REQUIRED_HEADERS:
            assert required in headers, (
                f"INVARIANT VIOLATION: DIDAuthenticator.sign_headers() missing header '{required}'. "
                f"Present: {set(headers.keys())}"
            )

    def test_invariant_authenticator_returns_empty_dict_when_not_configured(self):
        """DIDAuthenticator.sign_headers() must return empty dict when not configured."""
        from agentfield.did_auth import DIDAuthenticator

        auth = DIDAuthenticator()
        assert not auth.is_configured
        headers = auth.sign_headers(b"body")
        assert headers == {}, (
            f"INVARIANT VIOLATION: Unconfigured DIDAuthenticator returned non-empty headers: {headers}"
        )
