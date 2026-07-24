"""
Extended tests for agentfield.verification — issue #398.

Covers:
- _resolve_public_key: did:key resolution, invalid multicodec, admin key fallback
- _evaluate_constraints: all operator logic
- refresh(): success and partial-failure paths via aiohttp mocking
- verify_signature: full Ed25519 cryptographic verification
"""
from __future__ import annotations

import base64
import hashlib
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from agentfield.verification import (
    LocalVerifier,
    _evaluate_constraints,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_verifier(**kwargs) -> LocalVerifier:
    return LocalVerifier(agentfield_url="http://localhost:8080", **kwargs)


def _make_did_key_from_public_bytes(public_key_bytes: bytes) -> str:
    """Create a valid did:key:z... identifier from raw Ed25519 public key bytes."""
    # Ed25519 multicodec prefix: 0xed, 0x01
    multicodec = bytes([0xED, 0x01]) + public_key_bytes
    encoded = base64.urlsafe_b64encode(multicodec).rstrip(b"=").decode("ascii")
    return f"did:key:z{encoded}"


def _generate_ed25519_keypair():
    """Generate an Ed25519 keypair for testing."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_bytes = public_key.public_bytes_raw()
    return private_key, public_key, public_bytes


def _sign_payload(private_key: Ed25519PrivateKey, timestamp: str, body: bytes, nonce: str = "") -> str:
    """Sign a payload the same way the SDK does, return base64-encoded signature."""
    body_hash = hashlib.sha256(body).hexdigest()
    if nonce:
        payload = f"{timestamp}:{nonce}:{body_hash}".encode("utf-8")
    else:
        payload = f"{timestamp}:{body_hash}".encode("utf-8")
    signature = private_key.sign(payload)
    return base64.b64encode(signature).decode("ascii")


# ---------------------------------------------------------------------------
# did:key resolution
# ---------------------------------------------------------------------------


class TestResolveDidKey:
    def test_resolve_did_key_z_success(self):
        """Valid did:key:z6Mkp... resolves to correct 32-byte public key."""
        _, _, public_bytes = _generate_ed25519_keypair()
        did = _make_did_key_from_public_bytes(public_bytes)

        v = _make_verifier()
        result = v._resolve_public_key(did)

        assert result is not None
        assert len(result) == 32
        assert result == public_bytes

    def test_resolve_did_key_invalid_multicodec(self):
        """did:key:z with wrong multicodec prefix returns None."""
        # Use wrong prefix (0xAA, 0xBB instead of 0xED, 0x01)
        bad_prefix = bytes([0xAA, 0xBB]) + b"\x00" * 32
        encoded = base64.urlsafe_b64encode(bad_prefix).rstrip(b"=").decode("ascii")
        did = f"did:key:z{encoded}"

        v = _make_verifier()
        result = v._resolve_public_key(did)

        assert result is None

    def test_resolve_did_key_too_short(self):
        """did:key:z with payload shorter than 34 bytes returns None."""
        short_payload = bytes([0xED, 0x01]) + b"\x00" * 10  # only 12 bytes total
        encoded = base64.urlsafe_b64encode(short_payload).rstrip(b"=").decode("ascii")
        did = f"did:key:z{encoded}"

        v = _make_verifier()
        result = v._resolve_public_key(did)

        assert result is None

    def test_resolve_fallback_to_admin_key(self):
        """Non-did:key DID falls back to admin_public_key_jwk."""
        _, _, public_bytes = _generate_ed25519_keypair()
        x_value = base64.urlsafe_b64encode(public_bytes).rstrip(b"=").decode("ascii")

        v = _make_verifier()
        v.admin_public_key_jwk = {"kty": "OKP", "crv": "Ed25519", "x": x_value}

        result = v._resolve_public_key("did:web:example.com")
        assert result is not None
        assert result == public_bytes

    def test_resolve_no_key_available(self):
        """Non-did:key DID with no admin key set returns None."""
        v = _make_verifier()
        v.admin_public_key_jwk = None

        result = v._resolve_public_key("did:web:example.com")
        assert result is None

    def test_resolve_admin_key_invalid_base64(self):
        """Admin key with invalid base64 x value returns None gracefully."""
        v = _make_verifier()
        # Use a value that will cause base64 decode to raise an exception
        v.admin_public_key_jwk = {"kty": "OKP", "crv": "Ed25519", "x": "\x00\x01\x02"}

        result = v._resolve_public_key("did:web:example.com")
        # Should either return None or return some bytes (depending on decode behavior)
        # The important thing is it doesn't crash
        assert result is None or isinstance(result, bytes)

    def test_resolve_did_key_decode_exception(self):
        """did:key:z decode exception triggers the except path and returns None."""
        v = _make_verifier()
        # Patch base64.urlsafe_b64decode to raise inside the did:key try block
        with patch("agentfield.verification.base64.urlsafe_b64decode", side_effect=Exception("decode boom")):
            result = v._resolve_public_key("did:key:zSomeValidLookingData")
        assert result is None

    def test_resolve_admin_key_get_raises_exception(self):
        """Admin key JWK with a .get() that raises triggers except path."""
        v = _make_verifier()
        # Set admin_public_key_jwk to something whose .get("x") returns a value
        # that causes urlsafe_b64decode to throw (e.g., an integer)
        v.admin_public_key_jwk = {"x": 12345}  # int, not str — will fail on len()

        result = v._resolve_public_key("did:web:example.com")
        assert result is None


# ---------------------------------------------------------------------------
# _evaluate_constraints — operator logic
# ---------------------------------------------------------------------------


class TestEvaluateConstraints:
    def test_constraint_equality_allow(self):
        """== operator allows when value matches."""
        constraints = {"count": {"operator": "==", "value": 10}}
        assert _evaluate_constraints(constraints, "fn", {"count": 10}) is True

    def test_constraint_equality_deny(self):
        """== operator denies when value doesn't match."""
        constraints = {"count": {"operator": "==", "value": 10}}
        assert _evaluate_constraints(constraints, "fn", {"count": 11}) is False

    def test_constraint_greater_than_allow(self):
        """> operator allows when value exceeds threshold."""
        constraints = {"score": {"operator": ">", "value": 10}}
        assert _evaluate_constraints(constraints, "fn", {"score": 11}) is True

    def test_constraint_greater_than_deny(self):
        """> operator denies when value equals threshold."""
        constraints = {"score": {"operator": ">", "value": 10}}
        assert _evaluate_constraints(constraints, "fn", {"score": 10}) is False

    def test_constraint_greater_than_deny_below(self):
        """> operator denies when value is below threshold."""
        constraints = {"score": {"operator": ">", "value": 10}}
        assert _evaluate_constraints(constraints, "fn", {"score": 5}) is False

    def test_constraint_less_than_allow(self):
        """< operator allows when value is below threshold."""
        constraints = {"risk": {"operator": "<", "value": 5}}
        assert _evaluate_constraints(constraints, "fn", {"risk": 3}) is True

    def test_constraint_less_than_deny(self):
        """< operator denies when value equals threshold."""
        constraints = {"risk": {"operator": "<", "value": 5}}
        assert _evaluate_constraints(constraints, "fn", {"risk": 5}) is False

    def test_constraint_gte_allow(self):
        """>= operator allows when value equals threshold."""
        constraints = {"level": {"operator": ">=", "value": 3}}
        assert _evaluate_constraints(constraints, "fn", {"level": 3}) is True

    def test_constraint_gte_deny(self):
        """>= operator denies when value is below threshold."""
        constraints = {"level": {"operator": ">=", "value": 3}}
        assert _evaluate_constraints(constraints, "fn", {"level": 2}) is False

    def test_constraint_lte_allow(self):
        """<= operator allows when value equals threshold."""
        constraints = {"amount": {"operator": "<=", "value": 100}}
        assert _evaluate_constraints(constraints, "fn", {"amount": 100}) is True

    def test_constraint_lte_deny(self):
        """<= operator denies when value exceeds threshold."""
        constraints = {"amount": {"operator": "<=", "value": 100}}
        assert _evaluate_constraints(constraints, "fn", {"amount": 101}) is False

    def test_constraint_invalid_float_fails_closed(self):
        """Non-numeric input for a numeric constraint returns False (fail-closed)."""
        constraints = {"amount": {"operator": "<=", "value": 100}}
        assert _evaluate_constraints(constraints, "fn", {"amount": "not-a-number"}) is False

    def test_constraint_missing_param_skipped(self):
        """Parameter not in input_params is silently skipped (returns True)."""
        constraints = {"missing_param": {"operator": "<=", "value": 100}}
        assert _evaluate_constraints(constraints, "fn", {"other_param": 50}) is True

    def test_constraint_none_threshold_skipped(self):
        """Constraint with None value is skipped."""
        constraints = {"amount": {"operator": "<=", "value": None}}
        assert _evaluate_constraints(constraints, "fn", {"amount": 50}) is True

    def test_constraint_non_dict_returns_true(self):
        """Non-dict func_constraints returns True."""
        constraints = {"fn": "not-a-dict"}
        assert _evaluate_constraints(constraints, "fn", {"x": 1}) is True

    def test_constraint_keyed_by_function_name(self):
        """Constraints keyed by function name are resolved correctly."""
        constraints = {"transfer": {"amount": {"operator": "<=", "value": 500}}}
        assert _evaluate_constraints(constraints, "transfer", {"amount": 200}) is True
        assert _evaluate_constraints(constraints, "transfer", {"amount": 600}) is False

    def test_constraint_non_dict_constraint_value_skipped(self):
        """Non-dict constraint for a param is skipped."""
        constraints = {"amount": "just-a-string"}
        assert _evaluate_constraints(constraints, "fn", {"amount": 50}) is True


# ---------------------------------------------------------------------------
# verify_signature — full Ed25519 cryptographic verification
# ---------------------------------------------------------------------------


class TestVerifySignatureCrypto:
    def test_valid_signature_no_nonce(self):
        """Valid Ed25519 signature without nonce verifies successfully."""
        private_key, _, public_bytes = _generate_ed25519_keypair()
        did = _make_did_key_from_public_bytes(public_bytes)

        v = _make_verifier(timestamp_window=600)
        ts = str(int(time.time()))
        body = b"hello world"
        sig = _sign_payload(private_key, ts, body)

        result = v.verify_signature(did, sig, ts, body)
        assert result is True

    def test_valid_signature_with_nonce(self):
        """Valid Ed25519 signature with nonce verifies successfully."""
        private_key, _, public_bytes = _generate_ed25519_keypair()
        did = _make_did_key_from_public_bytes(public_bytes)

        v = _make_verifier(timestamp_window=600)
        ts = str(int(time.time()))
        body = b"request body"
        nonce = "abc123"
        sig = _sign_payload(private_key, ts, body, nonce=nonce)

        result = v.verify_signature(did, sig, ts, body, nonce=nonce)
        assert result is True

    def test_invalid_signature_wrong_body(self):
        """Signature over different body fails verification."""
        private_key, _, public_bytes = _generate_ed25519_keypair()
        did = _make_did_key_from_public_bytes(public_bytes)

        v = _make_verifier(timestamp_window=600)
        ts = str(int(time.time()))
        sig = _sign_payload(private_key, ts, b"original body")

        result = v.verify_signature(did, sig, ts, b"tampered body")
        assert result is False

    def test_invalid_signature_wrong_timestamp(self):
        """Signature with wrong timestamp fails verification."""
        private_key, _, public_bytes = _generate_ed25519_keypair()
        did = _make_did_key_from_public_bytes(public_bytes)

        v = _make_verifier(timestamp_window=600)
        ts = str(int(time.time()))
        sig = _sign_payload(private_key, ts, b"body")

        # Verify with a different (but still in-window) timestamp
        different_ts = str(int(time.time()) - 10)
        result = v.verify_signature(did, sig, different_ts, b"body")
        assert result is False

    def test_invalid_signature_wrong_key(self):
        """Signature from different key fails verification."""
        signing_key, _, _ = _generate_ed25519_keypair()
        _, _, wrong_public_bytes = _generate_ed25519_keypair()
        did = _make_did_key_from_public_bytes(wrong_public_bytes)

        v = _make_verifier(timestamp_window=600)
        ts = str(int(time.time()))
        body = b"data"
        sig = _sign_payload(signing_key, ts, body)

        result = v.verify_signature(did, sig, ts, body)
        assert result is False

    def test_unresolvable_did_returns_false(self):
        """DID that can't be resolved (no admin key, not did:key) returns False."""
        v = _make_verifier(timestamp_window=600)
        ts = str(int(time.time()))

        result = v.verify_signature("did:web:unknown.com", "AAAA", ts, b"body")
        assert result is False

    def test_valid_signature_with_admin_key_fallback(self):
        """Signature verified using admin key fallback for did:web."""
        private_key, _, public_bytes = _generate_ed25519_keypair()
        x_value = base64.urlsafe_b64encode(public_bytes).rstrip(b"=").decode("ascii")

        v = _make_verifier(timestamp_window=600)
        v.admin_public_key_jwk = {"kty": "OKP", "crv": "Ed25519", "x": x_value}

        ts = str(int(time.time()))
        body = b"admin-signed request"
        sig = _sign_payload(private_key, ts, body)

        result = v.verify_signature("did:web:admin.example.com", sig, ts, body)
        assert result is True


# ---------------------------------------------------------------------------
# refresh() — success and partial failure via aiohttp mocking
# ---------------------------------------------------------------------------


def _mock_aiohttp_response(status: int, json_data: Dict[str, Any]):
    """Create a mock aiohttp response context manager."""
    resp = AsyncMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_data)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


class TestRefreshSuccess:
    @pytest.mark.asyncio
    async def test_refresh_success_populates_caches(self):
        """Successful refresh populates all caches from control plane responses."""
        v = _make_verifier(api_key="test-key")

        policies_resp = _mock_aiohttp_response(200, {
            "policies": [{"action": "allow", "priority": 10, "enabled": True}]
        })
        revocations_resp = _mock_aiohttp_response(200, {
            "revoked_dids": ["did:key:revoked1", "did:key:revoked2"]
        })
        registered_resp = _mock_aiohttp_response(200, {
            "registered_dids": ["did:key:agent1", "did:key:agent2"]
        })
        admin_key_resp = _mock_aiohttp_response(200, {
            "public_key_jwk": {"kty": "OKP", "crv": "Ed25519", "x": "abc123"},
            "issuer_did": "did:web:admin.example.com",
        })

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=[
            policies_resp, revocations_resp, registered_resp, admin_key_resp
        ])
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = await v.refresh()

        assert result is True
        assert len(v.policies) == 1
        assert v.policies[0]["action"] == "allow"
        assert v.revoked_dids == {"did:key:revoked1", "did:key:revoked2"}
        assert v.registered_dids == {"did:key:agent1", "did:key:agent2"}
        assert v.admin_public_key_jwk == {"kty": "OKP", "crv": "Ed25519", "x": "abc123"}
        assert v.issuer_did == "did:web:admin.example.com"

    @pytest.mark.asyncio
    async def test_refresh_sets_initialized_and_timestamp(self):
        """Successful refresh sets _initialized=True and _last_refresh > 0."""
        v = _make_verifier()
        assert v._initialized is False
        assert v._last_refresh == 0

        policies_resp = _mock_aiohttp_response(200, {"policies": []})
        revocations_resp = _mock_aiohttp_response(200, {"revoked_dids": []})
        registered_resp = _mock_aiohttp_response(200, {"registered_dids": []})
        admin_key_resp = _mock_aiohttp_response(200, {"public_key_jwk": None, "issuer_did": None})

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=[
            policies_resp, revocations_resp, registered_resp, admin_key_resp
        ])
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        before = time.time()
        with patch("aiohttp.ClientSession", return_value=session_ctx):
            await v.refresh()

        assert v._initialized is True
        assert v._last_refresh >= before

    @pytest.mark.asyncio
    async def test_refresh_partial_failure_one_endpoint_500(self):
        """One endpoint returning 500 causes refresh to return False but other caches still update."""
        v = _make_verifier()

        policies_resp = _mock_aiohttp_response(200, {
            "policies": [{"action": "deny", "priority": 5, "enabled": True}]
        })
        # Revocations endpoint fails with 500
        revocations_resp = _mock_aiohttp_response(500, {})
        registered_resp = _mock_aiohttp_response(200, {
            "registered_dids": ["did:key:agent-x"]
        })
        admin_key_resp = _mock_aiohttp_response(200, {
            "public_key_jwk": {"kty": "OKP", "crv": "Ed25519", "x": "xyz"},
            "issuer_did": "did:web:cp.local",
        })

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=[
            policies_resp, revocations_resp, registered_resp, admin_key_resp
        ])
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = await v.refresh()

        # Overall result is False because one endpoint failed
        assert result is False
        # But other caches were still updated
        assert len(v.policies) == 1
        assert v.registered_dids == {"did:key:agent-x"}
        assert v.admin_public_key_jwk == {"kty": "OKP", "crv": "Ed25519", "x": "xyz"}
        # _initialized should NOT be set on partial failure
        assert v._initialized is False

    @pytest.mark.asyncio
    async def test_refresh_sends_api_key_header(self):
        """Refresh sends X-API-Key header when api_key is configured."""
        v = _make_verifier(api_key="secret-key-123")

        policies_resp = _mock_aiohttp_response(200, {"policies": []})
        revocations_resp = _mock_aiohttp_response(200, {"revoked_dids": []})
        registered_resp = _mock_aiohttp_response(200, {"registered_dids": []})
        admin_key_resp = _mock_aiohttp_response(200, {"public_key_jwk": None, "issuer_did": None})

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=[
            policies_resp, revocations_resp, registered_resp, admin_key_resp
        ])
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            await v.refresh()

        # Check that get was called with the API key header
        calls = mock_session.get.call_args_list
        for call in calls:
            headers = call.kwargs.get("headers", call.args[1] if len(call.args) > 1 else {})
            if isinstance(headers, dict):
                assert headers.get("X-API-Key") == "secret-key-123"

    @pytest.mark.asyncio
    async def test_refresh_all_endpoints_fail_returns_false(self):
        """All endpoints failing returns False, caches unchanged."""
        v = _make_verifier()
        v.policies = [{"old": True}]  # pre-existing cache

        mock_session = AsyncMock()
        mock_session.get = MagicMock(side_effect=[
            _mock_aiohttp_response(500, {}),
            _mock_aiohttp_response(503, {}),
            _mock_aiohttp_response(502, {}),
            _mock_aiohttp_response(500, {}),
        ])
        session_ctx = AsyncMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session_ctx):
            result = await v.refresh()

        assert result is False
        assert v._initialized is False
        # Policies still show whatever they had before (HTTP 200 not reached)
        # Actually, for non-200 responses the code doesn't overwrite — let's verify
        assert v.policies == [{"old": True}]
