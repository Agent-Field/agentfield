"""
Tests for agentfield.verification — LocalVerifier (security-critical).
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from agentfield.verification import LocalVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_verifier(**kwargs) -> LocalVerifier:
    return LocalVerifier(agentfield_url="http://localhost:8080", **kwargs)


# ---------------------------------------------------------------------------
# __init__ defaults
# ---------------------------------------------------------------------------


class TestLocalVerifierInit:
    def test_default_refresh_interval(self):
        v = _make_verifier()
        assert v.refresh_interval == 300

    def test_default_timestamp_window(self):
        v = _make_verifier()
        assert v.timestamp_window == 300

    def test_url_trailing_slash_stripped(self):
        v = LocalVerifier(agentfield_url="http://localhost:8080/")
        assert v.agentfield_url == "http://localhost:8080"

    def test_custom_refresh_and_window(self):
        v = _make_verifier(refresh_interval=60, timestamp_window=120)
        assert v.refresh_interval == 60
        assert v.timestamp_window == 120

    def test_initial_state_empty(self):
        v = _make_verifier()
        assert v.policies == []
        assert v.revoked_dids == set()
        assert v.registered_dids == set()
        assert v.admin_public_key_jwk is None
        assert v.issuer_did is None

    def test_not_initialized(self):
        v = _make_verifier()
        assert v._initialized is False
        assert v.needs_refresh is True

    def test_api_key_stored(self):
        v = _make_verifier(api_key="my-key")
        assert v.api_key == "my-key"


# ---------------------------------------------------------------------------
# needs_refresh
# ---------------------------------------------------------------------------


class TestNeedsRefresh:
    def test_needs_refresh_when_not_initialized(self):
        v = _make_verifier()
        assert v.needs_refresh is True

    def test_needs_refresh_after_stale_cache(self):
        v = _make_verifier(refresh_interval=10)
        v._initialized = True
        v._last_refresh = time.time() - 20  # 20s ago, interval is 10s
        assert v.needs_refresh is True

    def test_no_refresh_needed_fresh_cache(self):
        v = _make_verifier(refresh_interval=300)
        v._initialized = True
        v._last_refresh = time.time()
        assert v.needs_refresh is False


# ---------------------------------------------------------------------------
# check_revocation / check_registration
# ---------------------------------------------------------------------------


class TestRevocationAndRegistration:
    def test_not_revoked_when_set_empty(self):
        v = _make_verifier()
        assert v.check_revocation("did:key:abc") is False

    def test_revoked_did_detected(self):
        v = _make_verifier()
        v.revoked_dids = {"did:key:bad"}
        assert v.check_revocation("did:key:bad") is True

    def test_not_revoked_different_did(self):
        v = _make_verifier()
        v.revoked_dids = {"did:key:bad"}
        assert v.check_revocation("did:key:good") is False

    def test_registration_allows_when_cache_empty(self):
        v = _make_verifier()
        # Empty registered_dids means cache not populated — allow
        assert v.check_registration("did:key:anyone") is True

    def test_registration_allows_known_did(self):
        v = _make_verifier()
        v.registered_dids = {"did:key:known"}
        assert v.check_registration("did:key:known") is True

    def test_registration_denies_unknown_did(self):
        v = _make_verifier()
        v.registered_dids = {"did:key:known"}
        assert v.check_registration("did:key:stranger") is False


# ---------------------------------------------------------------------------
# verify_signature — timestamp window
# ---------------------------------------------------------------------------


class TestVerifySignatureTimestamp:
    def test_rejects_expired_timestamp_too_old(self):
        v = _make_verifier(timestamp_window=300)
        old_ts = str(int(time.time()) - 400)  # 400s ago, window=300
        result = v.verify_signature(
            caller_did="did:key:abc",
            signature_b64="AAAA",
            timestamp=old_ts,
            body=b"",
        )
        assert result is False

    def test_rejects_expired_timestamp_future(self):
        v = _make_verifier(timestamp_window=300)
        future_ts = str(int(time.time()) + 400)
        result = v.verify_signature(
            caller_did="did:key:abc",
            signature_b64="AAAA",
            timestamp=future_ts,
            body=b"",
        )
        assert result is False

    def test_rejects_invalid_timestamp_string(self):
        v = _make_verifier()
        result = v.verify_signature(
            caller_did="did:key:abc",
            signature_b64="AAAA",
            timestamp="not-a-number",
            body=b"",
        )
        assert result is False

    def test_rejects_none_timestamp(self):
        v = _make_verifier()
        result = v.verify_signature(
            caller_did="did:key:abc",
            signature_b64="AAAA",
            timestamp=None,  # type: ignore[arg-type]
            body=b"",
        )
        assert result is False


# ---------------------------------------------------------------------------
# verify_signature — cryptography not available
# ---------------------------------------------------------------------------


class TestVerifySignatureNoCrypto:
    def test_returns_false_without_cryptography(self):
        v = _make_verifier(timestamp_window=600)
        valid_ts = str(int(time.time()))
        with patch.dict("sys.modules", {"cryptography.hazmat.primitives.asymmetric.ed25519": None}):
            result = v.verify_signature(
                caller_did="did:key:z6Mk",
                signature_b64="AAAA",
                timestamp=valid_ts,
                body=b"test",
            )
            assert result is False


# ---------------------------------------------------------------------------
# evaluate_policy
# ---------------------------------------------------------------------------


class TestEvaluatePolicy:
    def test_no_policies_denies_all(self):
        v = _make_verifier()
        v.policies = []
        result = v.evaluate_policy(
            caller_tags=["agent"],
            target_tags=["service"],
            function_name="do_thing",
        )
        assert result is False

    def test_allow_policy_grants_access(self):
        v = _make_verifier()
        v.policies = [
            {
                "enabled": True,
                "action": "allow",
                "priority": 10,
                "caller_tags": [],
                "target_tags": [],
                "allow_functions": [],
                "deny_functions": [],
                "constraints": {},
            }
        ]
        assert v.evaluate_policy(["a"], ["b"], "run") is True

    def test_deny_policy_blocks_access(self):
        v = _make_verifier()
        v.policies = [
            {
                "enabled": True,
                "action": "deny",
                "priority": 10,
                "caller_tags": [],
                "target_tags": [],
                "allow_functions": [],
                "deny_functions": [],
                "constraints": {},
            }
        ]
        assert v.evaluate_policy(["a"], ["b"], "run") is False

    def test_deny_function_list_blocks_specific_function(self):
        v = _make_verifier()
        v.policies = [
            {
                "enabled": True,
                "action": "allow",
                "priority": 10,
                "caller_tags": [],
                "target_tags": [],
                "allow_functions": [],
                "deny_functions": ["dangerous_fn"],
                "constraints": {},
            }
        ]
        assert v.evaluate_policy([], [], "dangerous_fn") is False

    def test_allow_function_list_permits_listed(self):
        v = _make_verifier()
        v.policies = [
            {
                "enabled": True,
                "action": "allow",
                "priority": 10,
                "caller_tags": [],
                "target_tags": [],
                "allow_functions": ["safe_fn"],
                "deny_functions": [],
                "constraints": {},
            }
        ]
        assert v.evaluate_policy([], [], "safe_fn") is True

    def test_allow_function_list_skips_unlisted(self):
        # When allow_functions is set and function not in it, policy is skipped.
        # With no other matching policy, default is allow.
        v = _make_verifier()
        v.policies = [
            {
                "enabled": True,
                "action": "allow",
                "priority": 10,
                "caller_tags": [],
                "target_tags": [],
                "allow_functions": ["safe_fn"],
                "deny_functions": [],
                "constraints": {},
            }
        ]
        # "other_fn" not in allow_functions → policy skipped → default allow
        result = v.evaluate_policy([], [], "other_fn")
        assert result is True

    def test_disabled_policy_is_skipped(self):
        v = _make_verifier()
        v.policies = [
            {
                "enabled": False,
                "action": "deny",
                "priority": 10,
                "caller_tags": [],
                "target_tags": [],
                "allow_functions": [],
                "deny_functions": [],
                "constraints": {},
            }
        ]
        # disabled deny policy → no match → default allow
        assert v.evaluate_policy([], [], "fn") is True

    def test_caller_tag_mismatch_skips_policy(self):
        v = _make_verifier()
        v.policies = [
            {
                "enabled": True,
                "action": "deny",
                "priority": 10,
                "caller_tags": ["admin"],
                "target_tags": [],
                "allow_functions": [],
                "deny_functions": [],
                "constraints": {},
            }
        ]
        # caller doesn't have "admin" tag → policy skipped → default allow
        assert v.evaluate_policy(["user"], [], "fn") is True

    def test_target_tag_mismatch_skips_policy(self):
        v = _make_verifier()
        v.policies = [
            {
                "enabled": True,
                "action": "deny",
                "priority": 10,
                "caller_tags": [],
                "target_tags": ["protected"],
                "allow_functions": [],
                "deny_functions": [],
                "constraints": {},
            }
        ]
        assert v.evaluate_policy([], ["public"], "fn") is True

    def test_policy_priority_ordering(self):
        v = _make_verifier()
        v.policies = [
            {
                "enabled": True,
                "action": "deny",
                "priority": 5,
                "caller_tags": [],
                "target_tags": [],
                "allow_functions": [],
                "deny_functions": [],
                "constraints": {},
            },
            {
                "enabled": True,
                "action": "allow",
                "priority": 10,  # higher priority
                "caller_tags": [],
                "target_tags": [],
                "allow_functions": [],
                "deny_functions": [],
                "constraints": {},
            },
        ]
        # priority 10 allow evaluated first → access granted
        assert v.evaluate_policy([], [], "fn") is True

    def test_wildcard_deny_function(self):
        v = _make_verifier()
        v.policies = [
            {
                "enabled": True,
                "action": "allow",
                "priority": 10,
                "caller_tags": [],
                "target_tags": [],
                "allow_functions": [],
                "deny_functions": ["delete_*"],
                "constraints": {},
            }
        ]
        assert v.evaluate_policy([], [], "delete_user") is False

    def test_constraint_lte_violation_denies(self):
        v = _make_verifier()
        v.policies = [
            {
                "enabled": True,
                "action": "allow",
                "priority": 10,
                "caller_tags": [],
                "target_tags": [],
                "allow_functions": [],
                "deny_functions": [],
                "constraints": {
                    "amount": {"operator": "<=", "value": 100}
                },
            }
        ]
        # amount=150 > 100 → constraint violated → deny
        assert v.evaluate_policy([], [], "transfer", {"amount": 150}) is False

    def test_constraint_lte_satisfied_allows(self):
        v = _make_verifier()
        v.policies = [
            {
                "enabled": True,
                "action": "allow",
                "priority": 10,
                "caller_tags": [],
                "target_tags": [],
                "allow_functions": [],
                "deny_functions": [],
                "constraints": {
                    "amount": {"operator": "<=", "value": 100}
                },
            }
        ]
        assert v.evaluate_policy([], [], "transfer", {"amount": 50}) is True

    def test_no_policies_with_empty_list_denies(self):
        v = _make_verifier()
        assert v.evaluate_policy([], [], "anything") is False


# ---------------------------------------------------------------------------
# refresh — network error handling
# ---------------------------------------------------------------------------


class TestRefreshNetworkErrors:
    @pytest.mark.asyncio
    async def test_refresh_returns_false_on_connection_error(self):
        v = _make_verifier()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        # Simulate connection error on policies endpoint
        mock_session.get.side_effect = ConnectionError("network unreachable")

        with patch("aiohttp.ClientSession", return_value=mock_session):
            result = await v.refresh()

        assert result is False

    @pytest.mark.asyncio
    async def test_refresh_returns_false_when_aiohttp_missing(self):
        v = _make_verifier()
        with patch.dict("sys.modules", {"aiohttp": None}):
            result = await v.refresh()
        assert result is False

    @pytest.mark.asyncio
    async def test_refresh_preserves_stale_cache_on_failure(self):
        v = _make_verifier()
        v.policies = [{"action": "allow"}]
        v.revoked_dids = {"did:key:old"}

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.get.side_effect = OSError("timeout")

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await v.refresh()

        # stale cache should remain
        assert v.policies == [{"action": "allow"}]
        assert "did:key:old" in v.revoked_dids


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_evaluate_policy_empty_caller_and_target_tags(self):
        v = _make_verifier()
        v.policies = [
            {
                "enabled": True,
                "action": "allow",
                "priority": 0,
                "caller_tags": [],
                "target_tags": [],
                "allow_functions": [],
                "deny_functions": [],
                "constraints": {},
            }
        ]
        assert v.evaluate_policy([], [], "") is True

    def test_verify_signature_empty_caller_did(self):
        v = _make_verifier(timestamp_window=600)
        ts = str(int(time.time()))
        result = v.verify_signature("", "AAAA", ts, b"body")
        assert result is False

    def test_resolve_public_key_unknown_did_no_admin_key(self):
        v = _make_verifier()
        result = v._resolve_public_key("did:example:unknown")
        assert result is None

    def test_resolve_public_key_bad_did_key_format(self):
        v = _make_verifier()
        result = v._resolve_public_key("did:key:zNOTVALIDDATA!!!!")
        # Should return None gracefully, not raise
        assert result is None or isinstance(result, bytes)
