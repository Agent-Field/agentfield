"""
Local verification for AgentField SDK.

Provides decentralized verification of incoming requests by caching policies,
revocation lists, and the admin's public key from the control plane. Agents
can verify DID signatures and evaluate access policies locally without
hitting the control plane for every call.
"""

import base64
import hashlib
import time
from typing import Any, Dict, List, Optional, Set

from .logger import get_logger

logger = get_logger(__name__)

# DID auth headers (same as did_auth.py)
HEADER_CALLER_DID = "X-Caller-DID"
HEADER_DID_SIGNATURE = "X-DID-Signature"
HEADER_DID_TIMESTAMP = "X-DID-Timestamp"


class LocalVerifier:
    """
    Verifies incoming requests locally using cached policies, revocations,
    and the admin's Ed25519 public key.

    Periodically refreshes caches from the control plane. If the control plane
    is unreachable, continues using stale caches until TTL expires.
    """

    def __init__(
        self,
        agentfield_url: str,
        refresh_interval: int = 300,
        timestamp_window: int = 300,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the local verifier.

        Args:
            agentfield_url: Base URL of the AgentField control plane
            refresh_interval: Seconds between cache refreshes (default: 300 = 5 min)
            timestamp_window: Allowed timestamp skew in seconds (default: 300 = 5 min)
            api_key: Optional API key for authenticating with the control plane
        """
        self.agentfield_url = agentfield_url.rstrip("/")
        self.refresh_interval = refresh_interval
        self.timestamp_window = timestamp_window
        self.api_key = api_key

        # Cached data
        self.policies: List[Dict[str, Any]] = []
        self.revoked_dids: Set[str] = set()
        self.registered_dids: Set[str] = set()
        self.admin_public_key_jwk: Optional[Dict[str, Any]] = None
        self.issuer_did: Optional[str] = None

        # Cache metadata
        self._last_refresh: float = 0
        self._initialized: bool = False

    async def refresh(self) -> bool:
        """
        Fetch policies, revocations, and admin public key from the control plane.

        Returns:
            True if refresh succeeded, False otherwise (stale cache still used)
        """
        try:
            import aiohttp
        except ImportError:
            logger.warning("aiohttp not available, cannot refresh verification cache")
            return False

        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        success = True
        async with aiohttp.ClientSession() as session:
            # Fetch policies
            try:
                async with session.get(
                    f"{self.agentfield_url}/api/v1/policies",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.policies = data.get("policies", []) or []
                        logger.debug(f"Refreshed {len(self.policies)} policies")
                    else:
                        logger.warning(f"Failed to fetch policies: HTTP {resp.status}")
                        success = False
            except Exception as e:
                logger.warning(f"Failed to fetch policies: {e}")
                success = False

            # Fetch revocations
            try:
                async with session.get(
                    f"{self.agentfield_url}/api/v1/revocations",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.revoked_dids = set(data.get("revoked_dids", []))
                        logger.debug(f"Refreshed {len(self.revoked_dids)} revoked DIDs")
                    else:
                        logger.warning(f"Failed to fetch revocations: HTTP {resp.status}")
                        success = False
            except Exception as e:
                logger.warning(f"Failed to fetch revocations: {e}")
                success = False

            # Fetch registered DIDs
            try:
                async with session.get(
                    f"{self.agentfield_url}/api/v1/registered-dids",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.registered_dids = set(data.get("registered_dids", []))
                        logger.debug(f"Refreshed {len(self.registered_dids)} registered DIDs")
                    else:
                        logger.warning(f"Failed to fetch registered DIDs: HTTP {resp.status}")
                        success = False
            except Exception as e:
                logger.warning(f"Failed to fetch registered DIDs: {e}")
                success = False

            # Fetch admin public key
            try:
                async with session.get(
                    f"{self.agentfield_url}/api/v1/admin/public-key",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.admin_public_key_jwk = data.get("public_key_jwk")
                        self.issuer_did = data.get("issuer_did")
                        logger.debug(f"Refreshed admin public key (issuer: {self.issuer_did})")
                    else:
                        logger.warning(f"Failed to fetch admin public key: HTTP {resp.status}")
                        success = False
            except Exception as e:
                logger.warning(f"Failed to fetch admin public key: {e}")
                success = False

        if success:
            self._last_refresh = time.time()
            self._initialized = True

        return success

    @property
    def needs_refresh(self) -> bool:
        """Check if the cache is stale and needs refreshing."""
        if not self._initialized:
            return True
        return time.time() - self._last_refresh > self.refresh_interval

    def check_revocation(self, caller_did: str) -> bool:
        """
        Check if a caller DID is in the revocation list.

        Args:
            caller_did: The DID to check

        Returns:
            True if revoked, False if not revoked
        """
        return caller_did in self.revoked_dids

    def check_registration(self, caller_did: str) -> bool:
        """
        Check if a caller DID is registered with the control plane.

        Returns True if registered (known), False if unknown. When the
        registered DIDs cache is empty (not yet loaded), returns True to
        avoid blocking requests before the first refresh completes.
        """
        if not self.registered_dids:
            # Cache not yet populated — allow to avoid blocking before first refresh.
            return True
        return caller_did in self.registered_dids

    def verify_signature(
        self,
        caller_did: str,
        signature_b64: str,
        timestamp: str,
        body: bytes,
        nonce: str = "",
    ) -> bool:
        """
        Verify an Ed25519 DID signature on an incoming request.

        Resolves the caller's public key from their DID (did:key embeds the key
        directly; other methods fall back to the admin public key).

        Args:
            caller_did: Caller's DID identifier
            signature_b64: Base64-encoded Ed25519 signature
            timestamp: Unix timestamp string from the request
            body: Request body bytes
            nonce: Optional nonce from X-DID-Nonce header

        Returns:
            True if signature is valid, False otherwise
        """
        # Validate timestamp window
        try:
            ts = int(timestamp)
            now = int(time.time())
            if abs(now - ts) > self.timestamp_window:
                logger.debug(f"Timestamp expired: {now - ts}s drift (window: {self.timestamp_window}s)")
                return False
        except (ValueError, TypeError):
            logger.debug("Invalid timestamp format")
            return False

        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError:
            logger.warning("cryptography library not available for signature verification")
            return False

        try:
            # Resolve public key from the caller's DID
            public_key_bytes = self._resolve_public_key(caller_did)
            if public_key_bytes is None:
                logger.debug(f"Could not resolve public key for DID: {caller_did}")
                return False
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)

            # Reconstruct the signed payload: "{timestamp}[:{nonce}]:{sha256(body)}"
            # Must match the format used by SDK signing (did_auth.py)
            body_hash = hashlib.sha256(body).hexdigest()
            if nonce:
                payload = f"{timestamp}:{nonce}:{body_hash}".encode("utf-8")
            else:
                payload = f"{timestamp}:{body_hash}".encode("utf-8")

            # Decode the signature
            signature_bytes = base64.b64decode(signature_b64)

            # Verify
            public_key.verify(signature_bytes, payload)
            return True

        except Exception as e:
            logger.debug(f"Signature verification failed: {e}")
            return False

    def _resolve_public_key(self, caller_did: str) -> Optional[bytes]:
        """
        Resolve the public key bytes from a DID.

        For did:key, the public key is self-contained in the identifier:
          did:key:z<base64url(0xed01 + 32-byte-pubkey)>

        For other DID methods, falls back to the admin public key.
        """
        if caller_did.startswith("did:key:z"):
            try:
                encoded = caller_did[len("did:key:z"):]
                decoded = base64.urlsafe_b64decode(encoded + "==")
                # Verify Ed25519 multicodec prefix: 0xed, 0x01
                if len(decoded) >= 34 and decoded[0] == 0xED and decoded[1] == 0x01:
                    return decoded[2:34]
                logger.debug(f"Invalid multicodec prefix in did:key: {decoded[:2].hex()}")
                return None
            except Exception as e:
                logger.debug(f"Failed to decode did:key public key: {e}")
                return None

        # Fallback: use admin public key for non-did:key methods
        if self.admin_public_key_jwk:
            try:
                x_value = self.admin_public_key_jwk.get("x", "")
                padding = 4 - (len(x_value) % 4)
                if padding != 4:
                    x_value += "=" * padding
                return base64.urlsafe_b64decode(x_value)
            except Exception as e:
                logger.debug(f"Failed to decode admin public key: {e}")
                return None

        logger.debug("No public key available for verification")
        return None

    def evaluate_policy(
        self,
        caller_tags: List[str],
        target_tags: List[str],
        function_name: str,
        input_params: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate access policies locally.

        Finds matching policies based on caller/target tags and function name,
        then evaluates constraints.

        Args:
            caller_tags: Tags associated with the calling agent
            target_tags: Tags associated with the target agent
            function_name: Name of the function being called
            input_params: Input parameters for constraint evaluation

        Returns:
            True if access is allowed, False if denied
        """
        if not self.policies:
            # Fail closed: no policies loaded means we cannot verify access.
            # This prevents bypassing authorization when policies fail to load.
            return False

        # Sort policies by priority (descending)
        sorted_policies = sorted(
            self.policies,
            key=lambda p: p.get("priority", 0),
            reverse=True,
        )

        for policy in sorted_policies:
            if not policy.get("enabled", True):
                continue

            # Check if caller tags match
            policy_caller_tags = policy.get("caller_tags", [])
            if policy_caller_tags and not any(t in caller_tags for t in policy_caller_tags):
                continue

            # Check if target tags match
            policy_target_tags = policy.get("target_tags", [])
            if policy_target_tags and not any(t in target_tags for t in policy_target_tags):
                continue

            # Check function allow/deny lists
            allow_functions = policy.get("allow_functions", [])
            deny_functions = policy.get("deny_functions", [])

            # Check deny list first
            if deny_functions and _function_matches(function_name, deny_functions):
                return False

            # Check allow list
            if allow_functions and not _function_matches(function_name, allow_functions):
                continue

            # Check constraints
            constraints = policy.get("constraints", {})
            if constraints and input_params:
                if not _evaluate_constraints(constraints, function_name, input_params):
                    return False

            # Policy action
            action = policy.get("action", "allow")
            return action == "allow"

        # No matching policy found — allow by default
        return True


def _function_matches(function_name: str, patterns: List[str]) -> bool:
    """Check if a function name matches any of the patterns (supports * wildcards)."""
    import fnmatch

    for pattern in patterns:
        if fnmatch.fnmatch(function_name, pattern):
            return True
    return False


def _evaluate_constraints(
    constraints: Dict[str, Any],
    function_name: str,
    input_params: Dict[str, Any],
) -> bool:
    """Evaluate parameter constraints for a function call."""
    # Constraints can be keyed by function name or parameter name
    func_constraints = constraints.get(function_name, constraints)
    if not isinstance(func_constraints, dict):
        return True

    for param_name, constraint in func_constraints.items():
        if param_name not in input_params:
            continue

        value = input_params[param_name]
        if isinstance(constraint, dict):
            operator = constraint.get("operator", "")
            threshold = constraint.get("value")
            if threshold is None:
                continue

            try:
                value = float(value)
                threshold = float(threshold)
            except (ValueError, TypeError):
                # Fail closed: invalid constraint values should deny access
                # rather than silently skipping the constraint check
                return False

            if operator == "<=" and value > threshold:
                return False
            elif operator == ">=" and value < threshold:
                return False
            elif operator == "<" and value >= threshold:
                return False
            elif operator == ">" and value <= threshold:
                return False
            elif operator == "==" and value != threshold:
                return False

    return True
