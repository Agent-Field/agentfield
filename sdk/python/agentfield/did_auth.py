"""
DID Authentication for AgentField SDK

Provides cryptographic signing for agent-to-agent requests using Ed25519 signatures.
This module handles the creation of DID authentication headers for protected agent calls.
"""

import base64
import hashlib
import json
import os
import time
from typing import Dict, Optional, Tuple

from .logger import get_logger

logger = get_logger(__name__)

# Headers used for DID authentication
HEADER_CALLER_DID = "X-Caller-DID"
HEADER_DID_SIGNATURE = "X-DID-Signature"
HEADER_DID_TIMESTAMP = "X-DID-Timestamp"
HEADER_DID_NONCE = "X-DID-Nonce"


def _load_ed25519_private_key(private_key_jwk: str):
    """
    Load Ed25519 private key from JWK format.

    Args:
        private_key_jwk: JWK-formatted private key string

    Returns:
        Ed25519PrivateKey object

    Raises:
        ImportError: If cryptography library is not installed
        ValueError: If key format is invalid
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:
        raise ImportError(
            "The 'cryptography' library is required for DID authentication. "
            "Install it with: pip install cryptography"
        )

    try:
        jwk = json.loads(private_key_jwk) if isinstance(private_key_jwk, str) else private_key_jwk

        # Verify key type
        if jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519":
            raise ValueError("Invalid key type: expected Ed25519 OKP key")

        # Extract 'd' (private key bytes) from JWK
        d_value = jwk.get("d")
        if not d_value:
            raise ValueError("Missing 'd' (private key) in JWK")

        # Decode base64url-encoded private key
        # Add padding if needed for base64url decoding
        padding = 4 - (len(d_value) % 4)
        if padding != 4:
            d_value += "=" * padding

        private_key_bytes = base64.urlsafe_b64decode(d_value)

        return Ed25519PrivateKey.from_private_bytes(private_key_bytes)

    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JWK format: {e}")


def sign_request(
    body: bytes,
    private_key_jwk: str,
    did: str,
) -> Tuple[str, str, str]:
    """
    Sign a request body for DID authentication.

    Creates the signature payload as "{timestamp}:{nonce}:{sha256(body)}" and signs it
    with the Ed25519 private key. The nonce ensures each signature is unique even when
    the same body is signed within the same second.

    Args:
        body: Request body bytes to sign
        private_key_jwk: JWK-formatted private key string
        did: Caller's DID identifier

    Returns:
        Tuple of (signature_base64, timestamp_str, nonce, did)

    Raises:
        ImportError: If cryptography library is not installed
        ValueError: If key format is invalid
    """
    # Load private key
    private_key = _load_ed25519_private_key(private_key_jwk)

    # Get current timestamp
    timestamp = str(int(time.time()))

    # Generate per-request nonce to prevent replay detection when
    # multiple requests have the same body within the same second
    # (Ed25519 is deterministic, so same payload = same signature)
    nonce = os.urandom(16).hex()

    # Hash the body
    body_hash = hashlib.sha256(body).hexdigest()

    # Create payload: "{timestamp}:{nonce}:{body_hash}"
    payload = f"{timestamp}:{nonce}:{body_hash}".encode("utf-8")

    # Sign the payload
    signature = private_key.sign(payload)

    # Encode signature as base64
    signature_b64 = base64.b64encode(signature).decode("ascii")

    return signature_b64, timestamp, nonce, did


def create_did_auth_headers(
    body: bytes,
    private_key_jwk: str,
    did: str,
) -> Dict[str, str]:
    """
    Create DID authentication headers for a request.

    Args:
        body: Request body bytes
        private_key_jwk: JWK-formatted private key string
        did: Caller's DID identifier

    Returns:
        Dictionary with DID authentication headers

    Raises:
        ImportError: If cryptography library is not installed
        ValueError: If key format is invalid
    """
    signature, timestamp, nonce, caller_did = sign_request(body, private_key_jwk, did)

    return {
        HEADER_CALLER_DID: caller_did,
        HEADER_DID_SIGNATURE: signature,
        HEADER_DID_TIMESTAMP: timestamp,
        HEADER_DID_NONCE: nonce,
    }


class DIDAuthenticator:
    """
    Handles DID authentication for agent requests.

    This class manages the signing credentials and provides methods
    for creating authenticated request headers.
    """

    def __init__(self, did: Optional[str] = None, private_key_jwk: Optional[str] = None):
        """
        Initialize DID authenticator.

        Args:
            did: The agent's DID identifier
            private_key_jwk: JWK-formatted private key for signing
        """
        self._did = did
        self._private_key_jwk = private_key_jwk
        self._private_key = None

        # Pre-load the private key if provided
        if private_key_jwk:
            try:
                self._private_key = _load_ed25519_private_key(private_key_jwk)
            except (ImportError, ValueError) as e:
                logger.warning(f"Could not load private key for DID auth: {e}")

    @property
    def did(self) -> Optional[str]:
        """Get the DID identifier."""
        return self._did

    @property
    def is_configured(self) -> bool:
        """Check if DID authentication is configured."""
        return self._did is not None and self._private_key is not None

    def set_credentials(self, did: str, private_key_jwk: str) -> bool:
        """
        Set DID authentication credentials.

        Args:
            did: The agent's DID identifier
            private_key_jwk: JWK-formatted private key for signing

        Returns:
            True if credentials were set successfully, False otherwise
        """
        try:
            self._private_key = _load_ed25519_private_key(private_key_jwk)
            self._did = did
            self._private_key_jwk = private_key_jwk
            logger.debug(f"DID authentication configured for {did}")
            return True
        except (ImportError, ValueError) as e:
            logger.error(f"Failed to set DID credentials: {e}")
            return False

    def sign_headers(self, body: bytes) -> Dict[str, str]:
        """
        Create DID authentication headers for a request.

        Args:
            body: Request body bytes to sign

        Returns:
            Dictionary with DID authentication headers, empty if not configured

        Note:
            Returns empty dict if DID auth is not configured, allowing
            requests to proceed without authentication.
        """
        if not self.is_configured:
            return {}

        try:
            return create_did_auth_headers(body, self._private_key_jwk, self._did)
        except Exception as e:
            logger.error(f"Failed to sign request: {e}")
            return {}

    def get_auth_info(self) -> Dict[str, any]:
        """
        Get information about the authentication configuration.

        Returns:
            Dictionary with authentication info (no private key)
        """
        return {
            "configured": self.is_configured,
            "did": self._did,
        }
