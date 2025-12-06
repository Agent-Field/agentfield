"""
DID Manager for AgentField SDK

Handles Decentralized Identity (DID) and Verifiable Credentials (VC) functionality
for agent nodes, reasoners, and skills.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
from datetime import datetime

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class DIDIdentity:
    """Represents a DID identity with cryptographic keys."""

    did: str
    private_key_jwk: str
    public_key_jwk: str
    derivation_path: str
    component_type: str
    function_name: Optional[str] = None


@dataclass
class DIDIdentityPackage:
    """Complete DID identity package for an agent."""

    agent_did: DIDIdentity
    reasoner_dids: Dict[str, DIDIdentity]
    skill_dids: Dict[str, DIDIdentity]
    agentfield_server_id: str


@dataclass
class DIDExecutionContext:
    """Context for DID-enabled execution."""

    execution_id: str
    workflow_id: str
    session_id: str
    caller_did: str
    target_did: str
    agent_node_did: str
    timestamp: datetime


class DIDManager:
    """
    Manages DID operations for AgentField SDK agents.

    Handles:
    - Agent registration with AgentField Server
    - DID resolution and verification
    - Execution context creation
    - Integration with agent lifecycle
    """

    def __init__(
        self, agentfield_server_url: str, agent_node_id: str, api_key: Optional[str] = None
    ):
        """
        Initialize DID Manager.

        Args:
            agentfield_server_url: URL of the AgentField Server
            agent_node_id: Unique identifier for this agent node
            api_key: Optional API key for authentication
        """
        self.agentfield_server_url = agentfield_server_url.rstrip("/")
        self.agent_node_id = agent_node_id
        self.api_key = api_key
        self.identity_package: Optional[DIDIdentityPackage] = None
        self.enabled = False

    def _get_auth_headers(self) -> Dict[str, str]:
        """Return auth headers if API key is configured."""
        if not self.api_key:
            return {}
        return {"X-API-Key": self.api_key}

    def register_agent(
        self, reasoners: List[Dict[str, Any]], skills: List[Dict[str, Any]]
    ) -> bool:
        """
        Register agent with AgentField Server and obtain DID identity package.

        Args:
            reasoners: List of reasoner definitions
            skills: List of skill definitions

        Returns:
            True if registration successful, False otherwise
        """
        try:
            logger.debug(
                f"DID registration for agent: {self.agent_node_id} "
                f"({len(reasoners)} reasoners, {len(skills)} skills)"
            )

            # Prepare registration request
            registration_data = {
                "agent_node_id": self.agent_node_id,
                "reasoners": reasoners,
                "skills": skills,
            }

            # Send registration request to AgentField Server
            headers = {"Content-Type": "application/json"}
            headers.update(self._get_auth_headers())
            response = requests.post(
                f"{self.agentfield_server_url}/api/v1/did/register",
                json=registration_data,
                headers=headers,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    # Parse identity package
                    package_data = result["identity_package"]
                    self.identity_package = self._parse_identity_package(package_data)
                    self.enabled = True
                    logger.debug(
                        f"Agent {self.agent_node_id} successfully registered with DID system"
                    )
                    return True
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"DID registration failed: {error_msg}")
                    return False
            else:
                error_msg = f"{response.status_code} - {response.text}"
                logger.error(f"DID registration request failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"Error during DID registration: {e}")
            return False

    def create_execution_context(
        self,
        execution_id: str,
        workflow_id: str,
        session_id: str,
        caller_function: str,
        target_function: str,
    ) -> Optional[DIDExecutionContext]:
        """
        Create execution context for DID-enabled execution.

        Args:
            execution_id: Unique execution identifier
            workflow_id: Workflow identifier
            session_id: Session identifier
            caller_function: Name of calling function
            target_function: Name of target function

        Returns:
            ExecutionContext if successful, None otherwise
        """
        if not self.enabled or not self.identity_package:
            return None

        try:
            # Resolve caller DID
            caller_did = self._get_function_did(caller_function)
            if not caller_did:
                logger.warning(
                    f"Could not resolve DID for caller function: {caller_function}"
                )
                return None

            # Resolve target DID
            target_did = self._get_function_did(target_function)
            if not target_did:
                logger.warning(
                    f"Could not resolve DID for target function: {target_function}"
                )
                return None

            return DIDExecutionContext(
                execution_id=execution_id,
                workflow_id=workflow_id,
                session_id=session_id,
                caller_did=caller_did,
                target_did=target_did,
                agent_node_did=self.identity_package.agent_did.did,
                timestamp=datetime.utcnow(),
            )

        except Exception as e:
            logger.error(f"Error creating execution context: {e}")
            return None

    def get_agent_did(self) -> Optional[str]:
        """Get the agent node DID."""
        if self.identity_package:
            return self.identity_package.agent_did.did
        return None

    def get_function_did(self, function_name: str) -> Optional[str]:
        """
        Get DID for a specific function (reasoner or skill).

        Args:
            function_name: Name of the function

        Returns:
            DID string if found, None otherwise
        """
        return self._get_function_did(function_name)

    def resolve_did(self, did: str) -> Optional[Dict[str, Any]]:
        """
        Resolve a DID to get its public information.

        Args:
            did: DID to resolve

        Returns:
            DID document if successful, None otherwise
        """
        try:
            response = requests.get(
                f"{self.agentfield_server_url}/api/v1/did/resolve/{did}",
                headers=self._get_auth_headers(),
                timeout=10,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to resolve DID {did}: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error resolving DID {did}: {e}")
            return None

    def is_enabled(self) -> bool:
        """Check if DID system is enabled and configured."""
        return self.enabled and self.identity_package is not None

    def get_identity_summary(self) -> Dict[str, Any]:
        """
        Get summary of identity package for debugging/monitoring.

        Returns:
            Dictionary with identity information (no private keys)
        """
        if not self.identity_package:
            return {"enabled": False, "message": "No identity package available"}

        return {
            "enabled": True,
            "agent_did": self.identity_package.agent_did.did,
            "agentfield_server_id": self.identity_package.agentfield_server_id,
            "reasoner_count": len(self.identity_package.reasoner_dids),
            "skill_count": len(self.identity_package.skill_dids),
            "reasoner_dids": {
                name: identity.did
                for name, identity in self.identity_package.reasoner_dids.items()
            },
            "skill_dids": {
                name: identity.did
                for name, identity in self.identity_package.skill_dids.items()
            },
        }

    def _parse_identity_package(
        self, package_data: Dict[str, Any]
    ) -> DIDIdentityPackage:
        """Parse identity package from registration response."""
        # Parse agent DID
        agent_data = package_data["agent_did"]
        agent_did = DIDIdentity(
            did=agent_data["did"],
            private_key_jwk=agent_data["private_key_jwk"],
            public_key_jwk=agent_data["public_key_jwk"],
            derivation_path=agent_data["derivation_path"],
            component_type=agent_data["component_type"],
            function_name=agent_data.get("function_name"),
        )

        # Parse reasoner DIDs
        reasoner_dids = {}
        for name, reasoner_data in package_data["reasoner_dids"].items():
            reasoner_dids[name] = DIDIdentity(
                did=reasoner_data["did"],
                private_key_jwk=reasoner_data["private_key_jwk"],
                public_key_jwk=reasoner_data["public_key_jwk"],
                derivation_path=reasoner_data["derivation_path"],
                component_type=reasoner_data["component_type"],
                function_name=reasoner_data.get("function_name"),
            )

        # Parse skill DIDs
        skill_dids = {}
        for name, skill_data in package_data["skill_dids"].items():
            skill_dids[name] = DIDIdentity(
                did=skill_data["did"],
                private_key_jwk=skill_data["private_key_jwk"],
                public_key_jwk=skill_data["public_key_jwk"],
                derivation_path=skill_data["derivation_path"],
                component_type=skill_data["component_type"],
                function_name=skill_data.get("function_name"),
            )

        return DIDIdentityPackage(
            agent_did=agent_did,
            reasoner_dids=reasoner_dids,
            skill_dids=skill_dids,
            agentfield_server_id=package_data["agentfield_server_id"],
        )

    def _get_function_did(self, function_name: str) -> Optional[str]:
        """Get DID for a function by name."""
        if not self.identity_package:
            return None

        # Check reasoners
        if function_name in self.identity_package.reasoner_dids:
            return self.identity_package.reasoner_dids[function_name].did

        # Check skills
        if function_name in self.identity_package.skill_dids:
            return self.identity_package.skill_dids[function_name].did

        # Return agent DID as fallback
        return self.identity_package.agent_did.did
