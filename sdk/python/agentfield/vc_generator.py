"""
VC Generator for AgentField SDK

Handles Verifiable Credentials (VC) generation and verification for agent executions.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import requests
from datetime import datetime

from .logger import get_logger
from .status import normalize_status

logger = get_logger(__name__)


@dataclass
class ExecutionVC:
    """Represents a verifiable credential for an execution."""

    vc_id: str
    execution_id: str
    workflow_id: str
    session_id: str
    issuer_did: str
    target_did: str
    caller_did: str
    vc_document: Dict[str, Any]
    signature: str
    input_hash: str
    output_hash: str
    status: str
    created_at: datetime


@dataclass
class WorkflowVC:
    """Represents a workflow-level verifiable credential."""

    workflow_id: str
    session_id: str
    component_vcs: List[str]
    workflow_vc_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    total_steps: int
    completed_steps: int


class VCGenerator:
    """
    Generates and manages verifiable credentials for agent executions.

    Handles:
    - Execution VC generation
    - Workflow VC aggregation
    - VC verification
    - Integration with AgentField Server
    """

    def __init__(self, agentfield_server_url: str, api_key: Optional[str] = None):
        """
        Initialize VC Generator.

        Args:
            agentfield_server_url: URL of the AgentField Server
            api_key: Optional API key for authentication
        """
        self.agentfield_server_url = agentfield_server_url.rstrip("/")
        self.api_key = api_key
        self.enabled = False

    def _get_auth_headers(self) -> Dict[str, str]:
        """Return auth headers if API key is configured."""
        if not self.api_key:
            return {}
        return {"X-API-Key": self.api_key}

    def set_enabled(self, enabled: bool):
        """Enable or disable VC generation."""
        self.enabled = enabled

    def generate_execution_vc(
        self,
        execution_context: Any,
        input_data: Any,
        output_data: Any,
        status: str,
        error_message: Optional[str] = None,
        duration_ms: int = 0,
    ) -> Optional[ExecutionVC]:
        """
        Generate a verifiable credential for an execution.

        Args:
            execution_context: ExecutionContext from DIDManager
            input_data: Input data for the execution
            output_data: Output data from the execution
            status: Execution status (success, error, etc.)
            error_message: Error message if execution failed
            duration_ms: Execution duration in milliseconds

        Returns:
            ExecutionVC if successful, None otherwise
        """
        if not self.enabled:
            return None

        try:
            logger.debug(
                f"Generating VC for execution: {execution_context.execution_id}"
            )

            # Prepare VC generation request
            vc_data = {
                "execution_context": {
                    "execution_id": execution_context.execution_id,
                    "workflow_id": execution_context.workflow_id,
                    "session_id": execution_context.session_id,
                    "caller_did": execution_context.caller_did,
                    "target_did": execution_context.target_did,
                    "agent_node_did": execution_context.agent_node_did,
                    "timestamp": execution_context.timestamp.isoformat() + "Z"
                    if execution_context.timestamp.tzinfo is None
                    else execution_context.timestamp.isoformat(),
                },
                "input_data": self._serialize_data_for_json(input_data),
                "output_data": self._serialize_data_for_json(output_data),
                "status": normalize_status(status),
                "error_message": error_message,
                "duration_ms": duration_ms,
            }

            # Send VC generation request to AgentField Server
            headers = {"Content-Type": "application/json"}
            headers.update(self._get_auth_headers())
            response = requests.post(
                f"{self.agentfield_server_url}/api/v1/execution/vc",
                json=vc_data,
                headers=headers,
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                logger.debug(
                    f"VC generation successful for execution: {execution_context.execution_id}"
                )
                return self._parse_execution_vc(result)
            else:
                logger.warning(
                    f"Failed to generate execution VC: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Error generating execution VC: {e}")
            return None

    def verify_vc(self, vc_document: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Verify a verifiable credential.

        Args:
            vc_document: VC document to verify

        Returns:
            Verification result if successful, None otherwise
        """
        try:
            verification_data = {"vc_document": vc_document}

            headers = {"Content-Type": "application/json"}
            headers.update(self._get_auth_headers())
            response = requests.post(
                f"{self.agentfield_server_url}/api/v1/did/verify",
                json=verification_data,
                headers=headers,
                timeout=10,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(
                    f"Failed to verify VC: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Error verifying VC: {e}")
            return None

    def get_workflow_vc_chain(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the complete VC chain for a workflow.

        Args:
            workflow_id: Workflow identifier

        Returns:
            Workflow VC chain if successful, None otherwise
        """
        try:
            response = requests.get(
                f"{self.agentfield_server_url}/api/v1/did/workflow/{workflow_id}/vc-chain",
                headers=self._get_auth_headers(),
                timeout=10,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(
                    f"Failed to get workflow VC chain: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Error getting workflow VC chain: {e}")
            return None

    def create_workflow_vc(
        self, workflow_id: str, session_id: str, execution_vc_ids: List[str]
    ) -> Optional[WorkflowVC]:
        """
        Create a workflow-level VC that aggregates execution VCs.

        Args:
            workflow_id: Workflow identifier
            session_id: Session identifier
            execution_vc_ids: List of execution VC IDs to aggregate

        Returns:
            WorkflowVC if successful, None otherwise
        """
        try:
            workflow_data = {
                "session_id": session_id,
                "execution_vc_ids": execution_vc_ids,
            }

            headers = {"Content-Type": "application/json"}
            headers.update(self._get_auth_headers())
            response = requests.post(
                f"{self.agentfield_server_url}/api/v1/did/workflow/{workflow_id}/vc",
                json=workflow_data,
                headers=headers,
                timeout=10,
            )

            if response.status_code == 200:
                result = response.json()
                return self._parse_workflow_vc(result)
            else:
                logger.warning(
                    f"Failed to create workflow VC: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Error creating workflow VC: {e}")
            return None

    def export_vcs(
        self, filters: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Export VCs for external verification.

        Args:
            filters: Optional filters for VC export

        Returns:
            List of VCs if successful, None otherwise
        """
        try:
            params = filters or {}

            response = requests.get(
                f"{self.agentfield_server_url}/api/v1/did/export/vcs",
                params=params,
                headers=self._get_auth_headers(),
                timeout=30,
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(
                    f"Failed to export VCs: {response.status_code} - {response.text}"
                )
                return None

        except Exception as e:
            logger.error(f"Error exporting VCs: {e}")
            return None

    def is_enabled(self) -> bool:
        """Check if VC generation is enabled."""
        return self.enabled

    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for VC generation."""
        if data is None:
            return b""

        if isinstance(data, (str, bytes)):
            return data.encode() if isinstance(data, str) else data

        # For complex objects, serialize to JSON
        try:
            return json.dumps(data, sort_keys=True).encode()
        except Exception:
            return str(data).encode()

    def _serialize_data_for_json(self, data: Any) -> str:
        """Serialize data for JSON transmission as base64-encoded string."""
        import base64

        if data is None:
            return ""

        # Convert data to string first
        if isinstance(data, str):
            data_str = data
        elif isinstance(data, bytes):
            data_str = data.decode("utf-8", errors="replace")
        else:
            # For complex objects, serialize to JSON string
            try:
                data_str = json.dumps(data, sort_keys=True)
            except Exception:
                data_str = str(data)

        # Encode as base64 for transmission to Go server
        return base64.b64encode(data_str.encode("utf-8")).decode("ascii")

    def _parse_execution_vc(self, vc_data: Dict[str, Any]) -> ExecutionVC:
        """Parse execution VC from API response."""
        return ExecutionVC(
            vc_id=vc_data["vc_id"],
            execution_id=vc_data["execution_id"],
            workflow_id=vc_data["workflow_id"],
            session_id=vc_data["session_id"],
            issuer_did=vc_data["issuer_did"],
            target_did=vc_data["target_did"],
            caller_did=vc_data["caller_did"],
            vc_document=vc_data["vc_document"],
            signature=vc_data["signature"],
            input_hash=vc_data["input_hash"],
            output_hash=vc_data["output_hash"],
            status=vc_data["status"],
            created_at=datetime.fromisoformat(
                vc_data["created_at"].replace("Z", "+00:00")
            ),
        )

    def _parse_workflow_vc(self, vc_data: Dict[str, Any]) -> WorkflowVC:
        """Parse workflow VC from API response."""
        end_time = None
        if vc_data.get("end_time"):
            end_time = datetime.fromisoformat(
                vc_data["end_time"].replace("Z", "+00:00")
            )

        return WorkflowVC(
            workflow_id=vc_data["workflow_id"],
            session_id=vc_data["session_id"],
            component_vcs=vc_data["component_vcs"],
            workflow_vc_id=vc_data["workflow_vc_id"],
            status=vc_data["status"],
            start_time=datetime.fromisoformat(
                vc_data["start_time"].replace("Z", "+00:00")
            ),
            end_time=end_time,
            total_steps=vc_data["total_steps"],
            completed_steps=vc_data["completed_steps"],
        )


class VCContext:
    """
    Context manager for VC-enabled execution.

    Automatically generates VCs for code blocks when used as a context manager.
    """

    def __init__(
        self, vc_generator: VCGenerator, execution_context: Any, function_name: str
    ):
        """
        Initialize VC context.

        Args:
            vc_generator: VCGenerator instance
            execution_context: ExecutionContext from DIDManager
            function_name: Name of the function being executed
        """
        self.vc_generator = vc_generator
        self.execution_context = execution_context
        self.function_name = function_name
        self.start_time = None
        self.input_data = None
        self.output_data = None
        self.error_message = None
        self.status = "success"

    def __enter__(self):
        """Enter the context manager."""
        self.start_time = datetime.utcnow()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and generate VC."""
        if not self.vc_generator.is_enabled():
            return

        # Calculate duration
        if self.start_time:
            duration_ms = int(
                (datetime.utcnow() - self.start_time).total_seconds() * 1000
            )
        else:
            duration_ms = 0

        # Set status based on exception
        if exc_type is not None:
            self.status = "error"
            self.error_message = str(exc_val) if exc_val else "Unknown error"

        # Generate VC
        try:
            vc = self.vc_generator.generate_execution_vc(
                execution_context=self.execution_context,
                input_data=self.input_data,
                output_data=self.output_data,
                status=self.status,
                error_message=self.error_message,
                duration_ms=duration_ms,
            )

            if vc:
                logger.debug(
                    f"Generated VC {vc.vc_id} for execution {self.execution_context.execution_id}"
                )
            else:
                logger.warning(
                    f"Failed to generate VC for execution {self.execution_context.execution_id}"
                )

        except Exception as e:
            logger.error(f"Error in VC context manager: {e}")

    def set_input_data(self, data: Any):
        """Set input data for VC generation."""
        self.input_data = data

    def set_output_data(self, data: Any):
        """Set output data for VC generation."""
        self.output_data = data
