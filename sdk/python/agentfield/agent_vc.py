
from typing import Any, Dict, Optional

from agentfield.logger import log_debug, log_info, log_warn, log_error
from agentfield.did_manager import DIDManager
from agentfield.vc_generator import VCGenerator


class _AgentVCMixin:
    def _initialize_did_system(self):
        """Initialize DID and VC components."""
        try:
            # Initialize DID Manager
            self.did_manager = DIDManager(
                self.agentfield_server, self.node_id, self.api_key
            )

            # Initialize VC Generator
            self.vc_generator = VCGenerator(self.agentfield_server, self.api_key)

            if self.dev_mode:
                log_debug("DID system initialized")

        except Exception as e:
            if self.dev_mode:
                log_error(f"Failed to initialize DID system: {e}")
            self.did_manager = None
            self.vc_generator = None
    
    def _populate_execution_context_with_did(
        self, execution_context, did_execution_context
    ):
        """
        Populate the execution context with DID information.

        Args:
            execution_context: The main ExecutionContext
            did_execution_context: The DIDExecutionContext with DID info
        """
        if did_execution_context:
            execution_context.session_id = did_execution_context.session_id
            execution_context.caller_did = did_execution_context.caller_did
            execution_context.target_did = did_execution_context.target_did
            execution_context.agent_node_did = did_execution_context.agent_node_did

    def _agent_vc_default(self) -> bool:
        """Resolve the agent-level VC default, falling back to enabled."""
        return True if self._agent_vc_enabled is None else self._agent_vc_enabled
    
    def _set_reasoner_vc_override(
        self, reasoner_id: str, value: Optional[bool]
    ) -> None:
        if value is None:
            self._reasoner_vc_overrides.pop(reasoner_id, None)
        else:
            self._reasoner_vc_overrides[reasoner_id] = value

    def _set_skill_vc_override(self, skill_id: str, value: Optional[bool]) -> None:
        if value is None:
            self._skill_vc_overrides.pop(skill_id, None)
        else:
            self._skill_vc_overrides[skill_id] = value

    def _effective_component_vc_setting(
        self, component_id: str, overrides: Dict[str, bool]
    ) -> bool:
        if component_id in overrides:
            return overrides[component_id]
        return self._agent_vc_default()
    
    def _should_generate_vc(
        self, component_id: str, overrides: Dict[str, bool]
    ) -> bool:
        if (
            not self.did_enabled
            or not self.vc_generator
            or not self.vc_generator.is_enabled()
        ):
            return False
        return self._effective_component_vc_setting(component_id, overrides)
    
    def _build_agent_metadata(self) -> Optional[Dict[str, Any]]:
        """Build agent metadata (description, tags, author) for registration payload."""
        metadata: Dict[str, Any] = {}
        if self.description:
            metadata["description"] = self.description
        if self.agent_tags:
            metadata["tags"] = self.agent_tags
        if self.author:
            metadata["author"] = self.author
        return metadata if metadata else None
    
    def _build_vc_metadata(self) -> Dict[str, Any]:
        """Produce a serializable VC policy snapshot for control-plane visibility."""
        effective_reasoners = {
            reasoner["id"]: self._effective_component_vc_setting(
                reasoner["id"], self._reasoner_vc_overrides
            )
            for reasoner in self.reasoners
            if "id" in reasoner
        }
        effective_skills = {
            skill["id"]: self._effective_component_vc_setting(
                skill["id"], self._skill_vc_overrides
            )
            for skill in self.skills
            if "id" in skill
        }

        return {
            "agent_default": self._agent_vc_default(),
            "reasoner_overrides": dict(self._reasoner_vc_overrides),
            "skill_overrides": dict(self._skill_vc_overrides),
            "effective_reasoners": effective_reasoners,
            "effective_skills": effective_skills,
        }
    
    async def _generate_vc_async(
        self,
        vc_generator,
        did_execution_context,
        function_name,
        input_data,
        output_data,
        status="success",
        error_message=None,
        duration_ms=0,
    ):
        """
        Generate VC asynchronously without blocking execution.

        Args:
            vc_generator: VCGenerator instance
            did_execution_context: DID execution context
            function_name: Name of the executed function
            input_data: Input data for the execution
            output_data: Output data from the execution
            status: Execution status
            error_message: Error message if any
            duration_ms: Execution duration in milliseconds
        """
        try:
            if vc_generator and vc_generator.is_enabled():
                vc = vc_generator.generate_execution_vc(
                    execution_context=did_execution_context,
                    input_data=input_data,
                    output_data=output_data,
                    status=status,
                    error_message=error_message,
                    duration_ms=duration_ms,
                )
                if vc:
                    log_info(f"Generated VC {vc.vc_id} for {function_name}")
        except Exception as e:
            log_warn(f"Failed to generate VC for {function_name}: {e}")