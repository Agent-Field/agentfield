from typing import Any, Dict, Optional
from datetime import datetime, timezone
from agentfield.logger import log_debug

class _AgentDiscoveryMixin:
    def _handle_discovery(self) -> dict:
        """
        Handle discovery requests for serverless agent registration.

        Returns agent metadata including reasoners, skills, and configuration
        for automatic registration with the AgentField server.

        Returns:
            dict: Agent metadata for registration
        """
        return {
            "node_id": self.node_id,
            "version": self.version,
            "deployment_type": "serverless",
            "reasoners": [
                {
                    "id": r["id"],
                    "input_schema": r.get("input_schema", {}),
                    "output_schema": r.get("output_schema", {}),
                    "memory_config": r.get("memory_config", {}),
                    "tags": r.get("tags", []),
                }
                for r in self.reasoners
            ],
            "skills": [
                {
                    "id": s["id"],
                    "input_schema": s.get("input_schema", {}),
                    "tags": s.get("tags", []),
                }
                for s in self.skills
            ],
        }

    def _build_callback_discovery_payload(self) -> Optional[Dict[str, Any]]:
        """Prepare discovery metadata for agent registration."""

        if not self.callback_candidates:
            return None

        payload: Dict[str, Any] = {
            "mode": "python-sdk:auto",
            "preferred": self.base_url,
            "callback_candidates": self.callback_candidates,
            "container": self._is_running_in_container(),
            "submitted_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        }

        return payload
    
    def _apply_discovery_response(self, payload: Optional[Dict[str, Any]]) -> None:
        """Update agent networking state from AgentField discovery response."""

        if not payload:
            return

        discovery_section = (
            payload.get("callback_discovery") if isinstance(payload, dict) else None
        )

        resolved = None
        if isinstance(payload, dict):
            resolved = payload.get("resolved_base_url")
        if not resolved and isinstance(discovery_section, dict):
            resolved = (
                discovery_section.get("resolved")
                or discovery_section.get("selected")
                or discovery_section.get("preferred")
            )

        if resolved and resolved != self.base_url:
            log_debug(f"Applying resolved callback URL from AgentField: {resolved}")
            self.base_url = resolved

        if isinstance(discovery_section, dict):
            candidates = discovery_section.get("candidates")
            if isinstance(candidates, list):
                normalized = []
                for candidate in candidates:
                    if isinstance(candidate, str):
                        normalized.append(candidate)
                # Ensure resolved URL is first when present
                if resolved and resolved in normalized:
                    normalized.remove(resolved)
                    normalized.insert(0, resolved)
                elif resolved:
                    normalized.insert(0, resolved)

                if normalized:
                    self.callback_candidates = normalized