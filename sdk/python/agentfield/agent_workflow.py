import inspect
import time
from typing import Any, Callable, Dict, Optional

from agentfield.logger import log_debug, log_warn

from .execution_context import (
    ExecutionContext,
    get_current_context,
    set_execution_context,
    reset_execution_context,
)
from fastapi.encoders import jsonable_encoder


class AgentWorkflow:
    """Workflow helper that keeps local execution metadata in sync with AgentField."""

    def __init__(self, agent_instance):
        self.agent = agent_instance

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    def replace_function_references(
        self, original_func: Callable, tracked_func: Callable, func_name: str
    ) -> None:
        """Replace the agent attribute with the tracked wrapper."""
        setattr(self.agent, func_name, tracked_func)

    async def execute_with_tracking(
        self, original_func: Callable, args: tuple, kwargs: dict
    ) -> Any:
        """
        Execute the wrapped function with automatic workflow instrumentation.
        """

        reasoner_name = getattr(original_func, "__name__", "reasoner")

        parent_context = self._get_parent_context()
        execution_context = self._build_execution_context(reasoner_name, parent_context)

        # Ensure this execution is registered when running under an existing workflow
        execution_context = await self._ensure_execution_registered(
            execution_context, reasoner_name, parent_context
        )

        call_args = args
        call_kwargs = dict(kwargs or {})
        signature = self._safe_signature(original_func)

        if "execution_context" in signature.parameters:
            call_kwargs.setdefault("execution_context", execution_context)

        input_data = self._build_input_payload(signature, call_args, call_kwargs)

        previous_agent_context = getattr(self.agent, "_current_execution_context", None)
        client_context = getattr(self.agent, "client", None)
        previous_client_context = None
        if client_context is not None:
            previous_client_context = getattr(
                client_context, "_current_workflow_context", None
            )

        token = set_execution_context(execution_context)
        self.agent._current_execution_context = execution_context
        if client_context is not None:
            client_context._current_workflow_context = execution_context

        start_time = time.time()
        parent_execution_id = parent_context.execution_id if parent_context else None

        await self.notify_call_start(
            execution_context.execution_id,
            execution_context,
            reasoner_name,
            input_data,
            parent_execution_id=parent_execution_id,
        )

        try:
            result = original_func(*call_args, **call_kwargs)
            if inspect.isawaitable(result):
                result = await result
            duration_ms = int((time.time() - start_time) * 1000)
            await self.notify_call_complete(
                execution_context.execution_id,
                execution_context.workflow_id,
                result,
                duration_ms,
                execution_context,
                input_data=input_data,
                parent_execution_id=parent_execution_id,
            )
            return result
        except Exception as exc:  # pragma: no cover - re-raised
            duration_ms = int((time.time() - start_time) * 1000)
            await self.notify_call_error(
                execution_context.execution_id,
                execution_context.workflow_id,
                str(exc),
                duration_ms,
                execution_context,
                input_data=input_data,
                parent_execution_id=parent_execution_id,
            )
            raise
        finally:
            reset_execution_context(token)
            self.agent._current_execution_context = previous_agent_context
            if client_context is not None:
                client_context._current_workflow_context = previous_client_context

    async def notify_call_start(
        self,
        execution_id: str,
        context: ExecutionContext,
        reasoner_name: str,
        input_data: Dict[str, Any],
        *,
        parent_execution_id: Optional[str] = None,
    ) -> None:
        payload = self._build_event_payload(
            context,
            reasoner_name,
            status="running",
            parent_execution_id=parent_execution_id,
            input_data=input_data,
        )
        await self.fire_and_forget_update(payload)

    async def notify_call_complete(
        self,
        execution_id: str,
        workflow_id: str,
        result: Any,
        duration_ms: int,
        context: ExecutionContext,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        parent_execution_id: Optional[str] = None,
    ) -> None:
        payload = self._build_event_payload(
            context,
            context.reasoner_name,
            status="succeeded",
            parent_execution_id=parent_execution_id,
            input_data=input_data,
        )
        payload["result"] = result
        payload["duration_ms"] = duration_ms
        await self.fire_and_forget_update(payload)

    async def notify_call_error(
        self,
        execution_id: str,
        workflow_id: str,
        error: str,
        duration_ms: int,
        context: ExecutionContext,
        *,
        input_data: Optional[Dict[str, Any]] = None,
        parent_execution_id: Optional[str] = None,
    ) -> None:
        payload = self._build_event_payload(
            context,
            context.reasoner_name,
            status="failed",
            parent_execution_id=parent_execution_id,
            input_data=input_data,
        )
        payload["error"] = error
        payload["duration_ms"] = duration_ms
        await self.fire_and_forget_update(payload)

    async def fire_and_forget_update(self, payload: Dict[str, Any]) -> None:
        """Send workflow update to AgentField when a client is available."""

        client = getattr(self.agent, "client", None)
        base_url = getattr(self.agent, "agentfield_server", None)
        if not client or not hasattr(client, "_async_request") or not base_url:
            return

        url = base_url.rstrip("/") + "/api/v1/workflow/executions/events"
        try:
            safe_payload = jsonable_encoder(payload)
            await client._async_request("POST", url, json=safe_payload)
        except Exception:  # pragma: no cover - best effort logging
            if getattr(self.agent, "dev_mode", False):
                log_debug("Failed to publish workflow update", exc_info=True)

    # --------------------------------------------------------------------- #
    # Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _get_parent_context(self) -> Optional[ExecutionContext]:
        return (
            getattr(self.agent, "_current_execution_context", None)
            or get_current_context()
        )

    def _build_execution_context(
        self,
        reasoner_name: str,
        parent_context: Optional[ExecutionContext],
    ) -> ExecutionContext:
        if parent_context:
            context = parent_context.create_child_context()
            context.reasoner_name = reasoner_name
        else:
            context = ExecutionContext.create_new(
                getattr(self.agent, "node_id", "agent"), reasoner_name
            )
            context.reasoner_name = reasoner_name
        context.agent_instance = self.agent
        return context

    async def _ensure_execution_registered(
        self,
        context: ExecutionContext,
        reasoner_name: str,
        parent_context: Optional[ExecutionContext],
    ) -> ExecutionContext:
        if context.registered:
            return context

        client = getattr(self.agent, "client", None)
        base_url = getattr(self.agent, "agentfield_server", None)
        if not client or not hasattr(client, "_async_request") or not base_url:
            context.registered = True
            return context

        payload = {
            "execution_id": context.execution_id,
            "run_id": context.run_id,
            "workflow_id": context.workflow_id,
            "reasoner_name": reasoner_name,
            "node_id": getattr(self.agent, "node_id", None),
            "parent_execution_id": (
                parent_context.execution_id if parent_context else None
            ),
            "parent_workflow_id": (
                parent_context.workflow_id if parent_context else None
            ),
            "session_id": context.session_id,
            "caller_did": context.caller_did,
            "target_did": context.target_did,
            "agent_node_did": context.agent_node_did,
        }

        url = base_url.rstrip("/") + "/api/v1/workflow/executions"
        try:
            response = await client._async_request("POST", url, json=payload)
            body = response.json() if hasattr(response, "json") else response
            if isinstance(body, dict):
                context.execution_id = body.get("execution_id", context.execution_id)
                context.workflow_id = body.get("workflow_id", context.workflow_id)
                context.run_id = body.get("run_id", context.run_id)
        except Exception as exc:  # pragma: no cover - network failure path
            if getattr(self.agent, "dev_mode", False):
                log_warn(f"Workflow registration failed: {exc}")
        finally:
            context.registered = True

        return context

    @staticmethod
    def _safe_signature(func: Callable) -> inspect.Signature:
        try:
            return inspect.signature(func)
        except (TypeError, ValueError):
            return inspect.Signature()

    def _build_event_payload(
        self,
        context: ExecutionContext,
        reasoner_name: str,
        *,
        status: str,
        parent_execution_id: Optional[str],
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "execution_id": context.execution_id,
            "workflow_id": context.workflow_id,
            "run_id": context.run_id,
            "reasoner_id": reasoner_name,
            "agent_node_id": getattr(self.agent, "node_id", None),
            "status": status,
            "type": reasoner_name,
            "parent_execution_id": parent_execution_id,
            "parent_workflow_id": context.parent_workflow_id,
        }
        if input_data is not None:
            payload["input_data"] = input_data
        return payload

    @staticmethod
    def _build_input_payload(
        signature: inspect.Signature, args: tuple, kwargs: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not signature.parameters:
            return dict(kwargs)

        try:
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
        except Exception:
            # Fallback when binding fails (e.g., C extensions)
            payload = {f"arg_{idx}": value for idx, value in enumerate(args)}
            payload.update(kwargs)
            return payload

        payload = {}
        for name, value in bound.arguments.items():
            if name == "self":
                continue
            payload[name] = value
        return payload


__all__ = ["AgentWorkflow"]
