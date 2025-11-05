"""
Minimal execution context helpers for the simplified run-based pipeline.
"""

import contextvars
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional


_RUN_HEADER = "X-Run-ID"
_EXECUTION_HEADER = "X-Execution-ID"
_PARENT_EXECUTION_HEADER = "X-Parent-Execution-ID"
_SESSION_HEADER = "X-Session-ID"
_ACTOR_HEADER = "X-Actor-ID"
_CALLER_DID_HEADER = "X-Caller-DID"
_TARGET_DID_HEADER = "X-Target-DID"
_AGENT_DID_HEADER = "X-Agent-Node-DID"


@dataclass
class ExecutionContext:
    """Captures the inbound execution metadata for a reasoner invocation."""

    run_id: str
    execution_id: str
    agent_instance: Any
    reasoner_name: str
    parent_execution_id: Optional[str] = None
    depth: int = 0
    started_at: float = 0.0
    session_id: Optional[str] = None
    actor_id: Optional[str] = None
    caller_did: Optional[str] = None
    target_did: Optional[str] = None
    agent_node_did: Optional[str] = None
    # Compatibility fields retained for existing integrations
    workflow_id: Optional[str] = None
    parent_workflow_id: Optional[str] = None
    root_workflow_id: Optional[str] = None
    registered: bool = False

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = time.time()
        if not self.workflow_id:
            self.workflow_id = self.run_id

    # ------------------------------------------------------------------
    # Header helpers

    def to_headers(self) -> Dict[str, str]:
        """
        Produce the headers that should be forwarded for downstream executions.

        We only send the run identifier and the current execution as the parent.
        The AgentField backend issues fresh execution IDs for child nodes.
        """

        parent_execution = self.parent_execution_id or self.execution_id

        headers: Dict[str, str] = {
            _RUN_HEADER: self.run_id,
            "X-Workflow-ID": self.workflow_id or self.run_id,
            _PARENT_EXECUTION_HEADER: parent_execution,
            _EXECUTION_HEADER: self.execution_id,
            "X-Workflow-Run-ID": self.run_id,
        }

        if self.session_id:
            headers[_SESSION_HEADER] = self.session_id
        if self.actor_id:
            headers[_ACTOR_HEADER] = self.actor_id
        if self.parent_workflow_id:
            headers["X-Parent-Workflow-ID"] = self.parent_workflow_id
        if self.root_workflow_id:
            headers["X-Root-Workflow-ID"] = self.root_workflow_id
        if self.caller_did:
            headers[_CALLER_DID_HEADER] = self.caller_did
        if self.target_did:
            headers[_TARGET_DID_HEADER] = self.target_did
        if self.agent_node_did:
            headers[_AGENT_DID_HEADER] = self.agent_node_did

        return headers

    def child_context(self) -> "ExecutionContext":
        """
        Create an in-process child context for local tracking.

        The new execution ID is generated locally so callers can reference
        it while awaiting downstream responses. The AgentField server will still
        assign its own execution ID when the child request is submitted.
        """

        return ExecutionContext(
            run_id=self.run_id,
            execution_id=generate_execution_id(),
            agent_instance=self.agent_instance,
            reasoner_name=self.reasoner_name,
            parent_execution_id=self.execution_id,
            depth=self.depth + 1,
            session_id=self.session_id,
            actor_id=self.actor_id,
            caller_did=self.caller_did,
            target_did=self.target_did,
            agent_node_did=self.agent_node_did,
            workflow_id=self.workflow_id,
            parent_workflow_id=self.workflow_id,
            root_workflow_id=self.root_workflow_id or self.workflow_id,
        )

    def create_child_context(self) -> "ExecutionContext":
        """
        Backwards-compatible wrapper returning a derived child context.
        """

        return self.child_context()

    # ------------------------------------------------------------------
    # Factories

    @classmethod
    def from_request(cls, request, agent_node_id: str) -> "ExecutionContext":
        """
        Build an execution context from inbound FastAPI request headers.

        We accept both canonical and lowercase header names to match Starlette's
        header behavior.
        """

        headers = request.headers

        def _read(name: str) -> Optional[str]:
            lower = name.lower()
            return headers.get(lower) or headers.get(name)

        workflow_id = _read("X-Workflow-ID")
        run_id = _read(_RUN_HEADER) or workflow_id or generate_run_id()
        if not workflow_id:
            workflow_id = run_id
        execution_id = _read(_EXECUTION_HEADER) or generate_execution_id()
        parent_execution_id = _read(_PARENT_EXECUTION_HEADER)
        session_id = _read(_SESSION_HEADER)
        actor_id = _read(_ACTOR_HEADER)
        caller_did = _read(_CALLER_DID_HEADER)
        target_did = _read(_TARGET_DID_HEADER)
        agent_node_did = _read(_AGENT_DID_HEADER)
        parent_workflow_id = _read("X-Parent-Workflow-ID")
        root_workflow_id = _read("X-Root-Workflow-ID")

        from .agent_registry import get_current_agent_instance

        return cls(
            run_id=run_id,
            execution_id=execution_id,
            agent_instance=get_current_agent_instance(),
            reasoner_name="unknown",
            parent_execution_id=parent_execution_id,
            session_id=session_id,
            actor_id=actor_id,
            caller_did=caller_did,
            target_did=target_did,
            agent_node_did=agent_node_did,
            workflow_id=workflow_id,
            parent_workflow_id=parent_workflow_id,
            root_workflow_id=root_workflow_id,
            registered=True,
        )

    @classmethod
    def new_root(
        cls, agent_node_id: str, reasoner_name: str = "root"
    ) -> "ExecutionContext":
        """Create a brand-new root execution context for manual invocation."""

        from .agent_registry import get_current_agent_instance

        run_id = generate_run_id()
        return cls(
            run_id=run_id,
            execution_id=generate_execution_id(),
            agent_instance=get_current_agent_instance(),
            reasoner_name=reasoner_name,
            parent_execution_id=None,
            workflow_id=run_id,
            root_workflow_id=run_id,
        )

    @classmethod
    def create_new(cls, agent_node_id: str, workflow_name: str) -> "ExecutionContext":
        """
        Backwards-compatible wrapper for legacy code that expected create_new().
        Generates a fresh root execution context using the provided workflow name.
        """

        context = cls.new_root(agent_node_id, workflow_name)
        context.reasoner_name = workflow_name
        return context


class ExecutionContextManager:
    """Async-safe access to the current execution context."""

    def __init__(self) -> None:
        self._context_var: contextvars.ContextVar[Optional[ExecutionContext]] = (
            contextvars.ContextVar("execution_context", default=None)
        )

    def get_current_context(self) -> Optional[ExecutionContext]:
        return self._context_var.get()

    def set_context(self, context: ExecutionContext) -> contextvars.Token:
        return self._context_var.set(context)

    def reset_context(self, token: contextvars.Token) -> None:
        self._context_var.reset(token)


_context_manager = ExecutionContextManager()


def get_current_context() -> Optional[ExecutionContext]:
    return _context_manager.get_current_context()


def set_execution_context(context: ExecutionContext):
    return _context_manager.set_context(context)


def reset_execution_context(token: contextvars.Token) -> None:
    _context_manager.reset_context(token)


def generate_execution_id() -> str:
    timestamp = int(time.time() * 1000)
    return f"exec_{timestamp}_{uuid.uuid4().hex[:8]}"


def generate_run_id() -> str:
    timestamp = int(time.time() * 1000)
    return f"run_{timestamp}_{uuid.uuid4().hex[:8]}"
