from typing import Optional, Callable
import time
from agentfield.execution_context import ExecutionContext
from agentfield.logger import log_warn

class _AgentServerlessMixin:
    def handle_serverless(
        self, event: dict, adapter: Optional[Callable] = None
    ) -> dict:
        """
        Universal serverless handler for executing reasoners and skills.

        This method enables agents to run in serverless environments (AWS Lambda,
        Google Cloud Functions, Cloud Run, Kubernetes Jobs, etc.) by providing
        a simple entry point that parses the event, executes the target function,
        and returns the result.

        Special Endpoints:
            - /discover: Returns agent metadata for AgentField server registration
            - /execute: Executes reasoners and skills

        Args:
            event (dict): Serverless event containing:
                - path: Request path (/discover or /execute)
                - action: Alternative to path (discover or execute)
                - reasoner: Name of the reasoner to execute (for execution)
                - input: Input parameters for the function (for execution)

        Returns:
            dict: Execution result with status and output, or discovery metadata

        Example:
            ```python
            # AWS Lambda handler with API Gateway
            from agentfield import Agent

            app = Agent("my_agent", auto_register=False)

            @app.reasoner()
            async def analyze(text: str) -> dict:
                return {"result": text.upper()}

            def lambda_handler(event, context):
                # Handle both discovery and execution
                return app.handle_serverless(event)
            ```
        """
        import asyncio

        if adapter:
            try:
                event = adapter(event) or event
            except Exception as exc:  # pragma: no cover - adapter failures
                return {
                    "statusCode": 400,
                    "body": {"error": f"serverless adapter failed: {exc}"},
                }

        # Check if this is a discovery request
        path = event.get("path") or event.get("rawPath") or ""
        action = event.get("action", "")

        if path == "/discover" or path.endswith("/discover") or action == "discover":
            # Return agent metadata for AgentField server registration
            return self._handle_discovery()

        # Auto-register with AgentField if needed (for execution requests)
        if self.auto_register and not self.agentfield_connected:
            try:
                # Attempt registration (non-blocking)
                self.agentfield_handler._register_agent()
                self.agentfield_connected = True
            except Exception as e:
                if self.dev_mode:
                    log_warn(f"Auto-registration failed: {e}")

        # Serverless invocations arrive via the control plane; mark as connected so
        # cross-agent calls can route through the gateway without a lease loop.
        self.agentfield_connected = True
        # Serverless handlers should avoid async execute polling; force sync path.
        if getattr(self.async_config, "enable_async_execution", True):
            self.async_config.enable_async_execution = False

        # Parse event format for execution
        reasoner_name = (
            event.get("reasoner") or event.get("target") or event.get("skill")
        )
        if not reasoner_name and path:
            # Support paths like /execute/<target> or /reasoners/<name>
            cleaned_path = path.split("?", 1)[0].strip("/")
            parts = cleaned_path.split("/")
            if parts and parts[0] not in ("", "discover"):
                if len(parts) >= 2 and parts[0] in ("execute", "reasoners", "skills"):
                    reasoner_name = parts[1]
                elif parts[0] in ("execute", "reasoners", "skills"):
                    reasoner_name = None
                elif parts:
                    reasoner_name = parts[-1]

        input_data = event.get("input") or event.get("input_data", {})
        execution_context_data = (
            event.get("execution_context") or event.get("executionContext") or {}
        )

        if not reasoner_name:
            return {
                "statusCode": 400,
                "body": {"error": "Missing 'reasoner' or 'target' in event"},
            }

        # Create execution context
        exec_id = execution_context_data.get(
            "execution_id", f"exec_{int(time.time() * 1000)}"
        )
        run_id = execution_context_data.get("run_id") or execution_context_data.get(
            "workflow_id"
        )
        if not run_id:
            run_id = f"wf_{int(time.time() * 1000)}"
        workflow_id = execution_context_data.get("workflow_id", run_id)

        execution_context = ExecutionContext(
            run_id=run_id,
            execution_id=exec_id,
            agent_instance=self,
            agent_node_id=self.node_id,
            reasoner_name=reasoner_name,
            parent_execution_id=execution_context_data.get("parent_execution_id"),
            session_id=execution_context_data.get("session_id"),
            actor_id=execution_context_data.get("actor_id"),
            caller_did=execution_context_data.get("caller_did"),
            target_did=execution_context_data.get("target_did"),
            agent_node_did=execution_context_data.get(
                "agent_node_did", execution_context_data.get("agent_did")
            ),
            workflow_id=workflow_id,
            parent_workflow_id=execution_context_data.get("parent_workflow_id"),
            root_workflow_id=execution_context_data.get("root_workflow_id"),
        )

        # Set execution context
        self._current_execution_context = execution_context

        try:
            # Find and execute the target function
            if hasattr(self, reasoner_name):
                func = getattr(self, reasoner_name)

                # Execute function (sync or async)
                if asyncio.iscoroutinefunction(func):
                    result = asyncio.run(func(**input_data))
                else:
                    result = func(**input_data)

                return {"statusCode": 200, "body": result}
            else:
                return {
                    "statusCode": 404,
                    "body": {"error": f"Function '{reasoner_name}' not found"},
                }

        except Exception as e:
            return {"statusCode": 500, "body": {"error": str(e)}}
        finally:
            # Clean up execution context
            self._current_execution_context = None