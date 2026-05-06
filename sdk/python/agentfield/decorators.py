"""
Enhanced decorators for AgentField SDK with automatic workflow tracking.
Provides always-on workflow tracking for reasoner calls.
"""

import asyncio
import functools
import inspect
import time
from typing import Any, Callable, Dict, List, Optional, Union

from agentfield.logger import log_warn

from .execution_context import (
    ExecutionContext,
    get_current_context,
    set_execution_context,
    reset_execution_context,
)
from .agent_registry import get_current_agent_instance
from .triggers import EventTrigger, ScheduleTrigger, Trigger
from .types import ReasonerDefinition
from .pydantic_utils import convert_function_args, should_convert_args
from pydantic import ValidationError

# Attribute used by the @on_event / @on_schedule sugar to stage triggers on a
# function before the outer @reasoner consumes them. Keeping this off of the
# function signature lets the canonical `triggers=[...]` kwarg coexist with
# stacked sugar without either form leaking into the other.
_PENDING_TRIGGERS_ATTR = "_pending_triggers"


def _code_origin(fn) -> Optional[str]:
    """Capture the source code location (file:line) of a function.

    Returns the fully qualified path to the file and the line number where the
    function is declared, in the format 'path/to/file.py:42'.
    """
    try:
        src = inspect.getsourcefile(fn)
        line = fn.__code__.co_firstlineno
        if src and line:
            return f"{src}:{line}"
    except Exception:
        pass
    return None


def reasoner(
    func=None,
    *,
    path: Optional[str] = None,
    tags: Optional[List[str]] = None,
    description: Optional[str] = None,
    track_workflow: bool = True,
    triggers: Optional[List[Trigger]] = None,
    accepts_webhook: Optional[Union[bool, str]] = None,
    **kwargs,
):
    """Enhanced reasoner decorator with automatic workflow tracking and triggers.

    Examples:
        @reasoner
        async def hello(input, ctx): ...

        @reasoner(tags=["payments"])
        @reasoner(triggers=[EventTrigger(source="stripe", types=["payment_intent.succeeded"], secret_env="STRIPE_SECRET")])
        async def handle_payment(input, ctx): ...

        # Sugar (desugars to triggers=[...] internally):
        @reasoner()
        @on_schedule("*/5 * * * *")
        async def poll(input, ctx): ...

        # Explicitly opt into/out of webhook input:
        @reasoner(accepts_webhook=True)
        async def webhook_ready(input, ctx): ...

        @reasoner(accepts_webhook=False)
        async def not_for_webhooks(input, ctx): ...

    Args:
        triggers: Typed trigger bindings (EventTrigger / ScheduleTrigger) that
            cause this reasoner to fire when the named external Source emits
            a matching event. The control plane upserts a code-managed Trigger
            row per binding at agent registration time.
        path: Custom API endpoint path.
        tags: Tags for grouping and authorization.
        description: Human-readable description (defaults to the docstring).
        track_workflow: Whether to enable automatic workflow tracking.
        accepts_webhook: Controls whether UI-managed triggers can invoke this reasoner.
            True = explicitly opt in; False = explicitly refuse; None (default) = auto-detect
            from triggers (True if any declared) or default to "warn".
        **kwargs: Additional metadata stored on the function.
    """

    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        async def wrapper(*args, **kwargs):
            if track_workflow:
                # Execute with automatic workflow tracking
                return await _execute_with_tracking(f, *args, **kwargs)
            else:
                # Execute without tracking
                if asyncio.iscoroutinefunction(f):
                    return await f(*args, **kwargs)
                else:
                    return f(*args, **kwargs)

        # Store comprehensive metadata on the function
        wrapper._is_reasoner = True
        wrapper._track_workflow = track_workflow
        wrapper._reasoner_name = f.__name__
        wrapper._original_func = f
        wrapper._reasoner_path = path
        wrapper._reasoner_tags = tags or []
        wrapper._reasoner_description = (
            description or f.__doc__ or f"Reasoner: {f.__name__}"
        )

        # Merge canonical triggers= kwarg with any sugar-staged bindings on the
        # wrapped function. Sugar must run inside @reasoner so order is
        # @reasoner outermost, @on_event / @on_schedule inner.
        merged: List[Trigger] = []
        if triggers:
            merged.extend(triggers)
        pending = getattr(f, _PENDING_TRIGGERS_ATTR, None)
        if pending:
            merged.extend(pending)
            try:
                delattr(f, _PENDING_TRIGGERS_ATTR)
            except AttributeError:
                pass
        # Stamp code_origin on triggers that don't already have it

        origin = _code_origin(f)

        if origin:
            for trigger in merged:
                if not getattr(trigger, "code_origin", None):
                    trigger.code_origin = origin

        wrapper._reasoner_triggers = merged

        # Resolve accepts_webhook value:
        # - If explicitly passed (True/False/str), use that
        # - Else if any triggers declared, auto-set to True
        # - Else default to "warn"
        resolved_accepts_webhook: Union[bool, str]
        if accepts_webhook is not None:
            resolved_accepts_webhook = accepts_webhook
        elif merged:
            resolved_accepts_webhook = True
        else:
            resolved_accepts_webhook = "warn"
        wrapper._accepts_webhook = resolved_accepts_webhook

        # Store any additional metadata
        for key, value in kwargs.items():
            setattr(wrapper, f"_reasoner_{key}", value)

        return wrapper

    # Handle both @reasoner and @reasoner(...) syntax
    if func is None:
        # Called as @reasoner(track_workflow=False) or @reasoner(path="/custom")
        return decorator
    else:
        # Called as @reasoner (no parentheses)
        return decorator(func)


def on_event(
    source: str,
    *,
    types: Optional[List[str]] = None,
    secret_env: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
):
    """Sugar that stages an :class:`EventTrigger` for the next outer ``@reasoner``.

    Equivalent to passing ``triggers=[EventTrigger(...)]`` on ``@reasoner``.
    Order matters: ``@reasoner`` MUST be the outermost decorator. Use the
    canonical ``triggers=[...]`` form when you prefer to keep all reasoner
    config in a single place.
    """

    def decorator(func: Callable) -> Callable:
        binding = EventTrigger(
            source=source,
            types=list(types or []),
            secret_env=secret_env,
            config=dict(config or {}),
        )
        # Capture code origin automatically
        if not binding.code_origin:
            binding.code_origin = _code_origin(func)
        existing = getattr(func, _PENDING_TRIGGERS_ATTR, None)
        if existing is None:
            existing = []
            setattr(func, _PENDING_TRIGGERS_ATTR, existing)
        existing.append(binding)
        return func

    return decorator


def on_schedule(cron: str, *, timezone: str = "UTC"):
    """Sugar that stages a :class:`ScheduleTrigger` for the next outer ``@reasoner``.

    Equivalent to ``triggers=[ScheduleTrigger(cron=cron, timezone=timezone)]``.
    """

    def decorator(func: Callable) -> Callable:
        binding = ScheduleTrigger(cron=cron, timezone=timezone)
        # Capture code origin automatically
        if not binding.code_origin:
            binding.code_origin = _code_origin(func)
        existing = getattr(func, _PENDING_TRIGGERS_ATTR, None)
        if existing is None:
            existing = []
            setattr(func, _PENDING_TRIGGERS_ATTR, existing)
        existing.append(binding)
        return func

    return decorator


async def _execute_with_tracking(func: Callable, *args, **kwargs) -> Any:
    """
    Core function that handles automatic workflow tracking for reasoner calls.

    Args:
        func: The reasoner function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the function execution
    """
    # Get current execution context
    current_context = get_current_context()

    # Get agent instance (from context or global registry)
    agent_instance = get_current_agent_instance()

    if not agent_instance:
        # No agent context - execute without tracking
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    workflow_handler = getattr(agent_instance, "workflow_handler", None)
    reasoner_name = getattr(func, "__name__", "reasoner")

    # Generate execution metadata
    # Build a child context when executing under an existing workflow; otherwise create a root context
    if current_context:
        execution_context = current_context.create_child_context()
        execution_context.reasoner_name = reasoner_name
        parent_context = current_context
    else:
        workflow_name = reasoner_name
        if hasattr(agent_instance, "node_id"):
            workflow_name = f"{agent_instance.node_id}_{workflow_name}"
        execution_context = ExecutionContext.new_root(
            agent_node_id=getattr(agent_instance, "node_id", "agent"),
            reasoner_name=workflow_name,
        )
        execution_context.reasoner_name = reasoner_name
        execution_context.agent_instance = agent_instance
        parent_context = None

    # Align run/session metadata with the parent context so registration inherits the workflow run
    if parent_context:
        execution_context.run_id = parent_context.run_id
        execution_context.session_id = parent_context.session_id
        execution_context.caller_did = parent_context.caller_did
        execution_context.target_did = parent_context.target_did
        execution_context.agent_node_did = parent_context.agent_node_did
    execution_context.agent_instance = agent_instance

    if workflow_handler is not None:
        execution_context = await workflow_handler._ensure_execution_registered(
            execution_context, reasoner_name, parent_context
        )

    previous_agent_context = getattr(agent_instance, "_current_execution_context", None)
    agent_instance._current_execution_context = execution_context

    client = getattr(agent_instance, "client", None)
    previous_client_context = None
    if client is not None:
        previous_client_context = getattr(client, "_current_workflow_context", None)
        client._current_workflow_context = execution_context

    token = None
    start_time = time.time()
    parent_execution_id = parent_context.execution_id if parent_context else None

    sig = inspect.signature(func)
    call_kwargs = dict(kwargs or {})
    input_data: Dict[str, Any] = {}

    # Prepare DID-aware execution context so VC generation works for decorator-driven calls
    did_execution_context = None
    agent_has_did = getattr(agent_instance, "did_enabled", False) and getattr(
        agent_instance, "did_manager", None
    )
    if agent_has_did:
        try:
            session_id = execution_context.session_id or execution_context.workflow_id
            did_execution_context = agent_instance.did_manager.create_execution_context(
                execution_context.execution_id,
                execution_context.workflow_id,
                session_id,
                "agent",
                reasoner_name,
                parent_vc_id=getattr(execution_context, "parent_vc_id", None),
            )
            if did_execution_context and hasattr(
                agent_instance, "_populate_execution_context_with_did"
            ):
                agent_instance._populate_execution_context_with_did(
                    execution_context, did_execution_context
                )
        except Exception as exc:  # pragma: no cover - diagnostic only
            if getattr(agent_instance, "dev_mode", False):
                log_warn(f"Failed to build DID context for {reasoner_name}: {exc}")
            did_execution_context = None

    def _maybe_generate_vc(
        status: str, result_payload: Any, duration_ms: int, error_message: Optional[str]
    ) -> None:
        """Fire-and-forget VC generation so decorator parity matches HTTP path."""
        generate_vc = getattr(agent_instance, "_generate_vc_async", None)
        vc_generator = getattr(agent_instance, "vc_generator", None)
        if (
            did_execution_context
            and callable(generate_vc)
            and hasattr(agent_instance, "_should_generate_vc")
            and agent_instance._should_generate_vc(
                reasoner_name, getattr(agent_instance, "_reasoner_vc_overrides", {})
            )
        ):
            asyncio.create_task(
                generate_vc(
                    vc_generator,
                    did_execution_context,
                    reasoner_name,
                    input_data,
                    result_payload,
                    status=status,
                    error_message=error_message,
                    duration_ms=duration_ms,
                )
            )

    try:
        # Execute function with new context
        token = set_execution_context(execution_context)

        # Inject execution_context if the function accepts it
        if "execution_context" in sig.parameters:
            call_kwargs.setdefault("execution_context", execution_context)

        # Phase 5: Inject trigger context (webhook metadata)
        if "trigger" in sig.parameters:
            call_kwargs.setdefault("trigger", execution_context.trigger)
        if "webhook" in sig.parameters:
            call_kwargs.setdefault("webhook", execution_context.trigger)

        # 🔥 NEW: Automatic Pydantic model conversion (FastAPI-like behavior)
        try:
            if should_convert_args(func):
                converted_args, converted_kwargs = convert_function_args(
                    func, args, call_kwargs
                )
                args = converted_args
                call_kwargs = converted_kwargs
        except ValidationError as e:
            # Re-raise validation errors with context
            raise ValidationError(
                f"Pydantic validation failed for reasoner '{func.__name__}': {e}",
                model=getattr(e, "model", None),
            ) from e
        except Exception as e:
            # Log conversion errors but continue with original args for backward compatibility
            if hasattr(agent_instance, "dev_mode") and agent_instance.dev_mode:
                log_warn(f"Failed to convert arguments for {func.__name__}: {e}")

        input_data = _build_input_payload(sig, args, call_kwargs)

        start_payload = {
            "reasoner_name": reasoner_name,
            "args": list(args),
            "kwargs": dict(call_kwargs),
            "input_data": input_data,
            "parent_execution_id": parent_execution_id,
        }
        await asyncio.create_task(
            _send_workflow_start(
                agent_instance,
                execution_context,
                start_payload,
            )
        )

        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **call_kwargs)
        else:
            result = func(*args, **call_kwargs)

        duration_ms = int((time.time() - start_time) * 1000)
        completion_payload = {
            "input_data": input_data,
            "parent_execution_id": parent_execution_id,
        }
        await asyncio.create_task(
            _send_workflow_completion(
                agent_instance,
                execution_context,
                result,
                duration_ms,
                completion_payload,
            )
        )
        _maybe_generate_vc("success", result, duration_ms, None)
        return result
    except Exception as exc:
        duration_ms = int((time.time() - start_time) * 1000)
        error_payload = {
            "input_data": input_data,
            "parent_execution_id": parent_execution_id,
        }
        _maybe_generate_vc("error", None, duration_ms, str(exc))
        await asyncio.create_task(
            _send_workflow_error(
                agent_instance,
                execution_context,
                str(exc),
                duration_ms,
                error_payload,
            )
        )
        raise

    finally:
        if token is not None:
            reset_execution_context(token)
        agent_instance._current_execution_context = previous_agent_context
        if client is not None:
            client._current_workflow_context = previous_client_context


def _build_input_payload(
    signature: inspect.Signature, args: tuple, kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    if not signature.parameters:
        return dict(kwargs)

    try:
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
    except Exception:
        payload = {f"arg_{idx}": value for idx, value in enumerate(args)}
        payload.update(kwargs)
        return payload

    payload = {}
    for name, value in bound.arguments.items():
        if name == "self":
            continue
        payload[name] = value
    return payload


def _compose_event_payload(
    agent,
    context: ExecutionContext,
    reasoner_name: str,
    status: str,
    parent_execution_id: Optional[str],
    input_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    event: Dict[str, Any] = {
        "execution_id": context.execution_id,
        "workflow_id": context.workflow_id,
        "run_id": context.run_id,
        "reasoner_id": reasoner_name,
        "agent_node_id": getattr(agent, "node_id", None),
        "status": status,
        "type": reasoner_name,
        "parent_execution_id": parent_execution_id,
        "parent_workflow_id": context.parent_workflow_id,
    }
    if input_data is not None:
        event["input_data"] = input_data
    return event


def on_change(pattern: Union[str, List[str]]):
    """
    Decorator to mark a function as a memory event listener.

    Args:
        pattern: Memory pattern(s) to listen for changes

    Returns:
        Decorated function with memory event listener metadata
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        # Attach metadata to the function
        wrapper._memory_event_listener = True
        wrapper._memory_event_patterns = (
            pattern if isinstance(pattern, list) else [pattern]
        )
        return wrapper

    return decorator


# Legacy support for old reasoner decorator signature
async def _send_workflow_start(
    agent, context: ExecutionContext, payload: Dict[str, Any]
) -> None:
    handler = getattr(agent, "workflow_handler", None)
    if handler is None:
        return
    try:
        reasoner_name = payload.get("reasoner_name", context.reasoner_name)
        parent_execution_id = payload.get("parent_execution_id")
        input_data = payload.get("input_data") or {}

        if hasattr(handler, "notify_call_start"):
            await handler.notify_call_start(
                context.execution_id,
                context,
                reasoner_name,
                input_data,
                parent_execution_id=parent_execution_id,
            )
        elif hasattr(handler, "fire_and_forget_update"):
            event_payload = _compose_event_payload(
                agent,
                context,
                reasoner_name,
                "running",
                parent_execution_id,
                input_data=input_data,
            )
            await handler.fire_and_forget_update(event_payload)
    except Exception as exc:  # pragma: no cover - logging pathway
        if getattr(agent, "dev_mode", False):
            log_warn(f"Failed to emit workflow start: {exc}")


async def _send_workflow_completion(
    agent,
    context: ExecutionContext,
    result: Any,
    duration_ms: int,
    payload: Dict[str, Any],
) -> None:
    handler = getattr(agent, "workflow_handler", None)
    if handler is None:
        return
    try:
        parent_execution_id = payload.get("parent_execution_id")
        input_data = payload.get("input_data")
        reasoner_name = context.reasoner_name

        if hasattr(handler, "notify_call_complete"):
            await handler.notify_call_complete(
                context.execution_id,
                context.workflow_id,
                result,
                duration_ms,
                context,
                input_data=input_data,
                parent_execution_id=parent_execution_id,
            )
        elif hasattr(handler, "fire_and_forget_update"):
            event_payload = _compose_event_payload(
                agent,
                context,
                reasoner_name,
                "succeeded",
                parent_execution_id,
                input_data=input_data,
            )
            event_payload["result"] = result
            event_payload["duration_ms"] = duration_ms
            await handler.fire_and_forget_update(event_payload)
    except Exception as exc:  # pragma: no cover - logging pathway
        if getattr(agent, "dev_mode", False):
            log_warn(f"Failed to emit workflow completion: {exc}")


async def _send_workflow_error(
    agent,
    context: ExecutionContext,
    message: str,
    duration_ms: int,
    payload: Dict[str, Any],
) -> None:
    handler = getattr(agent, "workflow_handler", None)
    if handler is None:
        return
    try:
        parent_execution_id = payload.get("parent_execution_id")
        input_data = payload.get("input_data")
        reasoner_name = context.reasoner_name

        if hasattr(handler, "notify_call_error"):
            await handler.notify_call_error(
                context.execution_id,
                context.workflow_id,
                message,
                duration_ms,
                context,
                input_data=input_data,
                parent_execution_id=parent_execution_id,
            )
        elif hasattr(handler, "fire_and_forget_update"):
            event_payload = _compose_event_payload(
                agent,
                context,
                reasoner_name,
                "failed",
                parent_execution_id,
                input_data=input_data,
            )
            event_payload["error"] = message
            event_payload["duration_ms"] = duration_ms
            await handler.fire_and_forget_update(event_payload)
    except Exception as exc:  # pragma: no cover - logging pathway
        if getattr(agent, "dev_mode", False):
            log_warn(f"Failed to emit workflow error: {exc}")


def legacy_reasoner(reasoner_id: str, input_schema: dict, output_schema: dict):
    """
    Legacy reasoner decorator for backward compatibility.

    This is kept for compatibility with existing code that uses the old signature.
    New code should use the enhanced @reasoner decorator.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Attach metadata to the function
        wrapper._reasoner_def = ReasonerDefinition(
            id=reasoner_id, input_schema=input_schema, output_schema=output_schema
        )
        return wrapper

    return decorator
