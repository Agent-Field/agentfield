"""Claude Code provider using claude_agent_sdk (native Python SDK).

Uses lazy import - claude_agent_sdk is an optional dependency that's only
loaded when the claude-code provider is actually used.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from agentfield.harness._result import Metrics, RawResult
from agentfield.exceptions import HarnessProviderUnavailable


def _get_claude_sdk() -> Any:
    """Lazy import of claude_agent_sdk."""
    try:
        import claude_agent_sdk  # pyright: ignore[reportMissingImports]

        return claude_agent_sdk
    except ImportError as exc:
        raise ImportError(
            "claude_agent_sdk is required for the 'claude-code' provider. "
            "Install it with: pip install claude-agent-sdk"
        ) from exc


_PERMISSION_MAP = {
    "auto": "bypassPermissions",
    "plan": "plan",
}


def _parse_usage(usage: dict) -> Dict[str, int]:
    """Normalize a Claude Code result-message ``usage`` object into token counts.

    Claude Code emits Anthropic-native fields: ``input_tokens``,
    ``output_tokens``, ``cache_read_input_tokens``,
    ``cache_creation_input_tokens``.
    """

    def _int(name: str) -> int:
        val = usage.get(name)
        try:
            return int(val) if val is not None else 0
        except (TypeError, ValueError):
            return 0

    return {
        "input_tokens": _int("input_tokens"),
        "output_tokens": _int("output_tokens"),
        "cache_read_tokens": _int("cache_read_input_tokens"),
        "cache_creation_tokens": _int("cache_creation_input_tokens"),
    }


class ClaudeCodeProvider:
    """Claude Code provider using the native claude_agent_sdk."""

    async def execute(self, prompt: str, options: dict[str, object]) -> RawResult:
        """Execute a prompt via Claude Code SDK."""
        try:
            sdk = _get_claude_sdk()
        except ImportError as exc:
            raise HarnessProviderUnavailable(
                "claude-code",
                binary="claude_agent_sdk",
                install_command="pip install 'agentfield[harness-claude]'",
            ) from exc

        agent_options: dict[str, object] = {}
        if options.get("model") is not None:
            agent_options["model"] = options["model"]
        # Agent root: project_dir is the canonical field, fall back to cwd
        # (agentfield#686). The SDK's cwd is both the process dir and the root
        # the agent operates in; the runner places the schema output file under
        # this same root.
        root = options.get("project_dir") or options.get("cwd")
        if root is not None:
            agent_options["cwd"] = root
        if options.get("max_turns") is not None:
            agent_options["max_turns"] = options["max_turns"]
        if options.get("tools") is not None:
            agent_options["allowed_tools"] = options["tools"]
        if options.get("system_prompt") is not None:
            agent_options["system_prompt"] = options["system_prompt"]
        if options.get("max_budget_usd") is not None:
            agent_options["max_budget_usd"] = options["max_budget_usd"]
        if options.get("permission_mode") is not None:
            raw_mode = str(options["permission_mode"])
            agent_options["permission_mode"] = _PERMISSION_MAP.get(raw_mode, raw_mode)
        if options.get("env") is not None:
            agent_options["env"] = options["env"]

        resume_sid = options.get("resume_session_id")
        if resume_sid:
            agent_options["resume"] = str(resume_sid)

        messages: List[Dict[str, Any]] = []
        result_text: Optional[str] = None
        total_cost: Optional[float] = None
        num_turns = 0
        session_id = ""
        usage_tokens: Dict[str, int] = {}
        result_model: Optional[str] = None
        start_api = time.monotonic()

        try:
            # Capture stderr so CLI failures can be diagnosed
            stderr_lines: list[str] = []

            def _on_stderr(line: str) -> None:
                stderr_lines.append(line)

            opts = (
                sdk.ClaudeAgentOptions(**agent_options)
                if hasattr(sdk, "ClaudeAgentOptions")
                else agent_options
            )
            # Set stderr callback after construction to avoid polluting
            # agent_options dict (which tests may assert on).
            if hasattr(opts, "stderr"):
                opts.stderr = _on_stderr

            msg_count = 0
            async for msg in sdk.query(prompt=prompt, options=opts):
                msg_count += 1
                if isinstance(msg, dict):
                    msg_dict = msg
                elif hasattr(msg, "__dict__"):
                    msg_dict = dict(msg.__dict__)
                else:
                    msg_dict = {"raw": str(msg)}

                messages.append(msg_dict)

                msg_type = str(msg_dict.get("type", ""))
                msg_subtype = str(msg_dict.get("subtype", ""))
                if msg_type == "result" or msg_subtype == "success":
                    raw_result = msg_dict.get("result", msg_dict.get("text", ""))
                    result_text = (
                        raw_result if isinstance(raw_result, str) else str(raw_result)
                    )
                    sid = msg_dict.get("session_id", "")
                    session_id = sid if isinstance(sid, str) else str(sid)
                    cost_info = msg_dict.get("cost_usd") or msg_dict.get(
                        "total_cost_usd"
                    )
                    if cost_info is not None:
                        total_cost = float(cost_info)
                    # Claude Code result messages carry a full usage object:
                    # input_tokens, output_tokens, cache_read_input_tokens,
                    # cache_creation_input_tokens.
                    usage_obj = msg_dict.get("usage")
                    if isinstance(usage_obj, dict):
                        usage_tokens = _parse_usage(usage_obj)
                    model_field = msg_dict.get("model") or msg_dict.get("modelUsage")
                    if isinstance(model_field, str):
                        result_model = model_field
                    turns = msg_dict.get("num_turns")
                    num_turns = (
                        int(turns) if isinstance(turns, (int, float)) else len(messages)
                    )
                elif msg_type == "assistant" and result_text is None:
                    content = msg_dict.get("content")
                    message_obj = msg_dict.get("message")
                    if content is None and isinstance(message_obj, dict):
                        content = message_obj.get("content")

                    if isinstance(content, str):
                        result_text = content
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text")
                                if isinstance(text, str):
                                    result_text = text

            api_ms = int((time.monotonic() - start_api) * 1000)

            return RawResult(
                result=result_text,
                messages=messages,
                metrics=Metrics(
                    duration_ms=0,
                    duration_api_ms=api_ms,
                    num_turns=num_turns,
                    total_cost_usd=total_cost,
                    session_id=session_id,
                    usage=usage_tokens or None,
                    input_tokens=usage_tokens.get("input_tokens", 0),
                    output_tokens=usage_tokens.get("output_tokens", 0),
                    cache_read_tokens=usage_tokens.get("cache_read_tokens", 0),
                    cache_creation_tokens=usage_tokens.get("cache_creation_tokens", 0),
                    model=result_model,
                ),
                is_error=False,
            )
        except Exception as exc:
            import logging as _logging

            stderr_output = "\n".join(stderr_lines).strip()
            _logging.getLogger("agentfield.harness.claude").error(
                "ClaudeCodeProvider error: %s\nStderr output:\n%s",
                exc,
                stderr_output or "(no stderr captured)",
            )
            api_ms = int((time.monotonic() - start_api) * 1000)
            error_detail = str(exc)
            if stderr_output:
                error_detail = f"{error_detail}\nStderr output:\n{stderr_output}"
            return RawResult(
                result=None,
                messages=messages,
                metrics=Metrics(duration_api_ms=api_ms, session_id=session_id),
                is_error=True,
                error_message=error_detail,
            )
