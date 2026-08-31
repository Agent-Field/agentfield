# LLM observability in the Python SDK

The Python SDK can register LiteLLM observability callbacks and attach AgentField execution correlation data to text completions made through `app.ai`.

Set `AGENTFIELD_LITELLM_CALLBACKS` to a comma-separated list of LiteLLM callback names, for example:

```bash
export AGENTFIELD_LITELLM_CALLBACKS=langfuse,logfire
```

Names are trimmed, lowercased, deduplicated, and passed to LiteLLM. When the variable is unset or empty, AgentField does not import LiteLLM or change its callback state. Callback registration failures are logged and do not prevent the Agent or its LLM calls from running.

## Execution metadata

Inside a reasoner, each `app.ai` text completion receives a LiteLLM `metadata` dictionary containing the available values below. Empty or missing values are omitted.

```json
{
  "agentfield_execution_id": "exec_123",
  "agentfield_run_id": "run_123",
  "agentfield_agent_node_id": "support-agent",
  "agentfield_reasoner": "answer",
  "agentfield_session_id": "session_123",
  "agentfield_parent_execution_id": "exec_parent"
}
```

The execution ID, run ID, agent node ID, and reasoner name appear when they are available from the current execution context. Session and parent execution IDs appear only when present. Caller-provided metadata keys take precedence. `user_id` and `requester_metadata` are never added.

When AgentField itself successfully registers a LangFuse-family callback (`langfuse` or `langfuse_otel`), it also supplies that vendor's native aliases: `trace_id` is the run ID, `session_id` is the session ID, `trace_name` is the node ID, `generation_name` is the node and reasoner name, and `tags` identifies AgentField plus the available node and reasoner. These aliases are never added merely because application code registered a process-global LiteLLM callback itself, nor because AgentField registered some other vendor's callback — either would silently re-key, rename and re-tag an existing LangFuse setup's generations.

`metadata` is a LiteLLM-only parameter and is not included in the provider request body. To disable all AgentField execution metadata stamping while retaining callback registration, set:

```bash
export AGENTFIELD_LITELLM_METADATA=false
```

The values `0`, `false`, `no`, and `off` are treated as false, case-insensitively and with surrounding whitespace ignored. The stamp is enabled for every other value, including when the variable is unset.

## Scope and pitfalls

The stamp covers text completions made by `app.ai`, including its tool-calling turns. Image generation and harness schema-repair completions call LiteLLM directly and are out of scope. Text-to-speech is also out of scope: to trace TTS, pass `metadata=` explicitly into the `aspeech` call.

- LiteLLM callback state is process-global. A process hosting several Agents applies the union of their configured callbacks.
- `langfuse` and `logfire` are not AgentField SDK dependencies. When the corresponding package is missing, LiteLLM degrades the callback to a logged no-op.
- Enabling an OpenTelemetry-family callback yields a second trace tree disconnected from the control plane's own OTLP spans, because AgentField does not propagate W3C `traceparent` to agent nodes.
