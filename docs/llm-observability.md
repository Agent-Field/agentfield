# LLM observability in the Python SDK

The Python SDK can register LiteLLM observability callbacks and attach AgentField execution correlation data to text completions made through `app.ai`.

Set `AGENTFIELD_LITELLM_CALLBACKS` to a comma-separated list of LiteLLM callback names, for example:

```bash
export AGENTFIELD_LITELLM_CALLBACKS=langfuse,logfire
```

Names are trimmed, lowercased, deduplicated, and passed to LiteLLM. When the variable is unset or empty, AgentField does not import LiteLLM or change its callback state. Callback registration failures are logged and do not prevent the Agent or its LLM calls from running.

### LangFuse setup

Install AgentField's compatible LangFuse extra before enabling LiteLLM's standard `langfuse` callback:

```bash
pip install "agentfield[langfuse]"
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
# Optional for self-hosted LangFuse; defaults to LangFuse Cloud.
export LANGFUSE_HOST=https://cloud.langfuse.com
export AGENTFIELD_LITELLM_CALLBACKS=langfuse
```

The extra deliberately installs LangFuse 2.x, the client API supported by LiteLLM's standard `langfuse` callback. Installing an unbounded LangFuse 3.x or 4.x package with that callback can degrade to a logged no-op, so AgentField refuses the standard callback at registration time when the compatible client is missing or outside its tested range. Use the extra rather than installing `langfuse` separately. Other callback integrations remain separate packages; for example, install and configure `logfire` before selecting the `logfire` callback.

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

When AgentField itself successfully registers a LangFuse-family callback (`langfuse` or `langfuse_otel`), it also supplies that vendor's native aliases: `trace_id` is a stable W3C-compatible 32-hex ID derived from the run ID, `trace_metadata` carries the original AgentField correlation fields, `session_id` is the session ID, `trace_name` is the node ID, `generation_name` is the node and reasoner name, and `tags` identifies AgentField plus the available node and reasoner. The original run ID therefore remains available as `agentfield_run_id` in LangFuse trace metadata for joins with AgentField. These aliases are never added merely because application code registered a process-global LiteLLM callback itself, nor because AgentField registered some other vendor's callback — either would silently re-key, rename and re-tag an existing LangFuse setup's generations. If an AgentField-owned callback is later removed from LiteLLM's process-global callback list, AgentField drops its ownership record and stops adding the native aliases.

`metadata` is a LiteLLM-only parameter and is not included in the provider request body. Stamping is off when neither observability variable is configured. Setting `AGENTFIELD_LITELLM_CALLBACKS` opts into both the selected callbacks and the correlation stamp. To enable the stamp for an integration registered elsewhere, set:

```bash
export AGENTFIELD_LITELLM_METADATA=true
```

Set `AGENTFIELD_LITELLM_METADATA=false` to retain callback registration without AgentField's stamp. Only `1`, `true`, `yes`, and `on` enable the standalone flag, case-insensitively and with surrounding whitespace ignored; unset or unrecognized values do not opt in unless AgentField has a configured/active callback.

## Scope and pitfalls

The stamp covers text completions made by `app.ai`, including its tool-calling turns. Image generation and harness schema-repair completions call LiteLLM directly and are out of scope. Text-to-speech is also out of scope: to trace TTS, pass `metadata=` explicitly into the `aspeech` call.

- LiteLLM callback state is process-global. A process hosting several Agents applies the union of their configured callbacks.
- LangFuse is optional; install `agentfield[langfuse]` for the tested standard-callback path. Other callback packages such as `logfire` are not AgentField SDK dependencies. When a corresponding package is missing or incompatible, LiteLLM degrades the callback to a logged no-op.
- Enabling an OpenTelemetry-family callback yields a second trace tree disconnected from the control plane's own OTLP spans, because AgentField does not propagate W3C `traceparent` to agent nodes.
