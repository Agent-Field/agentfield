# Execute API

The control plane exposes synchronous and asynchronous execution endpoints:

- `POST /api/v1/execute/{agent}.{reasoner}`
- `POST /api/v1/execute/async/{agent}.{reasoner}`

## Request

```json
{
  "input": {"question": "What changed?"},
  "context": {"provider": "openai"},
  "webhook": {
    "url": "https://example.com/execution-events",
    "secret": "shared-secret",
    "headers": {"X-Tenant": "example"}
  }
}
```

| Field | Meaning |
|-------|---------|
| `input` | Object delivered to the local agent reasoner as its arguments. |
| `context` | Optional control-plane context. This is a reserved, control-plane-interpreted field, not a general user metadata bag. |
| `webhook` | Optional completion webhook registration (`url`, optional `secret`, and optional string `headers`). |

The control plane stores `input` and `context` together in `executions.input_payload` and includes `context` in replay matching. It inspects the reserved context keys `llm_endpoint`, `llm_backend`, `backend`, `provider`, and `model_provider` for LLM-endpoint gating. For external ARD targets it also interprets `operation` for policy enforcement and forwards the complete context object verbatim to the external endpoint.

`context` is never delivered to a local agent node: local dispatch sends only `input`. The Python, Go, and TypeScript execute helpers currently serialize only `input`, so callers that need these control-plane fields must use the REST endpoint directly. The restart helper is the exception and can send restart context.

## Responses

A successful synchronous request returns the terminal execution:

```json
{
  "execution_id": "exec_...",
  "run_id": "run_...",
  "status": "succeeded",
  "result": {},
  "duration_ms": 42,
  "finished_at": "2026-08-27T12:00:00Z",
  "webhook_registered": true
}
```

Failures may add `error_message` and `error_details`.

An accepted asynchronous request returns HTTP `202`:

```json
{
  "execution_id": "exec_...",
  "run_id": "run_...",
  "workflow_id": "run_...",
  "status": "queued",
  "target": "agent.reasoner",
  "type": "reasoner",
  "created_at": "2026-08-27T12:00:00Z",
  "enqueued_at": "2026-08-27T12:00:00Z",
  "webhook_registered": false
}
```

If webhook registration failed, the response may include `webhook_error`.

Poll `GET /api/v1/executions/{execution_id}`. Its response contains `execution_id`, `run_id`, `status`, `started_at`, and `webhook_registered`, plus applicable `status_reason`, `result`, `error`, `error_details`, `completed_at`, `duration_ms`, `webhook_events`, and approval fields.

The execute routes do not accept an idempotency key. Retrying a request can create another execution; use the [restart/replay API](EXECUTION_RESTART.md) when replaying an existing run.
