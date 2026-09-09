# Execution restart and replay API

`POST /api/v1/executions/{execution_id}/restart` starts a new execution from an existing run. It always mints a new `execution_id` and `run_id`; it never reuses or mutates the source execution ID. The new workflow run records backward lineage in `workflow_runs.metadata.lineage`.

The same operation is available through the UI API and the `af execution restart` command. The Python SDK exposes `client.restart_execution(...)`.

## Request

```json
{
  "scope": "workflow",
  "reuse": "succeeded-before",
  "fork": false,
  "input": {"question": "Try again"},
  "context": {"provider": "openai"},
  "webhook": {"url": "https://example.com/execution-events"}
}
```

| Field | Values and behavior |
|-------|---------------------|
| `scope` | `workflow` (default) restarts the source run's root; `execution` restarts the selected execution. |
| `reuse` | `succeeded-before` (default), `all-succeeded`, or `none`. |
| `fork` | Marks the lineage as a fork. Supplying replacement `input` or `context` also makes it a fork. |
| `input` | Optional replacement input; otherwise the restarted execution's stored input is used. |
| `context` | Optional replacement control-plane context; otherwise its stored context is used. See [Execute API](EXECUTE.md). |
| `webhook` | Optional webhook registration (`url`, optional `secret`, optional string `headers`). |

With workflow scope, `succeeded-before` replays matching successful children from the source run only before the selected source execution. `all-succeeded` allows any matching successful child in that run; `none` executes every child again. Execution scope converts `succeeded-before` to `all-succeeded`. Replay matching includes target, input, and context. The control plane propagates replay state internally with `X-AgentField-Replay-Source-Run-ID`, `X-AgentField-Replay-Before-Execution-ID`, and `X-AgentField-Replay-Mode` headers.

## Response

An accepted restart returns HTTP `202` with the new execution fields (`execution_id`, `run_id`, `workflow_id`, `status`, `target`, `type`, timestamps), source identifiers (`source_execution_id`, `source_run_id`, `restarted_execution_id`), and replay metadata (`replay_before_execution_id` when applicable, `replay_mode`, `scope`, `kind`). It also reports `webhook_registered` and an optional `webhook_error`.

## Status reasons

Operators polling execution state should branch on the stable category before any `:` suffix:

| `status_reason` | Meaning |
|-----------------|---------|
| `awaiting_agent_restart` | Dispatch is deliberately waiting for a restarting agent to return. |
| `agent_restart_orphaned[: ...]` | The old agent process is gone and its in-flight execution cannot be revived. |
| `replayed_from_execution:<id>` | The result was reused from the named source execution. |
| `waiting_for_approval` | Human/external approval is pending. |
| `approval_rejected[: ...]` | Approval was rejected; an optional suffix contains feedback. |
| `awaiting_child` | The parent is waiting for a child execution. |
| `agent_client_error:<status>` | The agent reported a client-facing HTTP 4xx failure. |
| `llm_unavailable`, `concurrency_limit`, `control_plane_shutdown`, `agent_timeout`, `agent_error`, `agent_unreachable`, `bad_response`, `internal_error`, `validation`, `permission_denied`, `node_unavailable`, `target_not_found` | Canonical failure categories used for operator routing and HTTP mapping. A pool that stops after restart persistence returns and stores `control_plane_shutdown`. |

Concurrency and LLM-circuit admission checks run before restart persistence. A rejected restart creates no execution or workflow-execution row. Queue-full restart responses return `503` with matching `Retry-After` and `retry_after` values.

The instance-scoped orphan reap also sweeps legacy rows whose `instance_id` is empty. Such a row can therefore have an `agent_restart_orphaned` `status_reason` naming an instance that never owned it. The explicit `instance_id` on execution reads distinguishes that legacy case from an execution created against the departed instance.

Do not emulate restart by re-submitting `/execute`: execute has no idempotency key, creates unrelated executions, and cannot establish restart lineage or replay boundaries.
