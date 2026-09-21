# Execution restart and replay API

`POST /api/v1/executions/{execution_id}/restart` starts a new execution from an existing run. It always mints a new `execution_id` and `run_id`; it never reuses or mutates the source execution ID. The new workflow run records backward lineage in `workflow_runs.metadata.lineage`, while the source execution and source run record a forward `restarted_as` pointer.

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

`GET /api/v1/executions/{execution_id}` includes the following field after that execution has been restarted, while the successor record still exists. A client polling only the source ID can follow it and continue polling the successor. If the successor no longer resolves, the field is omitted rather than returning a broken pointer:

```json
{
  "restarted_as": {
    "execution_id": "exec_...",
    "run_id": "run_..."
  }
}
```

## Status reasons

Operators polling execution state should branch on the stable category before any `:` suffix:

| `status_reason` | Meaning |
|-----------------|---------|
| `awaiting_agent_restart` | Dispatch is deliberately waiting for a restarting agent to return. |
| `agent_restart_orphaned[: ...]` | The old agent process is gone and its in-flight execution cannot be revived. |
| `agent_shutdown_cancelled` | An SDK cancelled the reasoner while its agent process drained for graceful shutdown. |
| `replayed_from_execution:<id>` | The result was reused from the named source execution. |
| `waiting_for_approval` | Human/external approval is pending. |
| `approval_rejected[: ...]` | Approval was rejected; an optional suffix contains feedback. |
| `awaiting_child` | The parent is waiting for a child execution. |
| `agent_client_error:<status>` | The agent reported a client-facing HTTP 4xx failure. |
| `llm_unavailable`, `concurrency_limit`, `control_plane_shutdown`, `agent_timeout`, `agent_error`, `agent_unreachable`, `bad_response`, `internal_error`, `validation`, `permission_denied`, `node_unavailable`, `target_not_found` | Canonical failure categories used for operator routing and HTTP mapping. A pool that stops after restart persistence returns and stores `control_plane_shutdown`. |

Concurrency and LLM-circuit admission checks run before restart persistence. A rejected restart creates no execution or workflow-execution row. Queue-full restart responses return `503` with matching `Retry-After` and `retry_after` values.

The instance-scoped orphan reap also sweeps legacy rows whose `instance_id` is empty. Such a row can therefore have an `agent_restart_orphaned` `status_reason` naming an instance that never owned it. The explicit `instance_id` on execution reads distinguishes that legacy case from an execution created against the departed instance.

Do not emulate restart by re-submitting `/execute`: execute has no idempotency key, creates unrelated executions, and cannot establish restart lineage or replay boundaries.

## Automatic handoff of interrupted runs

Set `AGENTFIELD_RESUME_INTERRUPTED_RUNS=true` to let the control plane hand an interrupted run root to the same workflow-scope restart operation automatically. Eligible reasons are `agent_restart_orphaned...`, `control_plane_shutdown`, and `agent_shutdown_cancelled`. User cancellations, validation or agent failures, timeouts, and interrupted child executions are not resumed independently.

The orchestrator reasoner re-executes from its first line. AgentField does not snapshot or restore Python stack state. The handoff uses `reuse: all-succeeded`, so already-succeeded `app.call` children from the source run are replayed instead of being executed again. If the orchestrator calls the same target with the same input more than once in a run, all of those calls replay the first recorded result.

| Environment variable | Default | Bound |
|---|---:|---|
| `AGENTFIELD_RESUME_INTERRUPTED_RUNS` | `false` | Feature gate; manual restart pointers remain unconditional. |
| `AGENTFIELD_RESUME_INTERRUPTED_MAX_ATTEMPTS` | `1` | Maximum `kind: resume` lineage depth. |
| `AGENTFIELD_RESUME_INTERRUPTED_WINDOW` | `1h` | Startup considers only executions updated within this window. |
| `AGENTFIELD_RESUME_INTERRUPTED_LIMIT` | `25` | Maximum startup candidates. |
| `AGENTFIELD_RESUME_INTERRUPTED_DELAY` | `15s` | How long the control plane waits for a replacement agent instance to register before dispatching each successor, both at startup and for inline handoffs. Set this to at least the time an agent pod takes to become ready. |

The defaults prevent crash loops from fanning out new runs. Each source can acquire only one forward pointer, and the successor's backward lineage records `kind: resume` plus its `resume_attempt`.

### Limitations

Automatic handoff is best-effort recovery, not a durable job queue. It assumes a single control plane: the eligibility check and forward-pointer write are not a compare-and-set, so two replicas racing on the same interrupted root can each create a successor. Both successors run, and only the last pointer write remains visible.

A handoff is attempted once. If the replacement agent is still unavailable when the delay expires, or if the asynchronous queue is full, the attempt is logged and dropped. It is not retried; only a later control-plane restart's startup sweep can pick up the run again. That startup sweep reads one bounded page per boot, controlled by `AGENTFIELD_RESUME_INTERRUPTED_LIMIT` and `AGENTFIELD_RESUME_INTERRUPTED_WINDOW`, so candidates outside that page wait for another restart.

The stale-execution sweep terminalizes executions as `timeout` without an interruption reason, so it never hands them off. This is the recovery path when an agent is killed without draining, its SDK predates `instance_id`, or orphan reaping is disabled.

The attempt bound comes from the successor run's lineage metadata, which is written best-effort. A manual restart or fork in the middle of a lineage resets the automatic counter.

The orphan-reap `created_at` cutoff uses wall-clock time and is approximate under control-plane clock skew. An execution created against the departing instance moments after its replacement registered is spared by the reap and later times out instead.
