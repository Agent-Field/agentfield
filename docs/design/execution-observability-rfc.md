# Execution Observability RFC

> **Status**: Draft RFC
> **Author**: Architecture review
> **Scope**: Control plane, execution detail UI, TypeScript SDK, Go SDK, Python SDK
> **Date**: 2026-04-06
> **Depends on**: PR #330 (`origin/feat/ui-revamp-product-research`)
> **Merge policy**: This branch and any PR created from it must not merge before PR #330 lands.

---

## 1. Summary

This RFC proposes a new execution-observability model for AgentField where the execution detail page becomes the primary debugging surface for live and recent execution behavior.

The model separates observability into three distinct layers:

1. **Execution logs**: structured, execution-correlated logs stamped by the SDK/runtime and surfaced as the primary log stream on the execution page.
2. **Lifecycle events**: control-plane execution and workflow events that anchor the execution timeline.
3. **Raw node logs**: process-level stdout/stderr logs exposed as an advanced debug surface for deeper inspection.

The key design decision is that **execution logs are the primary product surface**, while raw node logs remain available but clearly secondary. This gives users a useful, filterable, execution-scoped debugging experience without pretending that raw process logs are perfectly attributable to a single execution.

---

## 2. Product Intent

### 2.1 Main Product Goal

When a user opens an execution detail page, they should be able to answer:

- What happened during this execution?
- What node or reasoner produced each log line?
- What is happening right now if the execution is still live?
- What failed, where, and with what context?

### 2.2 Product Decisions Confirmed

This RFC reflects the following product decisions:

- **Execution logs are primary** on the execution page.
- **Raw node logs are behind advanced/debug UI** rather than front-and-center.
- **Control-plane lifecycle events are included**, but raw control-plane server logs are not in scope for v1.
- **Multi-node log rendering is chronological by default**.
- **Filtering by node, source, level, and text is required**.
- **Live streaming plus short retention** is the preferred operating model.
- **SDKs should expose a developer logger API and auto-generate runtime/system logs**.
- **Structured execution logs should also appear in stdout** for local developer ergonomics.
- **Fast rollout with partial coverage is acceptable**, but the architecture must preserve a path to stronger correlation and filtering.

---

## 3. Problem Statement

PR #330 adds node-level process log proxying and UI panels. That solves a real gap, but it does not solve execution-level observability.

Today, the system has two disjoint observability channels:

- **Durable execution/workflow events** in the control plane
- **Raw node/process logs** exposed by agents

These channels answer different questions:

- Lifecycle events explain **state transitions**
- Raw node logs explain **process behavior**

The missing piece is a structured execution-correlated log stream that can be safely rendered on an execution page and filtered in a useful way.

### 3.1 Why Raw Node Logs Alone Are Not Enough

Raw node logs are process-scoped, not execution-scoped.

If a node handles multiple executions concurrently:

- stdout/stderr lines can interleave
- lines may not be attributable to one execution
- filtering by execution becomes impossible unless the runtime stamps correlation metadata at emission time

This is the central architectural constraint.

---

## 4. Proposed User Experience

### 4.1 Execution Detail Page

The execution page should expose three distinct observability areas:

1. **Timeline**
   - execution started
   - node invoked
   - waiting / retried / failed / succeeded
   - powered by control-plane lifecycle and workflow events

2. **Execution Logs**
   - primary log stream
   - structured, execution-correlated
   - chronological
   - filterable by:
     - level
     - node
     - source
     - text query
     - system vs developer logs
   - supports:
     - live mode
     - paused mode
     - reconnect state
     - short-tail load

3. **Advanced Debug**
   - raw node logs
   - hidden behind an advanced/debug affordance
   - clearly labeled as process-level logs that may include unrelated node activity

### 4.2 Multi-Node Behavior

For v1, logs render in **strict chronological order**.

Each row should still visibly show:

- timestamp
- level
- node
- reasoner or source
- message

This preserves chronological readability while still making multi-node traffic understandable.

### 4.3 Live Mode

The logs component should behave similarly to node logs:

- collapsed or hidden by default if desired by page layout
- user opts into live mode
- connection indicator:
  - connected
  - reconnecting
  - disconnected
- tail fetch when opening
- live stream when enabled

---

## 5. Observability Model

### 5.1 Layer 1: Lifecycle Events

These are existing control-plane and workflow events:

- execution started
- execution waiting
- execution paused
- execution resumed
- execution cancelled
- execution failed
- execution completed
- workflow execution events emitted by SDK/runtime during local or nested execution paths

These anchor the timeline and should remain semantically distinct from logs.

### 5.2 Layer 2: Structured Execution Logs

These are new or newly formalized records emitted by the SDK/runtime and correlated to an execution.

They are the **main product surface** for debugging.

Characteristics:

- structured JSONL / NDJSON event envelope
- stamped with execution context
- live stream capable
- short retention
- stored or indexed in the control plane
- optimized for filtering and chronological rendering

### 5.3 Layer 3: Raw Node Logs

These remain:

- process-level stdout/stderr logs
- proxied from node to control plane UI
- useful for advanced debugging
- not reliable as the primary execution-correlation mechanism

Raw node logs should remain available because:

- some applications will still use plain stdout
- developers need local ergonomics
- production debugging often needs the raw stream

But they should not be presented as the canonical execution log model.

---

## 6. Canonical Structured Log Envelope

### 6.1 Required Fields

Each structured execution log record should include:

```json
{
  "v": 1,
  "ts": "2026-04-06T10:15:30.123Z",
  "execution_id": "exec_123",
  "workflow_id": "wf_123",
  "run_id": "run_123",
  "root_workflow_id": "wf_root_123",
  "parent_execution_id": "exec_parent_1",
  "agent_node_id": "claims-processor",
  "reasoner_id": "validate_claim",
  "level": "info",
  "source": "sdk.runtime",
  "event_type": "reasoner.started",
  "message": "Reasoner execution started",
  "attributes": {
    "sdk_language": "typescript",
    "input_size_bytes": 1420
  },
  "system_generated": true
}
```

Required top-level fields:

- `v`
- `ts`
- `execution_id`
- `workflow_id`
- `run_id`
- `agent_node_id`
- `level`
- `source`
- `message`

Strongly recommended:

- `root_workflow_id`
- `parent_execution_id`
- `reasoner_id`
- `event_type`
- `attributes`
- `system_generated`

### 6.2 Optional Fields

Optional fields for richer filtering and future compatibility:

- `attempt`
- `span_id`
- `step_id`
- `error_category`
- `sdk_language`
- `call_target`
- `transport`
- `request_id`

### 6.3 Level Semantics

Canonical levels:

- `debug`
- `info`
- `warn`
- `error`

Avoid runtime-specific level drift at the ingestion boundary.

### 6.4 Source Semantics

Source should identify who emitted the log:

- `sdk.runtime`
- `sdk.logger`
- `user.app`
- `control-plane.lifecycle`

For v1, `control-plane.lifecycle` can remain modeled as timeline events rather than entering the log store if that simplifies rollout.

---

## 7. SDK and Runtime Responsibilities

### 7.1 Hard Requirement

The SDK/runtime must stamp structured logs with the active execution context.

This is not optional. It is the only way to provide reliable execution-level observability.

### 7.2 Developer Logger API

Each SDK should expose an execution-aware logger API from the current context.

Examples:

```typescript
ctx.logger.info("Fetched customer profile", { customer_id: customerId });
ctx.logger.error("Tool call failed", { tool_name: "crm_lookup", retryable: true });
```

```python
app.ctx.logger.info("Fetched customer profile", customer_id=customer_id)
app.ctx.logger.error("Tool call failed", tool_name="crm_lookup", retryable=True)
```

```go
ctx.Logger().Info("Fetched customer profile", map[string]any{"customer_id": customerID})
ctx.Logger().Error("Tool call failed", map[string]any{"tool_name": "crm_lookup"})
```

Desired behavior:

- automatically enrich from current execution context
- accept structured attributes
- support common levels
- safe no-op or degraded behavior if execution context is unavailable

### 7.3 Automatic Runtime/System Logs

SDKs should also emit system-generated logs for key runtime transitions:

- reasoner started
- reasoner completed
- reasoner failed
- local child call started
- local child call completed
- downstream agent call started
- downstream agent call completed
- downstream agent call failed
- retry scheduled
- approval wait entered
- approval resolved
- transport timeout

This ensures v1 utility even before all application developers adopt the explicit logger API.

### 7.4 Stdout Mirroring

Structured execution logs should also appear in stdout for local developer ergonomics.

Recommended behavior:

- **Dev mode**: mirror structured logs to stdout by default
- **Production**: configurable

This allows local developers to continue using familiar terminal workflows without making stdout the canonical system of record.

### 7.5 Execution Context Propagation

The SDK audit should ensure that the following fields are propagated consistently:

- `execution_id`
- `workflow_id`
- `run_id`
- `parent_execution_id`
- `parent_workflow_id`
- `root_workflow_id`
- `agent_node_id`
- `reasoner_id`

Known implication:

- any path that emits logs without a current execution context must clearly degrade to node-level logging only

---

## 8. Storage and Transport Model

### 8.1 Transport

Use NDJSON / JSONL as the wire and storage format for structured execution logs.

Why:

- stream-friendly
- append-friendly
- compatible with live tailing
- already aligned with current node log patterns

### 8.2 Ingestion Model

The control plane should expose a structured execution-log ingestion path separate from raw node log proxying.

Possible v1 design:

- SDK/runtime emits structured execution logs to control plane over authenticated HTTP
- control plane appends them to a short-retention store keyed by execution
- control plane exposes:
  - tail fetch endpoint
  - live stream endpoint

This is preferable to scraping or reconstructing execution logs from stdout.

### 8.3 Retention

V1 should optimize for **live plus short retention**.

Suggested configurable controls:

- max log events per execution
- max bytes per execution
- retention duration
- live stream timeout
- default tail size

Retention should be exposed through settings, similar in spirit to node log proxy settings.

### 8.4 Query Shape

Execution logs should support queries by:

- `execution_id`
- `workflow_id`
- `agent_node_id`
- `level`
- `source`
- time window
- text search

Chronological order should be the default sort.

---

## 9. Control Plane Responsibilities

### 9.1 Timeline vs Logs

The control plane should preserve the semantic distinction:

- lifecycle and workflow events are for the timeline
- structured execution logs are for the logs surface
- raw node logs remain advanced debug

### 9.2 UI Requirements

Execution page logs component should support:

- initial tail fetch
- live mode
- reconnect state
- pause/resume live updates
- filters:
  - level
  - node
  - source
  - free-text
  - developer vs system logs
- clear visual treatment of:
  - lifecycle event
  - structured execution log
  - raw node log

### 9.3 Advanced Debug Surface

Raw node logs should:

- live under an advanced/debug affordance
- allow choosing the relevant node
- optionally constrain by recent time window or tail
- carry clear messaging that they are process-level and may include unrelated activity

---

## 10. Settings

The system should expose configuration for structured execution logs similarly to current node log settings.

Suggested settings:

- execution log retention duration
- max events per execution
- max bytes per execution
- live stream idle timeout
- default tail size
- stdout mirroring enabled
- system log emission enabled

Separate settings should remain for raw node logs.

---

## 11. Rollout Strategy

### Phase 1: RFC and Schema

- finalize canonical log envelope
- define SDK logger contract
- define ingestion and retention policy
- align terminology across UI and backend

### Phase 2: SDK Runtime Foundations

- add execution-aware logger API to TypeScript SDK
- add execution-aware logger API to Go SDK
- add execution-aware logger API to Python SDK
- add automatic runtime/system logs
- add stdout mirroring controls

### Phase 3: Control Plane Ingestion and Query

- add execution-log ingestion endpoint
- add short-retention storage
- add execution-log fetch and live stream APIs
- add settings

### Phase 4: Execution Page UI

- add primary execution logs component
- integrate lifecycle timeline with logs section
- add advanced raw node logs view

### Phase 5: Coverage and Hardening

- improve SDK auto-coverage
- refine filter UX
- validate multi-node chronological behavior
- add retention and load tests

---

## 12. Risks and Tradeoffs

### 12.1 Raw Node Logs Will Still Be Ambiguous

Even after this RFC, raw node logs may still include unrelated activity if a node serves multiple executions. This is acceptable because raw node logs are intentionally not the primary product surface.

### 12.2 Partial Coverage in V1

Fast rollout means:

- some runtime paths may not yet emit structured logs
- some developers will still rely on plain stdout
- perfect correlation may not exist for all user code immediately

This is acceptable as long as the official path is clearly defined and promoted.

### 12.3 Cross-SDK Drift

Without a shared schema and behavior contract, TS, Go, and Python could diverge in:

- field names
- level semantics
- runtime event names
- stdout mirroring behavior

This RFC exists in part to prevent that drift.

### 12.4 Storage Pressure

Execution logs can become high-volume quickly. That is why v1 should prefer:

- short retention
- bounded per-execution limits
- live-first UX

---

## 13. Out of Scope for V1

The following are intentionally excluded from v1:

- raw control-plane server logs on the execution page
- long-term archival observability strategy
- distributed tracing system integration
- tree-grouped or graph-grouped log rendering beyond chronological order
- perfect attribution of arbitrary stdout without structured logger adoption

---

## 14. Implementation Work Breakdown

This work should be split into parallelizable tracks.

### Track A: Architecture and Contracts

1. Finalize structured execution log schema
2. Finalize source, level, and event naming conventions
3. Finalize retention and settings contract

### Track B: SDK and Runtime

4. TypeScript SDK: execution-aware logger API
5. TypeScript SDK: automatic runtime/system log emission
6. Go SDK: execution-aware logger API
7. Go SDK: automatic runtime/system log emission
8. Python SDK: execution-aware logger API
9. Python SDK: automatic runtime/system log emission
10. Cross-SDK: stdout mirroring and config semantics
11. Cross-SDK: execution context propagation audit and fixes

### Track C: Control Plane Backend

12. Execution-log ingestion API
13. Execution-log storage and retention implementation
14. Execution-log fetch API
15. Execution-log live stream API
16. Settings API and runtime config wiring

### Track D: UI

17. Execution page primary logs component
18. Execution page live mode and reconnect state
19. Execution page filters
20. Advanced raw node logs panel
21. Timeline and logs visual integration

### Track E: Verification

22. Functional tests for single-node execution logs
23. Functional tests for multi-node chronological rendering
24. Functional tests for concurrent executions on one node
25. Functional tests for fallback behavior when only raw node logs exist
26. Load and retention boundary tests

---

## 15. Acceptance Criteria for V1

V1 is complete when:

- an execution page exposes a primary `Execution Logs` surface
- logs can stream live and fetch a recent tail
- logs are filterable by level, node, source, and text
- SDK/runtime structured logs are stamped with execution identifiers
- key runtime transitions emit automatic system logs
- structured logs can also appear in stdout in local/dev workflows
- raw node logs are available under advanced/debug UI
- lifecycle events remain available as timeline context

---

## 16. Open Questions

These should be resolved before implementation fans out:

1. Should structured execution logs be stored only in the control plane, or also cached locally on nodes for recovery?
2. Should ingestion be per-log-event, batched, or both?
3. How should backpressure behave when a live execution emits logs faster than the control plane can persist them?
4. Should system-generated logs be suppressible per node or per environment?
5. Should the UI visually separate system logs from developer logs by default, or only via filters?

---

## 17. Recommendation

Proceed with this RFC as the coordination artifact for the execution-observability workstream.

Build execution-correlated structured logs as the primary execution-page experience, preserve lifecycle events as timeline anchors, and keep raw node logs as an advanced debug layer. This achieves a useful v1 quickly while preserving a clean path to richer observability later.
