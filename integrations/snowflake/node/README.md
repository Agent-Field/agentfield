# Snowflake Capability Node

The Snowflake node is the primary user-facing integration runtime. It exposes
Snowflake capabilities to every AgentField agent through `app.call`.

## Why Node-Level

Routers are useful inside one Python or TypeScript app, but first-party
integrations need a cross-language surface. A node can be called by Python, Go,
TypeScript, or future SDKs; it can also be deployed, scaled, permissioned, and
upgraded independently.

Example callers:

```python
await app.call("snowflake-prod.semantic_ask", input={
    "question": "Why did revenue drop yesterday?",
    "semantic_model": "finance_metrics"
})
```

```go
result, err := a.Call(ctx, "snowflake-prod.query_readonly", map[string]any{
    "sql": "select current_timestamp()",
})
```

## Runtime Language

V1 is implemented in Go:

- deterministic API calls dominate the first surface,
- the node should start quickly and run with low memory,
- Go is a good fit for a long-running, separately deployed capability service,
- the public contract remains language-neutral through AgentField registration.

A Python router package can be added later for local developer workflows, but
it should not be the primary abstraction.

## Prompt Configuration

`.ai` based reasoners must not hide prompts in compiled code. The node should
load prompts in this precedence order:

1. Mounted YAML from `SNOWFLAKE_PROMPTS_FILE`.
2. Packaged defaults from `../prompts/default-prompts.yaml`.

The node reports the prompt source/version through `health` and prompt-backed
reasoner outputs. Do not log secret values or full customer prompts by default.

## Capability Layers

Primitive skills:

- `query_readonly`
- `describe_table`
- `search_columns`

Cortex skills:

- `cortex_analyst_message`
- `cortex_search_query`
- `cortex_complete`

Reasoners:

- `semantic_ask`
- `explain_result`
- `investigate_metric_change`

See `../contracts/capabilities.yaml` for input/output contracts.

## Guardrails

- Default to read-only.
- Use a dedicated Snowflake role with minimum required privileges.
- Enforce one-statement SQL.
- Deny DDL, DML, file transfer, role changes, and stored procedure calls in
  `query_readonly`.
- Enforce row limits and query timeouts.
- Return `NEEDS_REVIEW` when prompts or semantic context are insufficient.
- Include Snowflake request/query IDs in outputs when available.

## Package Entrypoint

```text
integrations/snowflake/node/
  cmd/snowflake-node/main.go
  internal/config/
  internal/prompts/
  internal/snowflake/
  internal/capabilities/
```

Do not let capability handlers depend on the control-plane source package. The
only shared contract between trigger and node should be YAML/docs schemas and
AgentField's public execution APIs.
