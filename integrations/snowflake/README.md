# Snowflake Integration Pack

Snowflake is the first integration pack organized around AgentField's node mesh.
It has two surfaces:

- **Snowflake trigger source** built into the control plane. It converts Snowflake data,
  task, alert, and event-table signals into durable AgentField trigger events.
- **Snowflake capability node** as an optional deployable agent node. It exposes
  Snowflake SQL, catalog, Cortex Analyst, Cortex Search, and explanation
  capabilities to every AgentField SDK through `app.call`.

This is intentionally node-first rather than router-first. A standalone node is
callable from Python, Go, TypeScript, and future SDKs; it can also be deployed,
scaled, permissioned, and upgraded separately from the agents that use it.

## Product Model

```text
Snowflake alert/task/table event
  -> AgentField snowflake source
    -> persisted inbound event + VC evidence
      -> target reasoner on any node
        -> app.call("snowflake-prod.semantic_ask", ...)
        -> app.call("snowflake-prod.query_readonly", ...)
        -> app.call("servicenow-prod.create_incident", ...)
```

Snowflake owns governed data execution and Cortex-native reasoning. AgentField
owns cross-system orchestration, dynamic call graphs, async execution, policy,
event replay, DAGs, and audit.

## Developer Experience

The integration has two independent setup paths that can be used together:

1. Configure event ingestion in the AgentField control plane.
2. Optionally deploy the Snowflake capability node for cross-agent data access.

### Trigger Setup

The Snowflake trigger source is built into the control plane. Operators create
it from **Integrations -> Snowflake -> Connect** or through the trigger API.
The target node and target reasoner must already be registered, because the
control plane validates them before enabling the loop source.

Example UI/API config:

```json
{
  "mode": "event_table_poll",
  "account_url": "https://<account>.snowflakecomputing.com",
  "database": "AGENTFIELD_TEST",
  "schema": "PUBLIC",
  "table": "AGENTFIELD_EVENTS",
  "warehouse": "AGENTFIELD_WH",
  "role": "AGENTFIELD_TEST_ROLE",
  "interval_seconds": 30,
  "max_batch_size": 100
}
```

The trigger stores only a secret reference such as `SNOWFLAKE_PAT`; the PAT
value lives in the control-plane environment or future connection storage.
When rows appear in the configured table, the control plane persists inbound
event history, deduplicates by `EVENT_ID`, and dispatches the normalized event
to the selected reasoner.

### Capability Node Setup

The Snowflake node is a standalone Go binary packaged in Docker. Running the
container registers a normal AgentField node, for example `snowflake-prod`.
After registration, any other AgentField node can call its capabilities.

```bash
docker run --name agentfield-snowflake \
  -p 8012:8012 \
  -e AGENTFIELD_SERVER=http://host.docker.internal:8080 \
  -e AGENTFIELD_NODE_ID=snowflake-prod \
  -e SNOWFLAKE_NODE_LISTEN=:8012 \
  -e SNOWFLAKE_NODE_PUBLIC_URL=http://localhost:8012 \
  -e SNOWFLAKE_ACCOUNT_URL=https://<account>.snowflakecomputing.com \
  -e SNOWFLAKE_PAT=<programmatic-access-token> \
  -e SNOWFLAKE_DATABASE=AGENTFIELD_TEST \
  -e SNOWFLAKE_SCHEMA=PUBLIC \
  -e SNOWFLAKE_WAREHOUSE=AGENTFIELD_WH \
  -e SNOWFLAKE_ROLE=AGENTFIELD_TEST_ROLE \
  agentfield-snowflake-node:latest
```

Other nodes call it through the normal cross-node call surface:

```go
result, err := a.Call(ctx, "snowflake-prod.query_readonly", map[string]any{
	"sql": "SELECT EVENT_ID, EVENT_TYPE FROM AGENTFIELD_EVENTS LIMIT 10",
})
```

```python
result = await app.call("snowflake-prod.semantic_ask", input={
    "question": "What changed in revenue yesterday?"
})
```

## V1 Scope

Control plane, always present when AgentField ships this integration:

- `snowflake` source with `event_table_poll` mode.
- `custom_query_poll` mode for advanced users.
- Account URL, role, warehouse, database, schema, and auth secret env-var config.
- Trigger event normalization into `event`, `_meta`, and inbound event history.
- Generated setup SQL shown in UI/API, not a CLI-generated app.

Capability node, deployed only when the user wants callable Snowflake tools:

- Read-only SQL and catalog operations.
- Cortex Analyst natural-language data questions.
- Cortex Search queries.
- `.ai` based explanation and follow-up reasoners with config-exposed prompts.
- No write queries by default.

## Non-Goals

- No router-first API for the primary product surface.
- No hardcoded customer prompts inside a compiled node binary.
- No direct Snowflake credential access from arbitrary user agents.
- No generic "run any SQL" write capability in the default pack.
- No CLI scaffolding as the only setup path.

## Files

| File | Purpose |
| --- | --- |
| `agentfield-package.yaml` | Runtime manifest for the optional Snowflake node. |
| `contracts/capabilities.yaml` | Public callable surface for the Snowflake node. |
| `contracts/trigger-source.yaml` | Control-plane source contract and event shapes. |
| `control-plane/README.md` | Control-plane implementation notes. |
| `node/README.md` | Capability node runtime design. |
| `prompts/default-prompts.yaml` | Default prompts and override keys for `.ai` reasoners. |
