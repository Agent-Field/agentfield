# Databricks Integration

Databricks is modeled as two AgentField surfaces:

- **Control-plane trigger source**: `databricks` receives Databricks notification destination webhooks and dispatches normalized events to webhook-enabled reasoners.
- **Deployable agent node**: `databricks-prod` is a Go node that exposes Databricks-native capabilities to every other AgentField node through `app.call`.

The node does not replace Databricks AI with AgentField `.ai`. It exposes Databricks AI Functions and Model Serving directly so an orchestrator can call:

```python
result = await app.call("databricks-prod.query_readonly", input={
    "sql": "SELECT event_type, count(*) AS n FROM workspace.agentfield_e2e.agentfield_e2e_events GROUP BY event_type",
    "warehouse_id": "9ebaa0cbec4347d6",
    "catalog": "workspace",
    "schema": "agentfield_e2e",
})
```

```python
answer = await app.call("databricks-prod.ai_query", input={
    "endpoint": "databricks-meta-llama-3-3-70b-instruct",
    "prompt": "Summarize this KPI movement in one paragraph: ...",
})
```

```python
prediction = await app.call("databricks-prod.invoke_serving_endpoint", input={
    "endpoint": "fraud-score-prod",
    "body": {"dataframe_records": [{"amount": 2500, "country": "US"}]},
})
```

## Node Capabilities

- `query_readonly`: guarded Databricks SQL Statement Execution API call.
- `describe_table`: table metadata through Databricks SQL.
- `search_columns`: Unity Catalog column search through `information_schema`.
- `ai_query`: Databricks AI Functions through SQL.
- `invoke_serving_endpoint`: Databricks Model Serving invocation.
- `explain_result`: prompt-backed explanation using Databricks `ai_query`.
- `investigate_metric_change`: bounded investigation plan with recommended calls.
- `handle_databricks_event`: webhook entrypoint for Databricks trigger events.

## Trigger Source

Use Databricks workspace notification destinations for Jobs and SQL alerts. In AgentField, create a trigger with:

- source: `databricks`
- target reasoner: `databricks-prod.handle_databricks_event` or your own webhook-enabled node reasoner
- config:

```json
{
  "auth_mode": "basic",
  "basic_username": "agentfield",
  "event_type_path": "run_state.life_cycle_state",
  "event_id_path": "run_id",
  "workspace_path": "workspace_id"
}
```

The AgentField trigger secret becomes the Databricks webhook destination password.

## Run The Node

```bash
cp integrations/databricks/node/.env.example integrations/databricks/node/.env
$EDITOR integrations/databricks/node/.env
docker compose -f integrations/databricks/docker-compose.example.yml up --build
```

Required node env:

- `DATABRICKS_HOST`
- `DATABRICKS_TOKEN`
- `DATABRICKS_WAREHOUSE_ID`

Optional defaults:

- `DATABRICKS_CATALOG`
- `DATABRICKS_SCHEMA`
- `DATABRICKS_AI_ENDPOINT`
- `DATABRICKS_PROMPTS_FILE`

## Example Caller Node

See `examples/incident-reviewer` for a Python node that uses the deployed
Databricks node instead of connecting to Databricks directly. It can receive a
Databricks trigger event and then call:

- `databricks-prod.query_readonly`
- `databricks-prod.ai_query`

This is the intended modular shape for application-specific agent logic.

## API Sources Checked

- Databricks SQL Statement Execution API 2.0 for warehouse SQL.
- Databricks AI Functions and `ai_query` for native AI over data.
- Databricks Model Serving endpoint invocation for deployed AI/ML endpoints.
- Databricks notification destinations for webhook trigger delivery.
