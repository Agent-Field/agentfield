# Databricks Node

This Go node exposes Databricks as an AgentField agent node. Other nodes call it with stable capability IDs instead of hardcoding Databricks REST paths, SQL API request shapes, or serving endpoint URLs.

## Capabilities

- `query_readonly`
- `describe_table`
- `search_columns`
- `ai_query`
- `invoke_serving_endpoint`
- `explain_result`
- `investigate_metric_change`
- `handle_databricks_event`

## Environment

```bash
AGENTFIELD_SERVER=http://host.docker.internal:8080
AGENTFIELD_NODE_ID=databricks-prod
DATABRICKS_HOST=https://dbc-xxxx.cloud.databricks.com
DATABRICKS_TOKEN=...
DATABRICKS_WAREHOUSE_ID=...
DATABRICKS_CATALOG=workspace
DATABRICKS_SCHEMA=agentfield_e2e
DATABRICKS_AI_ENDPOINT=databricks-meta-llama-3-3-70b-instruct
```

`DATABRICKS_PROMPTS_FILE` can point at a YAML override for prompt-backed calls. Prompt-backed calls still execute through Databricks `ai_query`; they do not call AgentField `.ai`.
