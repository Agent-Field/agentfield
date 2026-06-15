# Databricks Incident Reviewer Python Node

This example is a second AgentField Python node that receives Databricks
trigger events and calls the deployed Databricks node.

It demonstrates the intended integration shape:

1. Control plane owns the Databricks trigger.
2. `databricks-prod` owns Databricks SQL, `ai_query`, and Model Serving calls.
3. This reviewer node composes those capabilities through `app.call`.

It does not call AgentField `.ai`; the AI review goes through
`databricks-prod.ai_query`.

## Run

Start the Databricks node first, then run this Python node:

```bash
AGENTFIELD_URL=http://localhost:18080 \
AGENT_NODE_ID=databricks-incident-reviewer \
AGENT_CALLBACK_URL=http://localhost:18113 \
DATABRICKS_TARGET_NODE=databricks-prod \
DATABRICKS_AI_ENDPOINT=databricks-meta-llama-3-1-8b-instruct \
PORT=18113 \
python main.py
```

Create a Databricks trigger whose target is:

```text
databricks-incident-reviewer.review_databricks_event
```

When a Databricks notification arrives, the reviewer node queries live
Databricks metrics and asks Databricks `ai_query` for a compact review.

## Direct Call

Another node can also call:

```python
await app.call(
    "databricks-incident-reviewer.summarize_metric_change",
    metric_name="manual_review risk",
    time_window="last 24 hours",
)
```
