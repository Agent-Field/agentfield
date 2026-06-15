"""
Databricks incident reviewer node.

This is a separate AgentField node that uses the Databricks integration node.
It does not connect to Databricks directly and does not call AgentField .ai.

Flow:
  Databricks notification -> AgentField databricks trigger
    -> databricks-incident-reviewer.review_databricks_event
      -> databricks-prod.query_readonly
      -> databricks-prod.ai_query
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

from agentfield import Agent


app = Agent(
    node_id=os.getenv("AGENT_NODE_ID", "databricks-incident-reviewer"),
    agentfield_server=os.getenv("AGENTFIELD_URL", "http://localhost:18080"),
    dev_mode=True,
)

DATABRICKS_NODE = os.getenv("DATABRICKS_TARGET_NODE", "databricks-prod")
DATABRICKS_AI_ENDPOINT = os.getenv(
    "DATABRICKS_AI_ENDPOINT",
    "databricks-meta-llama-3-1-8b-instruct",
)
EVENT_SUMMARY_SQL = os.getenv(
    "DATABRICKS_EVENT_SUMMARY_SQL",
    """
SELECT event_type, COUNT(*) AS events, MAX(risk_score) AS max_risk, SUM(amount) AS total_amount
FROM workspace.agentfield_e2e.agentfield_e2e_events
GROUP BY event_type
ORDER BY max_risk DESC
LIMIT 10
""".strip(),
)


@app.reasoner(accepts_webhook=True)
async def review_databricks_event(
    event: dict[str, Any] | None = None,
    webhook: Any = None,
) -> dict[str, Any]:
    """
    Webhook-enabled reasoner for Databricks trigger events.

    Configure the AgentField trigger target as:

        databricks-incident-reviewer.review_databricks_event
    """
    normalized = event or {}
    payload = _dict(normalized.get("payload"))
    event_id = _first_string(
        normalized.get("event_id"),
        normalized.get("id"),
        payload.get("run_id"),
        getattr(webhook, "event_id", None),
    )
    event_type = _first_string(
        normalized.get("event_type"),
        normalized.get("type"),
        _nested(payload, "run_state").get("life_cycle_state"),
        getattr(webhook, "event_type", None),
    )

    metrics = await app.call(
        f"{DATABRICKS_NODE}.query_readonly",
        sql=EVENT_SUMMARY_SQL,
        max_rows=10,
    )

    metrics_preview = json.dumps(metrics, default=str)[:6000]
    review = await app.call(
        f"{DATABRICKS_NODE}.ai_query",
        endpoint=DATABRICKS_AI_ENDPOINT,
        prompt=(
            "Return compact JSON only with keys risk, reason, next_action. "
            "Use this Databricks event and live warehouse metric summary. "
            f"event_id={event_id or 'unknown'} event_type={event_type or 'unknown'} "
            f"metrics={metrics_preview}"
        ),
        fail_on_error=True,
    )

    return {
        "event_id": event_id,
        "event_type": event_type,
        "trigger_id": getattr(webhook, "trigger_id", None),
        "reviewed_by": app.node_id,
        "used_node": DATABRICKS_NODE,
        "databricks_metrics": metrics,
        "databricks_ai_query": review,
        "recommended_next_call": f"{DATABRICKS_NODE}.query_readonly",
    }


@app.reasoner()
async def summarize_metric_change(metric_name: str, time_window: str) -> dict[str, Any]:
    """
    Direct-call example for another node or operator.

    This composes Databricks-native capabilities exposed by the Databricks node.
    """
    plan = await app.call(
        f"{DATABRICKS_NODE}.investigate_metric_change",
        metric_name=metric_name,
        time_window=time_window,
    )
    review = await app.call(
        f"{DATABRICKS_NODE}.ai_query",
        endpoint=DATABRICKS_AI_ENDPOINT,
        prompt=(
            "Return compact JSON only with keys summary, likely_causes, next_query. "
            f"Metric={metric_name}. Window={time_window}. Plan={json.dumps(plan, default=str)}"
        ),
        fail_on_error=True,
    )
    return {
        "metric_name": metric_name,
        "time_window": time_window,
        "plan": plan,
        "databricks_ai_query": review,
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested(value: dict[str, Any], key: str) -> dict[str, Any]:
    return _dict(value.get(key))


def _first_string(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _log_heartbeat() -> None:
    n = 0
    while True:
        print(f"[{app.node_id}] reviewer heartbeat {n}", flush=True)
        n += 1
        time.sleep(15)


if __name__ == "__main__":
    threading.Thread(target=_log_heartbeat, daemon=True).start()
    port = int(os.getenv("PORT", "18113"))
    print(
        f"Databricks reviewer node using {DATABRICKS_NODE}; listening on :{port}",
        file=sys.stderr,
        flush=True,
    )
    app.run(host="0.0.0.0", port=port, auto_port=False)
