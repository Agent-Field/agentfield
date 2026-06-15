# Snowflake Control-Plane Source

The Snowflake control-plane integration is a trigger source, not a data access
node. It belongs in AgentField core because it owns durable event ingestion and
dispatch. It is not an optional node package; once included in the control
plane build, it appears in the normal source catalog.

## Implementation

```text
control-plane/internal/sources/snowflake/
  snowflake.go
  snowflake_test.go
```

Register it from:

```text
control-plane/internal/sources/all/all.go
```

The implementation satisfies `sources.LoopSource` first. Direct HTTP can be
added later only if a customer setup can safely produce arbitrary HTTPS
webhooks to AgentField.

## V1 Mode: event_table_poll

The source polls a configured Snowflake event table:

```sql
CREATE TABLE IF NOT EXISTS AGENTFIELD_EVENTS (
  EVENT_ID STRING DEFAULT UUID_STRING(),
  EVENT_TYPE STRING NOT NULL,
  PAYLOAD VARIANT NOT NULL,
  OCCURRED_AT TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP(),
  CONSUMED_AT TIMESTAMP_LTZ
);
```

Each unconsumed row becomes a `sources.Event`:

```json
{
  "event_id": "7f3...",
  "event_type": "snowflake.alert.metric_threshold",
  "occurred_at": "2026-06-11T17:00:00Z",
  "payload": { "metric": "revenue", "window": "1 hour" },
  "snowflake": {
    "database": "OBSERVABILITY",
    "schema": "AGENTFIELD",
    "table": "AGENTFIELD_EVENTS"
  }
}
```

Use `EVENT_ID` as the idempotency key. The control plane already dedupes on
`(source_name, idempotency_key)` and records event history before dispatch.

## Why Polling First

Snowflake supports notifications to queues, email, and webhooks, but direct
webhook notifications are documented around Slack, Microsoft Teams, and
PagerDuty-specific shapes. The general, portable path is to let Snowflake
write events into a table and let AgentField poll them through the Snowflake
SQL API.

Queue consumption can be added later by composing existing cloud queue sources
or by adding source modes that read SNS, Pub/Sub, or Azure Event Grid
notifications.

## Configuration Shape

The source config is documented in `../contracts/trigger-source.yaml`.

Keep credentials out of trigger config. Triggers should reference a connection
or secret env var:

- `connection_ref`: preferred future UI/API connection reference.
- `secret_env_var`: current source contract for secrets.

## UI/API Setup

The integration setup surface should show:

- connection test status,
- generated Snowflake setup SQL,
- selected table/query,
- polling interval and batch size,
- target node/reasoner,
- latest inbound events and dispatch statuses.

Do not require a CLI scaffold. A CLI may later call the same APIs, but the
source must be fully configurable through AgentField's control-plane surfaces.
