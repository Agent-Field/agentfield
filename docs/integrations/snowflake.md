# Snowflake Integration Design

Snowflake integration is split into a built-in control-plane source and an
optional deployable capability node.

## Control-Plane Source

The source ingests Snowflake-originated events and dispatches them to any
AgentField reasoner. The default mode is `event_table_poll`: Snowflake alerts
or tasks write to an `AGENTFIELD_EVENTS` table, and the control plane polls it
through Snowflake SQL API using the trigger's configured account URL and
secret env var.

This keeps AgentField setup UI/API-first and avoids CLI-generated hardcoding.

## Capability Node

The Snowflake node is deployed as a standalone AgentField node, for example
`snowflake-prod`. It registers typed capabilities that every SDK can call:

```text
snowflake-prod.query_readonly
snowflake-prod.describe_table
snowflake-prod.semantic_ask
snowflake-prod.cortex_analyst_message
snowflake-prod.cortex_search_query
snowflake-prod.explain_result
```

This is the cross-language abstraction. Routers can exist later for local
composition, but node-level capabilities are the product surface.

## Prompt Overrides

Reasoners that use `.ai` load prompt templates from configuration:

1. `SNOWFLAKE_PROMPTS_FILE`
2. packaged defaults

This prevents customer-specific behavior from being hardcoded in the node
binary.

## Source of Truth

Implementation contracts live under:

```text
integrations/snowflake/
```

The docs page is a pointer, not a second copy of the full contract.
