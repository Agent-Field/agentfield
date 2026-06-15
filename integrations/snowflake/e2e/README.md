# Snowflake E2E Harness

This folder contains a local smoke harness for the optional Snowflake node:

- `fake-snowflake`: local HTTP server that implements the Snowflake SQL API,
  Cortex REST chat-completions, Cortex Analyst, and Cortex Search paths used by
  the node.
- `caller-node`: second AgentField node that calls the Docker-hosted
  Snowflake node through the control plane.

The harness is intentionally credential-free. It proves AgentField registration,
cross-node execution, Docker hosting, and Snowflake API request shape without
requiring a live Snowflake account.
