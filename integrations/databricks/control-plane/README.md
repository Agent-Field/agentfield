# Databricks Control-Plane Source

The built-in `databricks` source receives Databricks notification destination webhooks. It is always available in the control plane after this package is compiled in.

The source verifies either:

- Basic auth: username from trigger config, password from the AgentField trigger secret.
- Bearer auth: `Authorization: Bearer <trigger secret>`.

The normalized event is dispatched to a webhook-enabled reasoner such as `databricks-prod.handle_databricks_event`.
