# Integrations

AgentField integrations are organized as modular packs in the repository.

Each pack can include:

- control-plane trigger source contracts,
- installable capability node manifests,
- node capability contracts,
- prompt configuration defaults for prompt-backed reasoners,
- implementation notes for provider-specific runtime code.

Current first-party integration design:

- [Snowflake](snowflake.md)
- [Databricks](databricks.md)

The canonical pack files live under `integrations/<provider>/`.
