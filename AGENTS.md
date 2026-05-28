## Learned User Preferences

- Open-source AgentField should prioritize stable APIs and primitives so integrators can build advanced observability themselves; large packaged business or fleet observability belongs in Enterprise.
- The embedded OSS UI should stay a lightweight convenience layer, not the primary surface for org-wide analytics or governance-heavy views.
- Developer-facing observability belongs in OSS; deeper reliability and governance programs may span OSS and Enterprise.
- Avoid empty or placeholder PRs when stacking branches; prefer draft PRs with real implementation, then thorough review before marking ready.
- When designing or documenting control plane behavior, treat YAML configuration (`config/agentfield.yaml` and `AGENTFIELD_CONFIG_FILE`) as a first-class surface alongside environment variables.

## Learned Workspace Facts

- Monorepo: Go control plane in `control-plane/`, SDKs in `sdk/`, embedded admin UI in `control-plane/web/client/`.
