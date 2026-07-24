## Learned User Preferences

- Open-source AgentField should prioritize stable APIs and primitives so integrators can build advanced observability themselves; large packaged business or fleet observability belongs in Enterprise.
- The embedded OSS UI should stay a lightweight convenience layer, not the primary surface for org-wide analytics or governance-heavy views.
- Developer-facing observability belongs in OSS; deeper reliability and governance programs may span OSS and Enterprise.
- Avoid empty or placeholder PRs when stacking branches; prefer draft PRs with real implementation, then thorough review before marking ready.
- When designing or documenting control plane behavior, treat YAML configuration (`config/agentfield.yaml` and `AGENTFIELD_CONFIG_FILE`) as a first-class surface alongside environment variables.
- AgentField Desktop targets GitHub-comfortable developers (not infra experts); primary jobs are installing agent nodes from GitHub and seeing runs/cost as a local sub-harness for coding agents.
- Desktop UI should use shared theme tokens rather than hardcoded page styles; treat Agents as a marketplace-style library (installed agents + add), and design Activity for high-volume dense/filterable runs rather than large cards.
- Locked desktop decisions: gold/amber accent; cold-launch to Home when agents exist (add/empty flow when none); usage totals on Home plus Activity per-row when the API allows; keep the update banner across views.

## Learned Workspace Facts

- Monorepo: Go control plane in `control-plane/`, SDKs in `sdk/`, embedded admin UI in `control-plane/web/client/`, Electron desktop app in `desktop/`.
- Agent-node manifests (`agentfield-package.yaml`) carry a `config_version` (schema version, e.g. `v1`; absent = `v0`) that is separate from the node's own `version:`. Bump `config_version` only for breaking format changes, never for additive fields. The single reader is `packages.ParsePackageMetadata` (`control-plane/internal/packages/installer.go`); the authoring contract lives in `docs/installing-agent-nodes.md`.
- Desktop design/product specs live in `DESIGN.md` and `PRODUCT.md`.
- Desktop featured-catalog copy is maintained in `desktop/src/shared/catalog.ts`; post-install descriptions come from each agent's `agentfield-package.yaml` (marketplace cards do not fetch YAML live).
