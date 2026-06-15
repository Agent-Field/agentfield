# Sentry Integration

This pack ships a first-party Sentry webhook source plus a deterministic Go capability node.

Use the source when Sentry should start an AgentField run from issue, alert, error, or comment webhooks. Use the node when an agent needs bounded Sentry API operations such as reading an issue, listing events, resolving an issue, or assigning ownership.

## Trigger Source

- Source name: `sentry`
- Kind: HTTP webhook
- Secret: `SENTRY_CLIENT_SECRET`
- Signature: `Sentry-Hook-Signature` HMAC-SHA256
- Idempotency: `Request-ID`, with a body hash fallback
- Event type: `<resource>.<action>`, lowercased, for example `issue.created`
- Timestamp validation: `Sentry-Hook-Timestamp` (300s tolerance by default, configurable)

## Capability Node

Required env:

```bash
export SENTRY_AUTH_TOKEN=sntrys_...
export SENTRY_ORG=agentfield
```

Useful capabilities:

- `list_issues`: list issues for a project and optional Sentry search query.
- `get_issue`: read one issue by issue ID.
- `list_issue_events` and `get_event`: inspect captured event payloads.
- `resolve_issue`, `assign_issue`, and `update_issue`: explicit issue mutations.

These are provider API calls. They only wrap explicit Sentry issue and event operations.

## Region

Sentry's API base URL depends on your organization's data-storage region. Set `SENTRY_BASE_URL` accordingly:

| Region | Value |
|--------|-------|
| Legacy US-only | `https://sentry.io` |
| US region | `https://us.sentry.io` |
| **EU region** | `https://de.sentry.io` |
| Self-hosted | Your install's hostname |

**EU customers must set `SENTRY_BASE_URL` explicitly.** The default `https://sentry.io` will fail for EU-region orgs.
