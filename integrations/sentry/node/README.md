# Sentry Node

Run locally:

```bash
export AGENTFIELD_SERVER=http://localhost:8080
export SENTRY_AUTH_TOKEN=sntrys_...
export SENTRY_ORG=agentfield
go run ./cmd/sentry-node
```

Set `SENTRY_NODE_PUBLIC_URL` when the control plane should call the node
through a tunnel, container hostname, or host-qualified listen address.

For local tests without a Sentry account, set `SENTRY_BASE_URL` to a mock Sentry API server. The unit tests use `httptest` and do not call Sentry.
