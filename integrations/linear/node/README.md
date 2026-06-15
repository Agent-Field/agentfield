# Linear Node

Run locally:

```bash
export AGENTFIELD_SERVER=http://localhost:8080
export LINEAR_API_KEY=lin_api_...
go run ./cmd/linear-node
```

Set `LINEAR_NODE_PUBLIC_URL` when the control plane should call the node
through a tunnel, container hostname, or host-qualified listen address.

For local tests without a Linear account, set `LINEAR_API_URL` to a mock GraphQL server. The unit tests use `httptest` and do not call Linear.
