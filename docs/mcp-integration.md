# Embedded MCP server

The AgentField control plane ships a built-in [Model Context
Protocol](https://modelcontextprotocol.io) server. It is **embedded in the
control plane itself and served on the same port as the REST API**, so any AI
harness that speaks MCP can discover and drive the agents you already have
running — with zero extra processes and zero setup.

- **Endpoint:** `POST <server>/mcp` (default `http://localhost:8080/mcp`)
- **Transport:** streamable HTTP, JSON-RPC 2.0
- **State:** stateless — no session negotiation; `Mcp-Session-Id` and
  `MCP-Protocol-Version` request headers are tolerated and ignored
- **On by default.** Disable with `AGENTFIELD_MCP_ENABLED=false`.

## Connecting a harness

Claude Code:

```bash
claude mcp add --transport http agentfield http://localhost:8080/mcp
```

Any other MCP client: point it at the same URL with the `http` (streamable-HTTP)
transport. Because the transport is stateless there is nothing else to
configure. If the control plane has an API key set (`AGENTFIELD_API_KEY`), the
`/mcp` endpoint requires it just like the REST API — pass it as an
`X-API-Key: <key>` header in the client's MCP configuration.

The `af` CLI and the REST API remain the full-power path (sessions, streaming,
cancel-tree, secrets, load-aware pacing). MCP covers the common discover →
execute → poll loop; reach for the CLI when a task needs more.

## Tools

| Tool | Arguments | Returns |
|---|---|---|
| `discover_agents` | `health?`: `"active"` (default) or `"all"` | Agents with `id`, `health_status`, `last_heartbeat`, and their `reasoners` (`id`, `description`, entrypoint `tags`, `target`). `"active"` lists only healthy agents; `"all"` includes unhealthy ones. |
| `get_reasoner_schema` | `node`, `reasoner` | The reasoner's input/output JSON Schema, description, and tags, exactly as the discovery surface serves them. |
| `execute_reasoner` | `target` (`"node.reasoner"`), `input` (object) | Validates the target exists (a clear error naming `discover_agents` if not), starts an **asynchronous** execution, and returns `{ run_id, status: "accepted" }`. |
| `get_run` | `run_id` | Current run `status`, `result`/`error`, and per-execution summaries (`reasoner_id`, `node_id`, `status`). |
| `wait_run` | `run_id`, `timeout_seconds?` (default 60, max 120) | Server-side poll until the run is terminal or the timeout elapses. Same shape as `get_run` plus a `timed_out` flag. The timeout is hard-capped so a single tool call can never hang the harness. |

Tool results are returned as a single MCP text content block containing compact
JSON — the shape harnesses parse most reliably. Validation and business
failures (unknown target, unknown run) come back as MCP tool errors
(`isError: true`) rather than transport errors, so the model sees and can react
to them.

## Example flow

1. `discover_agents` → find `swe-planner` and its `build` reasoner.
2. `get_reasoner_schema` `{ node: "swe-planner", reasoner: "build" }` → learn the
   input shape.
3. `execute_reasoner` `{ target: "swe-planner.build", input: { goal: "Add JWT auth" } }`
   → `{ run_id: "run_…", status: "accepted" }`.
4. `wait_run` `{ run_id: "run_…", timeout_seconds: 120 }` → block until the run
   finishes (or `timed_out: true`), then read `result`.

## Security

The MCP endpoint binds wherever the control plane binds and lives in the **same
trust domain as the REST API** — it starts async executions and reads run state
through the same service layer, subject to the same global API-key auth. Do not
expose the control plane's port to an untrusted network expecting `/mcp` to be
more restricted than `/api/v1`; it is not. Set `AGENTFIELD_MCP_ENABLED=false` to
remove the endpoint entirely (it then returns `404`).
