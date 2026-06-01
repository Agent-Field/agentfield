# Agent node process logs (NDJSON v1)

Agent nodes MAY expose process stdout/stderr for the control plane UI to proxy.

## Agent endpoint

`GET {agent_base_url}/agentfield/v1/logs`

### Authentication

`Authorization: Bearer <token>` where `<token>` equals `AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN` on both control plane and agent (same value as used for execution forwarding).

### Query parameters

| Parameter     | Description |
|---------------|-------------|
| `tail_lines`  | Last N lines (default 200 if no `since_seq` and no `follow`). |
| `since_seq`   | Return entries with `seq` greater than this (monotonic per process). |
| `follow`      | If `1` or `true`, stream chunked NDJSON until client disconnects or server cap. |

### Response

- `Content-Type: application/x-ndjson`
- Each line is a JSON object:

```json
{"v":1,"seq":1,"ts":"2026-04-05T12:00:00.000Z","stream":"stdout","line":"hello","level":"info","source":"process"}
```

| Field    | Type   | Description |
|----------|--------|-------------|
| `v`      | int    | Schema version (1). |
| `seq`    | int    | Monotonic sequence number. |
| `ts`     | string | RFC3339 UTC timestamp. |
| `stream` | string | `stdout` or `stderr`. |
| `line`   | string | Single line (no embedded newlines). |
| `level`  | string | Optional; SDKs MAY set `info` for stdout, `error` for stderr, `log` otherwise. |
| `source` | string | Optional; e.g. `process` for captured stdio. |
| `truncated` | bool | Optional; line was truncated at max length. |

### Errors

| Status | Meaning |
|--------|---------|
| 401    | Missing or invalid bearer token. |
| 404    | Logs API disabled (`AGENTFIELD_LOGS_ENABLED=false`). |
| 413    | Requested tail exceeds server cap. |

## Control plane proxy (UI)

`GET /api/ui/v1/nodes/:nodeId/logs`

Proxies to the agent with the same query string and injects the internal bearer token. Requires UI/API authentication consistent with other `/api/ui/v1` routes.
