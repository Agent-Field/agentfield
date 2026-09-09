# Agent node process logs (NDJSON v1)

Agent nodes MAY expose process stdout/stderr for the control plane UI to proxy.

Capture is enabled by default and bounded by `AGENTFIELD_LOG_BUFFER_BYTES`
(default 4194304 bytes) and `AGENTFIELD_LOG_MAX_LINE_BYTES` (default 16384
bytes). Python clamps integer line caps below 256 to 256; Go and TypeScript
reject them and use the default. Set `AGENTFIELD_LOGS_ENABLED=false` to disable
capture and this API. The Python, Go, and TypeScript SDKs mirror structured
records to stdout by default; set `AGENTFIELD_LOG_STDOUT=false` to disable that
mirror while retaining control-plane delivery for records with an execution ID.
All three SDKs skip that delivery for a record with no execution id, so such
records are stdout-only and are dropped when mirroring is disabled. Since the
node-log ring is fed by captured stdout, disabling the mirror also removes
structured records from `GET /agentfield/v1/logs`.

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
