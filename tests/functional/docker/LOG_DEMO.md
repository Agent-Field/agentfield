# Execution observability demo stack

This stack is the runnable demo path for the observability workstream.

Today it proves the raw node/process-log half of the story: the control plane
starts on port **8080**, three demo agents emit stdout/stderr, and the UI can
proxy those node logs under **Agents → expand row → Process logs**.

The execution-observability RFC adds the other half: structured execution logs
shown on the execution detail page as the primary surface, with raw node logs
kept behind an advanced/debug view.

## What the demo covers today

- Control plane + Python/Go/TypeScript demo agents
- NDJSON node/process log capture
- Live tailing and recent log history in the node log UI
- A sample execution-log fixture for downstream wiring and schema validation

## What the execution page should show after integration

- Chronological structured execution logs as the default view
- Filters by node, source, level, and free-text query
- Live/recent toggle for streaming and tailing
- Advanced raw node logs as a secondary debug surface

## Demo assets

- `make log-demo-up`
- `make log-demo-native-up`
- `tests/functional/docker/docker-compose.log-demo.yml`
- `scripts/run-log-demo-native.sh`
- `tests/functional/docker/execution-observability-sample.ndjson`

## Run

From the repository root:

```bash
make log-demo-up
# or:
docker compose -f tests/functional/docker/docker-compose.log-demo.yml up --build -d
```

### Docker Desktop not running (host stack)

The compose file uses `/data/...` paths that only exist inside containers. On the host, use:

```bash
make log-demo-native-up
```

This builds a local `agentfield-server` binary, stores SQLite/Bolt under `/tmp/agentfield-log-demo`, and starts the Python, Go, and Node demo agents on ports **8001-8003**. Stop with `make log-demo-native-down` (or `./scripts/stop-log-demo-native.sh`).

Open **http://localhost:8080/ui/agents** and expand:

| Node id            | Runtime |
|--------------------|---------|
| `demo-python-logs` | Python (`examples/python_agent_nodes/docker_hello_world`) |
| `demo-go-logs`     | Go (`examples/go_agent_nodes` via `Dockerfile.demo-go-agent`) |
| `demo-ts-logs`     | Node (`tests/functional/docker/log-demo-node/log-demo.mjs`) |

All services use `AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN=log-demo-internal-token` so the UI proxy can authenticate to each agent’s `GET /agentfield/v1/logs`.

Startup order: a one-shot **`wait-control-plane`** service polls `GET /api/v1/health` (with retries) before the three demo agents start, so they do not race the control plane on first boot. Demo agents use **`restart: unless-stopped`** so they recover if the CP restarts.

## Stop

```bash
make log-demo-down
```

## Validation

- `tests/functional/tests/test_ui_node_logs_proxy.py` validates the current NDJSON node-log proxy.
- `scripts/check-execution-observability-demo.sh` validates the structured execution-log sample fixture and keeps the docs aligned with the demo path.

## Integration blockers

The execution-detail UI and control-plane execution-log ingestion are not wired in this worktree yet. The demo assets in this branch therefore stop at the documentable seed stage: raw node logs are runnable now, and structured execution logs are represented by the sample fixture until the backend/UI integration lands.
