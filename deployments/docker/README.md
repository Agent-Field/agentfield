# Docker (local)

This folder contains a small Docker Compose setup for evaluating AgentField locally:

- Control plane (UI + REST API)
- PostgreSQL (pgvector)
- Optional demo agents (Go + Python)

## Quick start

```bash
cd deployments/docker
docker compose --profile python-demo up --build
```

Open the UI:
- `http://localhost:8080/ui/`

## Execute an agent via the control plane

Python demo agent (deterministic; no LLM keys required):

```bash
curl -X POST http://localhost:8080/api/v1/execute/demo-python-agent.hello \
  -H "Content-Type: application/json" \
  -d '{"input":{"name":"World"}}'
```

Go demo agent:

```bash
curl -X POST http://localhost:8080/api/v1/execute/demo-go-agent.demo_echo \
  -H "Content-Type: application/json" \
  -d '{"input":{"message":"Hello!"}}'
```

## Check Verifiable Credentials (VCs)

The Python SDK posts execution VC data back to the control plane. Grab the `run_id` and fetch the VC chain:

```bash
resp=$(curl -s -X POST http://localhost:8080/api/v1/execute/demo-python-agent.hello \
  -H "Content-Type: application/json" \
  -d '{"input":{"name":"VC"}}')
run_id=$(echo "$resp" | python3 -c 'import sys,json; print(json.load(sys.stdin)["run_id"])')
curl -s http://localhost:8080/api/v1/did/workflow/$run_id/vc-chain | head -c 1200
```

## Defaults (PostgreSQL)

- User / password / database: `agentfield` / `agentfield` / `agentfield`

## Anonymous Telemetry

The Docker stack enables anonymous usage telemetry by default to help us improve AgentField. It records coarse product signals such as startup, agent registration, SDK language, runtime type, and execution status buckets. The pseudonymous identifier represents the persisted installation, not a person or account.

The telemetry payload does not include prompts, inputs, outputs, logs, secrets, API keys, IP addresses, hostnames, user IDs, DIDs, or raw error text. Disable it with:

```bash
AGENTFIELD_TELEMETRY_ENABLED=false docker compose up
```

## Docker networking note (callback URL)

The control plane must be able to call your agent at the URL it registers.

- Same Compose network: use the service name (e.g. `http://demo-python-agent:8001`).
- Agent on host, control plane in Docker: use `host.docker.internal` (Python: `AGENT_CALLBACK_URL`, Go: `AGENT_PUBLIC_URL`).

## Cleanup

```bash
cd deployments/docker
docker compose --profile python-demo down -v
```
