# E2E Resilience Tests

Tests for execution resilience features described in [#316](https://github.com/Agent-Field/agentfield/issues/316) and tracked in [#318](https://github.com/Agent-Field/agentfield/issues/318).

## What's tested

| # | Test | What it validates |
|---|------|-------------------|
| 1 | **Basic execution** | Sanity check: agent execution works end-to-end |
| 2 | **LLM health monitoring** | Health endpoint reports correct LLM backend state |
| 3 | **Circuit breaker - failure** | Detects LLM errors, opens circuit |
| 4 | **Execution rejection + error classification** | Returns HTTP 503 with `error_category: llm_unavailable` |
| 5 | **Circuit breaker - recovery** | Closes circuit after LLM recovers |
| 6 | **Concurrency limits** | Rejects excess requests with 429 when max slots occupied |
| 7 | **Node status accuracy** | Agent health and node summary visible via API |
| 8 | **Agent kill/restart** | Execution fails detectably when agent dies; recovers after restart |
| 9 | **SSE log streaming** | Real-time execution event stream connects and delivers events |
| 10 | **LLM freeze simulation** | Exact #316 scenario: LLM hangs, circuit opens, unfreeze recovers |
| 11 | **Post-recovery execution** | Execution works after LLM recovery |
| 12 | **Agent crash visibility** | Crash errors are descriptive and include agent error details |
| 13 | **Timeout behavior** | Slow executions respect client timeout instead of hanging |
| 14 | **Error details in records** | Failed execution records contain error messages |
| 15 | **Queue depth visibility** | Queue status endpoint shows per-agent slot usage |

## Quick start

```bash
# From repo root
cd examples/e2e_resilience_tests
./run_tests.sh
```

This will:
1. Clean up stale state from previous runs
2. Start a mock LLM server (port 14000)
3. Build and start the control plane (port 18080) with LLM health monitoring enabled
4. Start 3 test agents (healthy, slow, flaky)
5. Run all 15 test groups (27 assertions)
6. Clean up everything on exit

## API endpoints tested

| Endpoint | Purpose |
|----------|---------|
| `GET /api/ui/v1/llm/health` | LLM backend health + circuit breaker state |
| `GET /api/ui/v1/queue/status` | Per-agent concurrency slot usage |
| `GET /api/ui/v1/nodes/summary` | Registered agent list |
| `GET /api/ui/v1/nodes/:nodeId/status` | Individual agent health status |
| `GET /api/ui/v1/executions/:id/logs/stream` | SSE execution log stream |
| `POST /api/v1/execute/:target` | Sync execution (returns `error_category` on failure) |
| `POST /api/v1/execute/async/:target` | Async execution |
| `GET /api/v1/executions/:id` | Execution record with error details |

## Error categories

Execution failures now include an `error_category` field in the response:

| Category | Meaning |
|----------|---------|
| `llm_unavailable` | LLM backend circuit breaker is open |
| `concurrency_limit` | Agent has reached max concurrent executions |
| `agent_timeout` | Agent didn't respond within timeout |
| `agent_unreachable` | Agent is down (connection refused/reset) |
| `agent_error` | Agent returned an HTTP error (5xx, 4xx) |
| `bad_response` | Agent returned invalid/non-JSON response |
| `internal_error` | Control plane internal error |

## Manual testing

### 1. Simulate LLM going down (the #316 scenario)

```bash
# Start mock LLM
python3 mock_llm_server.py 4000 &

# Start control plane with LLM health monitoring
AGENTFIELD_LLM_HEALTH_ENABLED=true \
AGENTFIELD_LLM_HEALTH_ENDPOINT=http://localhost:4000/health \
AGENTFIELD_LLM_HEALTH_ENDPOINT_NAME=litellm \
AGENTFIELD_LLM_HEALTH_CHECK_INTERVAL=5s \
AGENTFIELD_LLM_HEALTH_FAILURE_THRESHOLD=3 \
AGENTFIELD_LLM_HEALTH_RECOVERY_TIMEOUT=15s \
  go run ./cmd/agentfield-server

# Check LLM health
curl http://localhost:8080/api/ui/v1/llm/health | python3 -m json.tool

# Freeze the LLM (simulates LiteLLM hanging)
curl -X POST http://localhost:4000/freeze

# Wait ~15 seconds, check again — should show circuit_state: open
curl http://localhost:8080/api/ui/v1/llm/health | python3 -m json.tool

# Try to execute — should get 503 with error_category: llm_unavailable
curl -X POST http://localhost:8080/api/v1/execute/my-agent.my-skill \
  -H "Content-Type: application/json" \
  -d '{"input":{"text":"test"}}'

# Unfreeze, wait ~20s for recovery
curl -X POST http://localhost:4000/unfreeze
```

### 2. Test concurrency limits

```bash
# Start control plane with max 3 concurrent per agent
AGENTFIELD_MAX_CONCURRENT_PER_AGENT=3 go run ./cmd/agentfield-server

# Check queue status
curl http://localhost:8080/api/ui/v1/queue/status | python3 -m json.tool

# Submit 5 slow jobs in parallel — 4th and 5th get 429
for i in $(seq 1 5); do
  curl -s -o /dev/null -w "Request $i: %{http_code}\n" \
    -X POST http://localhost:8080/api/v1/execute/test-slow.slow_task \
    -H "Content-Type: application/json" \
    -d '{"input":{"duration_seconds":30}}' &
done
wait
```

### 3. Watch execution logs in real-time

```bash
# Submit a job
EXEC_ID=$(curl -s -X POST http://localhost:8080/api/v1/execute/async/my-agent.my-skill \
  -H "Content-Type: application/json" \
  -d '{"input":{}}' | python3 -c "import json,sys; print(json.load(sys.stdin)['execution_id'])")

# Stream logs
curl -N http://localhost:8080/api/ui/v1/executions/$EXEC_ID/logs/stream
```

## Configuration reference

| Env var | Default | Description |
|---------|---------|-------------|
| `AGENTFIELD_LLM_HEALTH_ENABLED` | false | Enable LLM health monitoring |
| `AGENTFIELD_LLM_HEALTH_ENDPOINT` | - | URL to probe (e.g. `http://localhost:4000/health`) |
| `AGENTFIELD_LLM_HEALTH_ENDPOINT_NAME` | default | Display name for the endpoint |
| `AGENTFIELD_LLM_HEALTH_CHECK_INTERVAL` | 15s | How often to check |
| `AGENTFIELD_LLM_HEALTH_CHECK_TIMEOUT` | 5s | Timeout per check |
| `AGENTFIELD_LLM_HEALTH_FAILURE_THRESHOLD` | 3 | Failures before circuit opens |
| `AGENTFIELD_LLM_HEALTH_RECOVERY_TIMEOUT` | 30s | How long circuit stays open before testing recovery |
| `AGENTFIELD_MAX_CONCURRENT_PER_AGENT` | 0 (unlimited) | Max concurrent executions per agent |
| `AGENTFIELD_EXECUTION_MAX_RETRIES` | 0 (disabled) | Max auto-retries for timed-out executions |
| `AGENTFIELD_EXECUTION_RETRY_BACKOFF` | 30s | Backoff between retries |

## Mock LLM server

`mock_llm_server.py` simulates an LLM backend with controllable state:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Returns 200 (healthy) or 500 (error) or hangs (frozen) |
| `GET /state` | Current server state |
| `POST /freeze` | Make all requests hang (simulates LiteLLM freeze) |
| `POST /unfreeze` | Resume normal operation |
| `POST /error` | Make `/health` return 500 |
| `POST /recover` | Make `/health` return 200 |
| `POST /v1/chat/completions` | Fake LLM response (or hang if frozen) |

## Test agents

| Agent | Purpose |
|-------|---------|
| `agent_healthy.py` | Instant responses (`echo`, `add`), baseline/control |
| `agent_slow.py` | Configurable delay (`slow_task`), tests concurrency and stale detection |
| `agent_flaky.py` | Intermittent failures (`maybe_fail`), crashes (`crash`), hangs (`timeout_then_fail`) |
