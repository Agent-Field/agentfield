# Permission Agent A (Caller)

Normal agent that attempts to call a protected agent (`permission-agent-b`) through the control plane.

## What it does

- `ping` — simple health check, no cross-agent call
- `call_payment_gateway` — calls `permission-agent-b.process_payment` via the control plane, triggering the VC authorization middleware

## Setup

Requires the control plane running with `authorization.enabled: true` in config.

```bash
# Terminal 1: Control plane
cd control-plane && go run ./cmd/af dev

# Terminal 2: Start the protected agent first
cd examples/python_agent_nodes/permission_agent_b && python main.py

# Terminal 3: Start this agent
cd examples/python_agent_nodes/permission_agent_a && python main.py
```

## Testing the permission flow

```bash
# Trigger Agent A to call Agent B (will be denied until approved)
curl -X POST http://localhost:8080/api/v1/execute/permission-agent-a.call_payment_gateway \
  -H "Content-Type: application/json" \
  -d '{"input": {"amount": 99.99, "currency": "USD"}}'

# Check pending permissions
curl http://localhost:8080/api/v1/admin/permissions/pending

# Approve (replace 1 with actual ID)
curl -X POST http://localhost:8080/api/v1/admin/permissions/1/approve \
  -H "Content-Type: application/json" \
  -d '{"duration_hours": 24}'

# Retry the call (should succeed now)
curl -X POST http://localhost:8080/api/v1/execute/permission-agent-a.call_payment_gateway \
  -H "Content-Type: application/json" \
  -d '{"input": {"amount": 99.99, "currency": "USD"}}'
```
