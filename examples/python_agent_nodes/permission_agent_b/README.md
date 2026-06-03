# Permission Agent B (Protected Target)

Protected agent that requires VC authorization before other agents can call it.

## Why it's protected

Matched by **two** protection rules in `control-plane/config/agentfield.yaml`:

1. `agent_id: permission-agent-b` — exact agent ID match
2. `tag: sensitive` — this agent is tagged `sensitive`

## What it does

- `process_payment` — processes a payment (amount + currency)
- `get_balance` — returns account balance

Both skills are protected because the entire agent is protected.

## Setup

```bash
# Terminal 1: Control plane with authorization enabled
cd control-plane && go run ./cmd/af dev

# Terminal 2: Start this agent
cd examples/python_agent_nodes/permission_agent_b && python main.py
```

Direct calls to this agent's skills through the control plane will be denied (403) unless the caller has an approved permission.
