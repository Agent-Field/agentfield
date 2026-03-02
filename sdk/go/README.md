# AgentField Go SDK

The AgentField Go SDK provides idiomatic Go bindings for interacting with the AgentField control plane.

## Installation

```bash
go get github.com/Agent-Field/agentfield/sdk/go
```

## Quick Start

```go
package main

import (
    "context"
    "log"

    agentfieldagent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

func main() {
    agent, err := agentfieldagent.New(agentfieldagent.Config{
        NodeID:   "example-agent",
        AgentFieldURL: "http://localhost:8080",
    })
    if err != nil {
        log.Fatal(err)
    }

    agent.RegisterSkill("health", func(ctx context.Context, _ map[string]any) (any, error) {
        return map[string]any{"status": "ok"}, nil
    })

    if err := agent.Run(context.Background()); err != nil {
        log.Fatal(err)
    }
}
```

## Modules

- `agent`: Build AgentField-compatible agents and register reasoners/skills.
- `client`: Low-level HTTP client for the AgentField control plane.
- `types`: Shared data structures and contracts.
- `ai`: Helpers for interacting with AI providers via the control plane.

## Human-in-the-Loop Approvals

The `client` package provides methods for requesting human approval, checking status, and waiting for decisions:

```go
import "github.com/Agent-Field/agentfield/sdk/go/client"

approvalClient := client.New("http://localhost:8080", nil)

// Request approval — transitions execution to "waiting"
_, err := approvalClient.RequestApproval(ctx, nodeID, executionID,
    client.RequestApprovalRequest{
        Title:          "Review Deployment",
        ProjectID:      "my-project",
        TemplateType:   "plan-review-v1",
        ExpiresInHours: 24,
    },
)

// Wait for human decision (uses context.Context for timeout)
waitCtx, cancel := context.WithTimeout(ctx, 1*time.Hour)
defer cancel()

result, err := approvalClient.WaitForApproval(waitCtx, nodeID, executionID,
    &client.WaitForApprovalOptions{
        PollInterval: 5 * time.Second,
        MaxInterval:  30 * time.Second,
    },
)
// result.Status is "approved", "rejected", or "expired"
```

**Methods:** `RequestApproval()`, `GetApprovalStatus()`, `WaitForApproval()`

## Testing

```bash
go test ./...
```

## License

Distributed under the Apache 2.0 License. See the repository root for full details.
