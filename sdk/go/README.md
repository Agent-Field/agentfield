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

## AI Tool Calling

Execute LLM tool-call loops with automatic capability discovery:

```go
import (
    "github.com/Agent-Field/agentfield/sdk/go/ai"
    "github.com/Agent-Field/agentfield/sdk/go/agent"
)

// Convert discovered capabilities to LLM tool definitions
tools := ai.CapabilitiesToToolDefinitions(discoveryResult.Capabilities)

// Execute tool-call loop with guardrails
response, trace, err := aiClient.ExecuteToolCallLoop(
    ctx,
    messages,
    tools,
    ai.ToolCallConfig{MaxTurns: 10, MaxToolCalls: 25},
    func(ctx context.Context, target string, input map[string]interface{}) (map[string]interface{}, error) {
        return agent.Call(ctx, target, input)
    },
)

fmt.Printf("Final: %s\n", response.Text())
fmt.Printf("Tool calls: %d, Turns: %d\n", trace.TotalToolCalls, trace.TotalTurns)
```

**Key features:**
- `CapabilitiesToToolDefinitions` — Convert discovery results to OpenAI tool schemas
- `ExecuteToolCallLoop` — Automatic LLM tool-call loop with turn/call limits
- `ToolCallTrace` — Per-call latency tracking and observability

## Human-in-the-Loop Approvals

The `client` package provides methods for requesting human approval, checking status, and waiting for decisions:

```go
import "github.com/Agent-Field/agentfield/sdk/go/client"

approvalClient := client.New("http://localhost:8080", nil)
approvalRequestID := "req-abc123"

// Create the human-facing approval request in your approval service first,
// then pass its ID/URL to AgentField so the execution transitions to "waiting".
_, err := approvalClient.RequestApproval(ctx, nodeID, executionID,
    client.RequestApprovalRequest{
        ApprovalRequestID:  approvalRequestID,
        ApprovalRequestURL: "https://approvals.example.com/review/" + approvalRequestID,
        ExpiresInHours:     24,
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

## Workspace Artifacts

A caller can attach a local folder to any reasoner execution. The platform seals
the folder, transports it to your node, materializes it into a per-execution
directory, runs your reasoner against it, and returns the file changes as a
staged diff the caller applies explicitly. **Reasoner authors write no
folder-handling code.**

The transport endpoints — `POST /api/v1/workspace/prepare`,
`PUT/GET /api/v1/workspace/blobs/{sha256}`, and the one-shot
`POST /api/v1/workspace/blobs/batch` (a gzip tar of many blobs) — are
**registered automatically** on every node server. There is nothing to wire up.

Inside a reasoner, read the workspace path from the context and set it as the
working directory of any subprocess you spawn:

```go
agent.RegisterReasoner("build", func(ctx context.Context, in map[string]any) (any, error) {
    dir, ok := agent.WorkspaceDir(ctx)
    if !ok {
        // No workspace attached — reasoner runs normally.
        return map[string]any{"skipped": true}, nil
    }

    // Read/write files relative to dir…
    data, _ := os.ReadFile(filepath.Join(dir, "go.mod"))

    // …and run tools inside the workspace via cmd.Dir.
    cmd := exec.CommandContext(ctx, "go", "test", "./...")
    cmd.Dir = dir
    out, err := cmd.CombinedOutput()

    return map[string]any{"log": string(out)}, err
})
```

When a workspace is attached, the SDK automatically diffs the directory after
your reasoner returns and adds a `workspace_diff` to the result
(`{"changed":[{path,sha256,size,mode}],"deleted":[path]}`). A map result gains
the key in place; any other result is wrapped as
`{"result": <original>, "workspace_diff": ...}`. Executions **without** a
workspace are completely unaffected.

`WorkspaceDir(ctx)` is a per-execution value carried on the context — the SDK
never sets a process-global env var or `chdir`, which would race across
concurrent requests. In a single Go process the working directory cannot be
switched per request, so the POC contract is `WorkspaceDir(ctx)` + `cmd.Dir`;
a per-execution isolation worker (matching the spec's worker semantics) is a
documented follow-up.

The `workspace` package (`github.com/Agent-Field/agentfield/sdk/go/workspace`)
is reusable standalone: `Seal`, `Materialize`, `ComputeDiff`, `Apply`,
`CanonicalJSON`/`ManifestID`, and the `CAS` content store all export clean
funcs.

## Testing

```bash
go test ./...
```

## License

Distributed under the Apache 2.0 License. See the repository root for full details.
