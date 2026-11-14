# AgentField Go SDK

## Graceful shutdown

Server agents notify the control plane, stop accepting requests, and drain in-flight executions on SIGTERM, SIGINT, or `POST /shutdown`. Set `AGENTFIELD_SHUTDOWN_TIMEOUT` to bare seconds (`30`) or a duration (`30s`, `5m`); the default is 30 seconds. `Config.ShutdownTimeout` takes precedence. In Kubernetes, set `terminationGracePeriodSeconds` higher than the drain timeout.

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
        NodeID:        "example-agent",
        Version:       "1.0.0",
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

approvalClient, err := client.New("http://localhost:8080")
if err != nil {
    log.Fatal(err)
}
approvalRequestID := "req-abc123"

// Create the human-facing approval request in your approval service first,
// then pass its ID/URL to AgentField so the execution transitions to "waiting".
_, err = approvalClient.RequestApproval(ctx, nodeID, executionID,
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

## Memory Change Subscriptions

With a `ControlPlaneMemoryBackend`, an agent can react to memory changes as they happen. Other backends return `agent.ErrMemoryEventsUnsupported`.

```go
a, err := agent.New(agent.Config{
    NodeID:        "watcher",
    Version:       "1.0.0",
    AgentFieldURL: "http://localhost:8080",
    MemoryBackend: agent.NewControlPlaneMemoryBackend("http://localhost:8080", apiKey, "watcher"),
})

sub, err := a.Memory().Watch(ctx, agent.WatchOptions{
    Patterns: []string{"user_*", "config.*"},
    Scope:    agent.ScopeSession, // empty = every scope
    ScopeID:  sessionID,          // empty = every ID in the scope
})
if err != nil {
    log.Fatal(err) // bad pattern, auth failure, or control plane unreachable
}
defer sub.Close()

for ev := range sub.Events() {
    switch ev.Action {
    case agent.MemoryActionSet: // ev.PreviousData is empty on a first write
        v, _ := ev.Value()
        fmt.Printf("set %s = %v\n", ev.Key, v)
    case agent.MemoryActionDelete:
        fmt.Printf("deleted %s\n", ev.Key)
    }
}
if err := sub.Err(); err != nil {
    log.Printf("subscription ended: %v", err)
}

// Callback form: one handler goroutine, events delivered in order.
sub2, err := a.Memory().OnChange(ctx, "user_*.preferences", func(ev agent.MemoryChangeEvent) {
    fmt.Println("preferences changed:", ev.Key)
})
```

- **Events:** `Action` is `set` or `delete`. `Data` and `PreviousData` are raw JSON; use `Value()` or `DecodeData(&dst)`. `Metadata` names the agent, actor and workflow behind the change. The `actor` scope is reported as `agent.ScopeUser`.
- **Patterns:** filtered by the control plane with Go's `path/filepath.Match` rules: `*` matches any characters except `/`, `?` one character, `[...]` a class. `.` is literal, so `config.*` does not match `configX`. Invalid patterns and commas are rejected by `Watch`.
- **Lifetime:** the subscription runs until `Close()` or until `ctx` is cancelled; then `Events()` closes and `Err()` is nil.
- **Reconnects:** if the connection drops, the SDK reconnects with backoff (1s doubling to 30s), up to `MaxReconnectAttempts` (default 5; negative disables). A 4xx answer such as 401 or 403 stops it at once. When it gives up, `Events()` closes and `Err()` says why.
- **No replay:** changes made while disconnected are not resent. Backfill with `Memory().History(ctx, agent.MemoryHistoryOptions{Since: lastEvent.Timestamp, ...})`.
- **Backpressure:** `Events()` is buffered (`BufferSize`, default 64). When it is full, the stream waits for you rather than dropping events, so keep handlers fast or hand work to another goroutine.

## Logging

- `AGENTFIELD_LOG_STDOUT` controls the on-by-default structured JSON mirror; `0`, `false`, `no`, or `off` disables it without disabling control-plane dispatch.
- `AGENTFIELD_LOGS_ENABLED` controls stdout/stderr capture and the node logs endpoint, not control-plane dispatch.
- `AGENTFIELD_LOG_MAX_LINE_BYTES` defaults to 16384 bytes. Values below 256 and non-integers use that default; valid integers of 256 or greater are accepted.

See the [environment-variable reference](../../docs/ENVIRONMENT_VARIABLES.md) for details.

## Testing

```bash
go test ./...
```

## License

Distributed under the Apache 2.0 License. See the repository root for full details.

## Realtime session turn detection

See [session turn detection and interruption](../../docs/session-turn-detection.md)
for VAD configuration, defaults, validation, and the WebRTC connection flow.
