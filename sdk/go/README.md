# Haxen Go SDK

The Haxen Go SDK provides idiomatic Go bindings for interacting with the Haxen control plane.

## Installation

```bash
go get github.com/agentfield/haxen/sdk/go
```

## Quick Start

```go
package main

import (
    "context"
    "log"

    haxenagent "github.com/agentfield/haxen/sdk/go/agent"
)

func main() {
    agent, err := haxenagent.New(haxenagent.Config{
        NodeID:   "example-agent",
        HaxenURL: "http://localhost:8080",
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

- `agent`: Build Haxen-compatible agents and register reasoners/skills.
- `client`: Low-level HTTP client for the Haxen control plane.
- `types`: Shared data structures and contracts.
- `ai`: Helpers for interacting with AI providers via the control plane.

## Testing

```bash
go test ./...
```

## License

Distributed under the Apache 2.0 License. See the repository root for full details.
