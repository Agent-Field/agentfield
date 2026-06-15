package main

import (
	"context"
	"log"
	"os"
	"strings"

	afagent "github.com/Agent-Field/agentfield/sdk/go/agent"
)

func main() {
	nodeID := env("AGENTFIELD_NODE_ID", "snowflake-caller-e2e")
	server := env("AGENTFIELD_SERVER", "http://localhost:18080")
	listen := env("CALLER_NODE_LISTEN", ":18112")
	publicURL := env("CALLER_NODE_PUBLIC_URL", "http://localhost:18112")
	target := env("SNOWFLAKE_TARGET", "snowflake-e2e")

	a, err := afagent.New(afagent.Config{
		NodeID:        nodeID,
		Version:       "0.1.0-e2e",
		AgentFieldURL: server,
		ListenAddress: listen,
		PublicURL:     publicURL,
		Tags:          []string{"e2e", "snowflake-caller"},
	})
	if err != nil {
		log.Fatal(err)
	}
	a.RegisterReasoner("ask_snowflake", func(ctx context.Context, input map[string]any) (any, error) {
		sqlResult, err := a.Call(ctx, target+".query_readonly", map[string]any{
			"sql":      "SELECT id, amount FROM revenue",
			"max_rows": 5,
		})
		if err != nil {
			return nil, err
		}
		cortexResult, err := a.Call(ctx, target+".cortex_complete", map[string]any{
			"model":  "claude-sonnet-4-5",
			"prompt": "Summarize the revenue query.",
		})
		if err != nil {
			return nil, err
		}
		return map[string]any{
			"question": input["question"],
			"sql":      sqlResult,
			"cortex":   cortexResult,
		}, nil
	}, afagent.WithDescription("E2E caller that invokes the Docker-hosted Snowflake node"))
	a.RegisterReasoner("handle_snowflake_event", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{
			"received": true,
			"event":    input,
		}, nil
	}, afagent.WithDescription("E2E receiver for Snowflake trigger events"), afagent.WithAcceptsWebhook("true"))
	if err := a.Run(context.Background()); err != nil {
		log.Fatal(err)
	}
}

func env(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}
