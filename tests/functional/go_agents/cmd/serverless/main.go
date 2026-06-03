package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
)

func main() {
	nodeID := envOr("AGENT_NODE_ID", "go-serverless-func")
	port := envOr("PORT", "8090")

	cfg := agent.Config{
		NodeID:               nodeID,
		Version:              "1.0.0",
		AgentFieldURL:        strings.TrimSpace(os.Getenv("AGENTFIELD_URL")),
		Token:                os.Getenv("AGENTFIELD_TOKEN"),
		ListenAddress:        ":" + port,
		DeploymentType:       "serverless",
		DisableLeaseLoop:     true,
		LeaseRefreshInterval: 0,
	}

	srv, err := agent.New(cfg)
	if err != nil {
		log.Fatal(err)
	}

	srv.RegisterReasoner("hello", func(ctx context.Context, input map[string]any) (any, error) {
		exec := agent.ExecutionContextFrom(ctx)
		name := strings.TrimSpace(defaultString(asString(input["name"]), "AgentField"))
		return map[string]any{
			"greeting":            "Hello, " + name + "!",
			"run_id":              exec.WorkflowID,
			"execution_id":        exec.ExecutionID,
			"parent_execution_id": exec.ParentExecutionID,
		}, nil
	})

	srv.RegisterReasoner("relay", func(ctx context.Context, input map[string]any) (any, error) {
		target := strings.TrimSpace(defaultString(asString(input["target"]), os.Getenv("CHILD_TARGET")))
		if target == "" {
			return map[string]any{"error": "target is required"}, nil
		}
		message := defaultString(asString(input["message"]), "ping")
		res, err := srv.Call(ctx, target, map[string]any{"name": message})
		if err != nil {
			return nil, err
		}
		return map[string]any{
			"target":     target,
			"downstream": res,
		}, nil
	})

	addr := srvAddress(cfg.ListenAddress)
	log.Printf("serverless handler listening on %s", addr)
	if err := http.ListenAndServe(addr, srv.Handler()); err != nil {
		log.Fatal(err)
	}
}

func envOr(name, fallback string) string {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		return fallback
	}
	return value
}

func srvAddress(listen string) string {
	if strings.HasPrefix(listen, ":") {
		return listen
	}
	if listen == "" {
		return ":8090"
	}
	return listen
}

func defaultString(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return strings.TrimSpace(value)
}

func asString(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	default:
		return ""
	}
}
