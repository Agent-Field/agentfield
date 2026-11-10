package main

import (
	"context"
	"log"
	"os"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
)

func envOr(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

func watchStopFile(stopFile string, cancel context.CancelFunc, done <-chan struct{}) {
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case <-done:
			return
		case <-ticker.C:
			if _, err := os.Stat(stopFile); err == nil {
				cancel()
				return
			}
		}
	}
}

func main() {
	nodeID := envOr("AGENT_NODE_ID", "smoke-go-130")
	cpURL := envOr("AGENTFIELD_URL", "http://127.0.0.1:18080")
	listenAddr := envOr("AGENT_LISTEN_ADDR", "127.0.0.1:18103")
	publicURL := envOr("AGENT_PUBLIC_URL", "http://"+listenAddr)
	version := envOr("AGENT_VERSION", "1.0.0")

	hbInterval := 5 * time.Second
	if raw := strings.TrimSpace(os.Getenv("SMOKE_HEARTBEAT_INTERVAL")); raw != "" {
		if parsed, err := time.ParseDuration(raw); err == nil && parsed > 0 {
			hbInterval = parsed
		}
	}

	node, err := agent.New(agent.Config{
		NodeID:               nodeID,
		Version:              version,
		AgentFieldURL:        cpURL,
		ListenAddress:        listenAddr,
		PublicURL:            publicURL,
		LeaseRefreshInterval: hbInterval,
	})
	if err != nil {
		log.Fatalf("create agent: %v", err)
	}

	node.RegisterReasoner("echo", func(ctx context.Context, input map[string]any) (any, error) {
		ec := agent.ExecutionContextFrom(ctx)
		echoed := make(map[string]any, len(input))
		for k, v := range input {
			echoed[k] = v
		}
		return map[string]any{
			"echo":        echoed,
			"node_id":     nodeID,
			"sdk":         "go",
			"sdk_version": "v0.1.130",
			"execution_context": map[string]any{
				"run_id":              ec.RunID,
				"execution_id":        ec.ExecutionID,
				"parent_execution_id": ec.ParentExecutionID,
				"session_id":          ec.SessionID,
				"actor_id":            ec.ActorID,
				"workflow_id":         ec.WorkflowID,
				"root_workflow_id":    ec.RootWorkflowID,
				"agent_node_id":       ec.AgentNodeID,
				"reasoner_name":       ec.ReasonerName,
			},
		}, nil
	}, agent.WithDescription("Deterministic smoke echo reasoner (sdk/go v0.1.130)"))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if stopFile := strings.TrimSpace(os.Getenv("SMOKE_STOP_FILE")); stopFile != "" {
		go watchStopFile(stopFile, cancel, ctx.Done())
	}

	if err := node.Serve(ctx); err != nil {
		log.Fatalf("serve agent: %v", err)
	}
}
