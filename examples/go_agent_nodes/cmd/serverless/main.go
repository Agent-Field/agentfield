package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
)

func main() {
	nodeID := strings.TrimSpace(os.Getenv("AGENT_NODE_ID"))
	if nodeID == "" {
		nodeID = "go-serverless-hello"
	}

	cfg := agent.Config{
		NodeID:               nodeID,
		Version:              "1.0.0",
		AgentFieldURL:        strings.TrimSpace(os.Getenv("AGENTFIELD_URL")),
		Token:                os.Getenv("AGENTFIELD_TOKEN"),
		DeploymentType:       "serverless",
		ListenAddress:        ":" + defaultString(os.Getenv("PORT"), "8085"),
		LeaseRefreshInterval: 0, // no lease loop in serverless mode
		DisableLeaseLoop:     true,
	}

	srv, err := agent.New(cfg)
	if err != nil {
		log.Fatal(err)
	}

	srv.RegisterReasoner("hello", func(ctx context.Context, input map[string]any) (any, error) {
		name := strings.TrimSpace(defaultString(toString(input["name"]), "AgentField"))
		return map[string]any{
			"greeting": "Hello, " + name + "!",
			"name":     name,
		}, nil
	})

	srv.RegisterReasoner("relay", func(ctx context.Context, input map[string]any) (any, error) {
		target := strings.TrimSpace(defaultString(toString(input["target"]), os.Getenv("CHILD_TARGET")))
		if target == "" {
			return map[string]any{"error": "target is required"}, nil
		}
		message := strings.TrimSpace(defaultString(toString(input["message"]), "ping"))

		res, err := srv.Call(ctx, target, map[string]any{"message": message})
		if err != nil {
			return nil, err
		}
		return map[string]any{
			"target":     target,
			"downstream": res,
		}, nil
	})

	// Optional local server for manual testing
	addr := cfg.ListenAddress
	log.Printf("serverless handler listening on %s", addr)
	if err := http.ListenAndServe(addr, srv.Handler()); err != nil {
		log.Fatal(err)
	}

	// For platforms that invoke a function instead of an HTTP listener, you can adapt
	// the event and delegate to the agent:
	//
	// func LambdaHandler(ctx context.Context, event map[string]any) (map[string]any, error) {
	//     result, status, err := srv.HandleServerlessEvent(ctx, event, nil) // adapter optional
	//     if err != nil {
	//         return map[string]any{"statusCode": 500, "body": map[string]any{"error": err.Error()}}, nil
	//     }
	//     return map[string]any{"statusCode": status, "body": result}, nil
	// }
}

func defaultString(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return strings.TrimSpace(value)
}

func toString(value any) string {
	if value == nil {
		return ""
	}
	if s, ok := value.(string); ok {
		return s
	}
	return strings.TrimSpace(strings.ReplaceAll(strings.TrimSpace(strings.Trim(fmt.Sprintf("%v", value), "\n")), "\t", " "))
}
