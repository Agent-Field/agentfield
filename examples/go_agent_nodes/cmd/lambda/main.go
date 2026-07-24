package main

import (
	"context"
	"log"
	"os"
	"strings"

	"github.com/aws/aws-lambda-go/lambda"
	"github.com/awslabs/aws-lambda-go-api-proxy/httpadapter"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
)

// This is the same agent.New(cfg) + srv.Handler() shape as cmd/serverless,
// just with lambda.Start(...) in place of http.ListenAndServe(...). No SDK
// changes are needed to run a serverless node on AWS Lambda: httpadapter
// wraps the standard http.Handler and translates Lambda Function URL /
// API Gateway HTTP API (payload format 2.0) events into it.
func main() {
	nodeID := strings.TrimSpace(os.Getenv("AGENT_NODE_ID"))
	if nodeID == "" {
		nodeID = "go-lambda-hello"
	}

	cfg := agent.Config{
		NodeID:               nodeID,
		Version:              "1.0.0",
		AgentFieldURL:        strings.TrimSpace(os.Getenv("AGENTFIELD_URL")),
		Token:                os.Getenv("AGENTFIELD_TOKEN"),
		DeploymentType:       "serverless",
		LeaseRefreshInterval: 0,
		DisableLeaseLoop:     true,
	}

	if token := strings.TrimSpace(os.Getenv("AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN")); token != "" {
		cfg.InternalToken = token
		cfg.RequireOriginAuth = true
	} else {
		log.Printf("warning: AGENTFIELD_AUTHORIZATION_INTERNAL_TOKEN not set - /execute is unauthenticated")
	}

	srv, err := agent.New(cfg)
	if err != nil {
		log.Fatal(err)
	}

	srv.RegisterReasoner("hello", func(ctx context.Context, input map[string]any) (any, error) {
		exec := agent.ExecutionContextFrom(ctx)
		name := strings.TrimSpace(toString(input["name"]))
		if name == "" {
			name = "AgentField"
		}
		return map[string]any{
			"greeting":            "Hello, " + name + "!",
			"name":                name,
			"execution_id":        exec.ExecutionID,
			"parent_execution_id": exec.ParentExecutionID,
		}, nil
	})

	srv.RegisterReasoner("relay", func(ctx context.Context, input map[string]any) (any, error) {
		exec := agent.ExecutionContextFrom(ctx)
		target := strings.TrimSpace(toString(input["target"]))
		if target == "" {
			target = strings.TrimSpace(os.Getenv("CHILD_TARGET"))
		}
		if target == "" {
			return map[string]any{"error": "target is required"}, nil
		}
		message := strings.TrimSpace(toString(input["message"]))
		if message == "" {
			message = "ping"
		}

		res, err := srv.Call(ctx, target, map[string]any{"name": message})
		if err != nil {
			return nil, err
		}
		return map[string]any{
			"target":              target,
			"downstream":          res,
			"execution_id":        exec.ExecutionID,
			"parent_execution_id": exec.ParentExecutionID,
		}, nil
	})

	adapter := httpadapter.NewV2(srv.Handler())
	lambda.Start(adapter.ProxyWithContext)
}

func toString(value any) string {
	if s, ok := value.(string); ok {
		return s
	}
	return ""
}
