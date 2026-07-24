// Permission Agent B (Protected Target) — Go SDK
//
// A protected agent with tags ["sensitive", "data-service"]. The "sensitive"
// tag triggers manual approval (tag_approval_rules in config), so this agent
// starts in "pending_approval" state until an admin approves its tags.
//
// Once approved, access policies control which callers can invoke which reasoners:
//   - analytics callers can call query_data and get_schema (allowed by policy)
//   - analytics callers are denied delete_records (deny_functions in policy)
//   - constraint violations (e.g. limit > 1000) are rejected
//
// Reasoners:
//   - query_data(query, limit)  — simulates a data query (allowed for analytics)
//   - delete_records(table)     — simulates record deletion (denied for analytics)
//   - get_schema                — returns the data schema
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
)

func main() {
	agentFieldURL := strings.TrimSpace(os.Getenv("AGENTFIELD_URL"))
	if agentFieldURL == "" {
		agentFieldURL = "http://localhost:8080"
	}

	listenAddr := strings.TrimSpace(os.Getenv("AGENT_LISTEN_ADDR"))
	if listenAddr == "" {
		listenAddr = ":8004"
	}

	publicURL := strings.TrimSpace(os.Getenv("AGENT_PUBLIC_URL"))
	if publicURL == "" {
		publicURL = "http://localhost" + listenAddr
	}

	cfg := agent.Config{
		NodeID:            "go-perm-target",
		Version:           "1.0.0",
		AgentFieldURL:     agentFieldURL,
		Token:             os.Getenv("AGENTFIELD_TOKEN"),
		InternalToken:     os.Getenv("AGENTFIELD_INTERNAL_TOKEN"),
		ListenAddress:     listenAddr,
		PublicURL:         publicURL,
		EnableDID:         true, // Auto-register DID during Initialize()
		VCEnabled:         true, // Generate VCs for audit trail
		Tags:              []string{"sensitive", "data-service"},
		RequireOriginAuth: true, // Only the control plane can invoke reasoners
	}

	b, err := agent.New(cfg)
	if err != nil {
		log.Fatal(err)
	}

	// Reasoner 1: query_data — simulates a data query.
	// Allowed for analytics callers by the access policy (query_* in allow_functions).
	// The "limit" parameter is constrained to <= 1000 by the policy.
	b.RegisterReasoner("query_data", func(ctx context.Context, input map[string]any) (any, error) {
		query := fmt.Sprintf("%v", input["query"])
		if query == "" || query == "<nil>" {
			query = "SELECT *"
		}

		limit := 100
		if l, ok := input["limit"]; ok {
			if lf, ok := l.(float64); ok {
				limit = int(lf)
			}
		}

		return map[string]any{
			"status":  "success",
			"agent":   "go-perm-target",
			"query":   query,
			"limit":   limit,
			"results": []map[string]any{{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}},
			"message": fmt.Sprintf("Query executed: %s (limit=%d)", query, limit),
		}, nil
	},
		agent.WithDescription("Execute a data query. Protected by access policy."),
		agent.WithReasonerTags("sensitive", "data-service"),
	)

	// Reasoner 2: delete_records — simulates record deletion.
	// Denied for analytics callers by the access policy (delete_* in deny_functions).
	b.RegisterReasoner("delete_records", func(ctx context.Context, input map[string]any) (any, error) {
		table := fmt.Sprintf("%v", input["table"])
		if table == "" || table == "<nil>" {
			table = "records"
		}

		return map[string]any{
			"status":  "deleted",
			"agent":   "go-perm-target",
			"table":   table,
			"message": fmt.Sprintf("Records deleted from %s", table),
		}, nil
	},
		agent.WithDescription("Delete records. Denied for analytics callers by policy."),
		agent.WithReasonerTags("data-service"),
	)

	// Reasoner 3: get_schema — returns the data schema.
	b.RegisterReasoner("get_schema", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{
			"status": "success",
			"agent":  "go-perm-target",
			"schema": map[string]any{
				"table": "records",
				"columns": []map[string]any{
					{"name": "id", "type": "integer", "primary_key": true},
					{"name": "name", "type": "text"},
					{"name": "created_at", "type": "timestamp"},
				},
			},
		}, nil
	},
		agent.WithDescription("Get the data schema."),
		agent.WithReasonerTags("data-service"),
	)

	fmt.Println("Permission Agent B (Protected) — Go SDK")
	fmt.Println("Node: go-perm-target")
	fmt.Printf("Server: %s\n", agentFieldURL)
	fmt.Println("Tags: sensitive, data-service")
	fmt.Println("Reasoners: query_data, delete_records, get_schema")

	if err := b.Run(context.Background()); err != nil {
		log.Fatal(err)
	}
}
