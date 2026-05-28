// Permission Agent A (Caller) — Go SDK
//
// An agent with tag "analytics" that demonstrates the policy engine:
//   - call_data_service   -> calls go-perm-target.query_data (ALLOWED by policy)
//   - call_large_query    -> calls go-perm-target.query_data with limit=5000 (DENIED: constraint)
//   - call_delete_records -> calls go-perm-target.delete_records (DENIED: deny_functions)
//
// The "analytics" tag auto-approves (tag_approval_rules), so this agent starts
// immediately in "active" state.
//
// Test flow:
//  1. Start control plane with authorization enabled
//  2. Start go-perm-target -> enters pending_approval
//  3. Admin approves go-perm-target's tags
//  4. Start go-perm-caller (this agent) -> auto-approved
//  5. POST /api/v1/execute/go-perm-caller.call_data_service -> 200 OK
//  6. POST /api/v1/execute/go-perm-caller.call_large_query -> 403 constraint
//  7. POST /api/v1/execute/go-perm-caller.call_delete_records -> 403 denied function
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
		listenAddr = ":8003"
	}

	publicURL := strings.TrimSpace(os.Getenv("AGENT_PUBLIC_URL"))
	if publicURL == "" {
		publicURL = "http://localhost" + listenAddr
	}

	cfg := agent.Config{
		NodeID:            "go-perm-caller",
		Version:           "1.0.0",
		AgentFieldURL:     agentFieldURL,
		Token:             os.Getenv("AGENTFIELD_TOKEN"),
		InternalToken:     os.Getenv("AGENTFIELD_INTERNAL_TOKEN"),
		ListenAddress:     listenAddr,
		PublicURL:         publicURL,
		EnableDID:         true,  // Auto-register DID during Initialize()
		VCEnabled:         true,  // Generate VCs for audit trail
		Tags:              []string{"analytics"},
		LocalVerification: true, // Verify DID signatures locally
		RequireOriginAuth: true,
	}

	a, err := agent.New(cfg)
	if err != nil {
		log.Fatal(err)
	}

	// Simple health check — no cross-agent call, should always work.
	a.RegisterReasoner("ping", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{
			"status": "ok",
			"agent":  "go-perm-caller",
		}, nil
	},
		agent.WithDescription("Simple health check"),
		agent.WithVCEnabled(false), // No VC needed for health checks
	)

	// Calls go-perm-target.query_data with a small limit.
	// Should succeed: analytics -> data-service, query_* is in allow_functions,
	// limit=100 satisfies the <= 1000 constraint.
	a.RegisterReasoner("call_data_service", func(ctx context.Context, input map[string]any) (any, error) {
		query := fmt.Sprintf("%v", input["query"])
		if query == "" || query == "<nil>" {
			query = "SELECT * FROM data"
		}

		result, err := a.Call(ctx, "go-perm-target.query_data", map[string]any{
			"query": query,
			"limit": 100,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to call go-perm-target.query_data: %w", err)
		}

		return map[string]any{
			"source":            "go-perm-caller",
			"test":              "allowed_query",
			"delegation_result": result,
		}, nil
	},
		agent.WithDescription("Calls go-perm-target.query_data (allowed)"),
		agent.WithReasonerTags("analytics"),
	)

	// Calls go-perm-target.query_data with limit=5000.
	// Should fail: limit=5000 violates the <= 1000 constraint.
	a.RegisterReasoner("call_large_query", func(ctx context.Context, input map[string]any) (any, error) {
		query := fmt.Sprintf("%v", input["query"])
		if query == "" || query == "<nil>" {
			query = "SELECT * FROM big_table"
		}

		result, err := a.Call(ctx, "go-perm-target.query_data", map[string]any{
			"query": query,
			"limit": 5000,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to call go-perm-target.query_data: %w", err)
		}

		return map[string]any{
			"source":            "go-perm-caller",
			"test":              "constraint_violation",
			"delegation_result": result,
		}, nil
	},
		agent.WithDescription("Calls go-perm-target.query_data with large limit (constraint violation)"),
		agent.WithReasonerTags("analytics"),
	)

	// Calls go-perm-target.delete_records.
	// Should fail: delete_* is in deny_functions for analytics->data-service.
	a.RegisterReasoner("call_delete_records", func(ctx context.Context, input map[string]any) (any, error) {
		table := fmt.Sprintf("%v", input["table"])
		if table == "" || table == "<nil>" {
			table = "sensitive_records"
		}

		result, err := a.Call(ctx, "go-perm-target.delete_records", map[string]any{
			"table": table,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to call go-perm-target.delete_records: %w", err)
		}

		return map[string]any{
			"source":            "go-perm-caller",
			"test":              "deny_function",
			"delegation_result": result,
		}, nil
	},
		agent.WithDescription("Calls go-perm-target.delete_records (denied by policy)"),
		agent.WithReasonerTags("analytics"),
	)

	fmt.Println("Permission Agent A (Caller) — Go SDK")
	fmt.Println("Node: go-perm-caller")
	fmt.Printf("Server: %s\n", agentFieldURL)
	fmt.Println("Tags: analytics")
	fmt.Println("Test reasoners: call_data_service (allow), call_large_query (constraint), call_delete_records (deny)")

	if err := a.Run(context.Background()); err != nil {
		log.Fatal(err)
	}
}
