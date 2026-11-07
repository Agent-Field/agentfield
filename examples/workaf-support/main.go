// workaf-support is a customer-support agent node.
//
// A ticket arrives on a webhook (generic_bearer trigger), and the node works it
// the way a support engineer would:
//
//	handle_ticket            the trigger's reasoner: runs the steps below in order
//	  triage                 plain code: what kind of ticket, which repository
//	  investigate            a coding agent reads the repository and finds the cause
//	    search_docs          plain code: greps the README and docs for the ticket's words
//	  draft_reply            a coding agent writes the customer reply and a suggested fix
//
// Every step is an ordinary reasoner called through the control plane, so each
// one is its own execution with its own parent link. Nothing here knows who is
// watching the run.
//
// Configuration is environment only:
//
//	AGENTFIELD_URL        control plane (default http://localhost:8080)
//	AGENTFIELD_API_KEY    control plane API key, or AGENTFIELD_API_KEY_FILE
//	AGENT_NODE_ID         node id (default support)
//	AGENT_LISTEN          listen address (default 127.0.0.1:8002)
//	AGENT_CALLBACK_URL    URL the control plane dispatches to (default http://<AGENT_LISTEN>)
//	SUPPORT_MODEL         harness model (default deepseek/deepseek-v4.1-flash)
//	CODEAF_BIN            codeaf binary for the harness (default: codeaf on PATH)
//	OPENROUTER_API_KEY    model key, or OPENROUTER_API_KEY_FILE
//	GITHUB_TOKEN          token that can clone the repositories, or GITHUB_TOKEN_FILE
//	SUPPORT_WORKDIR       where repositories are cloned (default: a temp dir)
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
	"github.com/Agent-Field/agentfield/sdk/go/triggers"
)

func main() {
	nodeID := envOr("AGENT_NODE_ID", "support")
	listen := envOr("AGENT_LISTEN", "127.0.0.1:8002")

	app, err := agent.New(agent.Config{
		NodeID:        nodeID,
		Version:       "0.1.0",
		AgentFieldURL: envOr("AGENTFIELD_URL", "http://localhost:8080"),
		APIKey:        secret("AGENTFIELD_API_KEY"),
		ListenAddress: listen,
		PublicURL:     envOr("AGENT_CALLBACK_URL", "http://"+listen),
		HarnessConfig: &agent.HarnessConfig{
			Provider: "codeaf",
			Model:    envOr("SUPPORT_MODEL", "deepseek/deepseek-v4.1-flash"),
			BinPath:  os.Getenv("CODEAF_BIN"),
			MaxTurns: 30,
			Timeout:  900,
			Env:      map[string]string{"OPENROUTER_API_KEY": secret("OPENROUTER_API_KEY")},
		},
	})
	if err != nil {
		log.Fatalf("workaf-support: %v", err)
	}

	s := &support{app: app, repos: newRepoCache(envOr("SUPPORT_WORKDIR", ""), secret("GITHUB_TOKEN"))}

	app.OnEvent(triggers.EventOpts{
		Source:    "generic_bearer",
		SecretEnv: "SUPPORT_INTAKE_BEARER",
		Config:    []byte(`{"idempotency_header":"X-Ticket-Id"}`),
	}, "handle_ticket", s.handleTicket)

	app.RegisterReasoner("triage", s.triage, agent.WithDescription("Classify a support ticket and pick the repository it is about."))
	app.RegisterReasoner("investigate", s.investigate, agent.WithDescription("Find the cause of a ticket in the repository."))
	app.RegisterReasoner("search_docs", s.searchDocs, agent.WithDescription("Grep a repository's README and docs for a ticket's words."))
	app.RegisterReasoner("draft_reply", s.draftReply, agent.WithDescription("Write the customer reply and a suggested fix."))

	if err := app.Run(context.Background()); err != nil {
		log.Fatalf("workaf-support: %v", err)
	}
}

type support struct {
	app   *agent.Agent
	repos *repoCache
}

// handleTicket is the trigger's reasoner. It accepts {"ticket": "..."} or
// {"subject": "...", "body": "..."} and returns the reply the customer gets.
func (s *support) handleTicket(ctx context.Context, in map[string]any) (any, error) {
	ticket := ticketText(in)
	if ticket == "" {
		return nil, fmt.Errorf("the event carried no ticket text")
	}

	tri, err := s.app.Call(ctx, "triage", map[string]any{"ticket": ticket})
	if err != nil {
		return nil, fmt.Errorf("triage: %w", err)
	}
	if str(tri, "repo") == "" {
		return map[string]any{"triage": tri, "reply": "Thanks for writing in. A person will pick this up, because it is not about a product we can look into automatically."}, nil
	}

	inv, err := s.app.Call(ctx, "investigate", map[string]any{"ticket": ticket, "repo": tri["repo"], "kind": tri["kind"]})
	if err != nil {
		return nil, fmt.Errorf("investigate: %w", err)
	}

	reply, err := s.app.Call(ctx, "draft_reply", map[string]any{"ticket": ticket, "repo": tri["repo"], "findings": inv})
	if err != nil {
		return nil, fmt.Errorf("draft_reply: %w", err)
	}

	return map[string]any{
		"ticket_id":     ticketID(ctx, in),
		"kind":          tri["kind"],
		"repo":          tri["repo"],
		"root_cause":    inv["root_cause"],
		"reply":         reply["reply"],
		"suggested_fix": reply["suggested_fix"],
		"diff":          reply["diff"],
	}, nil
}

func ticketText(in map[string]any) string {
	if t := strings.TrimSpace(str(in, "ticket")); t != "" {
		return t
	}
	return strings.TrimSpace(strings.TrimSpace(str(in, "subject")) + "\n\n" + strings.TrimSpace(str(in, "body")))
}

func ticketID(ctx context.Context, in map[string]any) string {
	if id := str(in, "id"); id != "" {
		return id
	}
	if tc := triggers.FromContext(ctx); tc != nil {
		if tc.IdempotencyKey != "" {
			return tc.IdempotencyKey
		}
		return tc.EventID
	}
	return ""
}

func str(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func envOr(key, fallback string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return fallback
}

// secret reads KEY, or the file named by KEY_FILE.
func secret(key string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	if path := os.Getenv(key + "_FILE"); path != "" {
		raw, err := os.ReadFile(path)
		if err != nil {
			log.Fatalf("workaf-support: %s_FILE: %v", key, err)
		}
		return strings.TrimSpace(string(raw))
	}
	return ""
}
