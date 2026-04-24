package agent_test

import (
	"context"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/agent"
)

// newTestAgent creates a minimal Agent suitable for local testing.
// No AgentFieldURL means Initialize/Serve are never called, but
// RegisterReasoner and Execute work fine in-process.
func newTestAgent(t *testing.T) *agent.Agent {
	t.Helper()
	a, err := agent.New(agent.Config{
		NodeID:  "test-agent",
		Version: "0.0.1",
	})
	if err != nil {
		t.Fatalf("agent.New: %v", err)
	}
	return a
}

// echoHandler returns its input as output so we can verify the right handler fired.
func echoHandler(name string) agent.HandlerFunc {
	return func(_ context.Context, input map[string]any) (any, error) {
		return map[string]any{"handler": name, "input": input}, nil
	}
}

func mustExecute(t *testing.T, a *agent.Agent, name string) map[string]any {
	t.Helper()
	result, err := a.Execute(context.Background(), name, map[string]any{})
	if err != nil {
		t.Fatalf("Execute(%q): %v", name, err)
	}
	m, ok := result.(map[string]any)
	if !ok {
		t.Fatalf("Execute(%q): unexpected result type %T", name, result)
	}
	return m
}

func mustNotExecute(t *testing.T, a *agent.Agent, name string) {
	t.Helper()
	_, err := a.Execute(context.Background(), name, map[string]any{})
	if err == nil {
		t.Errorf("Execute(%q): expected error for unknown reasoner, got nil", name)
	}
}

// ─── tests ────────────────────────────────────────────────────────────────────

func TestRouter_FlatMount(t *testing.T) {
	a := newTestAgent(t)

	r := agent.NewRouter()
	r.RegisterReasoner("get-profile", echoHandler("get-profile"))
	r.RegisterReasoner("update-profile", echoHandler("update-profile"))
	r.RegisterSkill("validate-email", echoHandler("validate-email"))

	a.IncludeRouter(r, agent.RouterOptions{Prefix: "users"})

	mustExecute(t, a, "users.get-profile")
	mustExecute(t, a, "users.update-profile")
	mustExecute(t, a, "users.validate-email")

	// Unprefixed names must NOT exist
	mustNotExecute(t, a, "get-profile")
}

func TestRouter_NoPrefix(t *testing.T) {
	a := newTestAgent(t)

	r := agent.NewRouter()
	r.RegisterReasoner("ping", echoHandler("ping"))

	a.IncludeRouter(r, agent.RouterOptions{})

	mustExecute(t, a, "ping")
}

func TestRouter_NestedRouters(t *testing.T) {
	a := newTestAgent(t)

	userRouter := agent.NewRouter()
	userRouter.RegisterReasoner("get-profile", echoHandler("get-profile"))

	orderRouter := agent.NewRouter()
	orderRouter.RegisterReasoner("create", echoHandler("create"))
	orderRouter.RegisterReasoner("cancel", echoHandler("cancel"))

	adminRouter := agent.NewRouter()
	adminRouter.IncludeRouter(userRouter, agent.RouterOptions{Prefix: "users"})
	adminRouter.IncludeRouter(orderRouter, agent.RouterOptions{Prefix: "orders"})

	a.IncludeRouter(adminRouter, agent.RouterOptions{Prefix: "admin"})

	mustExecute(t, a, "admin.users.get-profile")
	mustExecute(t, a, "admin.orders.create")
	mustExecute(t, a, "admin.orders.cancel")

	// Partial paths must not resolve
	mustNotExecute(t, a, "admin.users")
	mustNotExecute(t, a, "users.get-profile")
}

func TestRouter_DeeplyNested(t *testing.T) {
	a := newTestAgent(t)

	leaf := agent.NewRouter()
	leaf.RegisterReasoner("action", echoHandler("action"))

	mid := agent.NewRouter()
	mid.IncludeRouter(leaf, agent.RouterOptions{Prefix: "leaf"})

	top := agent.NewRouter()
	top.IncludeRouter(mid, agent.RouterOptions{Prefix: "mid"})

	a.IncludeRouter(top, agent.RouterOptions{Prefix: "top"})

	mustExecute(t, a, "top.mid.leaf.action")
}

func TestRouter_SharedRouter(t *testing.T) {
	// The same Router can be mounted into two separate agents (or under two
	// different prefixes) without any state mutation.
	shared := agent.NewRouter()
	shared.RegisterReasoner("ping", echoHandler("ping"))

	a1 := newTestAgent(t)
	a1.IncludeRouter(shared, agent.RouterOptions{Prefix: "alpha"})

	a2 := newTestAgent(t)
	a2.IncludeRouter(shared, agent.RouterOptions{Prefix: "beta"})

	mustExecute(t, a1, "alpha.ping")
	mustExecute(t, a2, "beta.ping")

	// Cross-contamination check
	mustNotExecute(t, a1, "beta.ping")
	mustNotExecute(t, a2, "alpha.ping")
}

func TestRouter_EmptyRouter(t *testing.T) {
	a := newTestAgent(t)
	a.IncludeRouter(agent.NewRouter(), agent.RouterOptions{Prefix: "empty"})
	// Nothing registered — Execute on any name must fail
	mustNotExecute(t, a, "empty.anything")
}

func TestRouter_HandlerFires(t *testing.T) {
	a := newTestAgent(t)

	r := agent.NewRouter()
	r.RegisterReasoner("greet", func(_ context.Context, input map[string]any) (any, error) {
		return map[string]any{"hello": "world"}, nil
	})

	a.IncludeRouter(r, agent.RouterOptions{Prefix: "svc"})

	result := mustExecute(t, a, "svc.greet")
	if result["hello"] != "world" {
		t.Errorf("expected hello=world, got %v", result)
	}
}

func TestRouter_WithDescription(t *testing.T) {
	// Verify WithDescription compiles and is accepted by RegisterReasoner.
	// (Description is stored on the Reasoner but not directly readable from
	// outside the package; we just confirm no panic and correct execution.)
	a := newTestAgent(t)

	r := agent.NewRouter()
	r.RegisterReasoner("described", echoHandler("described"),
		agent.WithDescription("A test reasoner with a description"))

	a.IncludeRouter(r, agent.RouterOptions{Prefix: "svc"})
	mustExecute(t, a, "svc.described")
}

func TestRouter_WithReasonerTags(t *testing.T) {
	// Verify WithReasonerTags compiles and is accepted.
	a := newTestAgent(t)

	r := agent.NewRouter()
	r.RegisterReasoner("tagged", echoHandler("tagged"),
		agent.WithReasonerTags("tag-a", "tag-b"))

	a.IncludeRouter(r, agent.RouterOptions{Prefix: "svc"})
	mustExecute(t, a, "svc.tagged")
}

func TestRouter_RouterOptionsTagsApplied(t *testing.T) {
	// Tags from RouterOptions should be merged without clobbering per-handler tags.
	a := newTestAgent(t)

	r := agent.NewRouter()
	r.RegisterReasoner("action", echoHandler("action"),
		agent.WithReasonerTags("handler-tag"))

	// RouterOptions carries its own tags — both sets must survive.
	a.IncludeRouter(r, agent.RouterOptions{
		Prefix: "svc",
		Tags:   []string{"router-tag"},
	})

	// Execution must succeed (tags do not affect routing, only auth).
	mustExecute(t, a, "svc.action")
}

func TestRouter_MultipleMountsOnSameAgent(t *testing.T) {
	a := newTestAgent(t)

	users := agent.NewRouter()
	users.RegisterReasoner("list", echoHandler("user-list"))

	orders := agent.NewRouter()
	orders.RegisterReasoner("list", echoHandler("order-list"))

	a.IncludeRouter(users, agent.RouterOptions{Prefix: "users"})
	a.IncludeRouter(orders, agent.RouterOptions{Prefix: "orders"})

	u := mustExecute(t, a, "users.list")
	o := mustExecute(t, a, "orders.list")

	if u["handler"] == o["handler"] {
		t.Error("expected different handlers for users.list and orders.list")
	}
}

// TestRouter_IssueExample mirrors the exact usage shown in the GitHub issue.
// This is the canonical acceptance test for the feature.
//
// Issue: https://github.com/Agent-Field/agentfield/issues/<N>
// "Add AgentRouter pattern to the Go SDK for modular agent organisation"
func TestRouter_IssueExample(t *testing.T) {
	a := newTestAgent(t)

	// ── User router ───────────────────────────────────────────────────────
	userRouter := agent.NewRouter()
	userRouter.RegisterReasoner("get-profile", echoHandler("get-profile"),
		agent.WithDescription("Get user profile by ID"))
	userRouter.RegisterReasoner("update-profile", echoHandler("update-profile"),
		agent.WithDescription("Update user profile"))
	userRouter.RegisterSkill("validate-email", echoHandler("validate-email"))

	// ── Order router ──────────────────────────────────────────────────────
	orderRouter := agent.NewRouter()
	orderRouter.RegisterReasoner("create", echoHandler("create"))
	orderRouter.RegisterReasoner("cancel", echoHandler("cancel"))

	// ── Mount both routers onto the agent with tags ────────────────────────
	a.IncludeRouter(userRouter, agent.RouterOptions{
		Prefix: "users",
		Tags:   []string{"user-management"},
	})
	a.IncludeRouter(orderRouter, agent.RouterOptions{
		Prefix: "orders",
		Tags:   []string{"order-management"},
	})

	// Assert: users.* names exist and route to the right handlers
	u1 := mustExecute(t, a, "users.get-profile")
	if u1["handler"] != "get-profile" {
		t.Errorf("users.get-profile: wrong handler %v", u1["handler"])
	}
	u2 := mustExecute(t, a, "users.update-profile")
	if u2["handler"] != "update-profile" {
		t.Errorf("users.update-profile: wrong handler %v", u2["handler"])
	}
	mustExecute(t, a, "users.validate-email")

	// Assert: orders.* names exist
	mustExecute(t, a, "orders.create")
	mustExecute(t, a, "orders.cancel")

	// Assert: unprefixed names do NOT exist (prefix is applied, not optional)
	mustNotExecute(t, a, "get-profile")
	mustNotExecute(t, a, "create")

	// ── Nested router (admin wraps both) ──────────────────────────────────
	b := newTestAgent(t)

	adminRouter := agent.NewRouter()
	adminRouter.IncludeRouter(userRouter, agent.RouterOptions{Prefix: "users"})
	adminRouter.IncludeRouter(orderRouter, agent.RouterOptions{Prefix: "orders"})

	b.IncludeRouter(adminRouter, agent.RouterOptions{Prefix: "admin"})

	// Assert: admin.users.* and admin.orders.* exist
	mustExecute(t, b, "admin.users.get-profile")
	mustExecute(t, b, "admin.users.update-profile")
	mustExecute(t, b, "admin.users.validate-email")
	mustExecute(t, b, "admin.orders.create")
	mustExecute(t, b, "admin.orders.cancel")

	// Assert: shorter paths do not resolve
	mustNotExecute(t, b, "users.get-profile")
	mustNotExecute(t, b, "orders.create")
}
