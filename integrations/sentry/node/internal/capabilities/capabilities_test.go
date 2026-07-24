package capabilities

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/integrations/sentry/node/internal/config"
	"github.com/Agent-Field/agentfield/integrations/sentry/node/internal/sentry"
)

func TestRuntimeCapabilitiesUseSentryAPI(t *testing.T) {
	var calls []string
	var gotAuth string
	var gotAssignBody map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")

		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/api/0/organizations/agentfield/projects/":
			calls = append(calls, "health")
			_, _ = w.Write([]byte(`[{"slug":"web"}]`))
		case r.Method == http.MethodGet && r.URL.Path == "/api/0/projects/agentfield/web/issues/":
			calls = append(calls, "list_issues")
			if r.URL.Query().Get("query") != "is:unresolved" {
				t.Fatalf("query=%q", r.URL.RawQuery)
			}
			_, _ = w.Write([]byte(`[{"id":"issue-1"}]`))
		case r.Method == http.MethodPut && r.URL.Path == "/api/0/organizations/agentfield/issues/issue-1/":
			calls = append(calls, "assign_issue")
			if err := json.NewDecoder(r.Body).Decode(&gotAssignBody); err != nil {
				t.Fatalf("decode assign body: %v", err)
			}
			_, _ = w.Write([]byte(`{"id":"issue-1","assignedTo":"user:alice@example.com"}`))
		default:
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.String())
		}
	}))
	defer server.Close()

	client, err := sentry.NewClient(sentry.Config{
		BaseURL:      server.URL,
		Organization: "agentfield",
		Token:        "sntrys_mock_token",
		HTTPClient:   server.Client(),
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	rt := Runtime{Config: config.Config{NodeID: "sentry-prod"}, Sentry: client}

	health, err := rt.health(context.Background(), nil)
	if err != nil {
		t.Fatalf("health: %v", err)
	}
	if health.(map[string]any)["status"] != "ok" {
		t.Fatalf("health=%v", health)
	}
	if _, err := rt.listIssues(context.Background(), map[string]any{"project": "web", "query": "is:unresolved", "limit": 10}); err != nil {
		t.Fatalf("listIssues: %v", err)
	}
	if _, err := rt.assignIssue(context.Background(), map[string]any{"issue_id": "issue-1", "assignee": "user:alice@example.com"}); err != nil {
		t.Fatalf("assignIssue: %v", err)
	}
	if gotAuth != "Bearer sntrys_mock_token" {
		t.Fatalf("auth=%q", gotAuth)
	}
	if gotAssignBody["assignedTo"] != "user:alice@example.com" {
		t.Fatalf("assign body=%v", gotAssignBody)
	}
	if strings.Join(calls, ",") != "health,list_issues,assign_issue" {
		t.Fatalf("calls=%v", calls)
	}
}

func TestRuntimeRejectsMissingSentryCapabilityInput(t *testing.T) {
	rt := Runtime{}
	if _, err := rt.listIssues(context.Background(), map[string]any{}); err == nil {
		t.Fatal("expected list_issues to require project")
	}
	if _, err := rt.assignIssue(context.Background(), map[string]any{"issue_id": "issue-1"}); err == nil {
		t.Fatal("expected assign_issue to require assignee")
	}
}
