package capabilities

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/integrations/linear/node/internal/config"
	"github.com/Agent-Field/agentfield/integrations/linear/node/internal/linear"
)

func TestRuntimeCapabilitiesUseLinearGraphQL(t *testing.T) {
	var calls []string
	var gotAuth string
	var gotCommentInput map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		var req struct {
			Query     string         `json:"query"`
			Variables map[string]any `json:"variables"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")

		switch {
		case strings.Contains(req.Query, "AgentFieldLinearHealth"):
			calls = append(calls, "health")
			_, _ = w.Write([]byte(`{"data":{"viewer":{"id":"usr-1","name":"AgentField"}}}`))
		case strings.Contains(req.Query, "AgentFieldGetIssue"):
			calls = append(calls, "get_issue")
			if req.Variables["id"] != "AF-1" {
				t.Fatalf("issue id variables=%v", req.Variables)
			}
			_, _ = w.Write([]byte(`{"data":{"issue":{"id":"issue-1","identifier":"AF-1"}}}`))
		case strings.Contains(req.Query, "AgentFieldCommentIssue"):
			calls = append(calls, "comment_issue")
			input, _ := req.Variables["input"].(map[string]any)
			gotCommentInput = input
			_, _ = w.Write([]byte(`{"data":{"commentCreate":{"success":true,"comment":{"id":"comment-1"}}}}`))
		default:
			t.Fatalf("unexpected query: %s", req.Query)
		}
	}))
	defer server.Close()

	client, err := linear.NewClient(linear.Config{
		APIURL:     server.URL,
		Token:      "lin_mock_token",
		HTTPClient: server.Client(),
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	rt := Runtime{Config: config.Config{NodeID: "linear-prod"}, Linear: client}

	health, err := rt.health(context.Background(), nil)
	if err != nil {
		t.Fatalf("health: %v", err)
	}
	if health.(map[string]any)["status"] != "ok" {
		t.Fatalf("health=%v", health)
	}
	if _, err := rt.getIssue(context.Background(), map[string]any{"id": "AF-1"}); err != nil {
		t.Fatalf("getIssue: %v", err)
	}
	if _, err := rt.commentIssue(context.Background(), map[string]any{"issue_id": "issue-1", "body": "confirmed"}); err != nil {
		t.Fatalf("commentIssue: %v", err)
	}
	if gotAuth != "lin_mock_token" {
		t.Fatalf("auth=%q", gotAuth)
	}
	if gotCommentInput["issueId"] != "issue-1" || gotCommentInput["body"] != "confirmed" {
		t.Fatalf("comment input=%v", gotCommentInput)
	}
	if strings.Join(calls, ",") != "health,get_issue,comment_issue" {
		t.Fatalf("calls=%v", calls)
	}
}

func TestRuntimeRejectsMissingLinearCapabilityInput(t *testing.T) {
	rt := Runtime{}
	if _, err := rt.getIssue(context.Background(), map[string]any{}); err == nil {
		t.Fatal("expected get_issue to require id")
	}
	if _, err := rt.commentIssue(context.Background(), map[string]any{"issue_id": "issue-1"}); err == nil {
		t.Fatal("expected comment_issue to require body")
	}
}
