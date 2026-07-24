package linear

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGraphQLClientSendsAuthAndVariables(t *testing.T) {
	var gotAuth string
	var gotQuery string
	var gotVariables map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		var req struct {
			Query     string         `json:"query"`
			Variables map[string]any `json:"variables"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		gotQuery = req.Query
		gotVariables = req.Variables
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":{"issue":{"id":"issue-1","identifier":"AF-1"}}}`))
	}))
	defer server.Close()

	client, err := NewClient(Config{APIURL: server.URL, Token: "lin_api_key", HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	out, err := client.GetIssue(context.Background(), "AF-1")
	if err != nil {
		t.Fatalf("GetIssue: %v", err)
	}
	if gotAuth != "lin_api_key" {
		t.Fatalf("auth=%q", gotAuth)
	}
	if !strings.Contains(gotQuery, "issue(id: $id)") {
		t.Fatalf("unexpected query: %s", gotQuery)
	}
	if gotVariables["id"] != "AF-1" || out["data"] == nil {
		t.Fatalf("bad variables/out: %v %v", gotVariables, out)
	}
}

func TestCreateIssueRequiresInput(t *testing.T) {
	client, err := NewClient(Config{APIURL: "http://linear.test/graphql", Token: "lin"})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if _, err := client.CreateIssue(context.Background(), nil); err == nil {
		t.Fatal("expected input error")
	}
}

func TestGraphQLErrorsFailCall(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"errors":[{"message":"bad request"}]}`))
	}))
	defer server.Close()

	client, err := NewClient(Config{APIURL: server.URL, Token: "lin", HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if _, err := client.ListIssues(context.Background(), 10); err == nil || !strings.Contains(err.Error(), "GraphQL errors") {
		t.Fatalf("expected GraphQL error, got %v", err)
	}
}

func TestGraphQLClientUsesBearerForOAuthTokens(t *testing.T) {
	var gotAuth string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":{"viewer":{"id":"user-1","name":"Test User"}}}`))
	}))
	defer server.Close()

	client, err := NewClient(Config{APIURL: server.URL, Token: "lin_oauth_example_token_xyz", HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	_, err = client.Health(context.Background())
	if err != nil {
		t.Fatalf("Health: %v", err)
	}
	if gotAuth != "Bearer lin_oauth_example_token_xyz" {
		t.Fatalf("auth=%q, want Bearer lin_oauth_example_token_xyz", gotAuth)
	}
}
