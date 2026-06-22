package sentry

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestClientSendsBearerAndBuildsIssueEventsPath(t *testing.T) {
	var gotAuth string
	var gotPath string
	var gotQuery string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[{"id":"evt-1"}]`))
	}))
	defer server.Close()

	client, err := NewClient(Config{BaseURL: server.URL, Organization: "agentfield", Token: "sntrys_token", HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	out, err := client.ListIssueEvents(context.Background(), "123", "level:error", 10)
	if err != nil {
		t.Fatalf("ListIssueEvents: %v", err)
	}
	if gotAuth != "Bearer sntrys_token" {
		t.Fatalf("auth=%q", gotAuth)
	}
	if gotPath != "/api/0/organizations/agentfield/issues/123/events/" {
		t.Fatalf("path=%q", gotPath)
	}
	if !strings.Contains(gotQuery, "query=level%3Aerror") {
		t.Fatalf("query=%q", gotQuery)
	}
	if out == nil {
		t.Fatal("expected parsed response")
	}
}

func TestUpdateIssueSendsBody(t *testing.T) {
	var gotMethod string
	var gotPath string
	var gotBody map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"123","status":"resolved"}`))
	}))
	defer server.Close()

	client, err := NewClient(Config{BaseURL: server.URL, Organization: "agentfield", Token: "tok", HTTPClient: server.Client()})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if _, err := client.ResolveIssue(context.Background(), "123"); err != nil {
		t.Fatalf("ResolveIssue: %v", err)
	}
	if gotMethod != http.MethodPut || gotPath != "/api/0/organizations/agentfield/issues/123/" {
		t.Fatalf("method/path=%s %s", gotMethod, gotPath)
	}
	if gotBody["status"] != "resolved" {
		t.Fatalf("body=%v", gotBody)
	}
}

func TestListIssuesRequiresOrganizationAndProject(t *testing.T) {
	client, err := NewClient(Config{BaseURL: "https://sentry.test", Token: "tok"})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if _, err := client.ListIssues(context.Background(), "web", "", 10); err == nil || !strings.Contains(err.Error(), "organization") {
		t.Fatalf("expected organization error, got %v", err)
	}
	client.cfg.Organization = "org"
	if _, err := client.ListIssues(context.Background(), "", "", 10); err == nil || !strings.Contains(err.Error(), "project") {
		t.Fatalf("expected project error, got %v", err)
	}
}
