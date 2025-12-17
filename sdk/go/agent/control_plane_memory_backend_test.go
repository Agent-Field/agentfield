package agent

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestControlPlaneMemoryBackend_SetSendsScopeHeaders(t *testing.T) {
	var gotPath string
	var gotWorkflow string
	var gotScope string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotWorkflow = r.Header.Get("X-Workflow-ID")

		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if s, ok := body["scope"].(string); ok {
			gotScope = s
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"key":"k","data":{},"scope":"workflow","scope_id":"wf-1","created_at":"now","updated_at":"now"}`))
	}))
	defer srv.Close()

	b := NewControlPlaneMemoryBackend(srv.URL, "", "agent-1")
	if err := b.Set(ScopeWorkflow, "wf-1", "k", map[string]any{"v": 1}); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if gotPath != "/api/v1/memory/set" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotWorkflow != "wf-1" {
		t.Fatalf("workflow header = %q", gotWorkflow)
	}
	if gotScope != "workflow" {
		t.Fatalf("scope body = %q", gotScope)
	}
}

func TestControlPlaneMemoryBackend_UserScopeMapsToActor(t *testing.T) {
	var gotActor string
	var gotScope string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotActor = r.Header.Get("X-Actor-ID")
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		gotScope, _ = body["scope"].(string)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"key":"k","data":"v","scope":"actor","scope_id":"u-1","created_at":"now","updated_at":"now"}`))
	}))
	defer srv.Close()

	b := NewControlPlaneMemoryBackend(srv.URL, "", "agent-1")
	if err := b.Set(ScopeUser, "u-1", "k", "v"); err != nil {
		t.Fatalf("Set: %v", err)
	}
	if gotActor != "u-1" {
		t.Fatalf("actor header = %q", gotActor)
	}
	if gotScope != "actor" {
		t.Fatalf("scope body = %q", gotScope)
	}
}

func TestControlPlaneMemoryBackend_GetNotFound(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":"not_found"}`))
	}))
	defer srv.Close()

	b := NewControlPlaneMemoryBackend(srv.URL, "", "agent-1")
	val, found, err := b.Get(ScopeSession, "s-1", "missing")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if found {
		t.Fatalf("expected not found")
	}
	if val != nil {
		t.Fatalf("expected nil val")
	}
}

func TestControlPlaneMemoryBackend_ListReturnsKeys(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.RawQuery, "scope=") {
			t.Fatalf("missing scope query: %q", r.URL.RawQuery)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`[
  {"key":"a","data":1,"scope":"global","scope_id":"global","created_at":"now","updated_at":"now"},
  {"key":"b","data":2,"scope":"global","scope_id":"global","created_at":"now","updated_at":"now"}
]`))
	}))
	defer srv.Close()

	b := NewControlPlaneMemoryBackend(srv.URL, "", "agent-1")
	keys, err := b.List(ScopeGlobal, "global")
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(keys) != 2 || keys[0] != "a" || keys[1] != "b" {
		t.Fatalf("keys = %#v", keys)
	}
}
