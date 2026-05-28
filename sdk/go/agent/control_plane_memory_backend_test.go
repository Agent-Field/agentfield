package agent

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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

func TestControlPlaneMemoryBackend_VectorOperationsRoundTrip(t *testing.T) {
	var sawAuthorization string
	var sawSession string
	var setVectorBody map[string]any
	var searchBody map[string]any

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sawAuthorization = r.Header.Get("Authorization")
		sawSession = r.Header.Get("X-Session-ID")

		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/api/v1/memory/vector":
			require.NoError(t, json.NewDecoder(r.Body).Decode(&setVectorBody))
			w.WriteHeader(http.StatusCreated)
		case r.Method == http.MethodGet && r.URL.Path == "/api/v1/memory/vector/vector-key":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"embedding":[1.25,2.5],"metadata":{"kind":"cached"}}`)
		case r.Method == http.MethodPost && r.URL.Path == "/api/v1/memory/vector/search":
			require.NoError(t, json.NewDecoder(r.Body).Decode(&searchBody))
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `[{"key":"vector-key","score":0.91,"metadata":{"kind":"cached"},"scope":"workflow","scope_id":"wf-1"}]`)
		case r.Method == http.MethodDelete && r.URL.Path == "/api/v1/memory/vector/vector-key":
			w.WriteHeader(http.StatusNoContent)
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer srv.Close()

	b := NewControlPlaneMemoryBackend(srv.URL, "token-123", "agent-1")
	require.NoError(t, b.SetVector(ScopeSession, "sess-1", "vector-key", []float64{1.25, 2.5}, map[string]any{"kind": "cached"}))

	embedding, metadata, found, err := b.GetVector(ScopeSession, "sess-1", "vector-key")
	require.NoError(t, err)
	assert.True(t, found)
	assert.InDeltaSlice(t, []float64{1.25, 2.5}, embedding, 1e-6)
	assert.Equal(t, map[string]any{"kind": "cached"}, metadata)

	results, err := b.SearchVector(ScopeSession, "sess-1", []float64{1.25, 2.5}, SearchOptions{Limit: 3, Threshold: 0.5, Filters: map[string]any{"kind": "cached"}, Scope: ScopeWorkflow})
	require.NoError(t, err)
	require.Len(t, results, 1)
	assert.Equal(t, "vector-key", results[0].Key)
	assert.Equal(t, ScopeWorkflow, results[0].Scope)
	assert.Equal(t, "wf-1", results[0].ScopeID)

	require.NoError(t, b.DeleteVector(ScopeSession, "sess-1", "vector-key"))
	assert.Equal(t, "Bearer token-123", sawAuthorization)
	assert.Equal(t, "sess-1", sawSession)
	assert.Equal(t, "session", setVectorBody["scope"])
	assert.Equal(t, "workflow", searchBody["scope"])
	assert.Equal(t, float64(3), searchBody["top_k"])
	assert.Equal(t, 2, len(setVectorBody["embedding"].([]any)))
}

func TestControlPlaneMemoryBackend_ErrorPathsAndHelpers(t *testing.T) {
	t.Run("vector get not found and delete not found are non-errors", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusNotFound)
		}))
		defer srv.Close()

		b := NewControlPlaneMemoryBackend(srv.URL, "", "agent-1")
		embedding, metadata, found, err := b.GetVector(ScopeWorkflow, "wf-1", "missing")
		require.NoError(t, err)
		assert.False(t, found)
		assert.Nil(t, embedding)
		assert.Nil(t, metadata)
		require.NoError(t, b.DeleteVector(ScopeWorkflow, "wf-1", "missing"))
	})

	t.Run("vector search surfaces server errors", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadGateway)
			_, _ = io.WriteString(w, "upstream failed")
		}))
		defer srv.Close()

		b := NewControlPlaneMemoryBackend(srv.URL, "", "agent-1")
		_, err := b.SearchVector(ScopeGlobal, "global", []float64{1}, SearchOptions{})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "vector memory search failed")
	})

	t.Run("scope helpers and mustJSONReader", func(t *testing.T) {
		b := NewControlPlaneMemoryBackend("http://example.com///", "token-123", "agent-1")
		assert.Equal(t, "http://example.com", b.baseURL)
		assert.Equal(t, "workflow", b.apiScope(ScopeWorkflow))
		assert.Equal(t, "session", b.apiScope(ScopeSession))
		assert.Equal(t, "actor", b.apiScope(ScopeUser))
		assert.Equal(t, "global", b.apiScope(ScopeGlobal))
		assert.Equal(t, "global", b.apiScope(MemoryScope("unexpected")))

		body, err := io.ReadAll(mustJSONReader(map[string]any{"ok": true}))
		require.NoError(t, err)
		assert.JSONEq(t, `{"ok":true}`, string(body))
	})
}
