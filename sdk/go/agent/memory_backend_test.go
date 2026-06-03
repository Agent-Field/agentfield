package agent

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestControlPlaneMemoryBackend_SetAppliesScopeHeaders(t *testing.T) {
	tests := []struct {
		name          string
		scope         MemoryScope
		scopeID       string
		wantScope     string
		wantHeaderKey string
		wantHeaderVal string
	}{
		{
			name:          "workflow scope",
			scope:         ScopeWorkflow,
			scopeID:       "wf-1",
			wantScope:     "workflow",
			wantHeaderKey: "X-Workflow-ID",
			wantHeaderVal: "wf-1",
		},
		{
			name:          "session scope",
			scope:         ScopeSession,
			scopeID:       "s-1",
			wantScope:     "session",
			wantHeaderKey: "X-Session-ID",
			wantHeaderVal: "s-1",
		},
		{
			name:      "global scope",
			scope:     ScopeGlobal,
			scopeID:   "",
			wantScope: "global",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var gotBody map[string]any
			var gotHeaders http.Header
			var gotMethod string

			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotMethod = r.Method
				gotHeaders = r.Header.Clone()
				require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))
				w.WriteHeader(http.StatusOK)
			}))
			defer server.Close()

			backend := NewControlPlaneMemoryBackend(server.URL, "token-1", "agent-1")
			err := backend.Set(tt.scope, tt.scopeID, "key", map[string]any{"ok": true})
			require.NoError(t, err)

			assert.Equal(t, http.MethodPost, gotMethod)
			assert.Equal(t, tt.wantScope, gotBody["scope"])
			assert.Equal(t, "Bearer token-1", gotHeaders.Get("Authorization"))
			assert.Equal(t, "agent-1", gotHeaders.Get("X-Agent-Node-ID"))
			if tt.wantHeaderKey == "" {
				assert.Empty(t, gotHeaders.Get("X-Workflow-ID"))
				assert.Empty(t, gotHeaders.Get("X-Session-ID"))
				assert.Empty(t, gotHeaders.Get("X-Actor-ID"))
			} else {
				assert.Equal(t, tt.wantHeaderVal, gotHeaders.Get(tt.wantHeaderKey))
			}
		})
	}
}

func TestControlPlaneMemoryBackend_GetHandlesNotFoundAndServerErrors(t *testing.T) {
	t.Run("not found", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, http.MethodPost, r.Method)
			assert.Equal(t, "/api/v1/memory/get", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}))
		defer server.Close()

		backend := NewControlPlaneMemoryBackend(server.URL, "", "")
		value, found, err := backend.Get(ScopeSession, "s-1", "missing")
		require.NoError(t, err)
		assert.Nil(t, value)
		assert.False(t, found)
	})

	t.Run("server error", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "broken", http.StatusInternalServerError)
		}))
		defer server.Close()

		backend := NewControlPlaneMemoryBackend(server.URL, "", "")
		value, found, err := backend.Get(ScopeSession, "s-1", "key")
		require.Error(t, err)
		assert.Nil(t, value)
		assert.False(t, found)
		assert.Contains(t, err.Error(), "memory get failed")
		assert.Contains(t, err.Error(), "500")
	})
}

func TestControlPlaneMemoryBackend_DeleteUsesPostEndpointAndScopeHeaders(t *testing.T) {
	var (
		gotMethod string
		gotPath   string
		gotHeader string
		gotBody   map[string]any
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotMethod = r.Method
		gotPath = r.URL.Path
		gotHeader = r.Header.Get("X-Session-ID")
		require.NoError(t, json.NewDecoder(r.Body).Decode(&gotBody))
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	backend := NewControlPlaneMemoryBackend(server.URL, "", "")

	// Current behavior is POST /api/v1/memory/delete rather than HTTP DELETE.
	err := backend.Delete(ScopeSession, "s-1", "key")
	require.NoError(t, err)

	assert.Equal(t, http.MethodPost, gotMethod)
	assert.Equal(t, "/api/v1/memory/delete", gotPath)
	assert.Equal(t, "s-1", gotHeader)
	assert.Equal(t, "session", gotBody["scope"])
	assert.Equal(t, "key", gotBody["key"])
}

func TestControlPlaneMemoryBackend_ListUsesScopeQueryAndHeaders(t *testing.T) {
	var (
		gotQuery  url.Values
		gotHeader string
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotQuery = r.URL.Query()
		gotHeader = r.Header.Get("X-Session-ID")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`[
			{"key":"a","scope":"session","scope_id":"s-1"},
			{"key":"","scope":"session","scope_id":"s-1"},
			{"key":"b","scope":"session","scope_id":"s-1"}
		]`))
	}))
	defer server.Close()

	backend := NewControlPlaneMemoryBackend(server.URL, "", "")
	keys, err := backend.List(ScopeSession, "s-1")
	require.NoError(t, err)

	assert.Equal(t, "session", gotQuery.Get("scope"))
	assert.Equal(t, "s-1", gotHeader)
	assert.Equal(t, []string{"a", "b"}, keys)
}

func TestControlPlaneMemoryBackend_EmptyScopeIDDoesNotErrorForNonGlobalScope(t *testing.T) {
	var gotSessionID string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotSessionID = r.Header.Get("X-Session-ID")
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	backend := NewControlPlaneMemoryBackend(server.URL, "", "")
	err := backend.Set(ScopeSession, "", "key", "value")
	require.NoError(t, err)

	// The backend does not resolve missing scope IDs from execution context.
	// It simply omits the header and lets the control plane decide what to do.
	assert.Empty(t, gotSessionID)
}
