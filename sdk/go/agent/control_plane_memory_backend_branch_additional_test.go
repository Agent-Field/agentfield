package agent

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestControlPlaneMemoryBackend_AdditionalHTTPBranches(t *testing.T) {
	t.Run("get delete and list success branches", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/v1/memory/get":
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `{"key":"k","data":{"ok":true},"scope":"session","scope_id":"sess-1","created_at":"now","updated_at":"now"}`)
			case "/api/v1/memory/delete":
				w.WriteHeader(http.StatusNoContent)
			case "/api/v1/memory/list":
				w.Header().Set("Content-Type", "application/json")
				_, _ = io.WriteString(w, `[{"key":"","data":1},{"key":"kept","data":2}]`)
			default:
				http.NotFound(w, r)
			}
		}))
		defer srv.Close()

		b := NewControlPlaneMemoryBackend(srv.URL, "", "agent-1")
		val, found, err := b.Get(ScopeSession, "sess-1", "k")
		require.NoError(t, err)
		require.True(t, found)
		assert.Equal(t, map[string]any{"ok": true}, val)

		require.NoError(t, b.Delete(ScopeSession, "sess-1", "k"))

		keys, err := b.List(ScopeSession, "sess-1")
		require.NoError(t, err)
		assert.Equal(t, []string{"kept"}, keys)
	})

	t.Run("set get delete and list surface server and decode failures", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/v1/memory/set":
				http.Error(w, "set failed", http.StatusBadRequest)
			case "/api/v1/memory/get":
				http.Error(w, "get failed", http.StatusInternalServerError)
			case "/api/v1/memory/delete":
				http.Error(w, "delete failed", http.StatusBadGateway)
			case "/api/v1/memory/list":
				_, _ = io.WriteString(w, `{`)
			default:
				http.NotFound(w, r)
			}
		}))
		defer srv.Close()

		b := NewControlPlaneMemoryBackend(srv.URL, "", "agent-1")

		err := b.Set(ScopeSession, "sess-1", "k", "v")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "memory set failed")

		_, _, err = b.Get(ScopeSession, "sess-1", "k")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "memory get failed")

		err = b.Delete(ScopeSession, "sess-1", "k")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "memory delete failed")

		_, err = b.List(ScopeSession, "sess-1")
		require.Error(t, err)
	})

	t.Run("vector operations surface server and decode failures", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch {
			case r.Method == http.MethodPost && r.URL.Path == "/api/v1/memory/vector":
				http.Error(w, "vector set failed", http.StatusUnprocessableEntity)
			case r.Method == http.MethodGet && r.URL.Path == "/api/v1/memory/vector/bad":
				http.Error(w, "vector get failed", http.StatusBadGateway)
			case r.Method == http.MethodGet && r.URL.Path == "/api/v1/memory/vector/decode":
				_, _ = io.WriteString(w, `{`)
			case r.Method == http.MethodPost && r.URL.Path == "/api/v1/memory/vector/search":
				_, _ = io.WriteString(w, `{`)
			case r.Method == http.MethodDelete && r.URL.Path == "/api/v1/memory/vector/bad":
				http.Error(w, "vector delete failed", http.StatusInternalServerError)
			default:
				http.NotFound(w, r)
			}
		}))
		defer srv.Close()

		b := NewControlPlaneMemoryBackend(srv.URL, "", "agent-1")

		err := b.SetVector(ScopeSession, "sess-1", "bad", []float64{1, 2}, nil)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "vector memory set failed")

		_, _, _, err = b.GetVector(ScopeSession, "sess-1", "bad")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "vector memory get failed")

		_, _, _, err = b.GetVector(ScopeSession, "sess-1", "decode")
		require.Error(t, err)

		_, err = b.SearchVector(ScopeSession, "sess-1", []float64{1}, SearchOptions{})
		require.Error(t, err)

		err = b.DeleteVector(ScopeSession, "sess-1", "bad")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "vector memory delete failed")
	})
}
