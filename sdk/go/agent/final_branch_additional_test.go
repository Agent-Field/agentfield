package agent

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLocalVerifier_FetchPoliciesAdditionalErrors(t *testing.T) {
	t.Run("non-200 response", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/api/v1/policies" {
				http.NotFound(w, r)
				return
			}
			http.Error(w, "policy failure", http.StatusBadGateway)
		}))
		defer srv.Close()

		v := NewLocalVerifier(srv.URL, 0, "")
		_, err := v.fetchPolicies(srv.Client())
		require.Error(t, err)
		assert.Contains(t, err.Error(), "HTTP 502")
	})

	t.Run("invalid json response", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/api/v1/policies" {
				http.NotFound(w, r)
				return
			}
			_, _ = io.WriteString(w, `{`)
		}))
		defer srv.Close()

		v := NewLocalVerifier(srv.URL, 0, "")
		_, err := v.fetchPolicies(srv.Client())
		require.Error(t, err)
	})
}

func TestMemory_SessionScopeFallsBackToRunID(t *testing.T) {
	backend := NewInMemoryBackend()
	memory := NewMemory(backend)
	ctx := contextWithExecution(context.Background(), ExecutionContext{RunID: "run-session-fallback"})

	require.NoError(t, memory.SessionScope().Set(ctx, "key", "value"))
	val, err := memory.SessionScope().Get(ctx, "key")
	require.NoError(t, err)
	assert.Equal(t, "value", val)
}
