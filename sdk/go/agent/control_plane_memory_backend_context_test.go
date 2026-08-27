package agent

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestControlPlaneMemoryBackend_ContextCancellation verifies that in-flight
// HTTP calls are cancelled promptly when the caller's context is cancelled.
// This is the core fix for issue #433.
func TestControlPlaneMemoryBackend_ContextCancellation(t *testing.T) {
	t.Run("Set returns context.Canceled", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		b := NewControlPlaneMemoryBackend(server.URL, "token", "node-1")
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately before the request

		err := b.Set(ctx, ScopeSession, "s1", "k", "v")
		require.Error(t, err)
		assert.ErrorIs(t, err, context.Canceled)
	})

	t.Run("Get returns context.Canceled", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		b := NewControlPlaneMemoryBackend(server.URL, "token", "node-1")
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		_, _, err := b.Get(ctx, ScopeSession, "s1", "k")
		require.Error(t, err)
		assert.ErrorIs(t, err, context.Canceled)
	})

	t.Run("Delete returns context.Canceled", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		b := NewControlPlaneMemoryBackend(server.URL, "token", "node-1")
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		err := b.Delete(ctx, ScopeSession, "s1", "k")
		require.Error(t, err)
		assert.ErrorIs(t, err, context.Canceled)
	})

	t.Run("List returns context.Canceled", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		b := NewControlPlaneMemoryBackend(server.URL, "token", "node-1")
		ctx, cancel := context.WithCancel(context.Background())
		cancel()

		_, err := b.List(ctx, ScopeSession, "s1")
		require.Error(t, err)
		assert.ErrorIs(t, err, context.Canceled)
	})

	t.Run("DeadlineExceeded returns promptly", func(t *testing.T) {
		var requestReceived atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			requestReceived.Add(1)
			// Simulate a slow server - sleep longer than the client deadline
			time.Sleep(5 * time.Second)
			w.WriteHeader(http.StatusOK)
		}))
		defer server.Close()

		b := NewControlPlaneMemoryBackend(server.URL, "token", "node-1")

		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()

		start := time.Now()
		err := b.Set(ctx, ScopeSession, "s1", "k", "v")
		elapsed := time.Since(start)

		require.Error(t, err)
		// Should return well before the 15s HTTP client timeout.
		assert.Less(t, elapsed, 2*time.Second)
		// The request should have been initiated (proving context cancellation
		// is what aborted it, not a pre-send check).
		assert.GreaterOrEqual(t, requestReceived.Load(), int32(1))
	})
}
