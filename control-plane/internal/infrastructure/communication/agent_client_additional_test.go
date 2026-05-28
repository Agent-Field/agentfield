package communication

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestHTTPAgentClient_ShutdownAgent(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	t.Run("success", func(t *testing.T) {
		t.Parallel()

		var sawContentType string
		var sawUserAgent string
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			assert.Equal(t, http.MethodPost, r.Method)
			assert.Equal(t, "/shutdown", r.URL.Path)
			sawContentType = r.Header.Get("Content-Type")
			sawUserAgent = r.Header.Get("User-Agent")

			body, err := io.ReadAll(r.Body)
			require.NoError(t, err)
			assert.JSONEq(t, `{"graceful":true,"timeout_seconds":15}`, string(body))

			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"status":"shutting_down",
				"graceful":true,
				"timeout_seconds":15,
				"estimated_shutdown_time":"15s",
				"message":"shutdown requested"
			}`))
		}))
		defer server.Close()

		provider := setupTestStorage(t, ctx)
		agentID := registerAgent(t, ctx, provider, server.URL)
		client := NewHTTPAgentClient(provider, time.Second)

		resp, err := client.ShutdownAgent(ctx, agentID, true, 15)
		require.NoError(t, err)
		assert.Equal(t, "application/json", sawContentType)
		assert.Equal(t, "AgentField-Server/1.0", sawUserAgent)
		assert.Equal(t, "shutting_down", resp.Status)
		assert.True(t, resp.Graceful)
		assert.Equal(t, 15, resp.TimeoutSeconds)
	})

	t.Run("storage error", func(t *testing.T) {
		t.Parallel()

		provider := &storageOverride{
			StorageProvider: setupTestStorage(t, ctx),
			override: func(context.Context, string) (*types.AgentNode, error) {
				return nil, errors.New("storage offline")
			},
		}
		client := NewHTTPAgentClient(provider, time.Second)

		resp, err := client.ShutdownAgent(ctx, "agent-test", true, 15)
		require.Nil(t, resp)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to get agent node agent-test")
		assert.Contains(t, err.Error(), "storage offline")
	})

	t.Run("invalid base url", func(t *testing.T) {
		t.Parallel()

		provider := &storageOverride{
			StorageProvider: setupTestStorage(t, ctx),
			override: func(context.Context, string) (*types.AgentNode, error) {
				return &types.AgentNode{ID: "agent-test", BaseURL: "://bad-url"}, nil
			},
		}
		client := NewHTTPAgentClient(provider, time.Second)

		resp, err := client.ShutdownAgent(ctx, "agent-test", false, 5)
		require.Nil(t, resp)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to create request")
	})

	t.Run("network error", func(t *testing.T) {
		t.Parallel()

		provider := &storageOverride{
			StorageProvider: setupTestStorage(t, ctx),
			override: func(context.Context, string) (*types.AgentNode, error) {
				return &types.AgentNode{ID: "agent-test", BaseURL: "http://agent.local"}, nil
			},
		}
		client := NewHTTPAgentClient(provider, time.Second)
		client.httpClient.Transport = roundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, errors.New("dial tcp failure")
		})

		resp, err := client.ShutdownAgent(ctx, "agent-test", false, 5)
		require.Nil(t, resp)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to call agent shutdown endpoint")
	})

	t.Run("status not found", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.NotFound(w, r)
		}))
		defer server.Close()

		provider := setupTestStorage(t, ctx)
		agentID := registerAgent(t, ctx, provider, server.URL)
		client := NewHTTPAgentClient(provider, time.Second)

		resp, err := client.ShutdownAgent(ctx, agentID, false, 5)
		require.Nil(t, resp)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "does not support HTTP shutdown endpoint")
	})

	t.Run("unexpected status", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadGateway)
		}))
		defer server.Close()

		provider := setupTestStorage(t, ctx)
		agentID := registerAgent(t, ctx, provider, server.URL)
		client := NewHTTPAgentClient(provider, time.Second)

		resp, err := client.ShutdownAgent(ctx, agentID, false, 5)
		require.Nil(t, resp)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "agent returned status 502")
	})

	t.Run("invalid response body", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"status":`))
		}))
		defer server.Close()

		provider := setupTestStorage(t, ctx)
		agentID := registerAgent(t, ctx, provider, server.URL)
		client := NewHTTPAgentClient(provider, time.Second)

		resp, err := client.ShutdownAgent(ctx, agentID, false, 5)
		require.Nil(t, resp)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to decode response")
	})
}

func TestHTTPAgentClient_GetAgentStatusAdditionalPaths(t *testing.T) {
	t.Parallel()

	ctx := context.Background()

	t.Run("storage error", func(t *testing.T) {
		t.Parallel()

		provider := &storageOverride{
			StorageProvider: setupTestStorage(t, ctx),
			override: func(context.Context, string) (*types.AgentNode, error) {
				return nil, errors.New("storage offline")
			},
		}
		client := NewHTTPAgentClient(provider, time.Second)

		resp, err := client.GetAgentStatus(ctx, "agent-test")
		require.Nil(t, resp)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to get agent node agent-test")
		assert.Contains(t, err.Error(), "storage offline")
	})

	t.Run("invalid base url", func(t *testing.T) {
		t.Parallel()

		provider := &storageOverride{
			StorageProvider: setupTestStorage(t, ctx),
			override: func(context.Context, string) (*types.AgentNode, error) {
				return &types.AgentNode{ID: "agent-test", BaseURL: "://bad-url"}, nil
			},
		}
		client := NewHTTPAgentClient(provider, time.Second)

		resp, err := client.GetAgentStatus(ctx, "agent-test")
		require.Nil(t, resp)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to create request")
	})

	t.Run("non retryable network error", func(t *testing.T) {
		t.Parallel()

		provider := &storageOverride{
			StorageProvider: setupTestStorage(t, ctx),
			override: func(context.Context, string) (*types.AgentNode, error) {
				return &types.AgentNode{ID: "agent-test", BaseURL: "http://agent.local"}, nil
			},
		}
		client := NewHTTPAgentClient(provider, time.Second)
		client.httpClient.Transport = roundTripFunc(func(*http.Request) (*http.Response, error) {
			return nil, errors.New("tls handshake failed")
		})
		client.httpClient.Timeout = 0

		resp, err := client.GetAgentStatus(ctx, "agent-test")
		require.Nil(t, resp)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "network failure calling agent status endpoint")
	})

	t.Run("status not found", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.NotFound(w, r)
		}))
		defer server.Close()

		provider := setupTestStorage(t, ctx)
		agentID := registerAgent(t, ctx, provider, server.URL)
		client := NewHTTPAgentClient(provider, time.Second)

		resp, err := client.GetAgentStatus(ctx, agentID)
		require.Nil(t, resp)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "does not support status endpoint")
	})

	t.Run("unexpected status", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusBadGateway)
		}))
		defer server.Close()

		provider := setupTestStorage(t, ctx)
		agentID := registerAgent(t, ctx, provider, server.URL)
		client := NewHTTPAgentClient(provider, time.Second)

		resp, err := client.GetAgentStatus(ctx, agentID)
		require.Nil(t, resp)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "agent returned status 502")
	})

	t.Run("invalid response body", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"status":`))
		}))
		defer server.Close()

		provider := setupTestStorage(t, ctx)
		agentID := registerAgent(t, ctx, provider, server.URL)
		client := NewHTTPAgentClient(provider, time.Second)

		resp, err := client.GetAgentStatus(ctx, agentID)
		require.Nil(t, resp)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to decode response")
	})

	t.Run("missing node id is allowed", func(t *testing.T) {
		t.Parallel()

		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"status":"running",
				"uptime":"1s",
				"uptime_seconds":1,
				"pid":123,
				"version":"1.0.0",
				"node_id":"",
				"last_activity":"2024-01-01T00:00:00Z",
				"resources":{}
			}`))
		}))
		defer server.Close()

		provider := setupTestStorage(t, ctx)
		agentID := registerAgent(t, ctx, provider, server.URL)
		client := NewHTTPAgentClient(provider, time.Second)

		resp, err := client.GetAgentStatus(ctx, agentID)
		require.NoError(t, err)
		assert.Equal(t, "running", resp.Status)
		assert.Empty(t, resp.NodeID)
	})
}

func TestIsRetryableError(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "nil", err: nil, want: false},
		{name: "connection refused", err: errors.New("Connection Refused"), want: true},
		{name: "network unreachable", err: errors.New("network is unreachable"), want: true},
		{name: "permanent failure", err: errors.New("certificate expired"), want: false},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.want, isRetryableError(tt.err))
		})
	}
}

var _ storage.StorageProvider = (*storageOverride)(nil)
