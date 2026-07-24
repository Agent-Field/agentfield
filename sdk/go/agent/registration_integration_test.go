package agent

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newRegistrationTestAgent(t *testing.T, agentFieldURL string) *Agent {
	t.Helper()

	a, err := New(Config{
		NodeID:           "node-1",
		Version:          "1.0.0",
		TeamID:           "team-1",
		AgentFieldURL:    agentFieldURL,
		PublicURL:        "https://agent.example.com",
		Logger:           log.New(io.Discard, "", 0),
		DisableLeaseLoop: true,
	})
	require.NoError(t, err)

	a.RegisterReasoner("test", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"ok": true}, nil
	})

	return a
}

func TestInitialize_RegistersNodeAndMarksReady(t *testing.T) {
	var (
		mu       sync.Mutex
		requests []string
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests = append(requests, r.Method+" "+r.URL.Path)
		mu.Unlock()

		switch r.URL.Path {
		case "/api/v1/nodes":
			require.Equal(t, http.MethodPost, r.Method)

			var payload types.NodeRegistrationRequest
			require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
			assert.Equal(t, "node-1", payload.ID)
			assert.Equal(t, "team-1", payload.TeamID)
			assert.Equal(t, "https://agent.example.com", payload.BaseURL)
			require.Len(t, payload.Reasoners, 1)
			assert.Equal(t, "test", payload.Reasoners[0].ID)

			w.WriteHeader(http.StatusOK)
			require.NoError(t, json.NewEncoder(w).Encode(types.NodeRegistrationResponse{
				ID:      "node-1",
				Success: true,
			}))
		case "/api/v1/nodes/node-1/status":
			require.Equal(t, http.MethodPatch, r.Method)

			var payload types.NodeStatusUpdate
			require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
			assert.Equal(t, "ready", payload.Phase)
			assert.Equal(t, "1.0.0", payload.Version)
			require.NotNil(t, payload.HealthScore)
			assert.Equal(t, 100, *payload.HealthScore)

			w.WriteHeader(http.StatusOK)
			require.NoError(t, json.NewEncoder(w).Encode(types.LeaseResponse{LeaseSeconds: 120}))
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()

	agent := newRegistrationTestAgent(t, server.URL)

	err := agent.Initialize(context.Background())
	require.NoError(t, err)
	assert.True(t, agent.initialized)

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, []string{
		"POST /api/v1/nodes",
		"PATCH /api/v1/nodes/node-1/status",
	}, requests)
}

func TestRegisterNode_ReturnsCleanErrorOnServerFailure(t *testing.T) {
	var registerCalls atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/nodes" {
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		registerCalls.Add(1)
		http.Error(w, "temporarily unavailable", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	agent := newRegistrationTestAgent(t, server.URL)

	// Current behavior is a single call with no retry on 5xx responses.
	err := agent.registerNode(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "503")
	assert.Equal(t, int32(1), registerCalls.Load())
}

func TestRegisterNode_PendingApprovalHonorsParentContextTimeout(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/nodes":
			w.WriteHeader(http.StatusOK)
			require.NoError(t, json.NewEncoder(w).Encode(types.NodeRegistrationResponse{
				ID:          "node-1",
				Success:     true,
				Status:      "pending_approval",
				PendingTags: []string{"sensitive"},
			}))
		case "/api/v1/nodes/node-1":
			// The source polls every 5s, so this path is usually not hit in this
			// test. If it is, keep the node pending.
			w.WriteHeader(http.StatusOK)
			require.NoError(t, json.NewEncoder(w).Encode(map[string]any{
				"id":               "node-1",
				"lifecycle_status": "pending_approval",
			}))
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()

	agent := newRegistrationTestAgent(t, server.URL)

	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()

	err := agent.registerNode(ctx)
	require.Error(t, err)
	// The source uses its own 5-minute timer for tag approval and reports it
	// with a fixed message; it does not wrap context.DeadlineExceeded. Assert
	// we got the approval-timeout path without insisting on the chain.
	assert.Contains(t, err.Error(), "tag approval")
	assert.Contains(t, err.Error(), "timed out")
}

func TestInitialize_WithoutAgentFieldURLReturnsClearError(t *testing.T) {
	agent := newRegistrationTestAgent(t, "")

	err := agent.Initialize(context.Background())
	require.Error(t, err)
	assert.EqualError(t, err, "AgentFieldURL is required when running in server mode")
}

func TestRegisterNode_FallsBackToLegacyEndpointOnNotFound(t *testing.T) {
	var (
		mu       sync.Mutex
		requests []string
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		requests = append(requests, r.URL.Path)
		mu.Unlock()

		switch r.URL.Path {
		case "/api/v1/nodes":
			http.NotFound(w, r)
		case "/api/v1/nodes/register":
			w.WriteHeader(http.StatusOK)
			require.NoError(t, json.NewEncoder(w).Encode(types.NodeRegistrationResponse{
				ID:      "node-1",
				Success: true,
			}))
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()

	agent := newRegistrationTestAgent(t, server.URL)

	err := agent.registerNode(context.Background())
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, []string{"/api/v1/nodes", "/api/v1/nodes/register"}, requests)
}

func TestRegisterNode_ConcurrentCallsDoNotPanic(t *testing.T) {
	var registerCalls atomic.Int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/v1/nodes" {
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
		registerCalls.Add(1)
		w.WriteHeader(http.StatusOK)
		require.NoError(t, json.NewEncoder(w).Encode(types.NodeRegistrationResponse{
			ID:      "node-1",
			Success: true,
		}))
	}))
	defer server.Close()

	agent := newRegistrationTestAgent(t, server.URL)

	errCh := make(chan error, 2)
	var wg sync.WaitGroup

	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() {
				if rec := recover(); rec != nil {
					errCh <- errors.New("registerNode panicked")
				}
			}()
			errCh <- agent.registerNode(context.Background())
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		assert.NoError(t, err)
	}
	assert.Equal(t, int32(2), registerCalls.Load())
}
