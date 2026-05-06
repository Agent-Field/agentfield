package agent

import (
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/did"
	"github.com/Agent-Field/agentfield/sdk/go/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestInitialize_AlreadyInitializedIsNoop(t *testing.T) {
	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://example.com",
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)

	// Once initialized, Initialize should return immediately.
	a.initialized = true
	require.NoError(t, a.Initialize(context.Background()))
}

func TestInitialize_WrapsRegisterNodeError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "temporarily unavailable", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.RegisterReasoner("demo", func(context.Context, map[string]any) (any, error) { return nil, nil })

	err = a.Initialize(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "register node:")
}

func TestInitialize_ContinuesWhenDIDOrReadyUpdatesFail(t *testing.T) {
	agentDID, _ := testDIDCredentials(t)
	var statusCalls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/nodes":
			w.WriteHeader(http.StatusOK)
			require.NoError(t, json.NewEncoder(w).Encode(types.NodeRegistrationResponse{
				ID:      "node-1",
				Success: true,
			}))
		case "/api/v1/nodes/node-1/status":
			statusCalls++
			http.Error(w, "status failed", http.StatusBadGateway)
		case "/api/v1/did/register":
			w.Header().Set("Content-Type", "application/json")
			// Successful DID registration followed by invalid credentials exercises
			// the warning-only path inside Initialize.
			require.NoError(t, json.NewEncoder(w).Encode(did.RegistrationResponse{
				Success: true,
				IdentityPackage: did.DIDIdentityPackage{
					AgentDID: did.DIDIdentity{
						DID:           agentDID,
						PrivateKeyJWK: "{invalid",
					},
				},
			}))
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()

	a, err := New(Config{
		NodeID:           "node-1",
		Version:          "1.0.0",
		AgentFieldURL:    server.URL,
		EnableDID:        true,
		DisableLeaseLoop: true,
		Logger:           log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.RegisterReasoner("demo", func(context.Context, map[string]any) (any, error) { return nil, nil })

	require.NoError(t, a.Initialize(context.Background()))
	assert.True(t, a.initialized)
	assert.Equal(t, 1, statusCalls)
}

func TestWaitForApproval_CompletesAfterPollAndLogsPollingErrors(t *testing.T) {
	var polls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/v1/nodes/node-1":
			polls++
			if polls == 1 {
				// The first poll failing should not abort the approval loop.
				http.Error(w, "try again", http.StatusBadGateway)
				return
			}
			w.WriteHeader(http.StatusOK)
			require.NoError(t, json.NewEncoder(w).Encode(map[string]any{
				"id":               "node-1",
				"lifecycle_status": "ready",
			}))
		default:
			t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
		}
	}))
	defer server.Close()

	a, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)

	require.NoError(t, a.waitForApproval(context.Background()))
	assert.GreaterOrEqual(t, polls, 2)
}

func TestShutdown_HandlesNilClientAndNilServer(t *testing.T) {
	a, err := New(Config{
		NodeID:  "node-1",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)

	require.NoError(t, a.shutdown(context.Background()))
}

func TestRegisteredHeartbeatInterval(t *testing.T) {
	a, err := New(Config{
		NodeID:               "node-1",
		Version:              "1.0.0",
		LeaseRefreshInterval: 15 * time.Second,
		Logger:               log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)

	assert.Equal(t, "15s", a.registeredHeartbeatInterval())
	a.cfg.DisableLeaseLoop = true
	assert.Equal(t, "0s", a.registeredHeartbeatInterval())
}
