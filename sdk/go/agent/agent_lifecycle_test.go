package agent

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
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

// Validation contract: a reasoner registered with WithDescription must carry
// that description in the node-registration payload sent to the control plane
// (it was previously local-only, used for CLI help).
func TestRegisterNode_TransmitsReasonerDescription(t *testing.T) {
	var payload types.NodeRegistrationRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			_ = json.NewDecoder(r.Body).Decode(&payload)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"registered"}`))
	}))
	defer server.Close()

	a, err := New(Config{
		NodeID:        "node-desc",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.RegisterReasoner("implement_issue",
		func(context.Context, map[string]any) (any, error) { return nil, nil },
		WithDescription("Implement one scoped issue on a branch"),
		WithReasonerTags("entrypoint"),
	)
	a.RegisterReasoner("run_coder",
		func(context.Context, map[string]any) (any, error) { return nil, nil },
	)

	require.NoError(t, a.registerNode(context.Background()))

	byID := map[string]types.ReasonerDefinition{}
	for _, r := range payload.Reasoners {
		byID[r.ID] = r
	}
	require.Len(t, byID, 2)
	assert.Equal(t, "Implement one scoped issue on a branch", byID["implement_issue"].Description)
	assert.Contains(t, byID["implement_issue"].Tags, "entrypoint")
	assert.Empty(t, byID["run_coder"].Description)
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

func TestResolveShutdownTimeout(t *testing.T) {
	for _, tc := range []struct {
		value string
		want  time.Duration
	}{
		{"", 30 * time.Second}, {"30", 30 * time.Second}, {"30s", 30 * time.Second},
		{"5m", 5 * time.Minute}, {"invalid", 30 * time.Second},
	} {
		t.Run(tc.value, func(t *testing.T) {
			assert.Equal(t, tc.want, resolveShutdownTimeout(tc.value, log.New(io.Discard, "", 0)))
		})
	}
}

func TestShutdownRouteAccepted(t *testing.T) {
	a, err := New(Config{NodeID: "node-1", Version: "1", Logger: log.New(io.Discard, "", 0)})
	require.NoError(t, err)
	recorder := httptest.NewRecorder()
	a.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/shutdown", strings.NewReader(`{"graceful":false,"timeout_seconds":1}`)))
	assert.Equal(t, http.StatusAccepted, recorder.Code)
}

func TestShutdownWaitsForAcceptedAsyncExecutionTerminalStatus(t *testing.T) {
	statusPosted := make(chan map[string]any, 1)
	cp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/v1/nodes/node-1/shutdown":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{}`)
		case r.URL.Path == "/api/v1/executions/exec-1/status":
			var payload map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
			statusPosted <- payload
			w.WriteHeader(http.StatusNoContent)
		default:
			http.NotFound(w, r)
		}
	}))
	defer cp.Close()

	release := make(chan struct{})
	started := make(chan struct{})
	a, err := New(Config{
		NodeID:          "node-1",
		Version:         "1.0.0",
		AgentFieldURL:   cp.URL,
		ShutdownTimeout: time.Second,
		Logger:          log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.httpClient = cp.Client()
	a.RegisterReasoner("slow", func(context.Context, map[string]any) (any, error) {
		close(started)
		<-release
		return map[string]any{"ok": true}, nil
	})

	req := httptest.NewRequest(http.MethodPost, "/reasoners/slow", strings.NewReader(`{}`))
	req.Header.Set("X-Execution-ID", "exec-1")
	recorder := httptest.NewRecorder()
	a.Handler().ServeHTTP(recorder, req)
	require.Equal(t, http.StatusAccepted, recorder.Code)
	<-started

	shutdownDone := make(chan error, 1)
	go func() { shutdownDone <- a.shutdown(context.Background()) }()
	select {
	case err := <-shutdownDone:
		t.Fatalf("shutdown returned before the execution completed: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	close(release)
	require.NoError(t, <-shutdownDone)
	select {
	case payload := <-statusPosted:
		assert.Equal(t, "succeeded", payload["status"])
	default:
		t.Fatal("terminal status was not posted before shutdown returned")
	}
}

func TestShutdownTimeoutCancelsAcceptedAsyncExecutionAndReportsTerminalStatus(t *testing.T) {
	statusPosted := make(chan map[string]any, 1)
	cp := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/v1/nodes/node-1/shutdown":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{}`)
		case r.URL.Path == "/api/v1/executions/exec-timeout/status":
			var payload map[string]any
			require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
			statusPosted <- payload
			w.WriteHeader(http.StatusNoContent)
		default:
			http.NotFound(w, r)
		}
	}))
	defer cp.Close()

	started := make(chan struct{})
	a, err := New(Config{
		NodeID:          "node-1",
		Version:         "1.0.0",
		AgentFieldURL:   cp.URL,
		ShutdownTimeout: 200 * time.Millisecond,
		Logger:          log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	a.httpClient = cp.Client()
	a.RegisterReasoner("cancel-aware", func(ctx context.Context, _ map[string]any) (any, error) {
		close(started)
		select {
		case <-time.After(2 * time.Second):
			return nil, errors.New("reasoner was not cancelled")
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	})

	req := httptest.NewRequest(http.MethodPost, "/reasoners/cancel-aware", strings.NewReader(`{}`))
	req.Header.Set("X-Execution-ID", "exec-timeout")
	recorder := httptest.NewRecorder()
	a.Handler().ServeHTTP(recorder, req)
	require.Equal(t, http.StatusAccepted, recorder.Code)
	<-started

	begin := time.Now()
	require.NoError(t, a.shutdown(context.Background()))
	assert.Less(t, time.Since(begin), time.Second)
	select {
	case payload := <-statusPosted:
		assert.Contains(t, []any{"failed", "cancelled"}, payload["status"])
	default:
		t.Fatal("terminal status was not posted before shutdown returned")
	}
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
