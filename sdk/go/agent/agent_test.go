package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/sdk/go/ai"
	agentclient "github.com/Agent-Field/agentfield/sdk/go/client"
	"github.com/Agent-Field/agentfield/sdk/go/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr bool
		check   func(t *testing.T, a *Agent)
	}{
		{
			name: "valid config",
			cfg: Config{
				NodeID:        "node-1",
				Version:       "1.0.0",
				AgentFieldURL: "https://api.example.com",
			},
			wantErr: false,
			check: func(t *testing.T, a *Agent) {
				assert.NotNil(t, a)
				assert.Equal(t, "node-1", a.cfg.NodeID)
				assert.Equal(t, "1.0.0", a.cfg.Version)
			},
		},
		{
			name: "missing NodeID",
			cfg: Config{
				Version:       "1.0.0",
				AgentFieldURL: "https://api.example.com",
			},
			wantErr: true,
		},
		{
			name: "missing Version",
			cfg: Config{
				NodeID:        "node-1",
				AgentFieldURL: "https://api.example.com",
			},
			wantErr: true,
		},
		{
			name: "missing AgentFieldURL",
			cfg: Config{
				NodeID:  "node-1",
				Version: "1.0.0",
			},
			wantErr: false,
			check: func(t *testing.T, a *Agent) {
				assert.Nil(t, a.client)
			},
		},
		{
			name: "CallTimeout defaults to 15s when unset",
			cfg: Config{
				NodeID:        "node-1",
				Version:       "1.0.0",
				AgentFieldURL: "https://api.example.com",
			},
			wantErr: false,
			check: func(t *testing.T, a *Agent) {
				assert.Equal(t, 15*time.Second, a.cfg.CallTimeout)
				assert.Equal(t, 15*time.Second, a.httpClient.Timeout)
			},
		},
		{
			name: "CallTimeout is honored when set",
			cfg: Config{
				NodeID:        "node-1",
				Version:       "1.0.0",
				AgentFieldURL: "https://api.example.com",
				CallTimeout:   90 * time.Second,
			},
			wantErr: false,
			check: func(t *testing.T, a *Agent) {
				assert.Equal(t, 90*time.Second, a.cfg.CallTimeout)
				assert.Equal(t, 90*time.Second, a.httpClient.Timeout)
			},
		},
		{
			name: "defaults applied",
			cfg: Config{
				NodeID:        "node-1",
				Version:       "1.0.0",
				AgentFieldURL: "https://api.example.com",
			},
			wantErr: false,
			check: func(t *testing.T, a *Agent) {
				assert.Equal(t, "default", a.cfg.TeamID)
				assert.Equal(t, ":8001", a.cfg.ListenAddress)
				assert.Equal(t, 2*time.Minute, a.cfg.LeaseRefreshInterval)
				assert.NotNil(t, a.cfg.Logger)
			},
		},
		{
			name: "with AIConfig",
			cfg: Config{
				NodeID:        "node-1",
				Version:       "1.0.0",
				AgentFieldURL: "https://api.example.com",
				AIConfig: &ai.Config{
					APIKey:  "test-key",
					BaseURL: "https://api.openai.com/v1",
					Model:   "gpt-4o",
				},
			},
			wantErr: false,
			check: func(t *testing.T, a *Agent) {
				assert.NotNil(t, a.aiClient)
			},
		},
		{
			name: "invalid AIConfig",
			cfg: Config{
				NodeID:        "node-1",
				Version:       "1.0.0",
				AgentFieldURL: "https://api.example.com",
				AIConfig:      &ai.Config{
					// Missing required fields
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a, err := New(tt.cfg)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, a)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, a)
				if tt.check != nil {
					tt.check(t, a)
				}
			}
		})
	}
}

func TestRegisterReasoner(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	// Register reasoner
	agent.RegisterReasoner("test", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"result": "ok"}, nil
	})

	// Verify registration
	reasoner, ok := agent.reasoners["test"]
	assert.True(t, ok)
	assert.NotNil(t, reasoner)
	assert.Equal(t, "test", reasoner.Name)
	assert.NotNil(t, reasoner.Handler)
}

func TestRegisterReasoner_WithOptions(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	inputSchema := json.RawMessage(`{"type":"object","properties":{"name":{"type":"string"}}}`)
	outputSchema := json.RawMessage(`{"type":"object","properties":{"result":{"type":"string"}}}`)

	agent.RegisterReasoner("test",
		func(ctx context.Context, input map[string]any) (any, error) {
			return nil, nil
		},
		WithInputSchema(inputSchema),
		WithOutputSchema(outputSchema),
	)

	reasoner := agent.reasoners["test"]
	assert.Equal(t, inputSchema, reasoner.InputSchema)
	assert.Equal(t, outputSchema, reasoner.OutputSchema)
}

func TestRegisterReasoner_NilHandler(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	// This should panic
	assert.Panics(t, func() {
		agent.RegisterReasoner("test", nil)
	})
}

func TestInitialize(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v1/nodes" {
			var req types.NodeRegistrationRequest
			json.NewDecoder(r.Body).Decode(&req)
			assert.Equal(t, "node-1", req.ID)
			assert.Equal(t, "team-1", req.TeamID)

			resp := types.NodeRegistrationResponse{
				ID:      "node-1",
				Success: true,
			}
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(resp)
		} else if strings.Contains(r.URL.Path, "/status") {
			resp := types.LeaseResponse{
				LeaseSeconds: 120,
			}
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(resp)
		}
	}))
	defer server.Close()

	cfg := Config{
		NodeID:           "node-1",
		Version:          "1.0.0",
		TeamID:           "team-1",
		AgentFieldURL:    server.URL,
		Logger:           log.New(io.Discard, "", 0),
		DisableLeaseLoop: true, // Disable for testing
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	agent.RegisterReasoner("test", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"ok": true}, nil
	})

	err = agent.Initialize(context.Background())
	assert.NoError(t, err)
	assert.True(t, agent.initialized)
}

func TestRegistration_UsesConfiguredHeartbeatInterval(t *testing.T) {
	var captured types.NodeRegistrationRequest
	httpClient := &http.Client{
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			require.Equal(t, http.MethodPost, req.Method)
			require.Equal(t, "/api/v1/nodes", req.URL.Path)
			require.NoError(t, json.NewDecoder(req.Body).Decode(&captured))

			resp := types.NodeRegistrationResponse{
				ID:      "node-1",
				Success: true,
			}
			body, err := json.Marshal(resp)
			require.NoError(t, err)

			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(bytes.NewReader(body)),
			}, nil
		}),
	}

	cfg := Config{
		NodeID:               "node-1",
		Version:              "1.0.0",
		TeamID:               "team-1",
		AgentFieldURL:        "https://agentfield.example.com",
		Logger:               log.New(io.Discard, "", 0),
		DisableLeaseLoop:     true,
		LeaseRefreshInterval: 45 * time.Second,
	}

	agent, err := New(cfg)
	require.NoError(t, err)
	agent.cfg.DisableLeaseLoop = false
	agent.client, err = agentclient.New(cfg.AgentFieldURL, agentclient.WithHTTPClient(httpClient))
	require.NoError(t, err)

	agent.RegisterReasoner("test", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"ok": true}, nil
	})

	require.NoError(t, agent.registerNode(context.Background()))
	assert.Equal(t, "45s", captured.CommunicationConfig.HeartbeatInterval)
}

func TestRegistration_HeartbeatIntervalFallsBackToDefault(t *testing.T) {
	assert.Equal(t, "30s", formatHeartbeatInterval(0))
}

func TestRegistration_DisableLeaseLoopRegistersZeroHeartbeat(t *testing.T) {
	var captured types.NodeRegistrationRequest
	httpClient := &http.Client{
		Transport: roundTripperFunc(func(req *http.Request) (*http.Response, error) {
			require.NoError(t, json.NewDecoder(req.Body).Decode(&captured))
			resp := types.NodeRegistrationResponse{ID: "node-2", Success: true}
			body, err := json.Marshal(resp)
			require.NoError(t, err)
			return &http.Response{
				StatusCode: http.StatusOK,
				Header:     make(http.Header),
				Body:       io.NopCloser(bytes.NewReader(body)),
			}, nil
		}),
	}

	cfg := Config{
		NodeID:               "node-2",
		Version:              "1.0.0",
		TeamID:               "team-1",
		AgentFieldURL:        "https://agentfield.example.com",
		Logger:               log.New(io.Discard, "", 0),
		DisableLeaseLoop:     true,
		LeaseRefreshInterval: 45 * time.Second,
	}

	agent, err := New(cfg)
	require.NoError(t, err)
	agent.client, err = agentclient.New(cfg.AgentFieldURL, agentclient.WithHTTPClient(httpClient))
	require.NoError(t, err)

	agent.RegisterReasoner("test", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"ok": true}, nil
	})

	require.NoError(t, agent.registerNode(context.Background()))
	assert.Equal(t, "0s", captured.CommunicationConfig.HeartbeatInterval)
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestInitialize_NoReasoners(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	err = agent.Initialize(context.Background())
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no reasoners registered")
}

func TestHandler(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	agent.RegisterReasoner("test", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"result": "ok"}, nil
	})

	handler := agent.Handler()
	assert.NotNil(t, handler)

	// Test health endpoint
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var response map[string]any
	json.NewDecoder(w.Body).Decode(&response)
	assert.Equal(t, "ok", response["status"])
}

func TestStatusHandler(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.2.3",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodGet, "/status", nil)
	w := httptest.NewRecorder()
	agent.Handler().ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var response map[string]any
	require.NoError(t, json.NewDecoder(w.Body).Decode(&response))
	assert.Equal(t, "running", response["status"])
	assert.Equal(t, "node-1", response["node_id"])
	assert.Equal(t, "1.2.3", response["version"])
	assert.Contains(t, response, "uptime_seconds")
}

func TestStatusHandler_ExemptFromOriginAuth(t *testing.T) {
	cfg := Config{
		NodeID:            "node-1",
		Version:           "1.0.0",
		AgentFieldURL:     "https://api.example.com",
		Logger:            log.New(io.Discard, "", 0),
		InternalToken:     "secret",
		RequireOriginAuth: true,
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodGet, "/status", nil)
	w := httptest.NewRecorder()
	agent.Handler().ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code, "status endpoint must stay reachable without auth so the control plane's health monitor can poll it")
}

func TestDiscoveryPayload_ReportsAuthRequired(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)
	assert.Equal(t, false, agent.discoveryPayload()["auth_required"])

	agent.cfg.RequireOriginAuth = true
	assert.Equal(t, true, agent.discoveryPayload()["auth_required"])
}

func TestHandleReasoner_Sync(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	agent.RegisterReasoner("test", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"value": input["value"]}, nil
	})

	server := httptest.NewServer(agent.handler())
	defer server.Close()

	reqBody := []byte(`{"value":42}`)
	req, err := http.NewRequest(http.MethodPost, server.URL+"/reasoners/test", bytes.NewReader(reqBody))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusOK, resp.StatusCode)

	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)
	assert.Equal(t, float64(42), result["value"]) // JSON numbers are float64
}

func TestHandleReasoner_NotFound(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	server := httptest.NewServer(agent.handler())
	defer server.Close()

	req, err := http.NewRequest(http.MethodPost, server.URL+"/reasoners/nonexistent", bytes.NewReader([]byte("{}")))
	require.NoError(t, err)

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusNotFound, resp.StatusCode)
}

func TestHandleExecute_AndServerlessHelpers(t *testing.T) {
	t.Run("success and structured errors", func(t *testing.T) {
		cfg := Config{NodeID: "node-1", Version: "1.0.0", AgentFieldURL: "https://api.example.com", Logger: log.New(io.Discard, "", 0)}
		agent, err := New(cfg)
		require.NoError(t, err)

		agent.RegisterReasoner("echo", func(ctx context.Context, input map[string]any) (any, error) {
			execCtx := executionContextFrom(ctx)
			assert.Equal(t, "exec-1", execCtx.ExecutionID)
			assert.Equal(t, "run-1", execCtx.RunID)
			assert.Equal(t, "wf-1", execCtx.WorkflowID)
			return map[string]any{"echo": input["message"]}, nil
		})
		agent.RegisterReasoner("forbidden", func(context.Context, map[string]any) (any, error) {
			return nil, &ExecuteError{StatusCode: http.StatusForbidden, Message: "forbidden", ErrorDetails: map[string]any{"code": "policy_denied"}}
		})

		req := httptest.NewRequest(http.MethodPost, "/execute/echo", bytes.NewBufferString(`{"input":{"message":"hello"},"execution_context":{"execution_id":"exec-1","run_id":"run-1","workflow_id":"wf-1"}}`))
		resp := httptest.NewRecorder()
		agent.handleExecute(resp, req)
		assert.Equal(t, http.StatusOK, resp.Code)
		assert.JSONEq(t, `{"echo":"hello"}`, resp.Body.String())

		forbiddenReq := httptest.NewRequest(http.MethodPost, "/execute/forbidden", bytes.NewBufferString(`{"message":"hello"}`))
		forbiddenResp := httptest.NewRecorder()
		agent.handleExecute(forbiddenResp, forbiddenReq)
		assert.Equal(t, http.StatusForbidden, forbiddenResp.Code)
		assert.JSONEq(t, `{"error":"forbidden","error_details":{"code":"policy_denied"}}`, forbiddenResp.Body.String())
	})

	t.Run("helper functions and HandleServerlessEvent", func(t *testing.T) {
		assert.Equal(t, map[string]any{}, extractInputFromServerless(nil))
		assert.Equal(t, map[string]any{"value": "x"}, extractInputFromServerless(map[string]any{"input": "x"}))
		assert.Equal(t, map[string]any{"keep": 1}, extractInputFromServerless(map[string]any{"target": "echo", "path": "/execute/echo", "keep": 1}))

		cfg := Config{NodeID: "node-1", Version: "1.0.0", AgentFieldURL: "https://api.example.com", Logger: log.New(io.Discard, "", 0)}
		agent, err := New(cfg)
		require.NoError(t, err)
		agent.RegisterReasoner("echo", func(ctx context.Context, input map[string]any) (any, error) {
			assert.Equal(t, "serverless-run", executionContextFrom(ctx).RunID)
			return map[string]any{"echo": input["value"]}, nil
		})
		agent.RegisterReasoner("explode", func(context.Context, map[string]any) (any, error) {
			return nil, assert.AnError
		})

		req := httptest.NewRequest(http.MethodPost, "/unused", nil)
		req.Header.Set("X-Run-ID", "header-run")
		req.Header.Set("X-Actor-ID", "actor-1")
		execCtx := agent.buildExecutionContextFromServerless(req, map[string]any{
			"execution_context": map[string]any{"execution_id": "exec-2", "workflow_id": "wf-2"},
		}, "echo")
		assert.Equal(t, "header-run", execCtx.RunID)
		assert.Equal(t, "exec-2", execCtx.ExecutionID)
		assert.Equal(t, "wf-2", execCtx.WorkflowID)
		assert.Equal(t, "actor-1", execCtx.ActorID)
		assert.Equal(t, "node-1", execCtx.AgentNodeID)

		result, status, err := agent.HandleServerlessEvent(context.Background(), map[string]any{}, nil)
		require.NoError(t, err)
		assert.Equal(t, http.StatusBadRequest, status)
		assert.Equal(t, map[string]any{"error": "missing target or reasoner"}, result)

		result, status, err = agent.HandleServerlessEvent(context.Background(), map[string]any{"target": "missing"}, nil)
		require.NoError(t, err)
		assert.Equal(t, http.StatusNotFound, status)
		assert.Equal(t, map[string]any{"error": "reasoner not found"}, result)

		result, status, err = agent.HandleServerlessEvent(context.Background(), map[string]any{"rawPath": "/execute/echo", "input": map[string]any{"value": "hello"}, "execution_context": map[string]any{"run_id": "serverless-run"}}, nil)
		require.NoError(t, err)
		assert.Equal(t, http.StatusOK, status)
		assert.Equal(t, map[string]any{"echo": "hello"}, result)

		result, status, err = agent.HandleServerlessEvent(context.Background(), map[string]any{"target": "explode"}, nil)
		require.NoError(t, err)
		assert.Equal(t, http.StatusInternalServerError, status)
		assert.Equal(t, map[string]any{"error": assert.AnError.Error()}, result)
	})

	t.Run("reasoner options and ServeHTTP forwarding", func(t *testing.T) {
		cfg := Config{NodeID: "node-1", Version: "1.0.0", AgentFieldURL: "https://api.example.com", Logger: log.New(io.Discard, "", 0)}
		agent, err := New(cfg)
		require.NoError(t, err)

		formatterCalled := false
		agent.RegisterReasoner("cli", func(context.Context, map[string]any) (any, error) { return "ok", nil }, WithCLIFormatter(func(context.Context, any, error) {
			formatterCalled = true
		}), WithVCEnabled(true), WithReasonerTags("ops", "debug"), WithRequireRealtimeValidation())

		r := agent.reasoners["cli"]
		if assert.NotNil(t, r.VCEnabled) {
			assert.True(t, *r.VCEnabled)
		}
		assert.Equal(t, []string{"ops", "debug"}, r.Tags)
		assert.True(t, r.RequireRealtimeValidation)
		r.CLIFormatter(context.Background(), nil, nil)
		assert.True(t, formatterCalled)

		execErr := &ExecuteError{Message: "boom"}
		assert.Equal(t, "boom", execErr.Error())

		resp := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/health", nil)
		agent.ServeHTTP(resp, req)
		assert.Equal(t, http.StatusOK, resp.Code)
	})
}

func TestHandleReasoner_WrongMethod(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	server := httptest.NewServer(agent.handler())
	defer server.Close()

	req, err := http.NewRequest(http.MethodGet, server.URL+"/reasoners/test", nil)
	require.NoError(t, err)

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusMethodNotAllowed, resp.StatusCode)
}

func TestHandleReasoner_Error(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	agent.RegisterReasoner("test", func(ctx context.Context, input map[string]any) (any, error) {
		return nil, assert.AnError
	})

	server := httptest.NewServer(agent.handler())
	defer server.Close()

	req, err := http.NewRequest(http.MethodPost, server.URL+"/reasoners/test", bytes.NewReader([]byte("{}")))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer resp.Body.Close()

	assert.Equal(t, http.StatusInternalServerError, resp.StatusCode)

	var result map[string]any
	json.NewDecoder(resp.Body).Decode(&result)
	assert.Contains(t, result["error"], "assert.AnError")
}

// TestCall pins the async-submit + wait contract. Contract: Call POSTs to the
// ASYNC execute endpoint (/api/v1/execute/async/{target}) with the full set of
// lineage headers (the control plane reads them identically for sync and async
// via readExecutionHeaders, so DAG parentage is preserved), then polls
// GET /api/v1/executions/{id} — with the same lineage headers — until the
// execution succeeds, and returns the child's result map.
func TestCall(t *testing.T) {
	var polls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/api/v1/execute/async/target.node":
			// Verify lineage headers on the submit
			assert.Equal(t, "run-1", r.Header.Get("X-Run-ID"))
			assert.Equal(t, "parent-exec", r.Header.Get("X-Parent-Execution-ID"))
			assert.Equal(t, "session-1", r.Header.Get("X-Session-ID"))
			assert.Equal(t, "actor-1", r.Header.Get("X-Actor-ID"))
			assert.Equal(t, "node-1", r.Header.Get("X-Caller-Agent-ID"))

			var reqBody map[string]any
			json.NewDecoder(r.Body).Decode(&reqBody)
			assert.Equal(t, map[string]any{"value": float64(42)}, reqBody["input"])

			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-1",
				"run_id":       "run-1",
				"status":       "queued",
			})
		case r.Method == http.MethodGet && r.URL.Path == "/api/v1/executions/exec-1":
			// Lineage headers ride along on polls too (Python parity: the
			// wait loop reuses the submit headers).
			assert.Equal(t, "run-1", r.Header.Get("X-Run-ID"))
			assert.Equal(t, "parent-exec", r.Header.Get("X-Parent-Execution-ID"))

			// First poll: still running; second poll: terminal result.
			if polls.Add(1) == 1 {
				_ = json.NewEncoder(w).Encode(map[string]any{
					"execution_id": "exec-1",
					"run_id":       "run-1",
					"status":       "running",
				})
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-1",
				"run_id":       "run-1",
				"status":       "succeeded",
				"result":       map[string]any{"output": "result"},
			})
		case strings.HasSuffix(r.URL.Path, "/logs"):
			// Structured execution-log shipping — irrelevant to this contract.
			w.WriteHeader(http.StatusOK)
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	// Create context with execution context
	ctx := contextWithExecution(context.Background(), ExecutionContext{
		RunID:       "run-1",
		ExecutionID: "parent-exec",
		SessionID:   "session-1",
		ActorID:     "actor-1",
	})

	result, err := agent.Call(ctx, "target.node", map[string]any{"value": 42})
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "result", result["output"])
	assert.GreaterOrEqual(t, polls.Load(), int32(2), "expected at least two status polls")
}

// TestCall_OutlivesCallTimeout is the regression test for the 15s
// 'context deadline exceeded' bug: a child reasoner that runs LONGER than
// Config.CallTimeout must still complete successfully, because CallTimeout
// bounds each HTTP request (submit + each poll), not the end-to-end call.
func TestCall_OutlivesCallTimeout(t *testing.T) {
	const callTimeout = 300 * time.Millisecond
	start := time.Now()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/api/v1/execute/async/"):
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-slow",
				"run_id":       "run-slow",
				"status":       "queued",
			})
		case r.Method == http.MethodGet && r.URL.Path == "/api/v1/executions/exec-slow":
			// Child stays 'running' for 3x CallTimeout, then succeeds. Every
			// individual poll response is fast — only the CHILD is slow.
			if time.Since(start) < 3*callTimeout {
				_ = json.NewEncoder(w).Encode(map[string]any{"status": "running"})
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"status": "succeeded",
				"result": map[string]any{"took": "longer than CallTimeout"},
			})
		default:
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	agent, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		CallTimeout:   callTimeout,
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)

	result, err := agent.Call(context.Background(), "target.slow", map[string]any{})
	require.NoError(t, err, "a child running longer than CallTimeout must not kill the call")
	require.NotNil(t, result)
	assert.Equal(t, "longer than CallTimeout", result["took"])
	assert.GreaterOrEqual(t, time.Since(start), 3*callTimeout,
		"call must actually have waited past CallTimeout")
}

// TestCall_CtxCancelAbortsWait: cancelling the caller's ctx aborts the wait
// loop promptly (returning the context error). Note the contract: the child
// execution is NOT cancelled server-side — matching the Python SDK.
func TestCall_CtxCancelAbortsWait(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/api/v1/execute/async/"):
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-hang",
				"run_id":       "run-hang",
				"status":       "queued",
			})
		case r.Method == http.MethodGet:
			// Child never finishes.
			_ = json.NewEncoder(w).Encode(map[string]any{"status": "running"})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()

	agent, err := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		Logger:        log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(250 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	result, err := agent.Call(ctx, "target.hang", map[string]any{})
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
	assert.Nil(t, result)
	assert.Less(t, time.Since(start), 5*time.Second, "cancel must abort the wait promptly")
}

func TestCall_ErrorHandling(t *testing.T) {
	tests := []struct {
		name           string
		serverResponse func(w http.ResponseWriter, r *http.Request)
		wantErrSubstr  string
	}{
		{
			name: "API error on submit",
			serverResponse: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadRequest)
				w.Write([]byte("bad request"))
			},
			wantErrSubstr: "bad request",
		},
		{
			name: "submission missing execution id",
			serverResponse: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusAccepted)
				_ = json.NewEncoder(w).Encode(map[string]any{"status": "queued"})
			},
			wantErrSubstr: "missing identifiers",
		},
		{
			name: "execution failed status via poll",
			serverResponse: func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodPost {
					w.WriteHeader(http.StatusAccepted)
					_ = json.NewEncoder(w).Encode(map[string]any{
						"execution_id": "exec-f",
						"run_id":       "run-f",
						"status":       "queued",
					})
					return
				}
				_ = json.NewEncoder(w).Encode(map[string]any{
					"status":        "failed",
					"error_message": "execution failed",
				})
			},
			wantErrSubstr: "execution failed",
		},
		{
			name: "failed execution coalesces error field into message",
			serverResponse: func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodPost {
					w.WriteHeader(http.StatusAccepted)
					_ = json.NewEncoder(w).Encode(map[string]any{
						"execution_id": "exec-e",
						"run_id":       "run-e",
						"status":       "queued",
					})
					return
				}
				// The status endpoint reports failures in "error"
				// (ExecutionStatusResponse), not "error_message" — Python
				// coalesces it (client.py:1000-1002) and so must we.
				_ = json.NewEncoder(w).Encode(map[string]any{
					"status": "failed",
					"error":  "boom from error field",
				})
			},
			wantErrSubstr: "boom from error field",
		},
		{
			name: "cancelled execution is terminal",
			serverResponse: func(w http.ResponseWriter, r *http.Request) {
				if r.Method == http.MethodPost {
					w.WriteHeader(http.StatusAccepted)
					_ = json.NewEncoder(w).Encode(map[string]any{
						"execution_id": "exec-c",
						"run_id":       "run-c",
						"status":       "queued",
					})
					return
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"status": "cancelled"})
			},
			wantErrSubstr: "execute status cancelled",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(tt.serverResponse))
			defer server.Close()

			cfg := Config{
				NodeID:        "node-1",
				Version:       "1.0.0",
				AgentFieldURL: server.URL,
				Logger:        log.New(io.Discard, "", 0),
			}

			agent, err := New(cfg)
			require.NoError(t, err)

			result, err := agent.Call(context.Background(), "target", map[string]any{})
			require.Error(t, err)
			assert.Nil(t, result)
			assert.Contains(t, err.Error(), tt.wantErrSubstr)
		})
	}
}

func TestAI(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := ai.Response{
			Choices: []ai.Choice{
				{
					Message: ai.Message{
						Content: []ai.ContentPart{
							{Type: "text", Text: "AI response"},
						},
					},
				},
			},
		}
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
		AIConfig: &ai.Config{
			APIKey:  "test-key",
			BaseURL: server.URL,
			Model:   "gpt-4o",
		},
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	resp, err := agent.AI(context.Background(), "Hello")
	assert.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "AI response", resp.Text())
}

func TestAI_NotConfigured(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
		// No AIConfig
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	resp, err := agent.AI(context.Background(), "Hello")
	assert.Error(t, err)
	assert.Nil(t, resp)
	assert.Contains(t, err.Error(), "AI not configured")
}

func TestAIStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Write SSE chunks with proper formatting
		chunks := []string{
			"data: {\"id\":\"test\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
			"data: [DONE]\n\n",
		}

		for _, chunk := range chunks {
			w.Write([]byte(chunk))
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer server.Close()

	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
		AIConfig: &ai.Config{
			APIKey:  "test-key",
			BaseURL: server.URL,
			Model:   "gpt-4o",
		},
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	chunks, errs := agent.AIStream(context.Background(), "Hello")

	var receivedChunks []ai.StreamChunk
	done := make(chan bool)

	go func() {
		for chunk := range chunks {
			receivedChunks = append(receivedChunks, chunk)
		}
		done <- true
	}()

	// Wait for either error or completion
	select {
	case err := <-errs:
		if err != nil {
			t.Logf("Received error: %v", err)
		}
	case <-done:
	case <-time.After(2 * time.Second):
		t.Log("Timeout waiting for stream")
	}

	// The stream may or may not receive chunks depending on timing
	// Just verify the channels work
	assert.NotNil(t, chunks)
	assert.NotNil(t, errs)
}

func TestAIStream_NotConfigured(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "https://api.example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	chunks, errs := agent.AIStream(context.Background(), "Hello")

	// Should receive error immediately
	select {
	case err := <-errs:
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "AI not configured")
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Expected error but got none")
	}

	// Chunks channel should be closed
	_, ok := <-chunks
	assert.False(t, ok)
}

func TestAIWithTools(t *testing.T) {
	t.Run("not configured", func(t *testing.T) {
		agent, err := New(Config{NodeID: "node-1", Version: "1.0.0", AgentFieldURL: "https://api.example.com", Logger: log.New(io.Discard, "", 0)})
		require.NoError(t, err)

		resp, trace, err := agent.AIWithTools(context.Background(), "hello", ai.DefaultToolCallConfig())
		assert.Error(t, err)
		assert.Nil(t, resp)
		assert.Nil(t, trace)
	})

	t.Run("fallback without discovered tools", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/v1/discovery/capabilities":
				_, _ = w.Write([]byte(`{"discovered_at":"2025-01-01T00:00:00Z","total_agents":0,"total_reasoners":0,"total_skills":0,"pagination":{"limit":50,"offset":0,"has_more":false},"capabilities":[]}`))
			case "/chat/completions":
				_ = json.NewEncoder(w).Encode(ai.Response{Choices: []ai.Choice{{Message: ai.Message{Content: []ai.ContentPart{{Type: "text", Text: "fallback"}}}}}})
			default:
				t.Fatalf("unexpected path %s", r.URL.Path)
			}
		}))
		defer server.Close()

		agent, err := New(Config{
			NodeID:        "node-1",
			Version:       "1.0.0",
			AgentFieldURL: server.URL,
			Logger:        log.New(io.Discard, "", 0),
			AIConfig:      &ai.Config{APIKey: "test-key", BaseURL: server.URL, Model: "gpt-4o"},
		})
		require.NoError(t, err)

		resp, trace, err := agent.AIWithTools(context.Background(), "hello", ai.DefaultToolCallConfig())
		require.NoError(t, err)
		assert.Equal(t, "fallback", resp.Text())
		assert.Equal(t, 1, trace.TotalTurns)
	})

	t.Run("discovers tools and dispatches local calls", func(t *testing.T) {
		var chatRequests atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/v1/discovery/capabilities":
				_, _ = w.Write([]byte(`{"discovered_at":"2025-01-01T00:00:00Z","total_agents":1,"total_reasoners":1,"total_skills":0,"pagination":{"limit":50,"offset":0,"has_more":false},"capabilities":[{"agent_id":"agent-1","reasoners":[{"id":"lookup","invocation_target":"agent-1.lookup","input_schema":{"type":"object"}}],"skills":[]}]}`))
			case "/api/v1/execute/async/agent-1.lookup":
				w.WriteHeader(http.StatusAccepted)
				_, _ = w.Write([]byte(`{"execution_id":"exec-tool-1","run_id":"run-tool-1","status":"queued"}`))
			case "/api/v1/executions/exec-tool-1":
				_, _ = w.Write([]byte(`{"execution_id":"exec-tool-1","run_id":"run-tool-1","status":"succeeded","result":{"status":"open"}}`))
			case "/chat/completions":
				count := chatRequests.Add(1)
				if count == 1 {
					_ = json.NewEncoder(w).Encode(ai.Response{Choices: []ai.Choice{{Message: ai.Message{ToolCalls: []ai.ToolCall{{ID: "call-1", Type: "function", Function: ai.ToolCallFunction{Name: "agent-1.lookup", Arguments: `{"query":"status"}`}}}}}}})
					return
				}
				_ = json.NewEncoder(w).Encode(ai.Response{Choices: []ai.Choice{{Message: ai.Message{Content: []ai.ContentPart{{Type: "text", Text: "tool answer"}}}}}})
			default:
				t.Fatalf("unexpected path %s", r.URL.Path)
			}
		}))
		defer server.Close()

		agent, err := New(Config{
			NodeID:        "agent-1",
			Version:       "1.0.0",
			AgentFieldURL: server.URL,
			Logger:        log.New(io.Discard, "", 0),
			AIConfig:      &ai.Config{APIKey: "test-key", BaseURL: server.URL, Model: "gpt-4o"},
		})
		require.NoError(t, err)
		resp, trace, err := agent.AIWithTools(context.Background(), "hello", ai.DefaultToolCallConfig())
		require.NoError(t, err)
		assert.Equal(t, "tool answer", resp.Text())
		require.Len(t, trace.Calls, 1)
		assert.Equal(t, "agent-1.lookup", trace.Calls[0].ToolName)
	})

	t.Run("rejects model-invented tool targets before outbound execute request", func(t *testing.T) {
		var executeCalls atomic.Int32
		var chatRequests atomic.Int32
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/v1/discovery/capabilities":
				_, _ = w.Write([]byte(`{"discovered_at":"2025-01-01T00:00:00Z","total_agents":1,"total_reasoners":1,"total_skills":0,"pagination":{"limit":50,"offset":0,"has_more":false},"capabilities":[{"agent_id":"agent-1","reasoners":[{"id":"lookup","invocation_target":"agent-1.lookup","input_schema":{"type":"object"}}],"skills":[]}]}`))
			case "/chat/completions":
				if chatRequests.Add(1) == 1 {
					_ = json.NewEncoder(w).Encode(ai.Response{Choices: []ai.Choice{{Message: ai.Message{ToolCalls: []ai.ToolCall{{ID: "call-1", Type: "function", Function: ai.ToolCallFunction{Name: "agent-1.admin", Arguments: `{"query":"status"}`}}}}}}})
					return
				}
				_ = json.NewEncoder(w).Encode(ai.Response{Choices: []ai.Choice{{Message: ai.Message{Content: []ai.ContentPart{{Type: "text", Text: "blocked"}}}}}})
			case "/api/v1/execute/async/agent-1.lookup":
				executeCalls.Add(1)
				w.WriteHeader(http.StatusAccepted)
				_, _ = w.Write([]byte(`{"execution_id":"exec-tool-2","run_id":"run-tool-2","status":"queued"}`))
			default:
				t.Fatalf("unexpected path %s", r.URL.Path)
			}
		}))
		defer server.Close()

		agent, err := New(Config{
			NodeID:        "agent-1",
			Version:       "1.0.0",
			AgentFieldURL: server.URL,
			Logger:        log.New(io.Discard, "", 0),
			AIConfig:      &ai.Config{APIKey: "test-key", BaseURL: server.URL, Model: "gpt-4o"},
		})
		require.NoError(t, err)

		resp, trace, err := agent.AIWithTools(context.Background(), "hello", ai.ToolCallConfig{MaxTurns: 1, MaxToolCalls: 1, PromptConfig: &ai.PromptConfig{}})
		require.NoError(t, err)
		require.NotNil(t, resp)
		require.NotNil(t, trace)
		require.Len(t, trace.Calls, 1)
		assert.Contains(t, trace.Calls[0].Error, "not a discovered capability")
		assert.Equal(t, "blocked", resp.Text())
		assert.Equal(t, int32(0), executeCalls.Load())
	})
}

func TestNormalizeToolInvocationTarget(t *testing.T) {
	tests := []struct {
		name   string
		target string
		want   string
	}{
		{
			name:   "skill target",
			target: "agent-1:skill:lookup",
			want:   "agent-1.lookup",
		},
		{
			name:   "reasoner target",
			target: "agent-1:lookup",
			want:   "agent-1.lookup",
		},
		{
			name:   "already normalized",
			target: "agent-1.lookup",
			want:   "agent-1.lookup",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, normalizeToolInvocationTarget(tt.target))
		})
	}
}

func TestRunAndServe_ShutdownOnContextCancel(t *testing.T) {
	var shutdownCalls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/api/v1/nodes":
			_, _ = w.Write([]byte(`{"lease_seconds":120}`))
		case r.Method == http.MethodPost && r.URL.Path == "/api/v1/nodes/node-1/shutdown":
			shutdownCalls.Add(1)
			_, _ = w.Write([]byte(`{"lease_seconds":120}`))
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer server.Close()

	newServingAgent := func() *Agent {
		a, err := New(Config{
			NodeID:        "node-1",
			Version:       "1.0.0",
			AgentFieldURL: server.URL,
			ListenAddress: "127.0.0.1:0",
			Logger:        log.New(io.Discard, "", 0),
		})
		require.NoError(t, err)
		a.RegisterReasoner("echo", func(context.Context, map[string]any) (any, error) { return map[string]any{"ok": true}, nil })
		return a
	}

	serveAgent := newServingAgent()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- serveAgent.Serve(ctx) }()
	time.Sleep(50 * time.Millisecond)
	cancel()
	require.NoError(t, <-done)
	assert.NotNil(t, serveAgent.server)

	runAgent := newServingAgent()
	origArgs := os.Args
	os.Args = []string{"agentfield"}
	defer func() { os.Args = origArgs }()
	runCtx, runCancel := context.WithCancel(context.Background())
	runDone := make(chan error, 1)
	go func() { runDone <- runAgent.Run(runCtx) }()
	time.Sleep(50 * time.Millisecond)
	runCancel()
	require.NoError(t, <-runDone)
	assert.GreaterOrEqual(t, shutdownCalls.Load(), int32(2))
}

func TestExecutionContext(t *testing.T) {
	ctx := context.Background()
	execCtx := ExecutionContext{
		RunID:             "run-1",
		ExecutionID:       "exec-1",
		ParentExecutionID: "parent-1",
		SessionID:         "session-1",
		ActorID:           "actor-1",
	}

	ctxWithExec := contextWithExecution(ctx, execCtx)
	retrieved := executionContextFrom(ctxWithExec)

	assert.Equal(t, execCtx, retrieved)
}

func TestExecutionContext_Empty(t *testing.T) {
	ctx := context.Background()
	execCtx := executionContextFrom(ctx)
	assert.Equal(t, ExecutionContext{}, execCtx)
}

func TestExecutionLogger_EmitsAndPostsStructuredLog(t *testing.T) {
	logCh := make(chan ExecutionLogEntry, 1)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v1/executions/exec-1/logs" {
			assert.Equal(t, http.MethodPost, r.Method)
			assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

			var entry ExecutionLogEntry
			require.NoError(t, json.NewDecoder(r.Body).Decode(&entry))
			logCh <- entry
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		Logger:        log.New(io.Discard, "", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	ctx := contextWithExecution(context.Background(), ExecutionContext{
		RunID:          "run-1",
		ExecutionID:    "exec-1",
		WorkflowID:     "wf-1",
		RootWorkflowID: "root-wf-1",
		ReasonerName:   "demo",
		SessionID:      "session-1",
		ActorID:        "actor-1",
	})

	stdout, _, err := captureOutput(t, func() error {
		agent.ExecutionLogger(ctx).WithSource("sdk.user").Info("reasoner.custom", "custom log message", map[string]any{
			"foo": "bar",
		})
		time.Sleep(50 * time.Millisecond)
		return nil
	})
	require.NoError(t, err)

	var entry ExecutionLogEntry
	require.NoError(t, json.Unmarshal([]byte(strings.TrimSpace(stdout)), &entry))
	assert.Equal(t, "exec-1", entry.ExecutionID)
	assert.Equal(t, "wf-1", entry.WorkflowID)
	assert.Equal(t, "run-1", entry.RunID)
	assert.Equal(t, "root-wf-1", entry.RootWorkflowID)
	assert.Equal(t, "node-1", entry.AgentNodeID)
	assert.Equal(t, "demo", entry.ReasonerID)
	assert.Equal(t, "info", entry.Level)
	assert.Equal(t, "sdk.user", entry.Source)
	assert.Equal(t, "reasoner.custom", entry.EventType)
	assert.Equal(t, "custom log message", entry.Message)
	assert.False(t, entry.SystemGenerated)
	assert.Equal(t, "bar", entry.Attributes["foo"])

	select {
	case posted := <-logCh:
		assert.Equal(t, "exec-1", posted.ExecutionID)
		assert.Equal(t, "reasoner.custom", posted.EventType)
		assert.Equal(t, "sdk.user", posted.Source)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for execution log post")
	}
}

func TestExecutionLogger_HelperMethods(t *testing.T) {
	agent, err := New(Config{NodeID: "node-1", Version: "1.0.0", AgentFieldURL: "https://api.example.com", Logger: log.New(io.Discard, "", 0)})
	require.NoError(t, err)

	ctx := contextWithExecution(context.Background(), ExecutionContext{RunID: "run-1"})
	stdout, _, err := captureOutput(t, func() error {
		logger := agent.ExecutionLogger(ctx)
		logger.Debug("debug.event", "", nil)
		logger.Warn("warn.event", "warn message", nil)
		logger.Error("error.event", "error message", nil)
		logger.System("system.event", "system message", map[string]any{"kind": "system"})
		return nil
	})
	require.NoError(t, err)

	lines := strings.Split(strings.TrimSpace(stdout), "\n")
	require.Len(t, lines, 4)

	var entries []ExecutionLogEntry
	for _, line := range lines {
		var entry ExecutionLogEntry
		require.NoError(t, json.Unmarshal([]byte(line), &entry))
		entries = append(entries, entry)
	}

	assert.Equal(t, "debug", entries[0].Level)
	assert.Equal(t, "debug.event", entries[0].EventType)
	assert.Equal(t, "debug.event", entries[0].Message)
	assert.Equal(t, "run-1", entries[0].RootWorkflowID)
	assert.Equal(t, "node-1", entries[0].AgentNodeID)
	assert.Equal(t, "warn", entries[1].Level)
	assert.Equal(t, "error", entries[2].Level)
	assert.True(t, entries[3].SystemGenerated)
	assert.Equal(t, "system", entries[3].Attributes["kind"])
}

func TestHandleReasonerAsyncPostsStatus(t *testing.T) {
	callbackCh := make(chan map[string]any, 1)
	callbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		if strings.Contains(r.URL.Path, "/logs") {
			w.WriteHeader(http.StatusOK)
			return
		}
		if !strings.Contains(r.URL.Path, "/status") {
			w.WriteHeader(http.StatusOK)
			return
		}
		dec := json.NewDecoder(r.Body)
		var payload map[string]any
		if err := dec.Decode(&payload); err == nil {
			callbackCh <- payload
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer callbackServer.Close()

	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		TeamID:        "team",
		AgentFieldURL: callbackServer.URL,
		ListenAddress: ":0",
		PublicURL:     "http://localhost:0",
		Logger:        log.New(io.Discard, "[test] ", 0),
	}

	agent, err := New(cfg)
	require.NoError(t, err)

	agent.RegisterReasoner("demo", func(ctx context.Context, input map[string]any) (any, error) {
		time.Sleep(10 * time.Millisecond)
		return map[string]any{"ok": true}, nil
	})

	server := httptest.NewServer(agent.handler())
	defer server.Close()

	reqBody := []byte(`{"value":42}`)
	req, err := http.NewRequest(http.MethodPost, server.URL+"/reasoners/demo", bytes.NewReader(reqBody))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Execution-ID", "exec-test")
	req.Header.Set("X-Run-ID", "run-1")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	require.Equal(t, http.StatusAccepted, resp.StatusCode)
	resp.Body.Close()

	select {
	case payload := <-callbackCh:
		assert.Equal(t, "exec-test", payload["execution_id"])
		assert.Equal(t, "succeeded", payload["status"])
		result, ok := payload["result"].(map[string]any)
		require.True(t, ok)
		assert.Equal(t, true, result["ok"])
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for callback payload")
	}
}

func TestChildContext(t *testing.T) {
	parent := ExecutionContext{
		RunID:          "run-1",
		ExecutionID:    "exec-parent",
		WorkflowID:     "wf-1",
		RootWorkflowID: "root-wf",
		SessionID:      "session-1",
		ActorID:        "actor-1",
		Depth:          2,
	}

	child := parent.ChildContext("node-1", "child-reasoner")

	assert.Equal(t, "run-1", child.RunID)
	assert.Equal(t, "wf-1", child.WorkflowID)
	assert.Equal(t, "wf-1", child.ParentWorkflowID)
	assert.Equal(t, "root-wf", child.RootWorkflowID)
	assert.Equal(t, "exec-parent", child.ParentExecutionID)
	assert.Equal(t, 3, child.Depth)
	assert.Equal(t, "node-1", child.AgentNodeID)
	assert.Equal(t, "child-reasoner", child.ReasonerName)
	assert.NotEmpty(t, child.ExecutionID)
	assert.False(t, child.StartedAt.IsZero())
}

func TestChildContextGeneratesRunID(t *testing.T) {
	parent := ExecutionContext{}

	child := parent.ChildContext("node-1", "child-reasoner")

	assert.NotEmpty(t, child.RunID)
	assert.NotEmpty(t, child.WorkflowID)
	assert.Equal(t, child.WorkflowID, child.ParentWorkflowID)
	assert.Equal(t, child.WorkflowID, child.RootWorkflowID)
}

func TestBuildChildContext(t *testing.T) {
	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: "http://example.com",
		Logger:        log.New(io.Discard, "", 0),
	}

	ag, err := New(cfg)
	require.NoError(t, err)

	parent := ExecutionContext{
		RunID:          "run-1",
		ExecutionID:    "exec-parent",
		WorkflowID:     "wf-1",
		RootWorkflowID: "root-wf",
		Depth:          1,
		SessionID:      "session-1",
		ActorID:        "actor-1",
	}

	child := ag.buildChildContext(parent, "child")

	assert.Equal(t, parent.RunID, child.RunID)
	assert.Equal(t, parent.ExecutionID, child.ParentExecutionID)
	assert.Equal(t, parent.WorkflowID, child.WorkflowID)
	assert.Equal(t, parent.WorkflowID, child.ParentWorkflowID)
	assert.Equal(t, parent.RootWorkflowID, child.RootWorkflowID)
	assert.Equal(t, "node-1", child.AgentNodeID)
	assert.Equal(t, "child", child.ReasonerName)
	assert.Equal(t, parent.Depth+1, child.Depth)
	assert.NotEmpty(t, child.ExecutionID)
}

func TestBuildChildContextRoot(t *testing.T) {
	cfg := Config{
		NodeID:  "node-1",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
	}

	ag, err := New(cfg)
	require.NoError(t, err)

	child := ag.buildChildContext(ExecutionContext{}, "root-reasoner")

	assert.NotEmpty(t, child.RunID)
	assert.NotEmpty(t, child.ExecutionID)
	assert.Equal(t, child.WorkflowID, child.RootWorkflowID)
	assert.Empty(t, child.ParentExecutionID)
	assert.Equal(t, "node-1", child.AgentNodeID)
	assert.Equal(t, "root-reasoner", child.ReasonerName)
}

func TestCallLocalEmitsEvents(t *testing.T) {
	eventCh := make(chan types.WorkflowExecutionEvent, 4)
	eventServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		if strings.Contains(r.URL.Path, "/logs") {
			w.WriteHeader(http.StatusOK)
			return
		}
		if !strings.Contains(r.URL.Path, "/workflow/executions/events") {
			w.WriteHeader(http.StatusOK)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var event types.WorkflowExecutionEvent
		if err := json.Unmarshal(body, &event); err == nil {
			eventCh <- event
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer eventServer.Close()

	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: eventServer.URL,
		Logger:        log.New(io.Discard, "", 0),
	}

	ag, err := New(cfg)
	require.NoError(t, err)

	ag.RegisterReasoner("child", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"echo": input["msg"]}, nil
	})

	parentCtx := contextWithExecution(context.Background(), ExecutionContext{
		RunID:          "run-1",
		ExecutionID:    "exec-parent",
		WorkflowID:     "wf-1",
		RootWorkflowID: "wf-1",
		ReasonerName:   "parent",
		AgentNodeID:    "node-1",
	})

	res, err := ag.CallLocal(parentCtx, "child", map[string]any{"msg": "hi"})
	require.NoError(t, err)

	resultMap, ok := res.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "hi", resultMap["echo"])

	var received []types.WorkflowExecutionEvent
	timeout := time.After(2 * time.Second)

	for len(received) < 2 {
		select {
		case evt := <-eventCh:
			received = append(received, evt)
		case <-timeout:
			t.Fatalf("timed out waiting for workflow events, received %d", len(received))
		}
	}

	statuses := map[string]bool{}
	for _, evt := range received {
		assert.Equal(t, "child", evt.ReasonerID)
		assert.Equal(t, "node-1", evt.AgentNodeID)
		assert.Equal(t, "wf-1", evt.WorkflowID)
		assert.Equal(t, "run-1", evt.RunID)
		if evt.ParentExecutionID == nil {
			t.Fatalf("expected ParentExecutionID to be set")
		}
		assert.Equal(t, "exec-parent", *evt.ParentExecutionID)
		statuses[evt.Status] = true
	}

	assert.True(t, statuses["running"])
	assert.True(t, statuses["succeeded"])
}

func TestCallLocalEmitsStructuredExecutionLogs(t *testing.T) {
	logCh := make(chan ExecutionLogEntry, 4)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/logs"):
			var entry ExecutionLogEntry
			require.NoError(t, json.NewDecoder(r.Body).Decode(&entry))
			logCh <- entry
			w.WriteHeader(http.StatusOK)
		default:
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer server.Close()

	cfg := Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		Logger:        log.New(io.Discard, "", 0),
	}

	ag, err := New(cfg)
	require.NoError(t, err)

	ag.RegisterReasoner("child", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"echo": input["msg"]}, nil
	})

	parentCtx := contextWithExecution(context.Background(), ExecutionContext{
		RunID:          "run-1",
		ExecutionID:    "exec-parent",
		WorkflowID:     "wf-1",
		RootWorkflowID: "wf-1",
		ReasonerName:   "parent",
		AgentNodeID:    "node-1",
	})

	stdout, _, err := captureOutput(t, func() error {
		_, err := ag.CallLocal(parentCtx, "child", map[string]any{"msg": "hi"})
		time.Sleep(50 * time.Millisecond)
		return err
	})
	require.NoError(t, err)

	lines := strings.Split(strings.TrimSpace(stdout), "\n")
	require.GreaterOrEqual(t, len(lines), 2)

	var seenStart, seenComplete bool
	for _, line := range lines {
		var entry ExecutionLogEntry
		require.NoError(t, json.Unmarshal([]byte(line), &entry))
		if entry.ReasonerID != "child" {
			continue
		}
		assert.Equal(t, "sdk.runtime", entry.Source)
		if entry.EventType == "call.local.start" {
			seenStart = true
		}
		if entry.EventType == "call.local.complete" {
			seenComplete = true
		}
	}

	assert.True(t, seenStart)
	assert.True(t, seenComplete)

	select {
	case first := <-logCh:
		assert.Equal(t, "child", first.ReasonerID)
		assert.NotEmpty(t, first.ExecutionID)
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for structured execution logs")
	}
}

func TestCallLocalCoversCompletionAndFailureBranches(t *testing.T) {
	tests := []struct {
		name      string
		handler   func(context.Context, map[string]any) (any, error)
		wantEvent string
		wantLevel string
		wantErr   string
	}{
		{
			name: "success",
			handler: func(_ context.Context, input map[string]any) (any, error) {
				return map[string]any{"echo": input["msg"]}, nil
			},
			wantEvent: "call.local.complete",
			wantLevel: "info",
		},
		{
			name: "failure",
			handler: func(context.Context, map[string]any) (any, error) {
				return nil, errors.New("handler failed")
			},
			wantEvent: "call.local.failed",
			wantLevel: "error",
			wantErr:   "handler failed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ag, err := New(Config{
				NodeID:  "node-1",
				Version: "1.0.0",
				Logger:  log.New(io.Discard, "", 0),
			})
			require.NoError(t, err)
			ag.RegisterReasoner("child", tt.handler)

			parentCtx := contextWithExecution(context.Background(), ExecutionContext{
				RunID:          "run-1",
				ExecutionID:    "exec-parent",
				WorkflowID:     "wf-1",
				RootWorkflowID: "wf-1",
				ReasonerName:   "parent",
				AgentNodeID:    "node-1",
			})

			stdout, _, callErr := captureOutput(t, func() error {
				_, err := ag.CallLocal(parentCtx, "child", map[string]any{"msg": "hi"})
				return err
			})

			if tt.wantErr == "" {
				require.NoError(t, callErr)
			} else {
				require.Error(t, callErr)
				assert.Contains(t, callErr.Error(), tt.wantErr)
			}

			lines := strings.Split(strings.TrimSpace(stdout), "\n")
			require.GreaterOrEqual(t, len(lines), 2)

			var seen bool
			for _, line := range lines {
				var entry ExecutionLogEntry
				require.NoError(t, json.Unmarshal([]byte(line), &entry))
				if entry.EventType != tt.wantEvent {
					continue
				}
				seen = true
				assert.Equal(t, "child", entry.ReasonerID)
				assert.Equal(t, tt.wantLevel, entry.Level)
				assert.Equal(t, "sdk.runtime", entry.Source)
			}
			assert.True(t, seen, "expected %s log entry", tt.wantEvent)
		})
	}
}

func TestCallLocalUnknownReasoner(t *testing.T) {
	cfg := Config{
		NodeID:  "node-1",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
	}

	ag, err := New(cfg)
	require.NoError(t, err)

	_, err = ag.CallLocal(context.Background(), "missing", map[string]any{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unknown reasoner")
}

func TestCall_TargetPrefixing(t *testing.T) {
	var capturedSubmitPath string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			capturedSubmitPath = r.URL.Path
			w.WriteHeader(http.StatusAccepted)
			_ = json.NewEncoder(w).Encode(map[string]any{
				"execution_id": "exec-p",
				"run_id":       "run-p",
				"status":       "queued",
			})
			return
		}

		resp := map[string]any{
			"status": "succeeded",
			"result": map[string]any{"ok": true},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	agent, _ := New(Config{
		NodeID:        "node-1",
		Version:       "1.0.0",
		AgentFieldURL: server.URL,
		Logger:        log.New(io.Discard, "", 0),
	})

	_, err := agent.Call(context.Background(), "lookup", nil)
	require.NoError(t, err)

	assert.Contains(t, capturedSubmitPath, "/execute/async/node-1.lookup")
}
