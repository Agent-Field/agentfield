package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/sdk/go/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRegisterSkill_HTTPDiscoveryAndLocalExecution(t *testing.T) {
	a, err := New(Config{
		NodeID:  "node-1",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)

	require.Error(t, a.RegisterSkill("bad", nil))
	require.NoError(t, a.RegisterSkill("add", func(ctx context.Context, input map[string]any) (any, error) {
		return map[string]any{"ok": input["ok"]}, nil
	}, WithReasonerTags("deterministic"), WithRequireRealtimeValidation()))

	_, marked := a.realtimeValidationFunctions["add"]
	assert.True(t, marked)

	result, err := a.Execute(context.Background(), "add", map[string]any{"ok": true})
	require.NoError(t, err)
	assert.Equal(t, map[string]any{"ok": true}, result)

	resp := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/skills/add", bytes.NewReader([]byte(`{"ok":true}`)))
	a.Handler().ServeHTTP(resp, req)
	assert.Equal(t, http.StatusOK, resp.Code)
	assert.JSONEq(t, `{"ok":true}`, resp.Body.String())

	payload := a.discoveryPayload()
	skills, ok := payload["skills"].([]map[string]any)
	require.True(t, ok)
	require.Len(t, skills, 1)
	assert.Equal(t, "add", skills[0]["id"])
	assert.Equal(t, []string{"deterministic"}, skills[0]["tags"])
}

func TestHandleSkill_ErrorBranches(t *testing.T) {
	a, err := New(Config{
		NodeID:  "node-1",
		Version: "1.0.0",
		Logger:  log.New(io.Discard, "", 0),
	})
	require.NoError(t, err)
	require.NoError(t, a.RegisterSkill("fail", func(context.Context, map[string]any) (any, error) {
		return nil, assert.AnError
	}))

	for _, tc := range []struct {
		name   string
		method string
		path   string
		body   string
		status int
	}{
		{"wrong method", http.MethodGet, "/skills/fail", `{"ok":true}`, http.StatusMethodNotAllowed},
		{"missing name", http.MethodPost, "/skills/", `{"ok":true}`, http.StatusNotFound},
		{"unknown skill", http.MethodPost, "/skills/missing", `{"ok":true}`, http.StatusNotFound},
		{"bad json", http.MethodPost, "/skills/fail", `{`, http.StatusBadRequest},
		{"handler error", http.MethodPost, "/skills/fail", `{"ok":true}`, http.StatusInternalServerError},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resp := httptest.NewRecorder()
			req := httptest.NewRequest(tc.method, tc.path, bytes.NewReader([]byte(tc.body)))
			a.Handler().ServeHTTP(resp, req)
			assert.Equal(t, tc.status, resp.Code)
		})
	}
}

func TestSkillRegistrationPayloadAndServeCleanupOnInitializeFailure(t *testing.T) {
	t.Run("registers skills in node payload", func(t *testing.T) {
		var payload types.NodeRegistrationRequest
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/v1/nodes":
				require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
				w.WriteHeader(http.StatusOK)
				require.NoError(t, json.NewEncoder(w).Encode(types.NodeRegistrationResponse{
					ID:      "node-1",
					Success: true,
				}))
			case "/api/v1/nodes/node-1/status":
				w.WriteHeader(http.StatusOK)
			default:
				t.Fatalf("unexpected request %s %s", r.Method, r.URL.Path)
			}
		}))
		defer server.Close()

		a, err := New(Config{
			NodeID:                      "node-1",
			Version:                     "1.0.0",
			AgentFieldURL:               server.URL,
			ListenAddress:               "127.0.0.1:0",
			DisableLeaseLoop:            true,
			Logger:                      log.New(io.Discard, "", 0),
			RequireOriginAuth:           false,
			LocalVerification:           false,
			VerificationRefreshInterval: 0,
		})
		require.NoError(t, err)
		require.NoError(t, a.RegisterSkill("cleanup", func(context.Context, map[string]any) (any, error) {
			return map[string]any{"done": true}, nil
		}, WithReasonerTags("maintenance")))

		require.NoError(t, a.Initialize(context.Background()))
		require.Len(t, payload.Skills, 1)
		assert.Equal(t, "cleanup", payload.Skills[0].ID)
		assert.Equal(t, []string{"maintenance"}, payload.Skills[0].Tags)
	})

	t.Run("serve shuts down listener when initialize fails", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "registration failed", http.StatusServiceUnavailable)
		}))
		defer server.Close()

		a, err := New(Config{
			NodeID:        "node-1",
			Version:       "1.0.0",
			AgentFieldURL: server.URL,
			ListenAddress: "127.0.0.1:0",
			Logger:        log.New(io.Discard, "", 0),
		})
		require.NoError(t, err)
		require.NoError(t, a.RegisterSkill("cleanup", func(context.Context, map[string]any) (any, error) {
			return map[string]any{"done": true}, nil
		}))

		err = a.Serve(context.Background())
		require.Error(t, err)
		assert.Contains(t, err.Error(), "register node")
	})
}

func TestLocalVerificationMiddleware_SkillRealtimeValidationBypass(t *testing.T) {
	a := &Agent{
		realtimeValidationFunctions: map[string]struct{}{"skill": {}},
		logger:                      log.New(io.Discard, "", 0),
	}
	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})

	req := httptest.NewRequest(http.MethodPost, "/skills/skill", bytes.NewReader([]byte(`{}`)))
	resp := httptest.NewRecorder()
	a.localVerificationMiddleware(next).ServeHTTP(resp, req)
	assert.Equal(t, http.StatusNoContent, resp.Code)
}
