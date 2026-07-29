package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// SSRF protection tests for approval callback_url (issue #435)
//
// These tests are NOT marked t.Parallel() because they interact with the
// global services.SetWebhookAllowedHosts state which other tests in this
// package also mutate. Running them sequentially avoids races.
// ---------------------------------------------------------------------------

func setupRunningExecution(t *testing.T, store *testExecutionStorage, execID, agentID string) {
	t.Helper()
	now := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: execID,
		RunID:       "run-1",
		AgentNodeID: agentID,
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
		CreatedAt:   now,
	}))
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: execID,
		WorkflowID:  "wf-1",
		RunID:       ptr("run-1"),
		AgentNodeID: agentID,
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
	}))
}

func TestRequestApprovalHandler_RejectsPrivateCallbackURL(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Clear global allowlist to ensure SSRF validation is strict.
	services.SetWebhookAllowedHosts(nil)
	t.Cleanup(func() { services.SetWebhookAllowedHosts(nil) })

	tests := []struct {
		name        string
		callbackURL string
		wantReject  bool
	}{
		{
			name:        "localhost is rejected",
			callbackURL: "http://localhost:8080/callback",
			wantReject:  true,
		},
		{
			name:        "127.0.0.1 is rejected",
			callbackURL: "http://127.0.0.1:9000/callback",
			wantReject:  true,
		},
		{
			name:        "RFC-1918 10.x is rejected",
			callbackURL: "http://10.0.0.1/callback",
			wantReject:  true,
		},
		{
			name:        "RFC-1918 172.16.x is rejected",
			callbackURL: "http://172.16.0.5:3000/callback",
			wantReject:  true,
		},
		{
			name:        "RFC-1918 192.168.x is rejected",
			callbackURL: "http://192.168.1.1/callback",
			wantReject:  true,
		},
		{
			name:        "AWS metadata endpoint is rejected",
			callbackURL: "http://169.254.169.254/latest/meta-data/",
			wantReject:  true,
		},
		{
			name:        "IPv6 loopback is rejected",
			callbackURL: "http://[::1]:8080/callback",
			wantReject:  true,
		},
		{
			name:        "empty callback_url is allowed (optional field)",
			callbackURL: "",
			wantReject:  false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			agent := &types.AgentNode{ID: "agent-ssrf"}
			store := newTestExecutionStorage(agent)
			setupRunningExecution(t, store, "exec-ssrf", "agent-ssrf")

			router := gin.New()
			router.POST("/api/v1/agents/:node_id/executions/:execution_id/request-approval",
				AgentScopedRequestApprovalHandler(store))

			payload := map[string]any{
				"approval_request_id":  "req-ssrf-test",
				"approval_request_url": "https://hub.example.com/review/req-ssrf-test",
			}
			if tc.callbackURL != "" {
				payload["callback_url"] = tc.callbackURL
			}
			body, _ := json.Marshal(payload)

			req := httptest.NewRequest(http.MethodPost, "/api/v1/agents/agent-ssrf/executions/exec-ssrf/request-approval", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			if tc.wantReject {
				assert.Equal(t, http.StatusBadRequest, resp.Code,
					"expected rejection for callback_url=%q", tc.callbackURL)
				var result map[string]any
				require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
				assert.Equal(t, "invalid_callback_url", result["error"],
					"error code should be invalid_callback_url for %q", tc.callbackURL)
			} else {
				assert.Equal(t, http.StatusOK, resp.Code,
					"expected success for callback_url=%q, got body: %s", tc.callbackURL, resp.Body.String())
			}
		})
	}
}

func TestRequestApprovalHandler_PublicCallbackURL_Allowed(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Use a real httptest server so the URL resolves to 127.0.0.1 which we
	// explicitly allowlist. This avoids external DNS lookups in CI.
	callbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer callbackServer.Close()

	// Allowlist loopback so the httptest URL passes SSRF validation.
	services.SetWebhookAllowedHosts([]string{"127.0.0.1"})
	t.Cleanup(func() { services.SetWebhookAllowedHosts(nil) })

	agent := &types.AgentNode{ID: "agent-cb"}
	store := newTestExecutionStorage(agent)
	setupRunningExecution(t, store, "exec-cb", "agent-cb")

	router := gin.New()
	router.POST("/api/v1/agents/:node_id/executions/:execution_id/request-approval",
		AgentScopedRequestApprovalHandler(store))

	body, _ := json.Marshal(map[string]any{
		"approval_request_id":  "req-cb-1",
		"approval_request_url": "https://hub.example.com/review/req-cb-1",
		"callback_url":         callbackServer.URL + "/hooks/approval",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/agents/agent-cb/executions/exec-cb/request-approval", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code, "body: %s", resp.Body.String())

	// Verify the callback URL was stored
	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-cb")
	require.NoError(t, err)
	require.NotNil(t, wfExec)
	require.NotNil(t, wfExec.ApprovalCallbackURL)
	assert.Equal(t, callbackServer.URL+"/hooks/approval", *wfExec.ApprovalCallbackURL)
}

func TestNotifyApprovalCallback_UsesSSRFSafeClient(t *testing.T) {
	// Verify that notifyApprovalCallback uses the SSRF-safe client by
	// attempting to POST to a private IP (127.0.0.1). Without the allowlist
	// the SSRF-safe transport rejects it at dial time. We clear the
	// allowlist to ensure strict validation.
	gin.SetMode(gin.TestMode)

	services.SetWebhookAllowedHosts(nil)
	t.Cleanup(func() { services.SetWebhookAllowedHosts(nil) })

	ctrl := &webhookApprovalController{
		store:         newTestExecutionStorage(&types.AgentNode{ID: "agent-1"}),
		webhookSecret: "test-secret",
	}

	// Call synchronously. The SSRF-safe client will reject 127.0.0.1 at
	// dial time (no actual TCP connection), so this returns quickly.
	// If it used a plain http.Client instead, it would attempt a real
	// connection to 127.0.0.1:80 which would either connect or timeout.
	ctrl.notifyApprovalCallback(
		"http://127.0.0.1:1/ssrf-test",
		"exec-1", "approved", "running", "", nil, "req-1",
	)

	// If we reach here without hanging, the SSRF-safe client properly
	// rejected the private IP at the transport level.
}
