package handlers

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// seedWaitingExecution creates a running execution, transitions it to waiting
// with the given approval_request_id, and returns the seeded store.
func seedWaitingExecution(t *testing.T, executionID, agentNodeID, approvalRequestID string) *testExecutionStorage {
	t.Helper()

	agent := &types.AgentNode{ID: agentNodeID}
	store := newTestExecutionStorage(agent)

	now := time.Now().UTC()
	status := "pending"
	runID := "run-1"

	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: executionID,
		RunID:       "run-1",
		AgentNodeID: agentNodeID,
		Status:      types.ExecutionStatusWaiting,
		StartedAt:   now,
		CreatedAt:   now,
	}))

	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID:         executionID,
		WorkflowID:          "wf-1",
		RunID:               &runID,
		AgentNodeID:         agentNodeID,
		Status:              types.ExecutionStatusWaiting,
		StartedAt:           now,
		ApprovalRequestID:   &approvalRequestID,
		ApprovalStatus:      &status,
		ApprovalRequestedAt: &now,
	}))

	return store
}

// ---------------------------------------------------------------------------
// Flat format webhook
// ---------------------------------------------------------------------------

func TestApprovalWebhook_Approved_FlatFormat(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := seedWaitingExecution(t, "exec-1", "agent-1", "req-abc")

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, ""))

	body, _ := json.Marshal(map[string]any{
		"requestId": "req-abc",
		"decision":  "approved",
		"feedback":  "Looks good!",
		"timestamp": time.Now().UTC().Format(time.RFC3339),
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "processed", result["status"])
	assert.Equal(t, "exec-1", result["execution_id"])
	assert.Equal(t, "approved", result["decision"])
	assert.Equal(t, types.ExecutionStatusRunning, result["new_status"])

	// Verify execution transitioned back to running
	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-1")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusRunning, wfExec.Status)
	assert.NotNil(t, wfExec.ApprovalStatus)
	assert.Equal(t, "approved", *wfExec.ApprovalStatus)
	assert.NotNil(t, wfExec.ApprovalRespondedAt)
	assert.NotNil(t, wfExec.ApprovalResponse)
	// ApprovalRequestID should be preserved for approved
	assert.NotNil(t, wfExec.ApprovalRequestID)

	// Verify lightweight execution record
	execRecord, err := store.GetExecutionRecord(context.Background(), "exec-1")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusRunning, execRecord.Status)
}

func TestApprovalWebhook_Rejected_FlatFormat(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := seedWaitingExecution(t, "exec-rej", "agent-1", "req-rej")

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, ""))

	body, _ := json.Marshal(map[string]any{
		"requestId": "req-rej",
		"decision":  "rejected",
		"feedback":  "Plan needs more detail.",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "rejected", result["decision"])
	assert.Equal(t, types.ExecutionStatusCancelled, result["new_status"])

	// Verify execution transitioned to cancelled
	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-rej")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusCancelled, wfExec.Status)
	assert.NotNil(t, wfExec.CompletedAt)
	assert.NotNil(t, wfExec.DurationMS)
}

func TestApprovalWebhook_RequestChanges_FlatFormat(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := seedWaitingExecution(t, "exec-rc", "agent-1", "req-rc")

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, ""))

	body, _ := json.Marshal(map[string]any{
		"requestId": "req-rc",
		"decision":  "request_changes",
		"feedback":  "Add error handling to step 2.",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "request_changes", result["decision"])
	assert.Equal(t, types.ExecutionStatusRunning, result["new_status"])

	// Verify execution transitioned back to running
	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-rc")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusRunning, wfExec.Status)

	// request_changes clears approval fields so agent can re-request
	assert.Nil(t, wfExec.ApprovalRequestID)
	assert.Nil(t, wfExec.ApprovalRequestURL)
	assert.Nil(t, wfExec.ApprovalCallbackURL)

	// Should NOT have CompletedAt (still running)
	assert.Nil(t, wfExec.CompletedAt)
}

func TestApprovalWebhook_Expired_FlatFormat(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := seedWaitingExecution(t, "exec-exp", "agent-1", "req-exp")

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, ""))

	body, _ := json.Marshal(map[string]any{
		"requestId": "req-exp",
		"decision":  "expired",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "expired", result["decision"])
	assert.Equal(t, types.ExecutionStatusCancelled, result["new_status"])

	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-exp")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusCancelled, wfExec.Status)
	assert.NotNil(t, wfExec.CompletedAt)
}

// ---------------------------------------------------------------------------
// hax-sdk envelope format webhook
// ---------------------------------------------------------------------------

func TestApprovalWebhook_HaxSDKEnvelopeFormat(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := seedWaitingExecution(t, "exec-hax", "agent-1", "req-hax")

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, ""))

	body, _ := json.Marshal(map[string]any{
		"id":        "evt_001",
		"type":      "completed",
		"createdAt": time.Now().UTC().Format(time.RFC3339),
		"data": map[string]any{
			"requestId": "req-hax",
			"response": map[string]any{
				"decision": "approved",
				"feedback": "Approved via Response Hub",
			},
		},
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "approved", result["decision"])
	assert.Equal(t, types.ExecutionStatusRunning, result["new_status"])

	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-hax")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusRunning, wfExec.Status)
	assert.Equal(t, "approved", *wfExec.ApprovalStatus)
}

func TestApprovalWebhook_HaxSDKExpiredEvent(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := seedWaitingExecution(t, "exec-hax-exp", "agent-1", "req-hax-exp")

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, ""))

	body, _ := json.Marshal(map[string]any{
		"id":        "evt_002",
		"type":      "expired",
		"createdAt": time.Now().UTC().Format(time.RFC3339),
		"data": map[string]any{
			"requestId": "req-hax-exp",
			"response":  map[string]any{},
		},
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-hax-exp")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusCancelled, wfExec.Status)
	assert.Equal(t, "expired", *wfExec.ApprovalStatus)
}

// ---------------------------------------------------------------------------
// Idempotency
// ---------------------------------------------------------------------------

func TestApprovalWebhook_IdempotentDuplicate(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := seedWaitingExecution(t, "exec-idem", "agent-1", "req-idem")

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, ""))

	makeBody := func() []byte {
		b, _ := json.Marshal(map[string]any{
			"requestId": "req-idem",
			"decision":  "approved",
		})
		return b
	}

	// First call — should process
	req1 := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(makeBody()))
	req1.Header.Set("Content-Type", "application/json")
	resp1 := httptest.NewRecorder()
	router.ServeHTTP(resp1, req1)
	require.Equal(t, http.StatusOK, resp1.Code)

	var result1 map[string]any
	require.NoError(t, json.Unmarshal(resp1.Body.Bytes(), &result1))
	assert.Equal(t, "processed", result1["status"])

	// Second call — should be idempotent
	req2 := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(makeBody()))
	req2.Header.Set("Content-Type", "application/json")
	resp2 := httptest.NewRecorder()
	router.ServeHTTP(resp2, req2)
	require.Equal(t, http.StatusOK, resp2.Code)

	var result2 map[string]any
	require.NoError(t, json.Unmarshal(resp2.Body.Bytes(), &result2))
	assert.Equal(t, "already_processed", result2["status"])
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

func TestApprovalWebhook_MissingRequestID(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(&types.AgentNode{ID: "agent-1"})

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, ""))

	body, _ := json.Marshal(map[string]any{
		"decision": "approved",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusBadRequest, resp.Code)
}

func TestApprovalWebhook_InvalidDecision(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(&types.AgentNode{ID: "agent-1"})

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, ""))

	body, _ := json.Marshal(map[string]any{
		"requestId": "req-bad",
		"decision":  "maybe",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusBadRequest, resp.Code)
}

func TestApprovalWebhook_UnknownRequestID(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(&types.AgentNode{ID: "agent-1"})

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, ""))

	body, _ := json.Marshal(map[string]any{
		"requestId": "req-unknown",
		"decision":  "approved",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusNotFound, resp.Code)
}

// ---------------------------------------------------------------------------
// HMAC signature verification
// ---------------------------------------------------------------------------

func TestApprovalWebhook_ValidSignature(t *testing.T) {
	gin.SetMode(gin.TestMode)

	secret := "test-webhook-secret"
	store := seedWaitingExecution(t, "exec-sig", "agent-1", "req-sig")

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, secret))

	body, _ := json.Marshal(map[string]any{
		"requestId": "req-sig",
		"decision":  "approved",
	})

	// Compute HMAC-SHA256 signature
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	signature := hex.EncodeToString(mac.Sum(nil))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Webhook-Signature", signature)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "approved", result["decision"])
}

func TestApprovalWebhook_InvalidSignature(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := seedWaitingExecution(t, "exec-badsig", "agent-1", "req-badsig")

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, "real-secret"))

	body, _ := json.Marshal(map[string]any{
		"requestId": "req-badsig",
		"decision":  "approved",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Webhook-Signature", "deadbeef")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusUnauthorized, resp.Code)
}

func TestApprovalWebhook_MissingSignatureWhenRequired(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := seedWaitingExecution(t, "exec-nosig", "agent-1", "req-nosig")

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, "real-secret"))

	body, _ := json.Marshal(map[string]any{
		"requestId": "req-nosig",
		"decision":  "approved",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	// No signature header
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusUnauthorized, resp.Code)
}

func TestApprovalWebhook_HaxSignatureFormat(t *testing.T) {
	gin.SetMode(gin.TestMode)

	secret := "hax-webhook-secret"
	store := seedWaitingExecution(t, "exec-hax-sig", "agent-1", "req-hax-sig")

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, secret))

	body, _ := json.Marshal(map[string]any{
		"requestId": "req-hax-sig",
		"decision":  "approved",
	})

	// Compute hax-sdk style signature: t=timestamp,v1=signature
	timestamp := fmt.Sprintf("%d", time.Now().Unix())
	signedPayload := timestamp + "." + string(body)
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(signedPayload))
	sig := hex.EncodeToString(mac.Sum(nil))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Hax-Signature", fmt.Sprintf("t=%s,v1=%s", timestamp, sig))
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
}

// ---------------------------------------------------------------------------
// Callback notification
// ---------------------------------------------------------------------------

func TestApprovalWebhook_CallbackNotification(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)

	// Track callback
	var callbackReceived atomic.Bool
	var callbackBody map[string]any
	callbackServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callbackReceived.Store(true)
		_ = json.NewDecoder(r.Body).Decode(&callbackBody)
		w.WriteHeader(http.StatusOK)
	}))
	defer callbackServer.Close()

	now := time.Now().UTC()
	reqID := "req-cb"
	status := "pending"
	callbackURL := callbackServer.URL + "/webhooks/approval"

	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: "exec-cb",
		RunID:       "run-1",
		AgentNodeID: "agent-1",
		Status:      types.ExecutionStatusWaiting,
		StartedAt:   now,
		CreatedAt:   now,
	}))
	cbRunID := "run-1"
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID:         "exec-cb",
		WorkflowID:          "wf-1",
		RunID:               &cbRunID,
		AgentNodeID:         "agent-1",
		Status:              types.ExecutionStatusWaiting,
		StartedAt:           now,
		ApprovalRequestID:   &reqID,
		ApprovalStatus:      &status,
		ApprovalRequestedAt: &now,
		ApprovalCallbackURL: &callbackURL,
	}))

	router := gin.New()
	router.POST("/api/v1/webhooks/approval-response", ApprovalWebhookHandler(store, ""))

	body, _ := json.Marshal(map[string]any{
		"requestId": "req-cb",
		"decision":  "approved",
		"feedback":  "LGTM",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	// Wait briefly for the async callback goroutine
	time.Sleep(200 * time.Millisecond)

	assert.True(t, callbackReceived.Load(), "callback should have been received")
	assert.Equal(t, "exec-cb", callbackBody["execution_id"])
	assert.Equal(t, "approved", callbackBody["decision"])
}

// ---------------------------------------------------------------------------
// Full end-to-end flow: request → waiting → webhook → resume
// ---------------------------------------------------------------------------

func TestApprovalFlow_EndToEnd(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-e2e"}
	store := newTestExecutionStorage(agent)

	now := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: "exec-e2e",
		RunID:       "run-e2e",
		AgentNodeID: "agent-e2e",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
		CreatedAt:   now,
	}))
	e2eRunID := "run-e2e"
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: "exec-e2e",
		WorkflowID:  "wf-e2e",
		RunID:       &e2eRunID,
		AgentNodeID: "agent-e2e",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
	}))

	router := gin.New()
	router.POST("/api/v1/agents/:node_id/executions/:execution_id/request-approval",
		AgentScopedRequestApprovalHandler(store))
	router.GET("/api/v1/agents/:node_id/executions/:execution_id/approval-status",
		AgentScopedGetApprovalStatusHandler(store))
	router.POST("/api/v1/webhooks/approval-response",
		ApprovalWebhookHandler(store, ""))

	// ---- Step 1: Request approval ----
	reqBody, _ := json.Marshal(map[string]any{
		"approval_request_id":  "req-e2e-flow",
		"approval_request_url": "https://hub.example.com/review/req-e2e-flow",
		"expires_in_hours":     48,
	})
	reqApproval := httptest.NewRequest(http.MethodPost,
		"/api/v1/agents/agent-e2e/executions/exec-e2e/request-approval",
		bytes.NewReader(reqBody))
	reqApproval.Header.Set("Content-Type", "application/json")
	respApproval := httptest.NewRecorder()
	router.ServeHTTP(respApproval, reqApproval)

	require.Equal(t, http.StatusOK, respApproval.Code, "request approval should succeed")

	// ---- Step 2: Verify status is pending ----
	reqStatus := httptest.NewRequest(http.MethodGet,
		"/api/v1/agents/agent-e2e/executions/exec-e2e/approval-status", nil)
	respStatus := httptest.NewRecorder()
	router.ServeHTTP(respStatus, reqStatus)

	require.Equal(t, http.StatusOK, respStatus.Code)
	var statusResult map[string]any
	require.NoError(t, json.Unmarshal(respStatus.Body.Bytes(), &statusResult))
	assert.Equal(t, "pending", statusResult["status"])

	// Verify execution is in waiting state
	wfExec, _ := store.GetWorkflowExecution(context.Background(), "exec-e2e")
	assert.Equal(t, types.ExecutionStatusWaiting, wfExec.Status)

	// ---- Step 3: Webhook resolves the approval ----
	webhookBody, _ := json.Marshal(map[string]any{
		"requestId": "req-e2e-flow",
		"decision":  "approved",
		"feedback":  "Ship it!",
	})
	reqWebhook := httptest.NewRequest(http.MethodPost,
		"/api/v1/webhooks/approval-response",
		bytes.NewReader(webhookBody))
	reqWebhook.Header.Set("Content-Type", "application/json")
	respWebhook := httptest.NewRecorder()
	router.ServeHTTP(respWebhook, reqWebhook)

	require.Equal(t, http.StatusOK, respWebhook.Code)
	var webhookResult map[string]any
	require.NoError(t, json.Unmarshal(respWebhook.Body.Bytes(), &webhookResult))
	assert.Equal(t, "approved", webhookResult["decision"])
	assert.Equal(t, types.ExecutionStatusRunning, webhookResult["new_status"])

	// ---- Step 4: Verify final state ----
	reqStatusFinal := httptest.NewRequest(http.MethodGet,
		"/api/v1/agents/agent-e2e/executions/exec-e2e/approval-status", nil)
	respStatusFinal := httptest.NewRecorder()
	router.ServeHTTP(respStatusFinal, reqStatusFinal)

	require.Equal(t, http.StatusOK, respStatusFinal.Code)
	var finalStatus map[string]any
	require.NoError(t, json.Unmarshal(respStatusFinal.Body.Bytes(), &finalStatus))
	assert.Equal(t, "approved", finalStatus["status"])
	assert.NotNil(t, finalStatus["responded_at"])

	// Verify execution is back to running
	wfExec, _ = store.GetWorkflowExecution(context.Background(), "exec-e2e")
	assert.Equal(t, types.ExecutionStatusRunning, wfExec.Status)
	assert.Nil(t, wfExec.CompletedAt, "approved execution should not be completed")

	// Verify lightweight execution record
	execRecord, _ := store.GetExecutionRecord(context.Background(), "exec-e2e")
	assert.Equal(t, types.ExecutionStatusRunning, execRecord.Status)
}

// ---------------------------------------------------------------------------
// End-to-end: request_changes flow (agent can re-request after changes)
// ---------------------------------------------------------------------------

func TestApprovalFlow_RequestChanges_ThenReapprove(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-rc"}
	store := newTestExecutionStorage(agent)

	now := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: "exec-rc-flow",
		RunID:       "run-rc",
		AgentNodeID: "agent-rc",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
		CreatedAt:   now,
	}))
	rcRunID := "run-rc"
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: "exec-rc-flow",
		WorkflowID:  "wf-rc",
		RunID:       &rcRunID,
		AgentNodeID: "agent-rc",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
	}))

	router := gin.New()
	router.POST("/api/v1/agents/:node_id/executions/:execution_id/request-approval",
		AgentScopedRequestApprovalHandler(store))
	router.POST("/api/v1/webhooks/approval-response",
		ApprovalWebhookHandler(store, ""))

	// ---- Round 1: Request approval ----
	reqBody1, _ := json.Marshal(map[string]any{
		"approval_request_id": "req-round1",
	})
	r1 := httptest.NewRequest(http.MethodPost,
		"/api/v1/agents/agent-rc/executions/exec-rc-flow/request-approval",
		bytes.NewReader(reqBody1))
	r1.Header.Set("Content-Type", "application/json")
	w1 := httptest.NewRecorder()
	router.ServeHTTP(w1, r1)
	require.Equal(t, http.StatusOK, w1.Code)

	// ---- Round 1: Reviewer requests changes ----
	webhookBody1, _ := json.Marshal(map[string]any{
		"requestId": "req-round1",
		"decision":  "request_changes",
		"feedback":  "Add tests please",
	})
	rw1 := httptest.NewRequest(http.MethodPost,
		"/api/v1/webhooks/approval-response",
		bytes.NewReader(webhookBody1))
	rw1.Header.Set("Content-Type", "application/json")
	ww1 := httptest.NewRecorder()
	router.ServeHTTP(ww1, rw1)
	require.Equal(t, http.StatusOK, ww1.Code)

	// Execution should be back to running with cleared approval fields
	wfExec, _ := store.GetWorkflowExecution(context.Background(), "exec-rc-flow")
	assert.Equal(t, types.ExecutionStatusRunning, wfExec.Status)
	assert.Nil(t, wfExec.ApprovalRequestID, "approval fields should be cleared after request_changes")

	// ---- Round 2: Agent re-requests approval ----
	reqBody2, _ := json.Marshal(map[string]any{
		"approval_request_id": "req-round2",
	})
	r2 := httptest.NewRequest(http.MethodPost,
		"/api/v1/agents/agent-rc/executions/exec-rc-flow/request-approval",
		bytes.NewReader(reqBody2))
	r2.Header.Set("Content-Type", "application/json")
	w2 := httptest.NewRecorder()
	router.ServeHTTP(w2, r2)
	require.Equal(t, http.StatusOK, w2.Code, "second approval request should succeed after request_changes")

	// ---- Round 2: Approved this time ----
	webhookBody2, _ := json.Marshal(map[string]any{
		"requestId": "req-round2",
		"decision":  "approved",
	})
	rw2 := httptest.NewRequest(http.MethodPost,
		"/api/v1/webhooks/approval-response",
		bytes.NewReader(webhookBody2))
	rw2.Header.Set("Content-Type", "application/json")
	ww2 := httptest.NewRecorder()
	router.ServeHTTP(ww2, rw2)
	require.Equal(t, http.StatusOK, ww2.Code)

	wfExec, _ = store.GetWorkflowExecution(context.Background(), "exec-rc-flow")
	assert.Equal(t, types.ExecutionStatusRunning, wfExec.Status)
	assert.Equal(t, "approved", *wfExec.ApprovalStatus)
}
