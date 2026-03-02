package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// RequestApprovalHandler
// ---------------------------------------------------------------------------

func ptr(s string) *string { return &s }

func TestRequestApprovalHandler_Success(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)

	// Seed a running execution
	now := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: "exec-1",
		RunID:       "run-1",
		AgentNodeID: "agent-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
		CreatedAt:   now,
	}))
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: "exec-1",
		WorkflowID:  "wf-1",
		RunID:       ptr("run-1"),
		AgentNodeID: "agent-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
	}))

	router := gin.New()
	router.POST("/api/v1/agents/:node_id/executions/:execution_id/request-approval",
		AgentScopedRequestApprovalHandler(store))

	body, _ := json.Marshal(map[string]any{
		"approval_request_id":  "req-abc-123",
		"approval_request_url": "https://hub.example.com/review/req-abc-123",
		"callback_url":         "https://agent.example.com/webhooks/approval",
		"expires_in_hours":     24,
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/agents/agent-1/executions/exec-1/request-approval", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "req-abc-123", result["approval_request_id"])
	assert.Equal(t, "https://hub.example.com/review/req-abc-123", result["approval_request_url"])
	assert.Equal(t, "pending", result["status"])

	// Verify execution state transitioned to waiting
	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-1")
	require.NoError(t, err)
	require.NotNil(t, wfExec)
	assert.Equal(t, types.ExecutionStatusWaiting, wfExec.Status)
	assert.NotNil(t, wfExec.ApprovalRequestID)
	assert.Equal(t, "req-abc-123", *wfExec.ApprovalRequestID)
	assert.NotNil(t, wfExec.ApprovalRequestURL)
	assert.Equal(t, "https://hub.example.com/review/req-abc-123", *wfExec.ApprovalRequestURL)
	assert.NotNil(t, wfExec.ApprovalStatus)
	assert.Equal(t, "pending", *wfExec.ApprovalStatus)
	assert.NotNil(t, wfExec.ApprovalRequestedAt)
	assert.NotNil(t, wfExec.ApprovalCallbackURL)
	assert.Equal(t, "https://agent.example.com/webhooks/approval", *wfExec.ApprovalCallbackURL)
	assert.NotNil(t, wfExec.ApprovalExpiresAt)

	// Verify lightweight execution record also transitioned
	execRecord, err := store.GetExecutionRecord(context.Background(), "exec-1")
	require.NoError(t, err)
	require.NotNil(t, execRecord)
	assert.Equal(t, types.ExecutionStatusWaiting, execRecord.Status)

	// Verify status reason
	require.NotNil(t, wfExec.StatusReason)
	assert.Equal(t, "waiting_for_approval", *wfExec.StatusReason)
}

func TestRequestApprovalHandler_MissingApprovalRequestID(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)

	// The execution must exist so ownership check passes and we reach body validation
	now := time.Now().UTC()
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: "exec-1",
		WorkflowID:  "wf-1",
		AgentNodeID: "agent-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
	}))

	router := gin.New()
	router.POST("/api/v1/agents/:node_id/executions/:execution_id/request-approval",
		AgentScopedRequestApprovalHandler(store))

	// Missing required approval_request_id
	body, _ := json.Marshal(map[string]any{
		"approval_request_url": "https://hub.example.com/review",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/agents/agent-1/executions/exec-1/request-approval", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusBadRequest, resp.Code)
}

func TestRequestApprovalHandler_ExecutionNotFound(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(&types.AgentNode{ID: "agent-1"})

	router := gin.New()
	router.POST("/api/v1/agents/:node_id/executions/:execution_id/request-approval",
		AgentScopedRequestApprovalHandler(store))

	body, _ := json.Marshal(map[string]any{
		"approval_request_id": "req-abc",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/agents/agent-1/executions/nonexistent/request-approval", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusNotFound, resp.Code)
}

func TestRequestApprovalHandler_ExecutionNotRunning(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)

	now := time.Now().UTC()
	// Execution in "succeeded" state — cannot request approval
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: "exec-done",
		WorkflowID:  "wf-1",
		AgentNodeID: "agent-1",
		Status:      types.ExecutionStatusSucceeded,
		StartedAt:   now,
	}))

	router := gin.New()
	router.POST("/api/v1/agents/:node_id/executions/:execution_id/request-approval",
		AgentScopedRequestApprovalHandler(store))

	body, _ := json.Marshal(map[string]any{
		"approval_request_id": "req-abc",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/agents/agent-1/executions/exec-done/request-approval", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusConflict, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "invalid_state", result["error"])
}

func TestRequestApprovalHandler_DuplicateRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)

	now := time.Now().UTC()
	existingReqID := "req-existing"
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID:       "exec-dup",
		WorkflowID:        "wf-1",
		AgentNodeID:       "agent-1",
		Status:            types.ExecutionStatusRunning,
		StartedAt:         now,
		ApprovalRequestID: &existingReqID,
	}))

	router := gin.New()
	router.POST("/api/v1/agents/:node_id/executions/:execution_id/request-approval",
		AgentScopedRequestApprovalHandler(store))

	body, _ := json.Marshal(map[string]any{
		"approval_request_id": "req-new",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/agents/agent-1/executions/exec-dup/request-approval", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusConflict, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "approval_already_requested", result["error"])
}

func TestRequestApprovalHandler_OwnershipMismatch(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)

	now := time.Now().UTC()
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: "exec-other",
		WorkflowID:  "wf-1",
		AgentNodeID: "agent-2", // belongs to a different agent
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
	}))

	router := gin.New()
	router.POST("/api/v1/agents/:node_id/executions/:execution_id/request-approval",
		AgentScopedRequestApprovalHandler(store))

	body, _ := json.Marshal(map[string]any{
		"approval_request_id": "req-abc",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/agents/agent-1/executions/exec-other/request-approval", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusForbidden, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "execution_ownership_mismatch", result["error"])
}

func TestRequestApprovalHandler_DefaultExpiresInHours(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)

	now := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: "exec-def",
		RunID:       "run-1",
		AgentNodeID: "agent-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
		CreatedAt:   now,
	}))
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: "exec-def",
		WorkflowID:  "wf-1",
		RunID:       ptr("run-1"),
		AgentNodeID: "agent-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
	}))

	router := gin.New()
	router.POST("/api/v1/agents/:node_id/executions/:execution_id/request-approval",
		AgentScopedRequestApprovalHandler(store))

	// No expires_in_hours — should default to 72
	body, _ := json.Marshal(map[string]any{
		"approval_request_id": "req-def",
	})

	req := httptest.NewRequest(http.MethodPost, "/api/v1/agents/agent-1/executions/exec-def/request-approval", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-def")
	require.NoError(t, err)
	require.NotNil(t, wfExec.ApprovalExpiresAt)

	// Expiry should be approximately 72 hours from now
	expectedExpiry := now.Add(72 * time.Hour)
	diff := wfExec.ApprovalExpiresAt.Sub(expectedExpiry)
	assert.Less(t, diff.Abs(), 5*time.Second, "expiry should be ~72 hours from now")
}

// ---------------------------------------------------------------------------
// GetApprovalStatusHandler
// ---------------------------------------------------------------------------

func TestGetApprovalStatusHandler_Pending(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)

	now := time.Now().UTC()
	reqID := "req-pending"
	reqURL := "https://hub.example.com/review/req-pending"
	status := "pending"
	expiresAt := now.Add(72 * time.Hour)
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID:        "exec-pend",
		WorkflowID:         "wf-1",
		AgentNodeID:        "agent-1",
		Status:             types.ExecutionStatusWaiting,
		StartedAt:          now,
		ApprovalRequestID:  &reqID,
		ApprovalRequestURL: &reqURL,
		ApprovalStatus:     &status,
		ApprovalRequestedAt: &now,
		ApprovalExpiresAt:  &expiresAt,
	}))

	router := gin.New()
	router.GET("/api/v1/agents/:node_id/executions/:execution_id/approval-status",
		AgentScopedGetApprovalStatusHandler(store))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/agents/agent-1/executions/exec-pend/approval-status", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "pending", result["status"])
	assert.Equal(t, "https://hub.example.com/review/req-pending", result["request_url"])
	assert.NotEmpty(t, result["requested_at"])
	assert.Nil(t, result["responded_at"])
	assert.NotEmpty(t, result["expires_at"])
}

func TestGetApprovalStatusHandler_Approved(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)

	now := time.Now().UTC()
	respondedAt := now.Add(1 * time.Hour)
	reqID := "req-approved"
	status := "approved"
	responseJSON := `{"decision":"approved","feedback":"Looks great!"}`
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID:         "exec-appr",
		WorkflowID:          "wf-1",
		AgentNodeID:         "agent-1",
		Status:              types.ExecutionStatusRunning,
		StartedAt:           now,
		ApprovalRequestID:   &reqID,
		ApprovalStatus:      &status,
		ApprovalResponse:    &responseJSON,
		ApprovalRequestedAt: &now,
		ApprovalRespondedAt: &respondedAt,
	}))

	router := gin.New()
	router.GET("/api/v1/agents/:node_id/executions/:execution_id/approval-status",
		AgentScopedGetApprovalStatusHandler(store))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/agents/agent-1/executions/exec-appr/approval-status", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "approved", result["status"])
	assert.NotNil(t, result["response"])
	assert.NotNil(t, result["responded_at"])
}

func TestGetApprovalStatusHandler_NoApprovalRequest(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)

	now := time.Now().UTC()
	// Execution exists but has no approval request
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: "exec-no-approval",
		WorkflowID:  "wf-1",
		AgentNodeID: "agent-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
	}))

	router := gin.New()
	router.GET("/api/v1/agents/:node_id/executions/:execution_id/approval-status",
		AgentScopedGetApprovalStatusHandler(store))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/agents/agent-1/executions/exec-no-approval/approval-status", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusNotFound, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "no_approval_request", result["error"])
}

func TestGetApprovalStatusHandler_ExecutionNotFound(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := newTestExecutionStorage(&types.AgentNode{ID: "agent-1"})

	router := gin.New()
	router.GET("/api/v1/agents/:node_id/executions/:execution_id/approval-status",
		AgentScopedGetApprovalStatusHandler(store))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/agents/agent-1/executions/nonexistent/approval-status", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusNotFound, resp.Code)
}
