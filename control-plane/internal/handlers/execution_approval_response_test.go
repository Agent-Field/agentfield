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

func resolveApprovalRouter(store ExecutionStore) *gin.Engine {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/approval-response", ResolveApprovalHandler(store))
	return router
}

func postResolveApproval(t *testing.T, router *gin.Engine, executionID string, body map[string]any) *httptest.ResponseRecorder {
	t.Helper()

	encoded, err := json.Marshal(body)
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/"+executionID+"/approval-response", bytes.NewReader(encoded))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	return resp
}

// Contract: approving a waiting execution transitions it back to running and
// records the decision, exactly like the HMAC webhook path.
func TestResolveApproval_Approved(t *testing.T) {
	store := seedWaitingExecution(t, "exec-1", "agent-1", "req-abc")
	router := resolveApprovalRouter(store)

	resp := postResolveApproval(t, router, "exec-1", map[string]any{
		"decision": "approved",
		"reason":   "Looks good!",
	})

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "processed", result["status"])
	assert.Equal(t, "exec-1", result["execution_id"])
	assert.Equal(t, "approved", result["decision"])
	assert.Equal(t, types.ExecutionStatusRunning, result["new_status"])

	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-1")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusRunning, wfExec.Status)
	require.NotNil(t, wfExec.ApprovalStatus)
	assert.Equal(t, "approved", *wfExec.ApprovalStatus)
	assert.NotNil(t, wfExec.ApprovalRespondedAt)
	require.NotNil(t, wfExec.ApprovalResponse)
	assert.Contains(t, *wfExec.ApprovalResponse, "Looks good!")

	execRecord, err := store.GetExecutionRecord(context.Background(), "exec-1")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusRunning, execRecord.Status)
}

// Contract: rejecting cancels the execution and stamps completion.
func TestResolveApproval_Rejected(t *testing.T) {
	store := seedWaitingExecution(t, "exec-1", "agent-1", "req-abc")
	router := resolveApprovalRouter(store)

	resp := postResolveApproval(t, router, "exec-1", map[string]any{
		"decision": "rejected",
		"reason":   "not safe",
	})

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "processed", result["status"])
	assert.Equal(t, types.ExecutionStatusCancelled, result["new_status"])

	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-1")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusCancelled, wfExec.Status)
	assert.NotNil(t, wfExec.CompletedAt)
	require.NotNil(t, wfExec.StatusReason)
	assert.Contains(t, *wfExec.StatusReason, "approval_rejected")
	assert.Contains(t, *wfExec.StatusReason, "not safe")
}

// Contract: decision synonyms accepted by the webhook normalize here too.
func TestResolveApproval_NormalizesDecisionSynonyms(t *testing.T) {
	store := seedWaitingExecution(t, "exec-1", "agent-1", "req-abc")
	router := resolveApprovalRouter(store)

	resp := postResolveApproval(t, router, "exec-1", map[string]any{"decision": "approve"})

	require.Equal(t, http.StatusOK, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "approved", result["decision"])
}

// Contract: an unknown decision is a 400 with a structured error code.
func TestResolveApproval_InvalidDecision(t *testing.T) {
	store := seedWaitingExecution(t, "exec-1", "agent-1", "req-abc")
	router := resolveApprovalRouter(store)

	resp := postResolveApproval(t, router, "exec-1", map[string]any{"decision": "maybe"})

	require.Equal(t, http.StatusBadRequest, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "invalid_decision", result["error"])
}

// Contract: "expired" is not an operator decision on this endpoint.
func TestResolveApproval_ExpiredRejected(t *testing.T) {
	store := seedWaitingExecution(t, "exec-1", "agent-1", "req-abc")
	router := resolveApprovalRouter(store)

	resp := postResolveApproval(t, router, "exec-1", map[string]any{"decision": "expired"})

	require.Equal(t, http.StatusBadRequest, resp.Code)
}

// Contract: missing decision fails request binding with a 400.
func TestResolveApproval_MissingDecision(t *testing.T) {
	store := seedWaitingExecution(t, "exec-1", "agent-1", "req-abc")
	router := resolveApprovalRouter(store)

	resp := postResolveApproval(t, router, "exec-1", map[string]any{})

	require.Equal(t, http.StatusBadRequest, resp.Code)
}

// Contract: unknown execution ID is a 404, not a panic or 500.
func TestResolveApproval_ExecutionNotFound(t *testing.T) {
	store := seedWaitingExecution(t, "exec-1", "agent-1", "req-abc")
	router := resolveApprovalRouter(store)

	resp := postResolveApproval(t, router, "exec-missing", map[string]any{"decision": "approved"})

	require.Equal(t, http.StatusNotFound, resp.Code)
}

// Contract: an execution without a pending approval request is a 404 with the
// same no_approval_request code approval-status uses.
func TestResolveApproval_NoApprovalRequest(t *testing.T) {
	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)

	now := time.Now().UTC()
	runID := "run-1"
	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: "exec-1",
		RunID:       runID,
		AgentNodeID: "agent-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
		CreatedAt:   now,
	}))
	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: "exec-1",
		WorkflowID:  "wf-1",
		RunID:       &runID,
		AgentNodeID: "agent-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
	}))

	router := resolveApprovalRouter(store)
	resp := postResolveApproval(t, router, "exec-1", map[string]any{"decision": "approved"})

	require.Equal(t, http.StatusNotFound, resp.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.Equal(t, "no_approval_request", result["error"])
}

// Contract: resolving twice is idempotent — the second call reports
// already_processed and does not change state again.
func TestResolveApproval_Idempotent(t *testing.T) {
	store := seedWaitingExecution(t, "exec-1", "agent-1", "req-abc")
	router := resolveApprovalRouter(store)

	first := postResolveApproval(t, router, "exec-1", map[string]any{"decision": "approved"})
	require.Equal(t, http.StatusOK, first.Code)

	second := postResolveApproval(t, router, "exec-1", map[string]any{"decision": "rejected"})
	require.Equal(t, http.StatusOK, second.Code)

	var result map[string]any
	require.NoError(t, json.Unmarshal(second.Body.Bytes(), &result))
	assert.Equal(t, "already_processed", result["status"])

	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-1")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusRunning, wfExec.Status)
	require.NotNil(t, wfExec.ApprovalStatus)
	assert.Equal(t, "approved", *wfExec.ApprovalStatus)
}
