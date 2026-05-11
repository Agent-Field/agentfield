package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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
// UpdateAwaiterStatusHandler — multi-hop pause propagation
//
// These tests pin the state-machine semantics of the awaiter-status endpoint
// that fixes the case where a 3+-deep call chain (e.g. implement_from_issue
// → build → plan → run_X where only run_X explicitly app.pause()-s) times
// out at wallclock on the great-grandparent because intermediate reasoners
// stay in RUNNING while blocked on awaiting a paused descendant.
//
// The endpoint exists so the SDK's wait_for_result can push its own
// execution to WAITING when it observes its awaited child in WAITING — and
// back to RUNNING when the child resumes. The contract HAS to be tight:
//   - never override an approval-driven WAITING
//   - never re-introduce non-terminal status on a terminal execution
//   - silently no-op on benign races (so the SDK doesn't need to special-case)
// ---------------------------------------------------------------------------

func seedRunningExecution(t *testing.T, store ExecutionStore, execID, agentID string) {
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

func postAwaiterStatus(t *testing.T, store ExecutionStore, agentID, execID, status, reason string) *httptest.ResponseRecorder {
	t.Helper()
	router := gin.New()
	router.POST("/api/v1/agents/:node_id/executions/:execution_id/awaiter-status",
		UpdateAwaiterStatusHandler(store))
	body, _ := json.Marshal(map[string]any{"status": status, "reason": reason})
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/agents/"+agentID+"/executions/"+execID+"/awaiter-status",
		bytes.NewReader(body),
	)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	return resp
}

func TestUpdateAwaiterStatusHandler_RunningToWaiting(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)
	seedRunningExecution(t, store, "exec-1", "agent-1")

	resp := postAwaiterStatus(t, store, "agent-1", "exec-1", "waiting", "awaiting child exec_xyz")
	require.Equal(t, http.StatusOK, resp.Code)

	var result AwaiterStatusResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.True(t, result.Applied, "RUNNING -> WAITING must be applied")
	assert.Equal(t, "waiting", result.Status)

	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-1")
	require.NoError(t, err)
	require.NotNil(t, wfExec)
	assert.Equal(t, types.ExecutionStatusWaiting, wfExec.Status)
	require.NotNil(t, wfExec.StatusReason)
	assert.Equal(t, awaiterStatusReason, *wfExec.StatusReason,
		"awaiter-driven WAITING must mark its status_reason so the RUNNING flip can recognize it")
}

func TestUpdateAwaiterStatusHandler_AwaiterWaitingToRunning(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)
	seedRunningExecution(t, store, "exec-1", "agent-1")

	// First put it into awaiter-driven WAITING.
	resp := postAwaiterStatus(t, store, "agent-1", "exec-1", "waiting", "awaiting child")
	require.Equal(t, http.StatusOK, resp.Code)

	// Now flip back to RUNNING.
	resp = postAwaiterStatus(t, store, "agent-1", "exec-1", "running", "awaiting child")
	require.Equal(t, http.StatusOK, resp.Code)

	var result AwaiterStatusResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.True(t, result.Applied, "awaiter-driven WAITING -> RUNNING must be applied")
	assert.Equal(t, "running", result.Status)

	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-1")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusRunning, wfExec.Status)
	assert.Nil(t, wfExec.StatusReason, "RUNNING transition clears the awaiter status_reason")
}

func TestUpdateAwaiterStatusHandler_DoesNotOverrideApprovalWaiting(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)
	seedRunningExecution(t, store, "exec-1", "agent-1")

	// Simulate approval flow: WAITING with status_reason=waiting_for_approval.
	approvalReason := "waiting_for_approval"
	_, err := store.UpdateExecutionRecord(context.Background(), "exec-1", func(c *types.Execution) (*types.Execution, error) {
		c.Status = types.ExecutionStatusWaiting
		c.StatusReason = &approvalReason
		return c, nil
	})
	require.NoError(t, err)
	require.NoError(t, store.UpdateWorkflowExecution(context.Background(), "exec-1", func(c *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		c.Status = types.ExecutionStatusWaiting
		c.StatusReason = &approvalReason
		return c, nil
	}))

	// Awaiter cascade requesting RUNNING must NOT resume an approval-WAITING.
	resp := postAwaiterStatus(t, store, "agent-1", "exec-1", "running", "awaiting child")
	require.Equal(t, http.StatusOK, resp.Code)

	var result AwaiterStatusResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.False(t, result.Applied, "awaiter cascade must NOT resume an approval-driven WAITING")

	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-1")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusWaiting, wfExec.Status,
		"approval-driven WAITING must survive an awaiter-cascade RUNNING attempt")
	require.NotNil(t, wfExec.StatusReason)
	assert.Equal(t, approvalReason, *wfExec.StatusReason,
		"approval status_reason must be preserved untouched")
}

func TestUpdateAwaiterStatusHandler_TerminalExecutionIsNoOp(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)
	seedRunningExecution(t, store, "exec-1", "agent-1")

	// Mark it succeeded.
	_, err := store.UpdateExecutionRecord(context.Background(), "exec-1", func(c *types.Execution) (*types.Execution, error) {
		c.Status = types.ExecutionStatusSucceeded
		return c, nil
	})
	require.NoError(t, err)
	require.NoError(t, store.UpdateWorkflowExecution(context.Background(), "exec-1", func(c *types.WorkflowExecution) (*types.WorkflowExecution, error) {
		c.Status = types.ExecutionStatusSucceeded
		return c, nil
	}))

	// Cascade arrives after terminal — must no-op, not error.
	resp := postAwaiterStatus(t, store, "agent-1", "exec-1", "waiting", "awaiting child")
	require.Equal(t, http.StatusOK, resp.Code)

	var result AwaiterStatusResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.False(t, result.Applied)
	assert.Equal(t, "succeeded", result.Status)

	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-1")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionStatusSucceeded, wfExec.Status,
		"terminal execution must not be unwound by an awaiter cascade")
}

func TestUpdateAwaiterStatusHandler_WaitingToWaitingIsIdempotentNoOp(t *testing.T) {
	gin.SetMode(gin.TestMode)

	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)
	seedRunningExecution(t, store, "exec-1", "agent-1")

	// First cascade: RUNNING -> WAITING.
	resp := postAwaiterStatus(t, store, "agent-1", "exec-1", "waiting", "awaiting child A")
	require.Equal(t, http.StatusOK, resp.Code)

	// Second cascade (e.g. parallel app.call children, or a duplicate fire):
	// already WAITING, should no-op cleanly.
	resp = postAwaiterStatus(t, store, "agent-1", "exec-1", "waiting", "awaiting child B")
	require.Equal(t, http.StatusOK, resp.Code)
	var result AwaiterStatusResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.False(t, result.Applied)
	assert.Equal(t, "waiting", result.Status)
}

func TestUpdateAwaiterStatusHandler_BadStatus400(t *testing.T) {
	gin.SetMode(gin.TestMode)
	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)
	seedRunningExecution(t, store, "exec-1", "agent-1")

	resp := postAwaiterStatus(t, store, "agent-1", "exec-1", "succeeded", "")
	assert.Equal(t, http.StatusBadRequest, resp.Code,
		"status must be 'waiting' or 'running' — anything else is a client bug")
}

func TestUpdateAwaiterStatusHandler_RunningWithoutWaitingIsNoOp(t *testing.T) {
	gin.SetMode(gin.TestMode)
	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)
	seedRunningExecution(t, store, "exec-1", "agent-1")

	// Already RUNNING — a stray "running" cascade should no-op.
	resp := postAwaiterStatus(t, store, "agent-1", "exec-1", "running", "")
	require.Equal(t, http.StatusOK, resp.Code)
	var result AwaiterStatusResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &result))
	assert.False(t, result.Applied)
}

func TestUpdateAwaiterStatusHandler_UnknownExecution404(t *testing.T) {
	gin.SetMode(gin.TestMode)
	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)

	resp := postAwaiterStatus(t, store, "agent-1", "exec-missing", "waiting", "")
	// verifyExecutionOwnership returns 404 before reaching handler logic.
	assert.Equal(t, http.StatusNotFound, resp.Code)
}

func TestUpdateAwaiterStatusHandler_MalformedJSON400(t *testing.T) {
	gin.SetMode(gin.TestMode)
	agent := &types.AgentNode{ID: "agent-1"}
	store := newTestExecutionStorage(agent)
	seedRunningExecution(t, store, "exec-1", "agent-1")

	router := gin.New()
	router.POST("/api/v1/agents/:node_id/executions/:execution_id/awaiter-status",
		UpdateAwaiterStatusHandler(store))
	// Garbage body — should be caught by ShouldBindJSON and return 400 without
	// touching storage.
	req := httptest.NewRequest(
		http.MethodPost,
		"/api/v1/agents/agent-1/executions/exec-1/awaiter-status",
		bytes.NewReader([]byte("{not-json")),
	)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	assert.Equal(t, http.StatusBadRequest, resp.Code)
}

// awaiterStatusGetWorkflowErrorStore returns an error from GetWorkflowExecution
// so we can exercise the lookup-failure branch.
type awaiterStatusGetWorkflowErrorStore struct {
	*testExecutionStorage
}

func (s *awaiterStatusGetWorkflowErrorStore) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	return nil, errors.New("storage unavailable")
}

func TestUpdateAwaiterStatusHandler_StoreLookupFailure500(t *testing.T) {
	gin.SetMode(gin.TestMode)
	agent := &types.AgentNode{ID: "agent-1"}
	store := &awaiterStatusGetWorkflowErrorStore{
		testExecutionStorage: newTestExecutionStorage(agent),
	}
	seedRunningExecution(t, store.testExecutionStorage, "exec-1", "agent-1")

	resp := postAwaiterStatus(t, store, "agent-1", "exec-1", "waiting", "")
	assert.Equal(t, http.StatusInternalServerError, resp.Code,
		"a storage error during workflow lookup must surface as 500 so the SDK swallows it; "+
			"a silent success would mask a real CP outage")
}

// awaiterStatusUpdateRecordErrorStore returns an error from UpdateExecutionRecord
// so we can exercise the record-update failure branch.
type awaiterStatusUpdateRecordErrorStore struct {
	*testExecutionStorage
}

func (s *awaiterStatusUpdateRecordErrorStore) UpdateExecutionRecord(ctx context.Context, executionID string, mutator func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	return nil, errors.New("record update failed")
}

func TestUpdateAwaiterStatusHandler_UpdateExecutionRecordFailure500(t *testing.T) {
	gin.SetMode(gin.TestMode)
	agent := &types.AgentNode{ID: "agent-1"}
	store := &awaiterStatusUpdateRecordErrorStore{
		testExecutionStorage: newTestExecutionStorage(agent),
	}
	seedRunningExecution(t, store.testExecutionStorage, "exec-1", "agent-1")

	resp := postAwaiterStatus(t, store, "agent-1", "exec-1", "waiting", "")
	assert.Equal(t, http.StatusInternalServerError, resp.Code)
}

// awaiterStatusUpdateWorkflowErrorStore lets UpdateExecutionRecord succeed but
// fails UpdateWorkflowExecution. The handler must surface 500.
type awaiterStatusUpdateWorkflowErrorStore struct {
	*testExecutionStorage
}

func (s *awaiterStatusUpdateWorkflowErrorStore) UpdateWorkflowExecution(ctx context.Context, executionID string, mutator func(*types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	return errors.New("workflow update failed")
}

func TestUpdateAwaiterStatusHandler_UpdateWorkflowExecutionFailure500(t *testing.T) {
	gin.SetMode(gin.TestMode)
	agent := &types.AgentNode{ID: "agent-1"}
	store := &awaiterStatusUpdateWorkflowErrorStore{
		testExecutionStorage: newTestExecutionStorage(agent),
	}
	seedRunningExecution(t, store.testExecutionStorage, "exec-1", "agent-1")

	resp := postAwaiterStatus(t, store, "agent-1", "exec-1", "waiting", "")
	assert.Equal(t, http.StatusInternalServerError, resp.Code)
}
