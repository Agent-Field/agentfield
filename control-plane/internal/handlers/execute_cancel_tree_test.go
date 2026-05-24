package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// cancelTreeStorage embeds StorageProvider and overrides only the methods
// the cancel-tree handler touches. Unimplemented methods will panic if
// called — keeps the surface tight and surfaces accidental new dependencies.
type cancelTreeStorage struct {
	storage.StorageProvider

	mu                 sync.Mutex
	executionRecords   map[string]*types.Execution
	workflowExecutions map[string]*types.WorkflowExecution
	workflowEvents     []*types.WorkflowExecutionEvent
}

func newCancelTreeStorage() *cancelTreeStorage {
	return &cancelTreeStorage{
		executionRecords:   make(map[string]*types.Execution),
		workflowExecutions: make(map[string]*types.WorkflowExecution),
	}
}

func (s *cancelTreeStorage) seedExecution(execID, runID, status string, parentID *string, startOffsetMs int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().UTC().Add(time.Duration(startOffsetMs) * time.Millisecond)
	rid := runID
	s.executionRecords[execID] = &types.Execution{
		ExecutionID:       execID,
		RunID:             runID,
		AgentNodeID:       "agent-1",
		ReasonerID:        "reasoner-1",
		Status:            status,
		StartedAt:         now,
		CreatedAt:         now,
		UpdatedAt:         now,
		ParentExecutionID: parentID,
	}
	s.workflowExecutions[execID] = &types.WorkflowExecution{
		ExecutionID: execID,
		WorkflowID:  "wf-" + runID,
		RunID:       &rid,
		AgentNodeID: "agent-1",
		Status:      status,
		StartedAt:   now,
	}
}

func (s *cancelTreeStorage) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	exec, ok := s.executionRecords[executionID]
	if !ok {
		return nil, nil
	}
	clone := *exec
	return &clone, nil
}

func (s *cancelTreeStorage) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	wfExec, ok := s.workflowExecutions[executionID]
	if !ok {
		return nil, nil
	}
	clone := *wfExec
	return &clone, nil
}

func (s *cancelTreeStorage) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	current, ok := s.executionRecords[executionID]
	if !ok {
		return nil, fmt.Errorf("execution %s not found", executionID)
	}
	clone := *current
	updated, err := update(&clone)
	if err != nil {
		return nil, err
	}
	if updated != nil {
		clone = *updated
	}
	s.executionRecords[executionID] = &clone
	out := clone
	return &out, nil
}

func (s *cancelTreeStorage) UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(*types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	current, ok := s.workflowExecutions[executionID]
	if !ok {
		return fmt.Errorf("execution %s not found", executionID)
	}
	clone := *current
	updated, err := updateFunc(&clone)
	if err != nil {
		return err
	}
	if updated != nil {
		clone = *updated
	}
	s.workflowExecutions[executionID] = &clone
	return nil
}

func (s *cancelTreeStorage) StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.workflowEvents = append(s.workflowEvents, event)
	return nil
}

func (s *cancelTreeStorage) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]*types.Execution, 0, len(s.executionRecords))
	for _, exec := range s.executionRecords {
		if filter.RunID != nil && exec.RunID != *filter.RunID {
			continue
		}
		clone := *exec
		out = append(out, &clone)
	}
	return out, nil
}

func TestCancelWorkflowTreeHandler_HappyPath(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// Tree:
	//   root (succeeded, depth 0)
	//   ├─ child-a (running, depth 1)
	//   │  └─ leaf-a (running, depth 2)
	//   └─ child-b (failed, depth 1)
	//      └─ leaf-b (succeeded, depth 2)
	store := newCancelTreeStorage()
	rootID := "root"
	childA := "child-a"
	leafA := "leaf-a"
	childB := "child-b"
	leafB := "leaf-b"
	store.seedExecution(rootID, "run-1", types.ExecutionStatusSucceeded, nil, 0)
	store.seedExecution(childA, "run-1", types.ExecutionStatusRunning, &rootID, 100)
	store.seedExecution(leafA, "run-1", types.ExecutionStatusRunning, &childA, 200)
	store.seedExecution(childB, "run-1", types.ExecutionStatusFailed, &rootID, 300)
	store.seedExecution(leafB, "run-1", types.ExecutionStatusSucceeded, &childB, 400)

	router := gin.New()
	router.POST("/api/v1/workflows/:workflowId/cancel-tree", CancelWorkflowTreeHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflows/run-1/cancel-tree", bytes.NewReader([]byte(`{"reason":"user clicked cancel"}`)))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload cancelTreeResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "run-1", payload.RunID)
	require.Equal(t, 5, payload.TotalNodes)
	require.Equal(t, 2, payload.CancelledCount, "exactly the two non-terminal nodes (child-a, leaf-a) should flip")
	require.Equal(t, 3, payload.SkippedCount, "root, child-b, leaf-b are terminal and skipped")
	require.Equal(t, 0, payload.ErrorCount)

	// Verify storage state.
	for _, id := range []string{childA, leafA} {
		exec, err := store.GetExecutionRecord(context.Background(), id)
		require.NoError(t, err)
		require.Equal(t, types.ExecutionStatusCancelled, exec.Status, id)
	}
	for id, want := range map[string]string{
		rootID: types.ExecutionStatusSucceeded,
		childB: types.ExecutionStatusFailed,
		leafB:  types.ExecutionStatusSucceeded,
	} {
		exec, err := store.GetExecutionRecord(context.Background(), id)
		require.NoError(t, err)
		require.Equal(t, want, exec.Status, id)
	}

	// Bottom-up: leaf-a (depth 2) appears before child-a (depth 1) in the
	// node list. Find their indices and assert ordering.
	indexOf := func(id string) int {
		for i, n := range payload.Nodes {
			if n.ExecutionID == id {
				return i
			}
		}
		return -1
	}
	require.Less(t, indexOf(leafA), indexOf(childA), "leaves must be processed before parents")
}

func TestCancelWorkflowTreeHandler_NotFound(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newCancelTreeStorage()
	router := gin.New()
	router.POST("/api/v1/workflows/:workflowId/cancel-tree", CancelWorkflowTreeHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflows/missing-run/cancel-tree", bytes.NewReader(nil))
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusNotFound, resp.Code)
}

func TestCancelWorkflowTreeHandler_AllTerminal(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newCancelTreeStorage()
	store.seedExecution("a", "run-2", types.ExecutionStatusSucceeded, nil, 0)
	store.seedExecution("b", "run-2", types.ExecutionStatusFailed, strPtr("a"), 100)

	router := gin.New()
	router.POST("/api/v1/workflows/:workflowId/cancel-tree", CancelWorkflowTreeHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflows/run-2/cancel-tree", bytes.NewReader(nil))
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	var payload cancelTreeResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, 0, payload.CancelledCount)
	require.Equal(t, 2, payload.SkippedCount)
}

// TestCancelWorkflowTreeHandler_MissingWorkflowID exercises the 400 path when
// neither :workflowId nor :workflow_id is present.
func TestCancelWorkflowTreeHandler_MissingWorkflowID(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newCancelTreeStorage()
	router := gin.New()
	// Register without the param so the handler sees an empty value.
	router.POST("/api/v1/workflows/cancel-tree", CancelWorkflowTreeHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflows/cancel-tree", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
	require.Contains(t, resp.Body.String(), "workflowId is required")
}

// TestCancelWorkflowTreeHandler_FallbackParamName verifies the handler also
// accepts the legacy :workflow_id parameter name.
func TestCancelWorkflowTreeHandler_FallbackParamName(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newCancelTreeStorage()
	store.seedExecution("a", "run-fallback", types.ExecutionStatusRunning, nil, 0)

	router := gin.New()
	router.POST("/api/v1/workflows/:workflow_id/cancel-tree", CancelWorkflowTreeHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflows/run-fallback/cancel-tree", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	var payload cancelTreeResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "run-fallback", payload.RunID)
	require.Equal(t, 1, payload.CancelledCount)
}

// TestCancelWorkflowTreeHandler_MalformedJSON covers the 400 path on a
// non-empty but invalid JSON request body.
func TestCancelWorkflowTreeHandler_MalformedJSON(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newCancelTreeStorage()
	store.seedExecution("a", "run-malformed", types.ExecutionStatusRunning, nil, 0)

	router := gin.New()
	router.POST("/api/v1/workflows/:workflowId/cancel-tree", CancelWorkflowTreeHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflows/run-malformed/cancel-tree", bytes.NewReader([]byte("{not json")))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
	require.Contains(t, resp.Body.String(), "invalid request body")
}

// TestCancelWorkflowTreeHandler_QueryError covers the 500 path when the
// underlying QueryExecutionRecords call fails.
type erroringQueryStore struct {
	cancelTreeStorage
}

func (s *erroringQueryStore) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	return nil, fmt.Errorf("simulated storage failure")
}

func TestCancelWorkflowTreeHandler_QueryError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &erroringQueryStore{cancelTreeStorage: *newCancelTreeStorage()}

	router := gin.New()
	router.POST("/api/v1/workflows/:workflowId/cancel-tree", CancelWorkflowTreeHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflows/run-x/cancel-tree", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusInternalServerError, resp.Code)
	require.Contains(t, resp.Body.String(), "failed to load run")
}

// TestCancelWorkflowTreeHandler_RaceToTerminal covers the errAlreadyTerminal
// branch — the initial Query returned a non-terminal status, but the
// UpdateExecutionRecord callback observed the row had since flipped to a
// terminal state. Should report the node as skipped (raced_to_terminal).
type racingStore struct {
	cancelTreeStorage
}

func (s *racingStore) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	// Always present the execution as already-terminal so the update callback
	// returns errAlreadyTerminal.
	terminalExec := &types.Execution{
		ExecutionID: executionID,
		Status:      types.ExecutionStatusSucceeded,
	}
	_, err := update(terminalExec)
	if err != nil {
		return nil, err
	}
	return terminalExec, nil
}

func TestCancelWorkflowTreeHandler_RaceToTerminal(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &racingStore{cancelTreeStorage: *newCancelTreeStorage()}
	store.seedExecution("a", "run-race", types.ExecutionStatusRunning, nil, 0)

	router := gin.New()
	router.POST("/api/v1/workflows/:workflowId/cancel-tree", CancelWorkflowTreeHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflows/run-race/cancel-tree", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	var payload cancelTreeResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, 1, payload.SkippedCount, "raced-to-terminal node must be skipped, not errored")
	require.Equal(t, 0, payload.ErrorCount)
	require.Equal(t, "raced_to_terminal", payload.Nodes[0].SkipReason)
}

// TestCancelWorkflowTreeHandler_UpdateError covers the generic update-error
// branch — UpdateExecutionRecord returns a non-errAlreadyTerminal error.
type erroringUpdateStore struct {
	cancelTreeStorage
}

func (s *erroringUpdateStore) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	return nil, fmt.Errorf("simulated update failure")
}

func TestCancelWorkflowTreeHandler_UpdateError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &erroringUpdateStore{cancelTreeStorage: *newCancelTreeStorage()}
	store.seedExecution("a", "run-err", types.ExecutionStatusRunning, nil, 0)

	router := gin.New()
	router.POST("/api/v1/workflows/:workflowId/cancel-tree", CancelWorkflowTreeHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflows/run-err/cancel-tree", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	var payload cancelTreeResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, 1, payload.ErrorCount, "update failure must surface as error_count")
	require.Equal(t, 0, payload.CancelledCount)
	require.Equal(t, "error", payload.Nodes[0].Status)
}

// TestCancelWorkflowTreeHandler_NilWorkflowExecution covers the cancelOneExecution
// branch where GetWorkflowExecution returns (nil, nil) — execution exists in the
// execution_records table but not in workflow_executions (e.g. older runs).
type missingWFExecStore struct {
	cancelTreeStorage
}

func (s *missingWFExecStore) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	return nil, nil
}

func TestCancelWorkflowTreeHandler_NilWorkflowExecution(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &missingWFExecStore{cancelTreeStorage: *newCancelTreeStorage()}
	store.seedExecution("a", "run-orphan", types.ExecutionStatusRunning, nil, 0)

	router := gin.New()
	router.POST("/api/v1/workflows/:workflowId/cancel-tree", CancelWorkflowTreeHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflows/run-orphan/cancel-tree", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	var payload cancelTreeResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, 1, payload.CancelledCount, "missing workflow_execution row must not block the cancel")
	require.Equal(t, 0, payload.ErrorCount)
}

// TestCancelWorkflowTreeHandler_EmptyReason covers the path where the request
// body is empty (io.EOF on bind) and reasonPtr stays nil. The cancel still
// proceeds.
func TestCancelWorkflowTreeHandler_EmptyReason(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newCancelTreeStorage()
	store.seedExecution("a", "run-empty", types.ExecutionStatusRunning, nil, 0)

	router := gin.New()
	router.POST("/api/v1/workflows/:workflowId/cancel-tree", CancelWorkflowTreeHandler(store))

	// Empty body — the `errors.Is(err, io.EOF)` guard should swallow the
	// "EOF" bind error and proceed with a nil reason.
	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflows/run-empty/cancel-tree", nil)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	var payload cancelTreeResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, 1, payload.CancelledCount)
}

func TestComputeExecutionDepths_OrphanParent(t *testing.T) {
	// Parent ID points to an execution outside the loaded slice — depth
	// should fall back to 0 rather than crashing.
	missing := "missing-parent"
	executions := []*types.Execution{
		{ExecutionID: "orphan", ParentExecutionID: &missing},
	}
	depths := computeExecutionDepths(executions)
	require.Equal(t, 0, depths["orphan"])
}

