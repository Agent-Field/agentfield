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

type cancelHandlerStorage struct {
	storage.StorageProvider

	mu                 sync.Mutex
	executionRecords   map[string]*types.Execution
	workflowExecutions map[string]*types.WorkflowExecution
	workflowEvents     []*types.WorkflowExecutionEvent
}

func newCancelHandlerStorage() *cancelHandlerStorage {
	return &cancelHandlerStorage{
		executionRecords:   make(map[string]*types.Execution),
		workflowExecutions: make(map[string]*types.WorkflowExecution),
		workflowEvents:     make([]*types.WorkflowExecutionEvent, 0),
	}
}

func (s *cancelHandlerStorage) seedExecution(executionID, status string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().UTC()
	runID := "run-1"
	statusCopy := status
	s.executionRecords[executionID] = &types.Execution{
		ExecutionID:  executionID,
		RunID:        runID,
		AgentNodeID:  "agent-1",
		Status:       statusCopy,
		StartedAt:    now,
		CreatedAt:    now,
		UpdatedAt:    now,
		StatusReason: nil,
	}
	s.workflowExecutions[executionID] = &types.WorkflowExecution{
		ExecutionID: executionID,
		WorkflowID:  "wf-1",
		RunID:       &runID,
		AgentNodeID: "agent-1",
		Status:      statusCopy,
		StartedAt:   now,
	}
}

func (s *cancelHandlerStorage) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	exec, ok := s.executionRecords[executionID]
	if !ok {
		return nil, nil
	}
	copy := *exec
	return &copy, nil
}

func (s *cancelHandlerStorage) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	wfExec, ok := s.workflowExecutions[executionID]
	if !ok {
		return nil, nil
	}
	copy := *wfExec
	return &copy, nil
}

func (s *cancelHandlerStorage) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
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

func (s *cancelHandlerStorage) UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(*types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
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

func (s *cancelHandlerStorage) StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.workflowEvents = append(s.workflowEvents, event)
	return nil
}

func TestCancelExecutionHandler_StateTransitions(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name         string
		initialState string
		executionID  string
		body         string
		statusCode   int
	}{
		{name: "cancel running execution", initialState: types.ExecutionStatusRunning, executionID: "exec-running", statusCode: http.StatusOK},
		{name: "cancel pending execution", initialState: types.ExecutionStatusPending, executionID: "exec-pending", statusCode: http.StatusOK},
		{name: "cancel paused execution", initialState: types.ExecutionStatusPaused, executionID: "exec-paused", statusCode: http.StatusOK},
		{name: "cancel queued execution", initialState: types.ExecutionStatusQueued, executionID: "exec-queued", statusCode: http.StatusOK},
		{name: "cancel waiting execution", initialState: types.ExecutionStatusWaiting, executionID: "exec-waiting", statusCode: http.StatusOK},
		{name: "cancel already succeeded execution", initialState: types.ExecutionStatusSucceeded, executionID: "exec-succeeded", statusCode: http.StatusConflict},
		{name: "cancel already cancelled execution", initialState: types.ExecutionStatusCancelled, executionID: "exec-cancelled", statusCode: http.StatusConflict},
		{name: "cancel non-existent execution", executionID: "exec-missing", statusCode: http.StatusNotFound},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := newCancelHandlerStorage()
			if tt.initialState != "" {
				store.seedExecution(tt.executionID, tt.initialState)
			}

			router := gin.New()
			router.POST("/api/v1/executions/:execution_id/cancel", CancelExecutionHandler(store))

			var bodyReader *bytes.Reader
			if tt.body != "" {
				bodyReader = bytes.NewReader([]byte(tt.body))
			} else {
				bodyReader = bytes.NewReader(nil)
			}

			req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/"+tt.executionID+"/cancel", bodyReader)
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()
			router.ServeHTTP(resp, req)

			require.Equal(t, tt.statusCode, resp.Code)

			if tt.statusCode == http.StatusOK {
				var payload cancelExecutionResponse
				require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
				require.Equal(t, tt.executionID, payload.ExecutionID)
				require.Equal(t, tt.initialState, payload.PreviousStatus)
				require.Equal(t, types.ExecutionStatusCancelled, payload.Status)
				require.NotEmpty(t, payload.CancelledAt)

				exec, err := store.GetExecutionRecord(context.Background(), tt.executionID)
				require.NoError(t, err)
				require.NotNil(t, exec)
				require.Equal(t, types.ExecutionStatusCancelled, exec.Status)

				wfExec, err := store.GetWorkflowExecution(context.Background(), tt.executionID)
				require.NoError(t, err)
				require.NotNil(t, wfExec)
				require.Equal(t, types.ExecutionStatusCancelled, wfExec.Status)
			}
		})
	}
}

func TestCancelExecutionHandler_WithReason(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newCancelHandlerStorage()
	store.seedExecution("exec-reason", types.ExecutionStatusRunning)

	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/cancel", CancelExecutionHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/exec-reason/cancel", bytes.NewReader([]byte(`{"reason":"operator requested stop"}`)))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload cancelExecutionResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.NotNil(t, payload.Reason)
	require.Equal(t, "operator requested stop", *payload.Reason)

	exec, err := store.GetExecutionRecord(context.Background(), "exec-reason")
	require.NoError(t, err)
	require.NotNil(t, exec.StatusReason)
	require.Equal(t, "operator requested stop", *exec.StatusReason)

	wfExec, err := store.GetWorkflowExecution(context.Background(), "exec-reason")
	require.NoError(t, err)
	require.NotNil(t, wfExec.StatusReason)
	require.Equal(t, "operator requested stop", *wfExec.StatusReason)
}

func TestCancelExecutionHandler_WithoutReasonOmitsReasonField(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newCancelHandlerStorage()
	store.seedExecution("exec-no-reason", types.ExecutionStatusRunning)

	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/cancel", CancelExecutionHandler(store))

	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/exec-no-reason/cancel", nil)
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)

	var payload map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	_, hasReason := payload["reason"]
	require.False(t, hasReason)
}
