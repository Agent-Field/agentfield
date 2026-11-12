package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type projectionFailureStore struct {
	*testExecutionStorage
	createFailures, updateFailures int
}

type projectionRecoveryDAGStore struct {
	*workflowDAGStorageStub
	records *testExecutionStorage
}

func (s *projectionRecoveryDAGStore) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	return s.records.QueryExecutionRecords(ctx, filter)
}

func (s *projectionFailureStore) StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error {
	if s.createFailures > 0 {
		s.createFailures--
		return fmt.Errorf("injected projection create failure")
	}
	return s.testExecutionStorage.StoreWorkflowExecution(ctx, execution)
}

func (s *projectionFailureStore) UpdateWorkflowExecution(ctx context.Context, id string, update func(*types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	if s.updateFailures > 0 {
		s.updateFailures--
		return fmt.Errorf("injected projection update failure")
	}
	return s.testExecutionStorage.UpdateWorkflowExecution(ctx, id, update)
}

func TestWorkflowEventRecoversProjectionFailure(t *testing.T) {
	gin.SetMode(gin.TestMode)
	for _, terminalOnly := range []bool{false, true} {
		t.Run(fmt.Sprintf("terminal_only_%v", terminalOnly), func(t *testing.T) {
			store := &projectionFailureStore{testExecutionStorage: newTestExecutionStorage(nil)}
			handler := WorkflowExecutionEventHandler(store)
			event := WorkflowExecutionEventRequest{ExecutionID: "child", RunID: "run", AgentNodeID: "node", ReasonerID: "child", ParentExecutionID: stringPointer("root"), Status: "running"}
			if !terminalOnly {
				require.Equal(t, http.StatusOK, postWorkflowExecutionEvent(t, handler, event).Code)
				store.updateFailures = 1
			} else {
				store.createFailures = 1
			}
			duration := int64(37)
			event.Status, event.Result, event.DurationMS = "succeeded", map[string]any{"answer": "done"}, &duration
			response := postWorkflowExecutionEvent(t, handler, event)
			assert.Equal(t, http.StatusOK, response.Code, response.Body.String())
			execution, err := store.GetExecutionRecord(context.Background(), "child")
			require.NoError(t, err)
			projection, err := store.GetWorkflowExecution(context.Background(), "child")
			require.NoError(t, err)
			require.NotNil(t, projection, "terminal creation must not acknowledge a lost projection")
			assert.Equal(t, "succeeded", projection.Status, "projection must recover after the execution committed")
			assert.Equal(t, execution.ResultPayload, projection.OutputData)
			assert.Equal(t, execution.DurationMS, projection.DurationMS)
			assert.Equal(t, execution.CompletedAt, projection.CompletedAt)
			// Delivery is idempotent; a delayed start cannot undo completion.
			for _, status := range []string{"succeeded", "running"} {
				event.Status, event.Result, event.DurationMS = status, nil, nil
				require.Equal(t, http.StatusOK, postWorkflowExecutionEvent(t, handler, event).Code)
				projection, err = store.GetWorkflowExecution(context.Background(), "child")
				require.NoError(t, err)
				assert.Equal(t, "succeeded", projection.Status)
				assert.Equal(t, execution.ResultPayload, projection.OutputData)
				assert.Equal(t, execution.CompletedAt, projection.CompletedAt)
			}
			router := gin.New()
			router.GET("/executions/:execution_id", GetExecutionStatusHandler(store))
			router.GET("/workflows/:workflowId/dag", GetWorkflowDAGHandler(&projectionRecoveryDAGStore{
				workflowDAGStorageStub: &workflowDAGStorageStub{}, records: store.testExecutionStorage,
			}))
			api := httptest.NewRecorder()
			router.ServeHTTP(api, httptest.NewRequest(http.MethodGet, "/executions/child", nil))
			assert.Equal(t, http.StatusOK, api.Code)
			assert.Contains(t, api.Body.String(), `"status":"succeeded"`)
			dag := httptest.NewRecorder()
			router.ServeHTTP(dag, httptest.NewRequest(http.MethodGet, "/workflows/run/dag", nil))
			require.Equal(t, http.StatusOK, dag.Code, dag.Body.String())
			var body WorkflowDAGResponse
			require.NoError(t, json.Unmarshal(dag.Body.Bytes(), &body))
			require.Len(t, body.Timeline, 1)
			assert.Equal(t, projection.Status, body.Timeline[0].Status)
		})
	}
}

func TestWorkflowEventRepairsProjectionFromPersistedTerminal(t *testing.T) {
	store := &projectionFailureStore{testExecutionStorage: newTestExecutionStorage(nil)}
	handler := WorkflowExecutionEventHandler(store)
	event := WorkflowExecutionEventRequest{ExecutionID: "child", RunID: "run", Status: "running"}
	require.Equal(t, http.StatusOK, postWorkflowExecutionEvent(t, handler, event).Code)
	store.updateFailures = 10
	event.Status, event.Result = "succeeded", map[string]any{"answer": "done"}
	assert.Equal(t, http.StatusInternalServerError, postWorkflowExecutionEvent(t, handler, event).Code)
	// A later delivery can contain an old running status. The persisted
	// execution, whose terminal state cannot regress, is authoritative.
	store.updateFailures = 0
	event.Status, event.Result = "running", nil
	require.Equal(t, http.StatusOK, postWorkflowExecutionEvent(t, handler, event).Code)
	execution, err := store.GetExecutionRecord(context.Background(), "child")
	require.NoError(t, err)
	projection, err := store.GetWorkflowExecution(context.Background(), "child")
	require.NoError(t, err)
	assert.Equal(t, execution.Status, projection.Status)
	assert.Equal(t, execution.ResultPayload, projection.OutputData)
	assert.Equal(t, execution.CompletedAt, projection.CompletedAt)
}
