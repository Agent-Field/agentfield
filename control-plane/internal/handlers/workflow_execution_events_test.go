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

func TestWorkflowExecutionEventHandler_CreateAndUpdate(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := newTestExecutionStorage(&types.AgentNode{ID: "deep_research"})
	handler := WorkflowExecutionEventHandler(storage)

	startPayload := WorkflowExecutionEventRequest{
		ExecutionID: "exec_child",
		RunID:       "run_123",
		ReasonerID:  "understand_query_deeply",
		AgentNodeID: "deep_research",
		Status:      "running",
		InputData: map[string]interface{}{
			"arg": "value",
		},
	}

	body, err := json.Marshal(startPayload)
	require.NoError(t, err)

	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/workflow/executions/events", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	c.Request = req

	handler(c)

	require.Equal(t, http.StatusOK, w.Code)

	exec, err := storage.GetExecutionRecord(context.Background(), "exec_child")
	require.NoError(t, err)
	require.NotNil(t, exec)
	assert.Equal(t, "run_123", exec.RunID)
	assert.Equal(t, string(types.ExecutionStatusRunning), exec.Status)
	assert.Nil(t, exec.CompletedAt)
	assert.WithinDuration(t, time.Now(), exec.StartedAt, time.Second)

	resultPayload := map[string]string{"result": "ok"}
	duration := int64(1500)
	completePayload := WorkflowExecutionEventRequest{
		ExecutionID: "exec_child",
		RunID:       "run_123",
		ReasonerID:  "understand_query_deeply",
		AgentNodeID: "deep_research",
		Status:      "succeeded",
		Result:      resultPayload,
		DurationMS:  &duration,
	}

	body, err = json.Marshal(completePayload)
	require.NoError(t, err)

	w = httptest.NewRecorder()
	c, _ = gin.CreateTestContext(w)
	req = httptest.NewRequest(http.MethodPost, "/api/v1/workflow/executions/events", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	c.Request = req

	handler(c)

	require.Equal(t, http.StatusOK, w.Code)

	exec, err = storage.GetExecutionRecord(context.Background(), "exec_child")
	require.NoError(t, err)
	require.NotNil(t, exec)
	assert.Equal(t, string(types.ExecutionStatusSucceeded), exec.Status)
	require.NotNil(t, exec.CompletedAt)
	assert.True(t, exec.CompletedAt.After(exec.StartedAt))
	require.NotNil(t, exec.ResultPayload)
	assert.Contains(t, string(exec.ResultPayload), "result")
	require.NotNil(t, exec.DurationMS)
	assert.Equal(t, duration, *exec.DurationMS)
}
