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
	"github.com/stretchr/testify/require"
)

func TestHandleStatusUpdateMarksSuccess(t *testing.T) {
	gin.SetMode(gin.TestMode)

	storage := newTestExecutionStorage(&types.AgentNode{ID: "agent-1"})
	now := time.Now().UTC().Add(-time.Minute)
	exec := &types.Execution{
		ExecutionID: "exec-123",
		RunID:       "run-123",
		AgentNodeID: "agent-1",
		ReasonerID:  "reasoner",
		NodeID:      "agent-1",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
		CreatedAt:   now,
		UpdatedAt:   now,
	}
	require.NoError(t, storage.CreateExecutionRecord(context.Background(), exec))

	workflow := &types.WorkflowExecution{
		ExecutionID: exec.ExecutionID,
		WorkflowID:  exec.RunID,
		StartedAt:   now,
		CreatedAt:   now,
		UpdatedAt:   now,
	}
	require.NoError(t, storage.StoreWorkflowExecution(context.Background(), workflow))

	payloadStore := services.NewFilePayloadStore(t.TempDir())
	handler := UpdateExecutionStatusHandler(storage, payloadStore, nil, 90*time.Second)

	body := map[string]interface{}{
		"status": "succeeded",
		"result": map[string]string{
			"value": "done",
		},
		"duration_ms": 1500,
	}
	payload, err := json.Marshal(body)
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/exec-123/status", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	ctx.Params = gin.Params{gin.Param{Key: "execution_id", Value: exec.ExecutionID}}
	ctx.Request = req.WithContext(context.Background())

	handler(ctx)

	require.Equal(t, http.StatusOK, w.Code)

	var resp ExecutionStatusResponse
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
	require.Equal(t, "succeeded", resp.Status)

	updated, err := storage.GetExecutionRecord(context.Background(), exec.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusSucceeded, updated.Status)
	require.NotNil(t, updated.CompletedAt)
	require.NotNil(t, updated.DurationMS)
	require.NotNil(t, updated.ResultPayload)
	require.NotNil(t, updated.ResultURI)
}
