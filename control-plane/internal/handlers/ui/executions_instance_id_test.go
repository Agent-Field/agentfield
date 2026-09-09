package ui

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestGetExecutionDetailsGlobalHandlerReturnsInstanceID(t *testing.T) {
	store, _ := setupUIHandlerStorage(t)
	now := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: "exec-1",
		RunID:       "run-1",
		AgentNodeID: "node-1",
		InstanceID:  "instance-1",
		ReasonerID:  "reasoner-a",
		Status:      types.ExecutionStatusRunning,
		StartedAt:   now,
		CreatedAt:   now,
		UpdatedAt:   now,
	}))

	handler := NewExecutionHandler(store, nil, nil)
	router := gin.New()
	router.GET("/api/ui/v1/executions/:execution_id/details", handler.GetExecutionDetailsGlobalHandler)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/exec-1/details", nil))

	require.Equal(t, http.StatusOK, resp.Code)
	var payload map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
	require.Equal(t, "node-1", payload["agent_node_id"])
	require.Equal(t, "instance-1", payload["instance_id"])
}
