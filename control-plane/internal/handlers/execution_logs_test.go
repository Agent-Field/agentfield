package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStructuredExecutionLogsHandlerAcceptsSingleAndBatch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newTestExecutionStorage(&types.AgentNode{ID: "node-1"})
	handler := StructuredExecutionLogsHandler(store, func() config.ExecutionLogsConfig {
		return config.ExecutionLogsConfig{
			MaxEntriesPerExecution: 100,
			RetentionPeriod:        24 * time.Hour,
		}
	})

	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/logs", handler)

	single := types.ExecutionLogEntry{
		ExecutionID: "exec-1",
		WorkflowID:  "wf-1",
		AgentNodeID: "node-1",
		Level:       "info",
		Source:      "sdk.typescript",
		Message:     "single message",
		Attributes:  json.RawMessage(`{"mode":"single"}`),
		EmittedAt:   time.Now().UTC(),
	}
	body, err := json.Marshal(single)
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/exec-1/logs", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusAccepted, resp.Code)

	batch := map[string]any{
		"entries": []map[string]any{
			{
				"execution_id":  "exec-1",
				"workflow_id":   "wf-1",
				"agent_node_id": "node-1",
				"level":         "warn",
				"source":        "sdk.runtime",
				"message":       "batch message",
				"ts":            time.Now().UTC().Format(time.RFC3339),
			},
		},
	}
	body, err = json.Marshal(batch)
	require.NoError(t, err)

	req = httptest.NewRequest(http.MethodPost, "/api/v1/executions/exec-1/logs", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp = httptest.NewRecorder()
	router.ServeHTTP(resp, req)
	require.Equal(t, http.StatusAccepted, resp.Code)

	entries, err := store.ListExecutionLogEntries(context.Background(), "exec-1", nil, 10, nil, nil, nil, "")
	require.NoError(t, err)
	require.Len(t, entries, 2)
	assert.Equal(t, "single message", entries[0].Message)
	assert.Equal(t, "batch message", entries[1].Message)
	assert.Equal(t, int64(1), entries[0].Sequence)
	assert.Equal(t, int64(2), entries[1].Sequence)
}

func TestStructuredExecutionLogsHandlerRejectsPathMismatch(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newTestExecutionStorage(&types.AgentNode{ID: "node-1"})
	handler := StructuredExecutionLogsHandler(store, func() config.ExecutionLogsConfig {
		return config.ExecutionLogsConfig{MaxEntriesPerExecution: 100}
	})

	router := gin.New()
	router.POST("/api/v1/executions/:execution_id/logs", handler)

	body := `{"execution_id":"exec-2","workflow_id":"wf-1","agent_node_id":"node-1","level":"info","source":"sdk.typescript","message":"oops"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/exec-1/logs", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
	assert.Contains(t, resp.Body.String(), "execution_id_mismatch")
}
