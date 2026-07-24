package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func seedExecutionForPauseResume(t *testing.T, store *testExecutionStorage, executionID, status string) {
	t.Helper()

	now := time.Now().UTC()
	runID := "run-1"

	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID: executionID,
		RunID:       runID,
		AgentNodeID: "agent-node-1",
		Status:      status,
		StartedAt:   now,
		CreatedAt:   now,
		UpdatedAt:   now,
	}))

	require.NoError(t, store.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		ExecutionID: executionID,
		WorkflowID:  "wf-1",
		RunID:       &runID,
		AgentNodeID: "agent-node-1",
		Status:      status,
		StartedAt:   now,
		CreatedAt:   now,
		UpdatedAt:   now,
	}))
}

func TestPauseExecutionHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name               string
		executionID        string
		initialStatus      string
		body               string
		expectedStatusCode int
		expectedStatus     string
		expectedReason     string
	}{
		{
			name:               "pause running execution",
			executionID:        "exec-pause-running",
			initialStatus:      types.ExecutionStatusRunning,
			expectedStatusCode: http.StatusOK,
			expectedStatus:     types.ExecutionStatusPaused,
		},
		{
			name:               "pause running execution with reason",
			executionID:        "exec-pause-reason",
			initialStatus:      types.ExecutionStatusRunning,
			body:               `{"reason":"manual intervention"}`,
			expectedStatusCode: http.StatusOK,
			expectedStatus:     types.ExecutionStatusPaused,
			expectedReason:     "manual intervention",
		},
		{
			name:               "pause pending execution returns conflict",
			executionID:        "exec-pause-pending",
			initialStatus:      types.ExecutionStatusPending,
			expectedStatusCode: http.StatusConflict,
		},
		{
			name:               "pause paused execution returns conflict",
			executionID:        "exec-pause-paused",
			initialStatus:      types.ExecutionStatusPaused,
			expectedStatusCode: http.StatusConflict,
		},
		{
			name:               "pause succeeded execution returns conflict",
			executionID:        "exec-pause-succeeded",
			initialStatus:      types.ExecutionStatusSucceeded,
			expectedStatusCode: http.StatusConflict,
		},
		{
			name:               "pause non-existent execution returns not found",
			executionID:        "exec-pause-missing",
			initialStatus:      "",
			expectedStatusCode: http.StatusNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := newTestExecutionStorage(&types.AgentNode{ID: "agent-node-1"})
			if tt.initialStatus != "" {
				seedExecutionForPauseResume(t, store, tt.executionID, tt.initialStatus)
			}

			router := gin.New()
			router.POST("/api/v1/executions/:execution_id/pause", PauseExecutionHandler(store))

			requestBody := tt.body
			if requestBody == "" {
				requestBody = "{}"
			}
			req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/"+tt.executionID+"/pause", strings.NewReader(requestBody))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()

			router.ServeHTTP(resp, req)

			require.Equal(t, tt.expectedStatusCode, resp.Code)

			if tt.expectedStatusCode != http.StatusOK {
				return
			}

			var payload map[string]interface{}
			require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
			require.Equal(t, tt.executionID, payload["execution_id"])
			require.Equal(t, types.ExecutionStatusRunning, payload["previous_status"])
			require.Equal(t, tt.expectedStatus, payload["status"])
			require.NotEmpty(t, payload["paused_at"])

			if tt.expectedReason == "" {
				_, exists := payload["reason"]
				require.False(t, exists)
			} else {
				require.Equal(t, tt.expectedReason, payload["reason"])
			}

			execRecord, err := store.GetExecutionRecord(context.Background(), tt.executionID)
			require.NoError(t, err)
			require.NotNil(t, execRecord)
			require.Equal(t, types.ExecutionStatusPaused, execRecord.Status)
			if tt.expectedReason != "" {
				require.NotNil(t, execRecord.StatusReason)
				require.Equal(t, tt.expectedReason, *execRecord.StatusReason)
			}

			wfExec, err := store.GetWorkflowExecution(context.Background(), tt.executionID)
			require.NoError(t, err)
			require.NotNil(t, wfExec)
			require.Equal(t, types.ExecutionStatusPaused, wfExec.Status)
			if tt.expectedReason != "" {
				require.NotNil(t, wfExec.StatusReason)
				require.Equal(t, tt.expectedReason, *wfExec.StatusReason)
			}
		})
	}
}

func TestResumeExecutionHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name               string
		executionID        string
		initialStatus      string
		body               string
		expectedStatusCode int
	}{
		{
			name:               "resume paused execution",
			executionID:        "exec-resume-paused",
			initialStatus:      types.ExecutionStatusPaused,
			expectedStatusCode: http.StatusOK,
		},
		{
			name:               "resume paused execution with reason",
			executionID:        "exec-resume-reason",
			initialStatus:      types.ExecutionStatusPaused,
			body:               `{"reason":"continue work"}`,
			expectedStatusCode: http.StatusOK,
		},
		{
			name:               "resume running execution returns conflict",
			executionID:        "exec-resume-running",
			initialStatus:      types.ExecutionStatusRunning,
			expectedStatusCode: http.StatusConflict,
		},
		{
			name:               "resume non-paused execution returns conflict",
			executionID:        "exec-resume-succeeded",
			initialStatus:      types.ExecutionStatusSucceeded,
			expectedStatusCode: http.StatusConflict,
		},
		{
			name:               "resume non-existent execution returns not found",
			executionID:        "exec-resume-missing",
			initialStatus:      "",
			expectedStatusCode: http.StatusNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := newTestExecutionStorage(&types.AgentNode{ID: "agent-node-1"})
			if tt.initialStatus != "" {
				seedExecutionForPauseResume(t, store, tt.executionID, tt.initialStatus)
			}

			router := gin.New()
			router.POST("/api/v1/executions/:execution_id/resume", ResumeExecutionHandler(store))

			requestBody := tt.body
			if requestBody == "" {
				requestBody = "{}"
			}
			req := httptest.NewRequest(http.MethodPost, "/api/v1/executions/"+tt.executionID+"/resume", strings.NewReader(requestBody))
			req.Header.Set("Content-Type", "application/json")
			resp := httptest.NewRecorder()

			router.ServeHTTP(resp, req)

			require.Equal(t, tt.expectedStatusCode, resp.Code)

			if tt.expectedStatusCode != http.StatusOK {
				return
			}

			var payload map[string]interface{}
			require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &payload))
			require.Equal(t, tt.executionID, payload["execution_id"])
			require.Equal(t, types.ExecutionStatusPaused, payload["previous_status"])
			require.Equal(t, types.ExecutionStatusRunning, payload["status"])
			require.NotEmpty(t, payload["resumed_at"])

			execRecord, err := store.GetExecutionRecord(context.Background(), tt.executionID)
			require.NoError(t, err)
			require.NotNil(t, execRecord)
			require.Equal(t, types.ExecutionStatusRunning, execRecord.Status)

			wfExec, err := store.GetWorkflowExecution(context.Background(), tt.executionID)
			require.NoError(t, err)
			require.NotNil(t, wfExec)
			require.Equal(t, types.ExecutionStatusRunning, wfExec.Status)
		})
	}
}
