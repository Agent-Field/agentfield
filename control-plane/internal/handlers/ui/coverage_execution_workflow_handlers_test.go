package ui

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type recordingWebhookDispatcher struct {
	notified []string
	err      error
}

func (d *recordingWebhookDispatcher) Start(ctx context.Context) error { return nil }
func (d *recordingWebhookDispatcher) Stop(ctx context.Context) error  { return nil }
func (d *recordingWebhookDispatcher) Notify(ctx context.Context, executionID string) error {
	if d.err != nil {
		return d.err
	}
	d.notified = append(d.notified, executionID)
	return nil
}

func setupUIHandlerStorage(t *testing.T) (*storage.LocalStorage, context.Context) {
	t.Helper()

	ctx := context.Background()
	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(t.TempDir(), "agentfield.db"),
			KVStorePath:  filepath.Join(t.TempDir(), "agentfield.bolt"),
		},
	}

	ls := storage.NewLocalStorage(storage.LocalStorageConfig{})
	err := ls.Initialize(ctx, cfg)
	if err != nil && strings.Contains(strings.ToLower(err.Error()), "fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)
	t.Cleanup(func() {
		_ = ls.Close(ctx)
	})

	return ls, ctx
}

func seedWorkflowExecutions(t *testing.T, ls *storage.LocalStorage, ctx context.Context) time.Time {
	t.Helper()

	now := time.Date(2026, 4, 8, 12, 0, 0, 0, time.UTC)
	runID := "run-123"
	otherRunID := "run-999"
	rootWorkflowID := "wf-root"
	rootExecID := "exec-root"
	sessionID := "session-1"
	actorID := "actor-1"
	approvalID := "approval-1"
	approvalURL := "https://example.test/approval/1"
	approvalStatus := "pending"
	waitReason := "waiting_for_approval"
	failReason := "provider_timeout"
	errorMessage := "upstream failed"
	rootDuration := int64(61000)
	childDuration := int64(3000)
	otherDuration := int64(1200)
	rootCompleted := now.Add(61 * time.Second)
	childCompleted := now.Add(64 * time.Second)
	otherCompleted := now.Add(-58 * time.Minute)

	require.NoError(t, ls.StoreWorkflowRun(ctx, &types.WorkflowRun{
		RunID:          runID,
		RootWorkflowID: rootWorkflowID,
		Status:         string(types.ExecutionStatusRunning),
		CreatedAt:      now,
		UpdatedAt:      now,
	}))
	require.NoError(t, ls.StoreWorkflowRun(ctx, &types.WorkflowRun{
		RunID:          otherRunID,
		RootWorkflowID: "wf-other",
		Status:         string(types.ExecutionStatusSucceeded),
		CreatedAt:      now.Add(-time.Hour),
		UpdatedAt:      now.Add(-time.Hour),
	}))

	require.NoError(t, ls.StoreWorkflowExecution(ctx, &types.WorkflowExecution{
		WorkflowID:          rootWorkflowID,
		ExecutionID:         rootExecID,
		AgentFieldRequestID: "req-root",
		RunID:               &runID,
		SessionID:           &sessionID,
		ActorID:             &actorID,
		AgentNodeID:         "agent-alpha",
		ReasonerID:          "planner",
		InputData:           json.RawMessage(`{"error":"corrupted_json_data","preview":"partial"}`),
		OutputData:          json.RawMessage(`{"result":"ok"}`),
		Status:              string(types.ExecutionStatusWaiting),
		StatusReason:        &waitReason,
		StartedAt:           now,
		CompletedAt:         &rootCompleted,
		DurationMS:          &rootDuration,
		ApprovalRequestID:   &approvalID,
		ApprovalRequestURL:  &approvalURL,
		ApprovalStatus:      &approvalStatus,
		CreatedAt:           now,
		UpdatedAt:           rootCompleted,
		Notes: []types.ExecutionNote{
			{Message: "queued", Timestamp: now.Add(10 * time.Second)},
			{Message: "awaiting approval", Timestamp: now.Add(20 * time.Second)},
		},
	}))
	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID:  rootExecID,
		RunID:        runID,
		AgentNodeID:  "agent-alpha",
		ReasonerID:   "planner",
		NodeID:       "agent-alpha",
		InputPayload: json.RawMessage(`{"error":"corrupted_json_data","preview":"partial"}`),
		ResultPayload: json.RawMessage(`{"result":"ok"}`),
		Status:       string(types.ExecutionStatusWaiting),
		StatusReason: &waitReason,
		StartedAt:    now,
		CompletedAt:  &rootCompleted,
		DurationMS:   &rootDuration,
		SessionID:    &sessionID,
		ActorID:      &actorID,
		Notes: []types.ExecutionNote{
			{Message: "queued", Timestamp: now.Add(10 * time.Second)},
			{Message: "awaiting approval", Timestamp: now.Add(20 * time.Second)},
		},
	}))

	require.NoError(t, ls.StoreWorkflowExecution(ctx, &types.WorkflowExecution{
		WorkflowID:          rootWorkflowID,
		ExecutionID:         "exec-child",
		AgentFieldRequestID: "req-child",
		RunID:               &runID,
		SessionID:           &sessionID,
		ActorID:             &actorID,
		AgentNodeID:         "agent-beta",
		ReasonerID:          "review",
		ParentExecutionID:   &rootExecID,
		RootWorkflowID:      &rootWorkflowID,
		WorkflowDepth:       1,
		InputData:           json.RawMessage(`{"step":2}`),
		OutputData:          json.RawMessage(`{"error":"failed"}`),
		Status:              string(types.ExecutionStatusFailed),
		StatusReason:        &failReason,
		ErrorMessage:        &errorMessage,
		StartedAt:           now.Add(time.Minute),
		CompletedAt:         &childCompleted,
		DurationMS:          &childDuration,
		CreatedAt:           now.Add(time.Minute),
		UpdatedAt:           childCompleted,
	}))
	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID:       "exec-child",
		RunID:             runID,
		ParentExecutionID: &rootExecID,
		AgentNodeID:       "agent-beta",
		ReasonerID:        "review",
		NodeID:            "agent-beta",
		InputPayload:      json.RawMessage(`{"step":2}`),
		ResultPayload:     json.RawMessage(`{"error":"failed"}`),
		ErrorMessage:      &errorMessage,
		Status:            string(types.ExecutionStatusFailed),
		StatusReason:      &failReason,
		StartedAt:         now.Add(time.Minute),
		CompletedAt:       &childCompleted,
		DurationMS:        &childDuration,
		SessionID:         &sessionID,
		ActorID:           &actorID,
	}))

	require.NoError(t, ls.StoreWorkflowExecution(ctx, &types.WorkflowExecution{
		WorkflowID:          "wf-other",
		ExecutionID:         "exec-other",
		AgentFieldRequestID: "req-other",
		RunID:               &otherRunID,
		AgentNodeID:         "agent-gamma",
		ReasonerID:          "summarize",
		InputData:           json.RawMessage(`{"input":"value"}`),
		OutputData:          json.RawMessage(`{"summary":"done"}`),
		Status:              string(types.ExecutionStatusSucceeded),
		StartedAt:           now.Add(-time.Hour),
		CompletedAt:         &otherCompleted,
		DurationMS:          &otherDuration,
		CreatedAt:           now.Add(-time.Hour),
		UpdatedAt:           otherCompleted,
	}))
	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID:   "exec-other",
		RunID:         otherRunID,
		AgentNodeID:   "agent-gamma",
		ReasonerID:    "summarize",
		NodeID:        "agent-gamma",
		InputPayload:  json.RawMessage(`{"input":"value"}`),
		ResultPayload: json.RawMessage(`{"summary":"done"}`),
		Status:        string(types.ExecutionStatusSucceeded),
		StartedAt:     now.Add(-time.Hour),
		CompletedAt:   &otherCompleted,
		DurationMS:    &otherDuration,
	}))

	require.NoError(t, ls.RegisterExecutionWebhook(ctx, &types.ExecutionWebhook{
		ExecutionID: "exec-root",
		URL:         "https://example.test/webhook",
		Status:      types.ExecutionWebhookStatusPending,
		Headers:     map[string]string{"X-Test": "1"},
		CreatedAt:   now,
		UpdatedAt:   now,
	}))
	require.NoError(t, ls.StoreExecutionWebhookEvent(ctx, &types.ExecutionWebhookEvent{
		ExecutionID: "exec-root",
		EventType:   types.WebhookEventExecutionCompleted,
		Status:      types.ExecutionWebhookStatusDelivered,
		Payload:     json.RawMessage(`{"ok":true}`),
		CreatedAt:   now.Add(2 * time.Minute),
	}))

	return now
}

func TestExecutionHandlerRealStorageCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	ls, ctx := setupUIHandlerStorage(t)
	now := seedWorkflowExecutions(t, ls, ctx)

	dispatcher := &recordingWebhookDispatcher{}

	handler := NewExecutionHandler(ls, nil, dispatcher)
	router := gin.New()
	router.GET("/api/ui/v1/agents/:agentId/executions", handler.ListExecutionsHandler)
	router.GET("/api/ui/v1/agents/:agentId/executions/:executionId", handler.GetExecutionDetailsHandler)
	router.GET("/api/ui/v1/executions/summary", handler.GetExecutionsSummaryHandler)
	router.GET("/api/ui/v1/executions/stats", handler.GetExecutionStatsHandler)
	router.GET("/api/ui/v1/executions/enhanced", handler.GetEnhancedExecutionsHandler)
	router.GET("/api/ui/v1/executions/:execution_id/details", handler.GetExecutionDetailsGlobalHandler)
	router.POST("/api/ui/v1/executions/:execution_id/webhook/retry", handler.RetryExecutionWebhookHandler)

	t.Run("list executions for agent", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/agents/agent-alpha/executions?page=1&pageSize=1&status=waiting&sortBy=duration", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		var body ExecutionListResponse
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
		require.Len(t, body.Executions, 1)
		require.Equal(t, "exec-root", body.Executions[0].ExecutionID)
		require.Equal(t, 2, body.TotalPages)
	})

	t.Run("get agent execution details", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/agents/agent-alpha/executions/exec-root", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		var body ExecutionDetailsResponse
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
		require.Equal(t, "exec-root", body.ExecutionID)
		require.Equal(t, "agent-alpha", body.AgentNodeID)
		require.Equal(t, map[string]interface{}{"error": corruptedJSONSentinel, "preview": "partial"}, body.InputData)
		require.Equal(t, 2, body.NotesCount)
		require.NotNil(t, body.LatestNote)
		require.Equal(t, "awaiting approval", body.LatestNote.Message)
		require.True(t, body.WebhookRegistered)
		require.Len(t, body.WebhookEvents, 1)
		require.NotNil(t, body.ApprovalRequestID)
	})

	t.Run("summary group by status and invalid time", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/summary?group_by=status&start_time=not-a-time", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusBadRequest, resp.Code)

		req = httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/summary?group_by=status&page_size=5&start_time="+now.Add(-2*time.Hour).Format(time.RFC3339), nil)
		resp = httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusOK, resp.Code)

		var body map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
		var grouped map[string][]ExecutionSummary
		require.NoError(t, json.Unmarshal(body["grouped"], &grouped))
		require.Contains(t, grouped, string(types.ExecutionStatusFailed))
		require.Contains(t, grouped, string(types.ExecutionStatusSucceeded))
	})

	t.Run("execution stats", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/stats", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		var body ExecutionStatsResponse
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
		require.Equal(t, 3, body.TotalExecutions)
		require.Equal(t, 1, body.SuccessfulCount)
		require.Equal(t, 1, body.FailedCount)
		require.Equal(t, 1, body.RunningCount)
		require.Equal(t, 1, body.ExecutionsByAgent["agent-alpha"])
	})

	t.Run("enhanced executions list", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/enhanced?limit=2&page=1&status=failed", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		var body EnhancedExecutionsResponse
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
		require.Len(t, body.Executions, 1)
		require.Equal(t, "exec-child", body.Executions[0].ExecutionID)
		require.Equal(t, "3.0s", body.Executions[0].DurationDisplay)
		require.False(t, body.HasMore)
	})

	t.Run("global execution details not found", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/missing/details", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusNotFound, resp.Code)
	})

	t.Run("retry execution webhook accepted", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/api/ui/v1/executions/exec-root/webhook/retry", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusAccepted, resp.Code)
		require.Equal(t, []string{"exec-root"}, dispatcher.notified)
	})

	t.Run("retry execution webhook bad request when missing registration", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/api/ui/v1/executions/exec-child/webhook/retry", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusBadRequest, resp.Code)
	})
}

func TestWorkflowRunHandlerRealStorageCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	ls, ctx := setupUIHandlerStorage(t)
	seedWorkflowExecutions(t, ls, ctx)

	handler := NewWorkflowRunHandler(ls)
	router := gin.New()
	router.GET("/api/ui/v2/workflow-runs", handler.ListWorkflowRunsHandler)
	router.GET("/api/ui/v2/workflow-runs/:run_id", handler.GetWorkflowRunDetailHandler)

	t.Run("list workflow runs", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v2/workflow-runs?page=1&page_size=10&sort_by=nodes", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		var body WorkflowRunListResponse
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
		require.NotEmpty(t, body.Runs)
		require.Equal(t, "run-123", body.Runs[0].RunID)
		require.Equal(t, "exec-root", body.Runs[0].RootExecutionID)
		require.Equal(t, string(types.ExecutionStatusRunning), body.Runs[0].Status)
		require.False(t, body.HasMore)
	})

	t.Run("workflow run detail with approval enrichment", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v2/workflow-runs/run-123", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)

		require.Equal(t, http.StatusOK, resp.Code)
		var body WorkflowRunDetailResponse
		require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
		require.Equal(t, "run-123", body.Run.RunID)
		require.Equal(t, "exec-root", body.Run.RootExecutionID)
		require.Equal(t, 2, body.Run.TotalSteps)
		require.Equal(t, 1, body.Run.FailedSteps)
		require.Equal(t, 0, body.Run.CompletedSteps)
		require.Len(t, body.Executions, 2)
		foundApproval := false
		for _, execution := range body.Executions {
			if execution.ExecutionID == "exec-root" {
				require.NotNil(t, execution.ApprovalRequestID)
				require.NotNil(t, execution.ApprovalStatus)
				foundApproval = true
			}
		}
		require.True(t, foundApproval)
	})

	t.Run("workflow run detail missing", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v2/workflow-runs/missing", nil)
		resp := httptest.NewRecorder()
		router.ServeHTTP(resp, req)
		require.Equal(t, http.StatusNotFound, resp.Code)
	})
}

func TestExecutionHandlerHelperCoverage(t *testing.T) {
	t.Run("parse time pointer", func(t *testing.T) {
		parsed, err := parseTimePtrValue("2026-04-08T12:00:00Z")
		require.NoError(t, err)
		require.NotNil(t, parsed)

		empty, err := parseTimePtrValue(" ")
		require.NoError(t, err)
		require.Nil(t, empty)

		_, err = parseTimePtrValue("invalid")
		require.Error(t, err)
	})

	t.Run("format duration display", func(t *testing.T) {
		cases := []struct {
			name     string
			duration *int64
			want     string
		}{
			{name: "nil", duration: nil, want: "—"},
			{name: "millis", duration: int64Ptr(250), want: "250ms"},
			{name: "seconds", duration: int64Ptr(3200), want: "3.2s"},
			{name: "minutes", duration: int64Ptr(2 * 60 * 1000), want: "2m"},
			{name: "minutes seconds", duration: int64Ptr(125000), want: "2m 5s"},
			{name: "hours", duration: int64Ptr(2 * 60 * 60 * 1000), want: "2h"},
			{name: "hours minutes", duration: int64Ptr(130 * 60 * 1000), want: "2h 10m"},
		}

		for _, tc := range cases {
			t.Run(tc.name, func(t *testing.T) {
				require.Equal(t, tc.want, formatDurationDisplay(tc.duration))
			})
		}
	})

	t.Run("group execution summaries", func(t *testing.T) {
		handler := &ExecutionHandler{}
		grouped := handler.groupExecutionSummaries([]ExecutionSummary{
			{ExecutionID: "1", Status: "failed", AgentNodeID: "agent-a", ReasonerID: "r1"},
			{ExecutionID: "2", Status: "failed", AgentNodeID: "agent-b", ReasonerID: "r2"},
			{ExecutionID: "3", Status: "succeeded", AgentNodeID: "agent-a", ReasonerID: "r1"},
		}, "agent")

		require.Len(t, grouped["agent-a"], 2)
		require.Len(t, grouped["agent-b"], 1)
		require.Len(t, handler.groupExecutionSummaries(nil, "unknown")["ungrouped"], 0)
	})

	t.Run("relative time string", func(t *testing.T) {
		now := time.Date(2026, 4, 8, 12, 0, 0, 0, time.UTC)
		require.Equal(t, "", formatRelativeTimeString(now, time.Time{}))
		require.Equal(t, "just now", formatRelativeTimeString(now, now.Add(-30*time.Second)))
		require.Equal(t, "5m ago", formatRelativeTimeString(now, now.Add(-5*time.Minute)))
		require.Equal(t, "3h ago", formatRelativeTimeString(now, now.Add(-3*time.Hour)))
		require.Equal(t, "2d ago", formatRelativeTimeString(now, now.Add(-48*time.Hour)))
	})

	t.Run("pagination helpers", func(t *testing.T) {
		require.Equal(t, 3, parsePositiveIntOrDefault("3", 1))
		require.Equal(t, 1, parsePositiveIntOrDefault("-1", 1))
		require.Equal(t, 5, parseBoundedIntOrDefault("50", 5, 1, 5))
		require.Equal(t, "reasoner_id", sanitizeExecutionSortField("reasoner"))
		require.Equal(t, "started_at", sanitizeExecutionSortField("unknown"))
		require.Equal(t, 3, computeTotalPages(5, 2))
		require.Equal(t, 1, computeTotalPages(0, 0))
	})
}
func int64Ptr(v int64) *int64 { return &v }
