package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	_ "unsafe"
)

type workflowRunOverrideStorage struct {
	storage.StorageProvider
	queryRunSummariesFn func(context.Context, types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error)
	queryExecutionRecordsFn func(context.Context, types.ExecutionFilter) ([]*types.Execution, error)
	getWorkflowExecutionFn func(context.Context, string) (*types.WorkflowExecution, error)
}

func (s *workflowRunOverrideStorage) QueryRunSummaries(ctx context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
	if s.queryRunSummariesFn != nil {
		return s.queryRunSummariesFn(ctx, filter)
	}
	return s.StorageProvider.QueryRunSummaries(ctx, filter)
}

func (s *workflowRunOverrideStorage) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	if s.queryExecutionRecordsFn != nil {
		return s.queryExecutionRecordsFn(ctx, filter)
	}
	return s.StorageProvider.QueryExecutionRecords(ctx, filter)
}

func (s *workflowRunOverrideStorage) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	if s.getWorkflowExecutionFn != nil {
		return s.getWorkflowExecutionFn(ctx, executionID)
	}
	return s.StorageProvider.GetWorkflowExecution(ctx, executionID)
}

type executionRecordOverrideStore struct {
	queryFn func(context.Context, types.ExecutionFilter) ([]*types.Execution, error)
	getFn   func(context.Context, string) (*types.Execution, error)
}

//go:linkname linkedConcurrencyLimiter github.com/Agent-Field/agentfield/control-plane/internal/handlers.concurrencyLimiter
var linkedConcurrencyLimiter *handlers.AgentConcurrencyLimiter

//go:linkname linkedConcurrencyLimiterOnce github.com/Agent-Field/agentfield/control-plane/internal/handlers.concurrencyLimiterOnce
var linkedConcurrencyLimiterOnce sync.Once

func (s *executionRecordOverrideStore) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	if s.queryFn != nil {
		return s.queryFn(ctx, filter)
	}
	return nil, nil
}

func (s *executionRecordOverrideStore) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	if s.getFn != nil {
		return s.getFn(ctx, executionID)
	}
	return nil, nil
}

func TestParseTimeRangeParamsAdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	now := time.Date(2026, 4, 7, 12, 34, 56, 0, time.UTC)
	rounded := time.Date(2026, 4, 7, 13, 0, 0, 0, time.UTC)

	tests := []struct {
		name       string
		target     string
		wantStart  time.Time
		wantEnd    time.Time
		wantPreset TimeRangePreset
	}{
		{
			name:       "1h preset",
			target:     "/dashboard?preset=1h",
			wantStart:  rounded.Add(-1 * time.Hour),
			wantEnd:    rounded,
			wantPreset: TimeRangePreset1h,
		},
		{
			name:       "7d preset",
			target:     "/dashboard?preset=7d",
			wantStart:  rounded.AddDate(0, 0, -7),
			wantEnd:    rounded,
			wantPreset: TimeRangePreset7d,
		},
		{
			name:       "30d preset",
			target:     "/dashboard?preset=30d",
			wantStart:  rounded.AddDate(0, 0, -30),
			wantEnd:    rounded,
			wantPreset: TimeRangePreset30d,
		},
		{
			name:       "custom missing end falls back",
			target:     "/dashboard?preset=custom&start_time=2026-04-01T00:00:00Z",
			wantStart:  now.Add(-24 * time.Hour),
			wantEnd:    now,
			wantPreset: TimeRangePreset24h,
		},
		{
			name:       "custom invalid end falls back",
			target:     "/dashboard?preset=custom&start_time=2026-04-01T00:00:00Z&end_time=bad",
			wantStart:  now.Add(-24 * time.Hour),
			wantEnd:    now,
			wantPreset: TimeRangePreset24h,
		},
		{
			name:       "unknown preset defaults to 24h rounded",
			target:     "/dashboard?preset=unknown",
			wantStart:  rounded.Add(-24 * time.Hour),
			wantEnd:    rounded,
			wantPreset: TimeRangePreset24h,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
			ctx.Request = httptest.NewRequest(http.MethodGet, tc.target, nil)

			start, end, preset, err := parseTimeRangeParams(ctx, now)
			require.NoError(t, err)
			require.Equal(t, tc.wantStart, start)
			require.Equal(t, tc.wantEnd, end)
			require.Equal(t, tc.wantPreset, preset)
		})
	}
}

func TestNodesHandlerLowCoveragePaths(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("details requires node id", func(t *testing.T) {
		handler := NewNodesHandler(services.NewUIService(setupTestStorage(t), nil, nil, nil))
		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/nodes/", nil)

		handler.GetNodeDetailsHandler(ctx)

		require.Equal(t, http.StatusBadRequest, rec.Code)
		require.Contains(t, rec.Body.String(), "nodeId is required")
	})

	t.Run("details success includes package info", func(t *testing.T) {
		store := setupTestStorage(t)
		ctx := context.Background()

		require.NoError(t, store.RegisterAgent(ctx, &types.AgentNode{
			ID:              "node-1",
			TeamID:          "team-1",
			Version:         "1.2.3",
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
		}))
		require.NoError(t, store.StoreAgentPackage(ctx, &types.AgentPackage{
			ID:                  "pkg-1",
			Version:             "9.9.9",
			Status:              types.PackageStatusInstalled,
			ConfigurationSchema: json.RawMessage(`{"agent_node":{"node_id":"node-1"}}`),
		}))

		handler := NewNodesHandler(services.NewUIService(store, nil, nil, nil))
		router := gin.New()
		router.GET("/api/ui/v1/nodes/:nodeId", handler.GetNodeDetailsHandler)

		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/nodes/node-1", nil)
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)

		var response services.NodeDetailsWithPackageInfo
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &response))
		require.Equal(t, "node-1", response.ID)
		require.NotNil(t, response.PackageInfo)
		require.Equal(t, "pkg-1", response.PackageInfo.PackageID)
		require.Equal(t, "9.9.9", response.PackageInfo.Version)
	})

	t.Run("summary surfaces storage errors", func(t *testing.T) {
		store := setupTestStorage(t)
		require.NoError(t, store.Close(context.Background()))

		handler := NewNodesHandler(services.NewUIService(store, nil, nil, nil))
		router := gin.New()
		router.GET("/api/ui/v1/nodes", handler.GetNodesSummaryHandler)

		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/nodes", nil)
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusInternalServerError, rec.Code)
		require.Contains(t, rec.Body.String(), "failed to get nodes summary")
	})
}

func TestWorkflowRunHandlerAdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("list handler returns query error details", func(t *testing.T) {
		base := setupTestStorage(t)
		store := &workflowRunOverrideStorage{
			StorageProvider: base,
			queryRunSummariesFn: func(context.Context, types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
				return nil, 0, errors.New("boom")
			},
		}

		router := gin.New()
		router.GET("/api/ui/v1/workflow-runs", NewWorkflowRunHandler(store).ListWorkflowRunsHandler)

		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/workflow-runs", nil)
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusInternalServerError, rec.Code)
		require.Contains(t, rec.Body.String(), "failed to query run summaries")
		require.Contains(t, rec.Body.String(), "boom")
	})

	t.Run("list handler parses filters and pagination", func(t *testing.T) {
		base := setupTestStorage(t)
		rootExecutionID := "exec-1"
		rootStatus := string(types.ExecutionStatusPaused)
		rootReasonerID := "planner"
		rootAgentNodeID := "agent-1"
		sessionID := "session-1"
		actorID := "actor-1"
		startTime := time.Date(2026, 4, 6, 10, 0, 0, 0, time.UTC)
		latest := startTime.Add(2 * time.Minute)

		store := &workflowRunOverrideStorage{
			StorageProvider: base,
			queryRunSummariesFn: func(_ context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
				require.Equal(t, 2, filter.Limit)
				require.Equal(t, 2, filter.Offset)
				require.Equal(t, "updated_at", filter.SortBy)
				require.False(t, filter.SortDescending)
				require.NotNil(t, filter.RunID)
				require.Equal(t, "run-1", *filter.RunID)
				require.NotNil(t, filter.Status)
				require.Equal(t, "paused", *filter.Status)
				require.NotNil(t, filter.SessionID)
				require.Equal(t, "session-1", *filter.SessionID)
				require.NotNil(t, filter.ActorID)
				require.Equal(t, "actor-1", *filter.ActorID)
				require.NotNil(t, filter.Search)
				require.Equal(t, "planner", *filter.Search)
				require.NotNil(t, filter.StartTime)
				require.Equal(t, time.Date(2026, 4, 1, 10, 0, 0, 0, time.UTC), *filter.StartTime)

				return []*storage.RunSummaryAggregation{{
					RunID:            "run-1",
					RootExecutionID:  &rootExecutionID,
					RootStatus:       &rootStatus,
					RootReasonerID:   &rootReasonerID,
					RootAgentNodeID:  &rootAgentNodeID,
					SessionID:        &sessionID,
					ActorID:          &actorID,
					StatusCounts:     map[string]int{string(types.ExecutionStatusPaused): 1},
					TotalExecutions:  3,
					MaxDepth:         2,
					ActiveExecutions: 0,
					EarliestStarted:  startTime,
					LatestStarted:    latest,
				}}, 0, nil
			},
		}

		router := gin.New()
		router.GET("/api/ui/v1/workflow-runs", NewWorkflowRunHandler(store).ListWorkflowRunsHandler)

		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/workflow-runs?page=2&page_size=2&sort_by=bogus&sort_order=asc&run_id=run-1&status=paused&session_id=session-1&actor_id=actor-1&since=2026-04-01T10:00:00Z&search=planner", nil)
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)

		var response WorkflowRunListResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &response))
		require.Equal(t, 1, response.TotalCount)
		require.Equal(t, 2, response.Page)
		require.Equal(t, 2, response.PageSize)
		require.False(t, response.HasMore)
		require.Len(t, response.Runs, 1)
		require.Equal(t, "paused", response.Runs[0].Status)
		require.Equal(t, "exec-1", response.Runs[0].RootExecutionID)
		require.Equal(t, "paused", response.Runs[0].RootExecutionStatus)
		require.Equal(t, "planner", response.Runs[0].DisplayName)
		require.False(t, response.Runs[0].Terminal)
	})

	t.Run("loadRunSummary handles success empty and error", func(t *testing.T) {
		base := setupTestStorage(t)
		expected := &storage.RunSummaryAggregation{RunID: "run-42"}

		successStore := &workflowRunOverrideStorage{
			StorageProvider: base,
			queryRunSummariesFn: func(_ context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
				require.NotNil(t, filter.RunID)
				require.Equal(t, "run-42", *filter.RunID)
				require.Equal(t, 1, filter.Limit)
				require.Equal(t, 0, filter.Offset)
				return []*storage.RunSummaryAggregation{expected}, 1, nil
			},
		}
		require.Same(t, expected, NewWorkflowRunHandler(successStore).loadRunSummary(context.Background(), "run-42"))

		emptyStore := &workflowRunOverrideStorage{
			StorageProvider: base,
			queryRunSummariesFn: func(context.Context, types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
				return nil, 0, nil
			},
		}
		require.Nil(t, NewWorkflowRunHandler(emptyStore).loadRunSummary(context.Background(), "run-empty"))

		errStore := &workflowRunOverrideStorage{
			StorageProvider: base,
			queryRunSummariesFn: func(context.Context, types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
				return nil, 0, errors.New("query failed")
			},
		}
		require.Nil(t, NewWorkflowRunHandler(errStore).loadRunSummary(context.Background(), "run-err"))
	})

	t.Run("detail handler validates missing id and query failures", func(t *testing.T) {
		base := setupTestStorage(t)
		handler := NewWorkflowRunHandler(&workflowRunOverrideStorage{
			StorageProvider: base,
			queryExecutionRecordsFn: func(context.Context, types.ExecutionFilter) ([]*types.Execution, error) {
				return nil, errors.New("records failed")
			},
		})

		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/workflow-runs/", nil)
		handler.GetWorkflowRunDetailHandler(ctx)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		router := gin.New()
		router.GET("/api/ui/v1/workflow-runs/:run_id", handler.GetWorkflowRunDetailHandler)
		rec = httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/workflow-runs/run-1", nil)
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})

	t.Run("detail handler falls back when run summary is unavailable", func(t *testing.T) {
		ls, ctx := setupUIHandlerStorage(t)
		seedWorkflowExecutions(t, ls, ctx)

		store := &workflowRunOverrideStorage{
			StorageProvider: ls,
			queryRunSummariesFn: func(context.Context, types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
				return nil, 0, errors.New("summary unavailable")
			},
		}

		router := gin.New()
		router.GET("/api/ui/v1/workflow-runs/:run_id", NewWorkflowRunHandler(store).GetWorkflowRunDetailHandler)

		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/workflow-runs/run-123", nil)
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		var response WorkflowRunDetailResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &response))
		require.Equal(t, "run-123", response.Run.RunID)
		require.Equal(t, 2, response.Run.TotalSteps)
		require.Equal(t, 1, response.Run.FailedSteps)
		require.NotEmpty(t, response.Run.StatusCounts)
	})
}

func TestObservabilityWebhookClearDeadLetterQueueError(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store, _, handler, _ := setupTestEnvironment(t)
	require.NoError(t, store.Close(context.Background()))

	router := gin.New()
	router.DELETE("/api/v1/settings/observability-webhook/dlq", handler.ClearDeadLetterQueueHandler)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodDelete, "/api/v1/settings/observability-webhook/dlq", nil)
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusInternalServerError, rec.Code)
	require.Contains(t, rec.Body.String(), "failed to clear dead letter queue")
}

func TestExecutionHandlerLowCoveragePaths(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("enhanced executions handler reports errors and applies filters", func(t *testing.T) {
		handler := &ExecutionHandler{
			store: &executionRecordOverrideStore{
				queryFn: func(context.Context, types.ExecutionFilter) ([]*types.Execution, error) {
					return nil, errors.New("query failed")
				},
			},
		}

		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/enhanced", nil)
		handler.GetEnhancedExecutionsHandler(ctx)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		sessionID := "session-9"
		actorID := "actor-9"
		completedAt := time.Date(2026, 4, 8, 12, 5, 0, 0, time.UTC)
		duration := int64(5000)
		var captured types.ExecutionFilter
		handler.store = &executionRecordOverrideStore{
			queryFn: func(_ context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
				captured = filter
				return []*types.Execution{
					nil,
					{
						ExecutionID: "exec-9",
						RunID:       "run-9",
						AgentNodeID: "agent-9",
						ReasonerID:  "agent-9.plan",
						Status:      "SUCCESS",
						StartedAt:   time.Date(2026, 4, 8, 12, 0, 0, 0, time.UTC),
						CompletedAt: &completedAt,
						DurationMS:  &duration,
						SessionID:   &sessionID,
						ActorID:     &actorID,
					},
				}, nil
			},
		}

		router := gin.New()
		router.GET("/api/ui/v1/executions/enhanced", handler.GetEnhancedExecutionsHandler)
		rec = httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/enhanced?page=2&limit=1&sort_by=workflow_id&sort_order=asc&status=SUCCESS&agent_id=agent-9&workflow_id=run-9&session_id=session-9&actor_id=actor-9&since=2026-04-01T10:00:00Z", nil)
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		require.Equal(t, 1, captured.Limit)
		require.Equal(t, 1, captured.Offset)
		require.Equal(t, "run_id", captured.SortBy)
		require.False(t, captured.SortDescending)
		require.NotNil(t, captured.Status)
		require.Equal(t, string(types.ExecutionStatusSucceeded), *captured.Status)
		require.NotNil(t, captured.AgentNodeID)
		require.Equal(t, "agent-9", *captured.AgentNodeID)
		require.NotNil(t, captured.RunID)
		require.Equal(t, "run-9", *captured.RunID)

		var response EnhancedExecutionsResponse
		require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &response))
		require.Len(t, response.Executions, 1)
		require.Equal(t, "exec-9", response.Executions[0].ExecutionID)
		require.False(t, response.HasMore)
		require.Equal(t, 3, response.TotalCount)
		require.Equal(t, 2, response.Page)
	})

	// NOTE: This test must NOT run in parallel — it resets the package-level
	// concurrencyLimiter via go:linkname because InitConcurrencyLimiter is
	// guarded by sync.Once and cannot be re-invoked otherwise.
	t.Run("queue status handler reports active slots", func(t *testing.T) {
		savedLimiter := linkedConcurrencyLimiter
		t.Cleanup(func() {
			linkedConcurrencyLimiter = savedLimiter
			linkedConcurrencyLimiterOnce = sync.Once{}
		})

		linkedConcurrencyLimiter = nil
		linkedConcurrencyLimiterOnce = sync.Once{}
		handlers.InitConcurrencyLimiter(2)
		limiter := handlers.GetConcurrencyLimiter()
		require.NotNil(t, limiter)
		require.NoError(t, limiter.Acquire("agent-a"))
		require.NoError(t, limiter.Acquire("agent-a"))
		require.NoError(t, limiter.Acquire("agent-b"))

		handler := NewExecutionLogsHandler(nil, nil, func() config.ExecutionLogsConfig { return config.ExecutionLogsConfig{} })
		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/api/ui/v1/executions/queue", nil)
		handler.GetExecutionQueueStatusHandler(ctx)

		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Body.String(), `"enabled":true`)
		require.Contains(t, rec.Body.String(), `"agent_node_id":"agent-a"`)
		require.Contains(t, rec.Body.String(), `"available":0`)
		require.Contains(t, rec.Body.String(), `"total_running":3`)
	})

	t.Run("toExecutionDetails enriches approvals and vcs", func(t *testing.T) {
		ls, ctx := setupUIHandlerStorage(t)
		seedWorkflowExecutions(t, ls, ctx)
		require.NoError(t, ls.StoreExecutionVC(ctx, "vc-1", "exec-root", "wf-root", "session-1", "did:issuer:1", "did:target:1", "did:caller:1", "input-hash", "output-hash", "verified", []byte(`{"vc":1}`), "sig", "s3://vc", 123))

		handler := NewExecutionHandler(ls, nil, nil)
		exec, err := ls.GetExecutionRecord(ctx, "exec-root")
		require.NoError(t, err)

		details := handler.toExecutionDetails(ctx, exec)
		require.NotNil(t, details.ApprovalRequestID)
		require.Equal(t, "approval-1", *details.ApprovalRequestID)
		require.NotNil(t, details.ApprovalRequestURL)
		require.NotNil(t, details.ApprovalStatus)
		require.NotNil(t, details.CallerDID)
		require.Equal(t, "did:caller:1", *details.CallerDID)
		require.NotNil(t, details.TargetDID)
		require.Equal(t, "did:target:1", *details.TargetDID)
		require.NotNil(t, details.InputHash)
		require.Equal(t, "input-hash", *details.InputHash)
		require.NotNil(t, details.OutputHash)
		require.Equal(t, "output-hash", *details.OutputHash)
	})
}

func TestEnvHandlerLowCoveragePaths(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("put env covers missing agent package errors and write failure", func(t *testing.T) {
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		handler := NewEnvHandler(storage, nil, t.TempDir())
		router := newEnvRouter(handler)

		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodPut, "/api/ui/v1/agents//env", nil)
		handler.PutEnvHandler(ctx)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		rec = performJSONRequest(router, http.MethodPut, "/api/ui/v1/agents/agent-1/env?packageId=pkg", map[string]any{"variables": map[string]string{"KEY": "value"}})
		require.Equal(t, http.StatusNotFound, rec.Code)

		storage.getAgentPackageFn = func(context.Context, string) (*types.AgentPackage, error) {
			return &types.AgentPackage{ID: "pkg", InstallPath: filepath.Join(t.TempDir(), "missing", "nested")}, nil
		}
		storage.validateAgentConfigurationFn = func(context.Context, string, string, map[string]interface{}) (*types.ConfigurationValidationResult, error) {
			return nil, errors.New("boom")
		}
		rec = performJSONRequest(router, http.MethodPut, "/api/ui/v1/agents/agent-1/env?packageId=pkg", map[string]any{"variables": map[string]string{"KEY": "value"}})
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		storage.validateAgentConfigurationFn = func(context.Context, string, string, map[string]interface{}) (*types.ConfigurationValidationResult, error) {
			return &types.ConfigurationValidationResult{Valid: true}, nil
		}
		rec = performJSONRequest(router, http.MethodPut, "/api/ui/v1/agents/agent-1/env?packageId=pkg", map[string]any{"variables": map[string]string{"KEY": "value"}})
		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})

	t.Run("patch and delete env cover more error paths", func(t *testing.T) {
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		handler := NewEnvHandler(storage, nil, t.TempDir())
		router := newEnvRouter(handler)

		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodDelete, "/api/ui/v1/agents/agent-1/env/", nil)
		ctx.Params = gin.Params{{Key: "agentId", Value: "agent-1"}}
		handler.DeleteEnvVarHandler(ctx)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		rec = performJSONRequest(router, http.MethodPatch, "/api/ui/v1/agents/agent-1/env?packageId=pkg", map[string]any{"variables": map[string]string{"A": "b"}})
		require.Equal(t, http.StatusNotFound, rec.Code)

		storage.getAgentPackageFn = func(context.Context, string) (*types.AgentPackage, error) {
			return &types.AgentPackage{ID: "pkg", InstallPath: filepath.Join(t.TempDir(), "missing", "nested")}, nil
		}
		storage.validateAgentConfigurationFn = func(context.Context, string, string, map[string]interface{}) (*types.ConfigurationValidationResult, error) {
			return nil, errors.New("bad validate")
		}
		rec = performJSONRequest(router, http.MethodPatch, "/api/ui/v1/agents/agent-1/env?packageId=pkg", map[string]any{"variables": map[string]string{"A": "b"}})
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		storage.validateAgentConfigurationFn = func(context.Context, string, string, map[string]interface{}) (*types.ConfigurationValidationResult, error) {
			return &types.ConfigurationValidationResult{Valid: true}, nil
		}
		rec = performJSONRequest(router, http.MethodPatch, "/api/ui/v1/agents/agent-1/env?packageId=pkg", map[string]any{"variables": map[string]string{"A": "b"}})
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		storage.getAgentPackageFn = func(context.Context, string) (*types.AgentPackage, error) {
			dir := t.TempDir()
			require.NoError(t, os.WriteFile(filepath.Join(dir, ".env"), []byte("KEEP=value\n"), 0600))
			return &types.AgentPackage{ID: "pkg", InstallPath: dir}, nil
		}
		storage.validateAgentConfigurationFn = func(context.Context, string, string, map[string]interface{}) (*types.ConfigurationValidationResult, error) {
			return nil, errors.New("delete validate")
		}
		rec = performJSONRequest(router, http.MethodDelete, "/api/ui/v1/agents/agent-1/env/KEEP?packageId=pkg", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})
}

func TestReasonersAndObservabilityLowCoveragePaths(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("reasoner handlers cover template validation and event stream", func(t *testing.T) {
		storage := &overrideStorage{StorageProvider: setupTestStorage(t)}
		handler := NewReasonersHandler(storage)
		router := gin.New()
		router.GET("/api/ui/v1/reasoners", handler.GetAllReasonersHandler)
		router.POST("/api/ui/v1/reasoners/:reasonerId/templates", handler.SaveExecutionTemplateHandler)
		router.GET("/api/ui/v1/reasoners/events", handler.StreamReasonerEventsHandler)

		storage.listAgentsFn = func(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
			return []*types.AgentNode{{
				ID:           "agent-offline",
				Version:      "v1",
				HealthStatus: types.HealthStatusInactive,
				Reasoners:    []types.ReasonerDefinition{{ID: "plan"}},
				LastHeartbeat: time.Now().UTC(),
			}}, nil
		}
		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/reasoners?status=offline&limit=5&offset=99", nil)
		require.Equal(t, http.StatusOK, rec.Code)

		reqBad := httptest.NewRequest(http.MethodPost, "/api/ui/v1/reasoners/agent-offline.plan/templates", bytes.NewBufferString("{"))
		reqBad.Header.Set("Content-Type", "application/json")
		rec = httptest.NewRecorder()
		router.ServeHTTP(rec, reqBad)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/reasoners/events", nil)
		ctx, cancel := context.WithCancel(req.Context())
		req = req.WithContext(ctx)
		resp := httptest.NewRecorder()
		done := make(chan struct{})
		go func() {
			router.ServeHTTP(resp, req)
			close(done)
		}()

		time.Sleep(20 * time.Millisecond)
		events.GlobalReasonerEventBus.Publish(events.ReasonerEvent{
			Type:       "updated",
			ReasonerID: "agent-offline.plan",
			NodeID:     "agent-offline",
			Timestamp:  time.Now().UTC(),
		})
		time.Sleep(20 * time.Millisecond)
		cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("reasoner stream did not finish")
		}
		require.Contains(t, resp.Body.String(), "Reasoner events stream connected")
		require.Contains(t, resp.Body.String(), `"reasoner_id":"agent-offline.plan"`)
	})

	t.Run("observability handlers cover error and reload-pending branches", func(t *testing.T) {
		store, fwd, handler, _ := setupTestEnvironment(t)
		router := gin.New()
		router.GET("/api/v1/settings/observability-webhook", handler.GetWebhookHandler)
		router.POST("/api/v1/settings/observability-webhook", handler.SetWebhookHandler)
		router.DELETE("/api/v1/settings/observability-webhook", handler.DeleteWebhookHandler)

		fwd.reloadErr = errors.New("reload later")
		rec := performJSONRequest(router, http.MethodPost, "/api/v1/settings/observability-webhook", map[string]any{
			"url": "https://reload.example.com/hook",
		})
		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Body.String(), "reload pending")

		require.NoError(t, store.Close(context.Background()))

		rec = performJSONRequest(router, http.MethodGet, "/api/v1/settings/observability-webhook", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		rec = performJSONRequest(router, http.MethodDelete, "/api/v1/settings/observability-webhook", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})
}
