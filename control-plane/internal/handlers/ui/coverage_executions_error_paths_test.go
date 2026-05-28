package ui

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	storagepkg "github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type execOverrideStorage struct {
	storagepkg.StorageProvider
	queryExecutionRecordsFn func(context.Context, types.ExecutionFilter) ([]*types.Execution, error)
	getExecutionRecordFn    func(context.Context, string) (*types.Execution, error)
	hasExecutionWebhookFn   func(context.Context, string) (bool, error)
}

func (s *execOverrideStorage) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	if s.queryExecutionRecordsFn != nil {
		return s.queryExecutionRecordsFn(ctx, filter)
	}
	return s.StorageProvider.QueryExecutionRecords(ctx, filter)
}

func (s *execOverrideStorage) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	if s.getExecutionRecordFn != nil {
		return s.getExecutionRecordFn(ctx, executionID)
	}
	return s.StorageProvider.GetExecutionRecord(ctx, executionID)
}

func (s *execOverrideStorage) HasExecutionWebhook(ctx context.Context, executionID string) (bool, error) {
	if s.hasExecutionWebhookFn != nil {
		return s.hasExecutionWebhookFn(ctx, executionID)
	}
	return s.StorageProvider.HasExecutionWebhook(ctx, executionID)
}

func TestExecutionHandlerAdditionalCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	base := setupTestStorage(t)
	store := &execOverrideStorage{StorageProvider: base}
	dispatcher := &recordingWebhookDispatcher{}
	handler := NewExecutionHandler(store, nil, dispatcher)
	router := gin.New()
	router.GET("/api/ui/v1/agents/:agentId/executions/:executionId", handler.GetExecutionDetailsHandler)
	router.GET("/api/ui/v1/executions/summary", handler.GetExecutionsSummaryHandler)
	router.GET("/api/ui/v1/executions/:execution_id/details", handler.GetExecutionDetailsGlobalHandler)
	router.POST("/api/ui/v1/executions/:execution_id/webhook/retry", handler.RetryExecutionWebhookHandler)

	t.Run("execution details handlers cover validation and load errors", func(t *testing.T) {
		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/", nil)
		handler.GetExecutionDetailsHandler(ctx)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		store.getExecutionRecordFn = func(ctx context.Context, executionID string) (*types.Execution, error) {
			return nil, errors.New("boom")
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/agent-1/executions/exec-1", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		store.getExecutionRecordFn = func(ctx context.Context, executionID string) (*types.Execution, error) {
			return &types.Execution{ExecutionID: executionID, AgentNodeID: "other-agent"}, nil
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/agents/agent-1/executions/exec-1", nil)
		require.Equal(t, http.StatusNotFound, rec.Code)

		rec = httptest.NewRecorder()
		ctx, _ = gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/", nil)
		handler.GetExecutionDetailsGlobalHandler(ctx)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		store.getExecutionRecordFn = func(ctx context.Context, executionID string) (*types.Execution, error) {
			return nil, errors.New("load failed")
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/exec-1/details", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		started := time.Date(2026, 4, 8, 12, 0, 0, 0, time.UTC)
		duration := int64(1500)
		store.getExecutionRecordFn = func(ctx context.Context, executionID string) (*types.Execution, error) {
			return &types.Execution{
				ExecutionID:  executionID,
				RunID:        "wf-1",
				AgentNodeID:  "agent-1",
				ReasonerID:   "planner",
				Status:       string(types.ExecutionStatusSucceeded),
				StartedAt:    started,
				DurationMS:   &duration,
				InputPayload: []byte(`{"ok":true}`),
			}, nil
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/exec-1/details", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		body := decodeJSONResponse[ExecutionDetailsResponse](t, rec)
		require.Equal(t, "exec-1", body.ExecutionID)
		require.Equal(t, "agent-1", body.AgentNodeID)
	})

	t.Run("executions summary covers standard and error paths", func(t *testing.T) {
		rec := performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/summary?end_time=bad", nil)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		start := time.Date(2026, 4, 8, 10, 0, 0, 0, time.UTC)
		end := start.Add(2 * time.Hour)
		duration := int64(500)
		store.queryExecutionRecordsFn = func(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
			require.Equal(t, 2, filter.Limit)
			require.Equal(t, 2, filter.Offset)
			require.NotNil(t, filter.Status)
			require.Equal(t, "failed", *filter.Status)
			require.NotNil(t, filter.RunID)
			require.Equal(t, "wf-1", *filter.RunID)
			require.NotNil(t, filter.AgentNodeID)
			require.Equal(t, "agent-1", *filter.AgentNodeID)
			require.NotNil(t, filter.SessionID)
			require.Equal(t, "session-1", *filter.SessionID)
			require.NotNil(t, filter.StartTime)
			require.True(t, filter.StartTime.Equal(start))
			require.NotNil(t, filter.EndTime)
			require.True(t, filter.EndTime.Equal(end))
			return []*types.Execution{{
				ExecutionID: "exec-summary",
				RunID:       "wf-1",
				AgentNodeID: "agent-1",
				ReasonerID:  "planner",
				Status:      string(types.ExecutionStatusFailed),
				StartedAt:   start,
				DurationMS:  &duration,
			}}, nil
		}

		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/summary?page=2&page_size=2&status=failed&workflow_id=wf-1&agent_node_id=agent-1&session_id=session-1&start_time=2026-04-08T10:00:00Z&end_time=2026-04-08T12:00:00Z", nil)
		require.Equal(t, http.StatusOK, rec.Code)
		summary := decodeJSONResponse[ExecutionsSummaryResponse](t, rec)
		require.Len(t, summary.Executions, 1)
		require.Equal(t, 1, summary.Total)

		store.queryExecutionRecordsFn = func(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
			return nil, errors.New("query failed")
		}
		rec = performJSONRequest(router, http.MethodGet, "/api/ui/v1/executions/summary", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})

	t.Run("retry execution webhook covers error branches", func(t *testing.T) {
		noWebhookHandler := NewExecutionHandler(store, nil, nil)
		noWebhookRouter := gin.New()
		noWebhookRouter.POST("/api/ui/v1/executions/:execution_id/webhook/retry", noWebhookHandler.RetryExecutionWebhookHandler)
		rec := performJSONRequest(noWebhookRouter, http.MethodPost, "/api/ui/v1/executions/exec-1/webhook/retry", nil)
		require.Equal(t, http.StatusServiceUnavailable, rec.Code)

		rec = httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodPost, "/", nil)
		handler.RetryExecutionWebhookHandler(ctx)
		require.Equal(t, http.StatusBadRequest, rec.Code)

		store.getExecutionRecordFn = func(ctx context.Context, executionID string) (*types.Execution, error) {
			return nil, errors.New("load failed")
		}
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/executions/exec-1/webhook/retry", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		store.getExecutionRecordFn = func(ctx context.Context, executionID string) (*types.Execution, error) {
			return nil, nil
		}
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/executions/exec-1/webhook/retry", nil)
		require.Equal(t, http.StatusNotFound, rec.Code)

		store.getExecutionRecordFn = func(ctx context.Context, executionID string) (*types.Execution, error) {
			return &types.Execution{ExecutionID: executionID, AgentNodeID: "agent-1"}, nil
		}
		store.hasExecutionWebhookFn = func(ctx context.Context, executionID string) (bool, error) {
			return false, errors.New("lookup failed")
		}
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/executions/exec-1/webhook/retry", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)

		store.hasExecutionWebhookFn = func(ctx context.Context, executionID string) (bool, error) {
			return true, nil
		}
		dispatcher.err = errors.New("queue failed")
		rec = performJSONRequest(router, http.MethodPost, "/api/ui/v1/executions/exec-1/webhook/retry", nil)
		require.Equal(t, http.StatusInternalServerError, rec.Code)
		dispatcher.err = nil
	})
}
