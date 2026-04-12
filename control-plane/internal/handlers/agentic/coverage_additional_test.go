package agentic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

type handlerTestStorage struct {
	*mockStatusStorage
	getAgentFn          func(context.Context, string) (*types.AgentNode, error)
	queryRunSummariesFn func(context.Context, types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error)
	queryWorkflowsFn    func(context.Context, types.WorkflowFilters) ([]*types.Workflow, error)
	querySessionsFn     func(context.Context, types.SessionFilters) ([]*types.Session, error)
}

func (s *handlerTestStorage) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	if s.getAgentFn != nil {
		return s.getAgentFn(ctx, id)
	}
	return s.mockStatusStorage.GetAgent(ctx, id)
}

func (s *handlerTestStorage) QueryRunSummaries(ctx context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
	if s.queryRunSummariesFn != nil {
		return s.queryRunSummariesFn(ctx, filter)
	}
	return s.mockStatusStorage.QueryRunSummaries(ctx, filter)
}

func (s *handlerTestStorage) QueryWorkflows(ctx context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error) {
	if s.queryWorkflowsFn != nil {
		return s.queryWorkflowsFn(ctx, filters)
	}
	return s.mockStatusStorage.QueryWorkflows(ctx, filters)
}

func (s *handlerTestStorage) QuerySessions(ctx context.Context, filters types.SessionFilters) ([]*types.Session, error) {
	if s.querySessionsFn != nil {
		return s.querySessionsFn(ctx, filters)
	}
	return s.mockStatusStorage.QuerySessions(ctx, filters)
}

func decodeEnvelope(t *testing.T, body *bytes.Buffer) AgenticResponse {
	t.Helper()

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(body.Bytes(), &resp))
	return resp
}

func TestRespondErrorWithDetails(t *testing.T) {
	router := gin.New()
	router.GET("/test", func(c *gin.Context) {
		respondErrorWithDetails(c, http.StatusConflict, "conflict", "request conflicted", gin.H{"field": "status"})
	})

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusConflict, rec.Code)
	resp := decodeEnvelope(t, rec.Body)
	require.False(t, resp.OK)
	require.NotNil(t, resp.Error)
	assert.Equal(t, "conflict", resp.Error.Code)
	assert.Equal(t, "request conflicted", resp.Error.Message)
	assert.Equal(t, "status", resp.Error.Details.(map[string]interface{})["field"])
}

func TestGetAuthLevel_AdditionalFallbacks(t *testing.T) {
	router := gin.New()
	router.GET("/test", func(c *gin.Context) {
		if c.Query("bad_ctx") == "1" {
			c.Set("auth_level", 123)
		}
		c.JSON(http.StatusOK, gin.H{"level": getAuthLevel(c)})
	})

	tests := []struct {
		name     string
		url      string
		headers  map[string]string
		expected string
	}{
		{
			name:     "authorization bearer header",
			url:      "/test",
			headers:  map[string]string{"Authorization": "Bearer token"},
			expected: "api_key",
		},
		{
			name:     "api key query parameter",
			url:      "/test?api_key=abc",
			expected: "api_key",
		},
		{
			name:     "non string context falls back to public",
			url:      "/test?bad_ctx=1",
			expected: "public",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, tt.url, nil)
			for k, v := range tt.headers {
				req.Header.Set(k, v)
			}
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			var body map[string]string
			require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
			assert.Equal(t, tt.expected, body["level"])
		})
	}
}

func TestAgentSummaryHandler(t *testing.T) {
	t.Run("missing agent id", func(t *testing.T) {
		store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
		router := gin.New()
		router.GET("/agents/summary", AgentSummaryHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/agents/summary", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusBadRequest, rec.Code)
		resp := decodeEnvelope(t, rec.Body)
		assert.Equal(t, "missing_agent_id", resp.Error.Code)
	})

	t.Run("query failed", func(t *testing.T) {
		store := &handlerTestStorage{
			mockStatusStorage: &mockStatusStorage{},
			getAgentFn: func(context.Context, string) (*types.AgentNode, error) {
				return nil, errors.New("boom")
			},
		}
		router := gin.New()
		router.GET("/agents/:agent_id/summary", AgentSummaryHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/agents/agent-1/summary", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusInternalServerError, rec.Code)
		resp := decodeEnvelope(t, rec.Body)
		assert.Equal(t, "query_failed", resp.Error.Code)
	})

	t.Run("agent not found", func(t *testing.T) {
		store := &handlerTestStorage{
			mockStatusStorage: &mockStatusStorage{},
			getAgentFn: func(context.Context, string) (*types.AgentNode, error) {
				return nil, nil
			},
		}
		router := gin.New()
		router.GET("/agents/:agent_id/summary", AgentSummaryHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/agents/agent-404/summary", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusNotFound, rec.Code)
		resp := decodeEnvelope(t, rec.Body)
		assert.Equal(t, "agent_not_found", resp.Error.Code)
	})

	t.Run("success with metrics", func(t *testing.T) {
		duration := int64(120)
		completedAt := time.Now()
		store := &handlerTestStorage{
			mockStatusStorage: &mockStatusStorage{},
			getAgentFn: func(context.Context, string) (*types.AgentNode, error) {
				return &types.AgentNode{ID: "agent-1", Version: "v1"}, nil
			},
		}
		store.On("QueryExecutionRecords", mock.Anything, mock.MatchedBy(func(filter types.ExecutionFilter) bool {
			return filter.AgentNodeID != nil && *filter.AgentNodeID == "agent-1" && filter.StartTime != nil
		})).Return([]*types.Execution{
			{ExecutionID: "exec-1", Status: "completed", CompletedAt: &completedAt, DurationMS: &duration},
			{ExecutionID: "exec-2", Status: "completed"},
			{ExecutionID: "exec-3", Status: "failed"},
		}, nil)

		router := gin.New()
		router.GET("/agents/:agent_id/summary", AgentSummaryHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/agents/agent-1/summary", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		resp := decodeEnvelope(t, rec.Body)
		data := resp.Data.(map[string]interface{})
		metrics := data["metrics_24h"].(map[string]interface{})
		assert.Equal(t, float64(3), metrics["total_executions"])
		assert.Equal(t, float64(1), metrics["completed_count"])
		assert.Equal(t, float64(120), metrics["avg_duration_ms"])
		assert.Equal(t, float64(2), metrics["status_counts"].(map[string]interface{})["completed"])
		assert.Equal(t, float64(1), metrics["status_counts"].(map[string]interface{})["failed"])
		store.AssertExpectations(t)
	})
}

func TestRunOverviewHandler(t *testing.T) {
	t.Run("missing run id", func(t *testing.T) {
		store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
		router := gin.New()
		router.GET("/runs", RunOverviewHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/runs", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusBadRequest, rec.Code)
		resp := decodeEnvelope(t, rec.Body)
		assert.Equal(t, "missing_run_id", resp.Error.Code)
	})

	t.Run("query failed", func(t *testing.T) {
		store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
		store.On("QueryExecutionRecords", mock.Anything, mock.Anything).Return(nil, errors.New("db down"))

		router := gin.New()
		router.GET("/runs/:run_id", RunOverviewHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/runs/run-1", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusInternalServerError, rec.Code)
		resp := decodeEnvelope(t, rec.Body)
		assert.Equal(t, "query_failed", resp.Error.Code)
		store.AssertExpectations(t)
	})

	t.Run("run not found", func(t *testing.T) {
		store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
		store.On("QueryExecutionRecords", mock.Anything, mock.Anything).Return([]*types.Execution{}, nil)

		router := gin.New()
		router.GET("/runs/:run_id", RunOverviewHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/runs/run-404", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusNotFound, rec.Code)
		resp := decodeEnvelope(t, rec.Body)
		assert.Equal(t, "run_not_found", resp.Error.Code)
		store.AssertExpectations(t)
	})

	t.Run("success aggregates agents statuses and notes", func(t *testing.T) {
		now := time.Now()
		store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
		store.On("QueryExecutionRecords", mock.Anything, mock.MatchedBy(func(filter types.ExecutionFilter) bool {
			return filter.RunID != nil && *filter.RunID == "run-1"
		})).Return([]*types.Execution{
			{
				ExecutionID: "exec-1",
				RunID:       "run-1",
				AgentNodeID: "agent-a",
				Status:      "completed",
				Notes:       []types.ExecutionNote{{Message: "first", Timestamp: now}},
			},
			{
				ExecutionID: "exec-2",
				RunID:       "run-1",
				AgentNodeID: "agent-b",
				Status:      "failed",
				Notes:       []types.ExecutionNote{{Message: "second", Timestamp: now}},
			},
			{
				ExecutionID: "exec-3",
				RunID:       "run-1",
				AgentNodeID: "agent-a",
				Status:      "completed",
			},
		}, nil)

		router := gin.New()
		router.GET("/runs/:run_id", RunOverviewHandler(store))

		req := httptest.NewRequest(http.MethodGet, "/runs/run-1", nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		resp := decodeEnvelope(t, rec.Body)
		data := resp.Data.(map[string]interface{})
		assert.Equal(t, "run-1", data["run_id"])
		assert.Len(t, data["executions"].([]interface{}), 3)
		assert.Len(t, data["agents"].([]interface{}), 2)
		assert.Len(t, data["notes"].([]interface{}), 2)
		summary := data["summary"].(map[string]interface{})
		assert.Equal(t, float64(3), summary["total_executions"])
		assert.Equal(t, float64(2), summary["unique_agents"])
		assert.Equal(t, float64(2), summary["status_counts"].(map[string]interface{})["completed"])
		assert.Equal(t, float64(1), summary["status_counts"].(map[string]interface{})["failed"])
		store.AssertExpectations(t)
	})
}

func TestQueryHandler_SuccessPaths(t *testing.T) {
	now := time.Now().UTC()
	since := now.Add(-time.Hour).Format(time.RFC3339)
	until := now.Format(time.RFC3339)
	status := "completed"
	agentID := "agent-1"
	runID := "run-1"
	sessionID := "session-1"
	actorID := "actor-1"

	tests := []struct {
		name     string
		body     string
		store    *handlerTestStorage
		validate func(*testing.T, map[string]interface{})
	}{
		{
			name: "runs",
			body: `{"resource":"runs","filters":{"status":"completed","agent_id":"agent-1","since":"` + since + `"},"limit":0,"offset":3}`,
			store: &handlerTestStorage{
				mockStatusStorage: &mockStatusStorage{},
				queryRunSummariesFn: func(_ context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
					require.NotNil(t, filter.Status)
					require.NotNil(t, filter.AgentNodeID)
					require.NotNil(t, filter.StartTime)
					assert.Equal(t, status, *filter.Status)
					assert.Equal(t, agentID, *filter.AgentNodeID)
					return []*storage.RunSummaryAggregation{{RunID: runID, TotalExecutions: 2}}, 9, nil
				},
			},
			validate: func(t *testing.T, data map[string]interface{}) {
				assert.Equal(t, "runs", data["resource"])
				assert.Equal(t, float64(9), data["total"])
				assert.Equal(t, float64(20), data["limit"])
				assert.Equal(t, float64(3), data["offset"])
				assert.Len(t, data["results"].([]interface{}), 1)
			},
		},
		{
			name: "executions",
			body: `{"resource":"executions","filters":{"status":"completed","agent_id":"agent-1","run_id":"run-1","session_id":"session-1","actor_id":"actor-1","since":"` + since + `","until":"` + until + `"},"limit":1,"offset":1}`,
			store: func() *handlerTestStorage {
				store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
				store.On("QueryExecutionRecords", mock.Anything, mock.MatchedBy(func(filter types.ExecutionFilter) bool {
					return filter.Status != nil && *filter.Status == status &&
						filter.AgentNodeID != nil && *filter.AgentNodeID == agentID &&
						filter.RunID != nil && *filter.RunID == runID &&
						filter.SessionID != nil && *filter.SessionID == sessionID &&
						filter.ActorID != nil && *filter.ActorID == actorID &&
						filter.StartTime != nil && filter.EndTime != nil
				})).Return([]*types.Execution{
					{ExecutionID: "exec-1"},
					{ExecutionID: "exec-2"},
					{ExecutionID: "exec-3"},
				}, nil)
				return store
			}(),
			validate: func(t *testing.T, data map[string]interface{}) {
				assert.Equal(t, "executions", data["resource"])
				assert.Equal(t, float64(3), data["total"])
				assert.Equal(t, float64(1), data["limit"])
				assert.Equal(t, float64(1), data["offset"])
				results := data["results"].([]interface{})
				assert.Len(t, results, 1)
				assert.Equal(t, "exec-2", results[0].(map[string]interface{})["execution_id"])
			},
		},
		{
			name: "agents",
			body: `{"resource":"agents","limit":1,"offset":1}`,
			store: func() *handlerTestStorage {
				store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
				store.On("ListAgents", mock.Anything, mock.Anything).Return([]*types.AgentNode{
					{ID: "agent-0"},
					{ID: "agent-1"},
					{ID: "agent-2"},
				}, nil)
				return store
			}(),
			validate: func(t *testing.T, data map[string]interface{}) {
				assert.Equal(t, "agents", data["resource"])
				assert.Equal(t, float64(3), data["total"])
				results := data["results"].([]interface{})
				assert.Len(t, results, 1)
				assert.Equal(t, "agent-1", results[0].(map[string]interface{})["id"])
			},
		},
		{
			name: "workflows",
			body: `{"resource":"workflows","filters":{"status":"completed","session_id":"session-1","actor_id":"actor-1","since":"` + since + `","until":"` + until + `"},"limit":1,"offset":1}`,
			store: &handlerTestStorage{
				mockStatusStorage: &mockStatusStorage{},
				queryWorkflowsFn: func(_ context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error) {
					require.NotNil(t, filters.Status)
					require.NotNil(t, filters.SessionID)
					require.NotNil(t, filters.ActorID)
					require.NotNil(t, filters.StartTime)
					require.NotNil(t, filters.EndTime)
					assert.Equal(t, status, *filters.Status)
					assert.Equal(t, sessionID, *filters.SessionID)
					assert.Equal(t, actorID, *filters.ActorID)
					return []*types.Workflow{
						{WorkflowID: "wf-0"},
						{WorkflowID: "wf-1"},
					}, nil
				},
			},
			validate: func(t *testing.T, data map[string]interface{}) {
				assert.Equal(t, "workflows", data["resource"])
				assert.Equal(t, float64(2), data["total"])
				results := data["results"].([]interface{})
				assert.Len(t, results, 1)
				assert.Equal(t, "wf-1", results[0].(map[string]interface{})["workflow_id"])
			},
		},
		{
			name: "sessions",
			body: `{"resource":"sessions","filters":{"actor_id":"actor-1","since":"` + since + `","until":"` + until + `"},"limit":1,"offset":1}`,
			store: &handlerTestStorage{
				mockStatusStorage: &mockStatusStorage{},
				querySessionsFn: func(_ context.Context, filters types.SessionFilters) ([]*types.Session, error) {
					require.NotNil(t, filters.ActorID)
					require.NotNil(t, filters.StartTime)
					require.NotNil(t, filters.EndTime)
					assert.Equal(t, actorID, *filters.ActorID)
					return []*types.Session{
						{SessionID: "sess-0"},
						{SessionID: "sess-1"},
					}, nil
				},
			},
			validate: func(t *testing.T, data map[string]interface{}) {
				assert.Equal(t, "sessions", data["resource"])
				assert.Equal(t, float64(2), data["total"])
				results := data["results"].([]interface{})
				assert.Len(t, results, 1)
				assert.Equal(t, "sess-1", results[0].(map[string]interface{})["session_id"])
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := gin.New()
			router.POST("/query", QueryHandler(tt.store))

			req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(tt.body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			require.Equal(t, http.StatusOK, rec.Code)
			resp := decodeEnvelope(t, rec.Body)
			require.True(t, resp.OK)
			tt.validate(t, resp.Data.(map[string]interface{}))
			tt.store.AssertExpectations(t)
		})
	}
}

func TestQueryHandler_ErrorPaths(t *testing.T) {
	t.Run("invalid json", func(t *testing.T) {
		store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
		router := gin.New()
		router.POST("/query", QueryHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(`{"resource":`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusBadRequest, rec.Code)
		resp := decodeEnvelope(t, rec.Body)
		assert.Equal(t, "invalid_request", resp.Error.Code)
	})

	t.Run("invalid resource", func(t *testing.T) {
		store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
		router := gin.New()
		router.POST("/query", QueryHandler(store))

		req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(`{"resource":"unknown"}`))
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusBadRequest, rec.Code)
		resp := decodeEnvelope(t, rec.Body)
		assert.Equal(t, "invalid_resource", resp.Error.Code)
	})

	tests := []struct {
		name  string
		body  string
		store *handlerTestStorage
	}{
		{
			name: "runs query failed",
			body: `{"resource":"runs"}`,
			store: &handlerTestStorage{
				mockStatusStorage: &mockStatusStorage{},
				queryRunSummariesFn: func(context.Context, types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
					return nil, 0, errors.New("runs failed")
				},
			},
		},
		{
			name: "executions query failed",
			body: `{"resource":"executions"}`,
			store: func() *handlerTestStorage {
				store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
				store.On("QueryExecutionRecords", mock.Anything, mock.Anything).Return(nil, errors.New("execs failed"))
				return store
			}(),
		},
		{
			name: "agents query failed",
			body: `{"resource":"agents"}`,
			store: func() *handlerTestStorage {
				store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
				store.On("ListAgents", mock.Anything, mock.Anything).Return(nil, errors.New("agents failed"))
				return store
			}(),
		},
		{
			name: "workflows query failed",
			body: `{"resource":"workflows"}`,
			store: &handlerTestStorage{
				mockStatusStorage: &mockStatusStorage{},
				queryWorkflowsFn: func(context.Context, types.WorkflowFilters) ([]*types.Workflow, error) {
					return nil, errors.New("workflows failed")
				},
			},
		},
		{
			name: "sessions query failed",
			body: `{"resource":"sessions"}`,
			store: &handlerTestStorage{
				mockStatusStorage: &mockStatusStorage{},
				querySessionsFn: func(context.Context, types.SessionFilters) ([]*types.Session, error) {
					return nil, errors.New("sessions failed")
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			router := gin.New()
			router.POST("/query", QueryHandler(tt.store))

			req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(tt.body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			require.Equal(t, http.StatusInternalServerError, rec.Code)
			resp := decodeEnvelope(t, rec.Body)
			assert.Equal(t, "query_failed", resp.Error.Code)
			tt.store.AssertExpectations(t)
		})
	}
}
