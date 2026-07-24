package agentic

import (
	"bytes"
	"context"
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

func TestQueryHandler_DefaultLimit(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		expLimit float64
	}{
		{"limit zero defaults to 20", `{"resource":"agents","limit":0}`, 20},
		{"negative limit defaults to 20", `{"resource":"agents","limit":-5}`, 20},
		{"limit over 100 clamped to 20", `{"resource":"agents","limit":999}`, 20},
		{"valid limit preserved", `{"resource":"agents","limit":3}`, 3},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
			store.On("ListAgents", mock.Anything, mock.Anything).Return([]*types.AgentNode{}, nil)

			router := gin.New()
			router.POST("/query", QueryHandler(store))

			req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(tt.body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			require.Equal(t, http.StatusOK, rec.Code)
			resp := decodeEnvelope(t, rec.Body)
			data := resp.Data.(map[string]interface{})
			assert.Equal(t, tt.expLimit, data["limit"])
			store.AssertExpectations(t)
		})
	}
}

func TestQueryHandler_WorkflowsInvalidSince(t *testing.T) {
	store := &handlerTestStorage{
		mockStatusStorage: &mockStatusStorage{},
		queryWorkflowsFn: func(_ context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error) {
			assert.Nil(t, filters.StartTime, "invalid since should not be parsed")
			return []*types.Workflow{{WorkflowID: "wf-1"}}, nil
		},
	}

	router := gin.New()
	router.POST("/query", QueryHandler(store))

	body := `{"resource":"workflows","filters":{"since":"not-a-date"}}`
	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	resp := decodeEnvelope(t, rec.Body)
	require.True(t, resp.OK)
	data := resp.Data.(map[string]interface{})
	assert.Equal(t, "workflows", data["resource"])
}

func TestQueryHandler_WorkflowsInvalidUntil(t *testing.T) {
	store := &handlerTestStorage{
		mockStatusStorage: &mockStatusStorage{},
		queryWorkflowsFn: func(_ context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error) {
			assert.Nil(t, filters.EndTime, "invalid until should not be parsed")
			return []*types.Workflow{{WorkflowID: "wf-1"}}, nil
		},
	}

	router := gin.New()
	router.POST("/query", QueryHandler(store))

	body := `{"resource":"workflows","filters":{"until":"garbage"}}`
	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	resp := decodeEnvelope(t, rec.Body)
	require.True(t, resp.OK)
}

func TestQueryHandler_SessionsInvalidSinceUntil(t *testing.T) {
	store := &handlerTestStorage{
		mockStatusStorage: &mockStatusStorage{},
		querySessionsFn: func(_ context.Context, filters types.SessionFilters) ([]*types.Session, error) {
			assert.Nil(t, filters.StartTime, "invalid since should not be parsed")
			assert.Nil(t, filters.EndTime, "invalid until should not be parsed")
			return []*types.Session{{SessionID: "sess-1"}}, nil
		},
	}

	router := gin.New()
	router.POST("/query", QueryHandler(store))

	body := `{"resource":"sessions","filters":{"since":"bogus","until":"also-bogus"}}`
	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	resp := decodeEnvelope(t, rec.Body)
	require.True(t, resp.OK)
	data := resp.Data.(map[string]interface{})
	assert.Equal(t, "sessions", data["resource"])
}

func TestQueryHandler_RunsInvalidSince(t *testing.T) {
	store := &handlerTestStorage{
		mockStatusStorage: &mockStatusStorage{},
		queryRunSummariesFn: func(_ context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
			assert.Nil(t, filter.StartTime, "invalid since should not be parsed")
			return []*storage.RunSummaryAggregation{{RunID: "run-1"}}, 1, nil
		},
	}

	router := gin.New()
	router.POST("/query", QueryHandler(store))

	body := `{"resource":"runs","filters":{"since":"not-a-timestamp"}}`
	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	resp := decodeEnvelope(t, rec.Body)
	require.True(t, resp.OK)
}

func TestQueryHandler_ExecutionsInvalidSinceUntil(t *testing.T) {
	store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
	store.On("QueryExecutionRecords", mock.Anything, mock.Anything).Return([]*types.Execution{
		{ExecutionID: "exec-1"},
	}, nil)

	router := gin.New()
	router.POST("/query", QueryHandler(store))

	body := `{"resource":"executions","filters":{"since":"bad-date","until":"bad-date"}}`
	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	resp := decodeEnvelope(t, rec.Body)
	require.True(t, resp.OK)
	store.AssertExpectations(t)
}

func TestQueryHandler_AgentOffsetOutOfBounds(t *testing.T) {
	store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
	store.On("ListAgents", mock.Anything, mock.Anything).Return([]*types.AgentNode{
		{ID: "agent-1"},
		{ID: "agent-2"},
	}, nil)

	router := gin.New()
	router.POST("/query", QueryHandler(store))

	body := `{"resource":"agents","limit":5,"offset":10}`
	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	resp := decodeEnvelope(t, rec.Body)
	require.True(t, resp.OK)
	data := resp.Data.(map[string]interface{})
	assert.Equal(t, float64(2), data["total"])
	assert.Empty(t, data["results"])
	store.AssertExpectations(t)
}

func TestQueryHandler_ValidRFC3339FiltersAreForwarded(t *testing.T) {
	since := "2026-01-02T03:04:05Z"
	until := "2026-01-03T04:05:06Z"
	store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
	store.On("QueryExecutionRecords", mock.Anything, mock.MatchedBy(func(filter types.ExecutionFilter) bool {
		return filter.Status != nil && *filter.Status == "completed" &&
			filter.AgentNodeID != nil && *filter.AgentNodeID == "agent-1" &&
			filter.RunID != nil && *filter.RunID == "run-1" &&
			filter.SessionID != nil && *filter.SessionID == "session-1" &&
			filter.ActorID != nil && *filter.ActorID == "actor-1" &&
			filter.StartTime != nil && filter.StartTime.Equal(parseRFC3339(t, since)) &&
			filter.EndTime != nil && filter.EndTime.Equal(parseRFC3339(t, until))
	})).Return([]*types.Execution{}, nil)

	router := gin.New()
	router.POST("/query", QueryHandler(store))
	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(`{"resource":"executions","filters":{"status":"completed","agent_id":"agent-1","run_id":"run-1","session_id":"session-1","actor_id":"actor-1","since":"`+since+`","until":"`+until+`"}}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	store.AssertExpectations(t)
}

func parseRFC3339(t *testing.T, value string) time.Time {
	t.Helper()
	parsed, err := time.Parse(time.RFC3339, value)
	require.NoError(t, err)
	return parsed
}

func TestQueryHandler_ResourceRequiresField(t *testing.T) {
	store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}

	router := gin.New()
	router.POST("/query", QueryHandler(store))

	body := `{}`
	req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusBadRequest, rec.Code)
	resp := decodeEnvelope(t, rec.Body)
	assert.Equal(t, "invalid_request", resp.Error.Code)
}

func TestQueryHandler_ResponseStructure(t *testing.T) {
	tests := []struct {
		name     string
		body     string
		validate func(*testing.T, map[string]interface{})
	}{
		{
			name: "runs includes all keys",
			body: `{"resource":"runs","limit":10,"offset":0}`,
			validate: func(t *testing.T, data map[string]interface{}) {
				assert.Contains(t, data, "resource")
				assert.Contains(t, data, "results")
				assert.Contains(t, data, "total")
				assert.Contains(t, data, "limit")
				assert.Contains(t, data, "offset")
			},
		},
		{
			name: "executions includes all keys",
			body: `{"resource":"executions","limit":10,"offset":0}`,
			validate: func(t *testing.T, data map[string]interface{}) {
				assert.Contains(t, data, "resource")
				assert.Contains(t, data, "results")
				assert.Contains(t, data, "total")
				assert.Contains(t, data, "limit")
				assert.Contains(t, data, "offset")
			},
		},
		{
			name: "workflows includes all keys",
			body: `{"resource":"workflows","limit":10,"offset":0}`,
			validate: func(t *testing.T, data map[string]interface{}) {
				assert.Contains(t, data, "resource")
				assert.Contains(t, data, "results")
				assert.Contains(t, data, "total")
				assert.Contains(t, data, "limit")
				assert.Contains(t, data, "offset")
			},
		},
		{
			name: "sessions includes all keys",
			body: `{"resource":"sessions","limit":10,"offset":0}`,
			validate: func(t *testing.T, data map[string]interface{}) {
				assert.Contains(t, data, "resource")
				assert.Contains(t, data, "results")
				assert.Contains(t, data, "total")
				assert.Contains(t, data, "limit")
				assert.Contains(t, data, "offset")
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			store := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
			store.On("QueryExecutionRecords", mock.Anything, mock.Anything).Return([]*types.Execution{}, nil)
			store.On("ListAgents", mock.Anything, mock.Anything).Return([]*types.AgentNode{}, nil)

			store.queryRunSummariesFn = func(context.Context, types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
				return nil, 0, nil
			}
			store.queryWorkflowsFn = func(context.Context, types.WorkflowFilters) ([]*types.Workflow, error) {
				return nil, nil
			}
			store.querySessionsFn = func(context.Context, types.SessionFilters) ([]*types.Session, error) {
				return nil, nil
			}

			router := gin.New()
			router.POST("/query", QueryHandler(store))

			req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(tt.body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			require.Equal(t, http.StatusOK, rec.Code)
			resp := decodeEnvelope(t, rec.Body)
			require.True(t, resp.OK)
			tt.validate(t, resp.Data.(map[string]interface{}))
		})
	}
}

func TestQueryHandler_OffsetPastEndReturnsEmptyPage(t *testing.T) {
	execStore := &handlerTestStorage{mockStatusStorage: &mockStatusStorage{}}
	execStore.On("QueryExecutionRecords", mock.Anything, mock.Anything).Return([]*types.Execution{
		{ExecutionID: "exec-1"},
		{ExecutionID: "exec-2"},
	}, nil)

	cases := []struct {
		name  string
		body  string
		store *handlerTestStorage
	}{
		{
			name:  "executions",
			body:  `{"resource":"executions","limit":5,"offset":10}`,
			store: execStore,
		},
		{
			name: "workflows",
			body: `{"resource":"workflows","limit":5,"offset":10}`,
			store: &handlerTestStorage{
				mockStatusStorage: &mockStatusStorage{},
				queryWorkflowsFn: func(context.Context, types.WorkflowFilters) ([]*types.Workflow, error) {
					return []*types.Workflow{{WorkflowID: "wf-1"}, {WorkflowID: "wf-2"}}, nil
				},
			},
		},
		{
			name: "sessions",
			body: `{"resource":"sessions","limit":5,"offset":10}`,
			store: &handlerTestStorage{
				mockStatusStorage: &mockStatusStorage{},
				querySessionsFn: func(context.Context, types.SessionFilters) ([]*types.Session, error) {
					return []*types.Session{{SessionID: "sess-1"}, {SessionID: "sess-2"}}, nil
				},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			router := gin.New()
			router.POST("/query", QueryHandler(tc.store))

			req := httptest.NewRequest(http.MethodPost, "/query", bytes.NewBufferString(tc.body))
			req.Header.Set("Content-Type", "application/json")
			rec := httptest.NewRecorder()
			router.ServeHTTP(rec, req)

			require.Equal(t, http.StatusOK, rec.Code)
			resp := decodeEnvelope(t, rec.Body)
			require.True(t, resp.OK)
			data := resp.Data.(map[string]interface{})
			assert.Equal(t, float64(2), data["total"])
			assert.Empty(t, data["results"])
		})
	}
}
