package agentic

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

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
	store.AssertExpectations(t)
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
