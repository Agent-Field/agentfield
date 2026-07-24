package agentic

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
)

func reasonerTestAgents() []*types.AgentNode {
	return []*types.AgentNode{
		{
			ID:           "pr-af",
			HealthStatus: types.HealthStatusActive,
			Reasoners: []types.ReasonerDefinition{
				{ID: "review_pull_request", Tags: []string{"pr", "code-review"}},
				{ID: "summarize_diff", Tags: []string{"pr"}},
			},
		},
		{
			ID:           "weather",
			HealthStatus: types.HealthStatusInactive,
			Reasoners: []types.ReasonerDefinition{
				{ID: "get_forecast", Tags: []string{"weather"}},
			},
		},
	}
}

func serveReasoners(t *testing.T, store *mockStatusStorage, rawQuery string) (*httptest.ResponseRecorder, AgenticResponse) {
	t.Helper()
	router := gin.New()
	router.GET("/api/v1/agentic/reasoners", ReasonersHandler(store))

	req := httptest.NewRequest("GET", "/api/v1/agentic/reasoners?"+rawQuery, nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	var resp AgenticResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	return rec, resp
}

func TestReasonersHandler_Results(t *testing.T) {
	store := new(mockStatusStorage)
	store.On("ListAgents", mock.Anything, mock.Anything).Return(reasonerTestAgents(), nil)

	rec, resp := serveReasoners(t, store, "q=review+pull+request")
	require.Equal(t, http.StatusOK, rec.Code)
	assert.True(t, resp.OK)

	data := resp.Data.(map[string]interface{})
	assert.Equal(t, "review pull request", data["query"])
	assert.Equal(t, float64(3), data["total_indexed"])

	results := data["results"].([]interface{})
	require.NotEmpty(t, results)
	first := results[0].(map[string]interface{})
	assert.Equal(t, "review_pull_request", first["reasoner_id"])
	assert.Equal(t, "pr-af", first["agent_id"])
	assert.Equal(t, "pr-af:review_pull_request", first["invocation_target"])
	assert.Equal(t, "active", first["agent_health"])
	assert.Greater(t, first["score"].(float64), 0.0)
}

func TestReasonersHandler_EmptyQuery(t *testing.T) {
	store := new(mockStatusStorage)
	// ListAgents must not be called when q is empty.

	rec, resp := serveReasoners(t, store, "")
	require.Equal(t, http.StatusBadRequest, rec.Code)
	assert.False(t, resp.OK)
	require.NotNil(t, resp.Error)
	assert.Equal(t, "missing_query", resp.Error.Code)
	store.AssertNotCalled(t, "ListAgents", mock.Anything, mock.Anything)
}

func TestReasonersHandler_AgentFilter(t *testing.T) {
	store := new(mockStatusStorage)
	store.On("ListAgents", mock.Anything, mock.Anything).Return(reasonerTestAgents(), nil)

	rec, resp := serveReasoners(t, store, "q=forecast&agent=weather")
	require.Equal(t, http.StatusOK, rec.Code)

	data := resp.Data.(map[string]interface{})
	// Only the weather agent's single reasoner is indexed.
	assert.Equal(t, float64(1), data["total_indexed"])
	results := data["results"].([]interface{})
	require.Len(t, results, 1)
	first := results[0].(map[string]interface{})
	assert.Equal(t, "get_forecast", first["reasoner_id"])
	assert.Equal(t, "weather", first["agent_id"])
}

func TestReasonersHandler_LimitRespected(t *testing.T) {
	store := new(mockStatusStorage)
	store.On("ListAgents", mock.Anything, mock.Anything).Return(reasonerTestAgents(), nil)

	// Both pr-af reasoners are tagged "pr"; limit=1 must return exactly one.
	rec, resp := serveReasoners(t, store, "q=pr&limit=1")
	require.Equal(t, http.StatusOK, rec.Code)

	data := resp.Data.(map[string]interface{})
	results := data["results"].([]interface{})
	require.Len(t, results, 1)
}

func TestReasonersHandler_LimitClampedToMax(t *testing.T) {
	// 60 reasoners all matching "task"; a limit above the max (50) is clamped.
	reasoners := make([]types.ReasonerDefinition, 0, 60)
	for i := 0; i < 60; i++ {
		reasoners = append(reasoners, types.ReasonerDefinition{
			ID:   fmt.Sprintf("task_%d", i),
			Tags: []string{"task"},
		})
	}
	agents := []*types.AgentNode{{ID: "bulk", HealthStatus: types.HealthStatusActive, Reasoners: reasoners}}

	store := new(mockStatusStorage)
	store.On("ListAgents", mock.Anything, mock.Anything).Return(agents, nil)

	rec, resp := serveReasoners(t, store, "q=task&limit=100")
	require.Equal(t, http.StatusOK, rec.Code)

	data := resp.Data.(map[string]interface{})
	assert.Equal(t, float64(60), data["total_indexed"])
	results := data["results"].([]interface{})
	assert.Len(t, results, reasonerSearchMaxLimit)
}

func TestReasonersHandler_StorageError(t *testing.T) {
	store := new(mockStatusStorage)
	store.On("ListAgents", mock.Anything, mock.Anything).Return(nil, errors.New("db down"))

	rec, resp := serveReasoners(t, store, "q=anything")
	require.Equal(t, http.StatusInternalServerError, rec.Code)
	assert.False(t, resp.OK)
	require.NotNil(t, resp.Error)
	assert.Equal(t, "query_failed", resp.Error.Code)
}

func TestReasonersHandler_DescriptionIndexed(t *testing.T) {
	// A term that appears only in the metadata description must still match.
	agents := []*types.AgentNode{
		{
			ID:           "docs",
			HealthStatus: types.HealthStatusActive,
			Reasoners:    []types.ReasonerDefinition{{ID: "handler_a", Tags: []string{"a"}}},
			Metadata: types.AgentMetadata{
				Custom: map[string]interface{}{
					"descriptions": map[string]interface{}{
						"handler_a": "generates quarterly compliance paperwork",
					},
				},
			},
		},
	}
	store := new(mockStatusStorage)
	store.On("ListAgents", mock.Anything, mock.Anything).Return(agents, nil)

	rec, resp := serveReasoners(t, store, "q=compliance")
	require.Equal(t, http.StatusOK, rec.Code)
	data := resp.Data.(map[string]interface{})
	results := data["results"].([]interface{})
	require.Len(t, results, 1)
	assert.Equal(t, "handler_a", results[0].(map[string]interface{})["reasoner_id"])
}
