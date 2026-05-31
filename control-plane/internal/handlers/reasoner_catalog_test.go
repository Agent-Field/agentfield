package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type reasonerCatalogStoreStub struct {
	agents     []*types.AgentNode
	executions []*types.Execution
}

func (s *reasonerCatalogStoreStub) ListAgents(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
	return s.agents, nil
}

func (s *reasonerCatalogStoreStub) QueryExecutionRecords(context.Context, types.ExecutionFilter) ([]*types.Execution, error) {
	return s.executions, nil
}

func TestListReasonersHandlerRecencyAndFilters(t *testing.T) {
	gin.SetMode(gin.TestMode)
	now := time.Now().UTC()
	store := &reasonerCatalogStoreStub{
		agents: []*types.AgentNode{
			{
				ID:           "sec-af",
				HealthStatus: types.HealthStatusActive,
				Reasoners: []types.ReasonerDefinition{
					{ID: "hunt"},
					{ID: "prove"},
				},
			},
			{
				ID:           "contract-af",
				HealthStatus: types.HealthStatusInactive,
				Reasoners:    []types.ReasonerDefinition{{ID: "review"}},
			},
		},
		executions: []*types.Execution{
			{AgentNodeID: "sec-af", ReasonerID: "hunt", StartedAt: now},
			{AgentNodeID: "contract-af", ReasonerID: "review", StartedAt: now.Add(-time.Hour)},
		},
	}
	router := gin.New()
	router.GET("/api/v1/reasoners", ListReasonersHandler(store))

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/reasoners?query=hunt&live=true", nil)
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	var body ReasonerCatalogResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	require.Equal(t, 1, body.Total)
	require.Equal(t, "sec-af", body.Reasoners[0].Node)
	require.Equal(t, "hunt", body.Reasoners[0].Reasoner)
	require.Equal(t, "live", body.Reasoners[0].Status)
	require.NotNil(t, body.Reasoners[0].LastRunAt)
}

func TestStreamExecutionEventsHandlerSnapshot(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newTestExecutionStorage(nil)
	completed := time.Now().UTC()
	require.NoError(t, store.CreateExecutionRecord(context.Background(), &types.Execution{
		ExecutionID:   "exec-1",
		RunID:         "run-1",
		AgentNodeID:   "node-1",
		ReasonerID:    "hunt",
		Status:        string(types.ExecutionStatusSucceeded),
		ResultPayload: json.RawMessage(`{"ok":true}`),
		StartedAt:     completed.Add(-time.Second),
		CompletedAt:   &completed,
		CreatedAt:     completed.Add(-time.Second),
		UpdatedAt:     completed,
	}))

	router := gin.New()
	router.GET("/api/v1/executions/:execution_id/events", StreamExecutionEventsHandler(store))

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/executions/exec-1/events", nil)
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)
	require.Contains(t, rec.Header().Get("Content-Type"), "text/event-stream")
	require.Contains(t, rec.Body.String(), "data:")
	require.Contains(t, rec.Body.String(), `"status":"succeeded"`)
}

func TestStreamExecutionEventsHandlerLiveEvent(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := newTestExecutionStorage(nil)
	router := gin.New()
	router.GET("/api/v1/executions/:execution_id/events", StreamExecutionEventsHandler(store))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/executions/exec-live/events", nil)
	rec := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		router.ServeHTTP(rec, req)
		close(done)
	}()

	time.Sleep(20 * time.Millisecond)
	store.GetExecutionEventBus().Publish(events.ExecutionEvent{
		Type:        events.ExecutionCompleted,
		ExecutionID: "exec-live",
		WorkflowID:  "run-live",
		AgentNodeID: "node",
		Status:      string(types.ExecutionStatusSucceeded),
		Timestamp:   time.Now().UTC(),
	})

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("stream did not close on terminal event")
	}
	require.True(t, strings.Contains(rec.Body.String(), `"execution_id":"exec-live"`), rec.Body.String())
}
