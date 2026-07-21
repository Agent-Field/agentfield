package handlers

import (
	"context"
	"encoding/json"
	"errors"
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
	agentsErr  error
	execsErr   error
}

func (s *reasonerCatalogStoreStub) ListAgents(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
	if s.agentsErr != nil {
		return nil, s.agentsErr
	}
	return s.agents, nil
}

func (s *reasonerCatalogStoreStub) QueryExecutionRecords(context.Context, types.ExecutionFilter) ([]*types.Execution, error) {
	if s.execsErr != nil {
		return nil, s.execsErr
	}
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

// Validation contract: rows carry the reasoner's description (record field
// first, legacy metadata map as fallback) and tags; entrypoints=true filters
// to entrypoint-tagged reasoners; among never-run rows, entry points list
// first so a fresh registry shows callable surfaces on top.
func TestListReasonersHandlerDescriptionsAndEntrypoints(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &reasonerCatalogStoreStub{
		agents: []*types.AgentNode{
			{
				ID:           "swe-planner",
				HealthStatus: types.HealthStatusActive,
				Reasoners: []types.ReasonerDefinition{
					{ID: "run_coder", Description: "Internal coding role"},
					{
						ID:          "implement_issue",
						Description: "Implement one scoped issue on a branch",
						Tags:        []string{"swe-issue", types.TagEntrypoint},
					},
					{ID: "legacy_only"},
				},
				Metadata: types.AgentMetadata{
					Custom: map[string]interface{}{
						"descriptions": map[string]interface{}{
							"legacy_only": "From the metadata map",
						},
					},
				},
			},
		},
	}
	router := gin.New()
	router.GET("/api/v1/reasoners", ListReasonersHandler(store))

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/v1/reasoners", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	var body ReasonerCatalogResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	require.Equal(t, 3, body.Total)
	// No rows have runs -> the entrypoint-tagged reasoner sorts first.
	require.Equal(t, "implement_issue", body.Reasoners[0].Reasoner)
	require.Equal(t, "Implement one scoped issue on a branch", body.Reasoners[0].Description)
	require.Contains(t, body.Reasoners[0].Tags, "entrypoint")
	byName := map[string]ReasonerCatalogRow{}
	for _, row := range body.Reasoners {
		byName[row.Reasoner] = row
	}
	require.Equal(t, "From the metadata map", byName["legacy_only"].Description)

	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/api/v1/reasoners?entrypoints=true", nil))
	require.Equal(t, http.StatusOK, rec.Code)
	body = ReasonerCatalogResponse{}
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	require.Equal(t, 1, body.Total)
	require.Equal(t, "implement_issue", body.Reasoners[0].Reasoner)
}

func TestListReasonersHandlerErrorsAndTruncation(t *testing.T) {
	gin.SetMode(gin.TestMode)

	for _, tc := range []struct {
		name  string
		store *reasonerCatalogStoreStub
	}{
		{name: "agents error", store: &reasonerCatalogStoreStub{agentsErr: errors.New("boom")}},
		{name: "executions error", store: &reasonerCatalogStoreStub{agents: []*types.AgentNode{}, execsErr: errors.New("boom")}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := gin.New()
			router.GET("/api/v1/reasoners", ListReasonersHandler(tc.store))
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodGet, "/api/v1/reasoners", nil)
			router.ServeHTTP(rec, req)
			require.Equal(t, http.StatusInternalServerError, rec.Code)
		})
	}

	agents := []*types.AgentNode{nil}
	for i := 0; i < 25; i++ {
		agents = append(agents, &types.AgentNode{
			ID:           "node",
			HealthStatus: types.HealthStatusUnknown,
			Reasoners:    []types.ReasonerDefinition{{ID: "reasoner-" + string(rune('a'+i))}},
		})
	}
	store := &reasonerCatalogStoreStub{
		agents: agents,
		executions: []*types.Execution{
			nil,
			{AgentNodeID: "", ReasonerID: "skip", StartedAt: time.Now().UTC()},
			{AgentNodeID: "node", ReasonerID: "reasoner-b", StartedAt: time.Now().UTC()},
		},
	}
	router := gin.New()
	router.GET("/api/v1/reasoners", ListReasonersHandler(store))
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/reasoners", nil)
	router.ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code)
	var body ReasonerCatalogResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &body))
	require.Equal(t, 25, body.Total)
	require.Equal(t, 20, body.Shown)
	require.Equal(t, "stale", body.Reasoners[0].Status)
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

func TestReasonerCatalogHelpers(t *testing.T) {
	require.Equal(t, "down", reasonerNodeStatus(&types.AgentNode{HealthStatus: types.HealthStatusInactive}))
	require.Equal(t, "stale", reasonerNodeStatus(&types.AgentNode{HealthStatus: types.HealthStatusUnknown}))
	require.True(t, fuzzyReasonerMatch("sec-af.hunt", "sah"))
	require.False(t, fuzzyReasonerMatch("sec-af.hunt", "zzz"))
	require.True(t, parseQueryBool("yes"))
	require.False(t, parseQueryBool("no"))
	require.Equal(t, 0, parseFromStep(""))
	require.Equal(t, 0, parseFromStep("-2"))
	require.Equal(t, 0, parseFromStep("bad"))
	require.Equal(t, 3, parseFromStep("3"))
	require.Equal(t, events.ExecutionFailed, eventTypeForStatus(string(types.ExecutionStatusFailed)))
	require.Equal(t, events.ExecutionCancelledEvent, eventTypeForStatus(string(types.ExecutionStatusCancelled)))
	require.Equal(t, events.ExecutionUpdated, eventTypeForStatus("running"))

	ts := time.Now().UTC().Format(time.RFC3339)
	parsed, ok := parseCatalogTime(&ts)
	require.True(t, ok)
	require.False(t, parsed.IsZero())
	_, ok = parseCatalogTime(nil)
	require.False(t, ok)
	bad := "not-time"
	_, ok = parseCatalogTime(&bad)
	require.False(t, ok)
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
