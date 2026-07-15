package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type fakeRunSummaryStore struct {
	gotFilter types.ExecutionFilter
	summaries []*storage.RunSummaryAggregation
	total     int
	err       error
}

func (f *fakeRunSummaryStore) QueryRunSummaries(_ context.Context, filter types.ExecutionFilter) ([]*storage.RunSummaryAggregation, int, error) {
	f.gotFilter = filter
	return f.summaries, f.total, f.err
}

func TestActiveExecutionsHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)

	started := time.Date(2024, 3, 1, 8, 0, 0, 0, time.UTC)
	latest := started.Add(5 * time.Minute)
	store := &fakeRunSummaryStore{
		summaries: []*storage.RunSummaryAggregation{
			{
				RunID:           "run-1",
				TotalExecutions: 27,
				// The aggregation's own ActiveExecutions column excludes
				// paused; the handler must count from StatusCounts instead
				// (3 running + 1 paused = 4 in flight here).
				ActiveExecutions: 3,
				StatusCounts:     map[string]int{"running": 3, "paused": 1, "succeeded": 23},
				EarliestStarted:  started,
				LatestStarted:    latest,
				RootExecutionID:  strPtr("exec-root"),
				RootStatus:       strPtr("running"),
				RootAgentNodeID:  strPtr("pr-af-go"),
				RootReasonerID:   strPtr("review"),
				SessionID:        strPtr("session-1"),
			},
		},
		total: 1,
	}

	router := gin.New()
	router.GET("/api/v1/executions/active", ActiveExecutionsHandler(store))

	req := httptest.NewRequest(http.MethodGet, "/api/v1/executions/active?agent_id=pr-af-go&session_id=session-1", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusOK, rec.Code)

	var resp ActiveExecutionsResponse
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &resp))
	require.Equal(t, 1, resp.Count)
	require.Len(t, resp.Runs, 1)

	run := resp.Runs[0]
	require.Equal(t, "run-1", run.RunID)
	require.Equal(t, "pr-af-go.review", run.Target)
	require.Equal(t, "running", run.RootStatus)
	require.Equal(t, 4, run.ActiveExecutions)
	require.Equal(t, 27, run.TotalExecutions)
	require.Equal(t, "session-1", run.SessionID)
	require.Equal(t, started, run.StartedAt)
	require.Equal(t, latest, run.LatestActivity)

	// The handler must request run-level active filtering and pass the query
	// filters through.
	require.True(t, store.gotFilter.ActiveOnly)
	require.NotNil(t, store.gotFilter.AgentNodeID)
	require.Equal(t, "pr-af-go", *store.gotFilter.AgentNodeID)
	require.NotNil(t, store.gotFilter.SessionID)
	require.Equal(t, "session-1", *store.gotFilter.SessionID)
	require.Equal(t, 100, store.gotFilter.Limit)
}

func TestActiveExecutionsHandlerEmptyAndLimit(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := &fakeRunSummaryStore{summaries: nil, total: 0}
	router := gin.New()
	router.GET("/api/v1/executions/active", ActiveExecutionsHandler(store))

	// Empty state: count 0 and an empty (non-null) runs array.
	req := httptest.NewRequest(http.MethodGet, "/api/v1/executions/active", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code)
	require.JSONEq(t, `{"count":0,"runs":[]}`, rec.Body.String())

	// Limit is clamped to 200.
	req = httptest.NewRequest(http.MethodGet, "/api/v1/executions/active?limit=5000", nil)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	require.Equal(t, http.StatusOK, rec.Code)
	require.Equal(t, 200, store.gotFilter.Limit)

	// Invalid limit is a 400, not a silent default.
	req = httptest.NewRequest(http.MethodGet, "/api/v1/executions/active?limit=abc", nil)
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	require.Equal(t, http.StatusBadRequest, rec.Code)
}
