package ui

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func seedDashboardData(t *testing.T) (*DashboardHandler, *gin.Engine) {
	t.Helper()

	ls, ctx := setupUIHandlerStorage(t)
	now := time.Now().UTC().Truncate(time.Minute)

	require.NoError(t, ls.RegisterAgent(ctx, &types.AgentNode{
		ID:              "agent-alpha",
		GroupID:         "agent-alpha",
		TeamID:          "team-a",
		BaseURL:         "http://alpha",
		Version:         "v1",
		Reasoners:       []types.ReasonerDefinition{{ID: "planner"}},
		Skills:          []types.SkillDefinition{{ID: "review"}},
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		LastHeartbeat:   now.Add(-time.Minute),
		RegisteredAt:    now.Add(-time.Hour),
	}))
	require.NoError(t, ls.RegisterAgent(ctx, &types.AgentNode{
		ID:              "agent-beta",
		GroupID:         "agent-beta",
		TeamID:          "team-b",
		BaseURL:         "http://beta",
		Version:         "v2",
		Reasoners:       []types.ReasonerDefinition{{ID: "summarize"}},
		Skills:          []types.SkillDefinition{{ID: "draft"}},
		HealthStatus:    types.HealthStatusInactive,
		LifecycleStatus: types.AgentStatusDegraded,
		LastHeartbeat:   now.Add(-2 * time.Hour),
		RegisteredAt:    now.Add(-2 * time.Hour),
	}))

	errorMessage := "timeout while calling provider"
	durationFast := int64(1500)
	durationSlow := int64(65000)

	records := []*types.Execution{
		{
			ExecutionID:   "dash-1",
			RunID:         "run-dashboard",
			AgentNodeID:   "agent-alpha",
			ReasonerID:    "planner",
			NodeID:        "agent-alpha",
			Status:        string(types.ExecutionStatusSucceeded),
			StartedAt:     now.Add(-30 * time.Minute),
			CompletedAt:   timePtr(now.Add(-29 * time.Minute)),
			DurationMS:    &durationFast,
			ResultPayload: json.RawMessage(`{"ok":true}`),
		},
		{
			ExecutionID:   "dash-2",
			RunID:         "run-dashboard",
			AgentNodeID:   "agent-alpha",
			ReasonerID:    "planner",
			NodeID:        "agent-alpha",
			Status:        string(types.ExecutionStatusFailed),
			ErrorMessage:  &errorMessage,
			StartedAt:     now.Add(-10 * time.Minute),
			CompletedAt:   timePtr(now.Add(-9 * time.Minute)),
			DurationMS:    &durationSlow,
			ResultPayload: json.RawMessage(`{"error":"timeout"}`),
		},
		{
			ExecutionID: "dash-3",
			RunID:       "run-live",
			AgentNodeID: "agent-alpha",
			ReasonerID:  "review",
			NodeID:      "agent-alpha",
			Status:      string(types.ExecutionStatusRunning),
			StartedAt:   now.Add(-5 * time.Minute),
		},
		{
			ExecutionID: "dash-4",
			RunID:       "run-waiting",
			AgentNodeID: "agent-beta",
			ReasonerID:  "summarize",
			NodeID:      "agent-beta",
			Status:      string(types.ExecutionStatusWaiting),
			StartedAt:   now.Add(-2 * time.Minute),
		},
	}
	for _, record := range records {
		require.NoError(t, ls.CreateExecutionRecord(ctx, record))
	}

	mockAgentService := &MockAgentServiceForUI{}
	mockAgentService.On("GetAgentStatus", "agent-alpha").Return(&domain.AgentStatus{IsRunning: true, Uptime: "2h"}, nil)

	handler := NewDashboardHandler(ls, mockAgentService)
	router := gin.New()
	router.GET("/api/ui/v1/dashboard/enhanced", handler.GetEnhancedDashboardSummaryHandler)

	return handler, router
}

func TestEnhancedDashboardSummaryHandlerCoverage(t *testing.T) {
	gin.SetMode(gin.TestMode)

	handler, router := seedDashboardData(t)

	req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/dashboard/enhanced?preset=24h&compare=true", nil)
	resp := httptest.NewRecorder()
	router.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	var body EnhancedDashboardResponse
	require.NoError(t, json.Unmarshal(resp.Body.Bytes(), &body))
	require.Equal(t, 2, body.Overview.TotalAgents)
	require.Equal(t, 1, body.Overview.ActiveAgents)
	require.Equal(t, 1, body.Overview.DegradedAgents)
	require.Equal(t, 4, body.Overview.ExecutionsLast24h)
	require.NotNil(t, body.Comparison)
	require.NotEmpty(t, body.ExecutionTrends.Last7Days)
	require.NotEmpty(t, body.Workflows.TopWorkflows)
	require.NotEmpty(t, body.Incidents)
	require.NotEmpty(t, body.Hotspots.TopFailingReasoners)
	require.Len(t, body.ActivityPatterns.HourlyHeatmap, 7)

	cachedReq := httptest.NewRequest(http.MethodGet, "/api/ui/v1/dashboard/enhanced?preset=24h&compare=true", nil)
	cachedResp := httptest.NewRecorder()
	router.ServeHTTP(cachedResp, cachedReq)
	require.Equal(t, http.StatusOK, cachedResp.Code)

	handler.agentService.(*MockAgentServiceForUI).AssertExpectations(t)
}

func TestDashboardPureHelpersCoverage(t *testing.T) {
	now := time.Date(2026, 4, 8, 12, 0, 0, 0, time.UTC)
	durationA := int64(1000)
	durationB := int64(65000)
	errorMessage := "timeout from provider with a very long message that should be truncated when included in hotspot summaries because it is noisy"

	executions := []*types.Execution{
		{ExecutionID: "1", RunID: "run-a", AgentNodeID: "a", ReasonerID: "plan", Status: string(types.ExecutionStatusSucceeded), StartedAt: now.Add(-2 * time.Hour), DurationMS: &durationA},
		{ExecutionID: "2", RunID: "run-a", AgentNodeID: "a", ReasonerID: "plan", Status: string(types.ExecutionStatusFailed), StartedAt: now.Add(-90 * time.Minute), DurationMS: &durationB, ErrorMessage: &errorMessage},
		{ExecutionID: "3", RunID: "run-b", AgentNodeID: "b", ReasonerID: "review", Status: string(types.ExecutionStatusRunning), StartedAt: now.Add(-10 * time.Minute)},
		{ExecutionID: "4", RunID: "run-c", AgentNodeID: "c", ReasonerID: "draft", Status: string(types.ExecutionStatusCancelled), StartedAt: now.Add(-48 * time.Hour)},
	}

	t.Run("time range parsing", func(t *testing.T) {
		c, _ := gin.CreateTestContext(httptest.NewRecorder())
		c.Request = httptest.NewRequest(http.MethodGet, "/?preset=custom", nil)
		start, end, preset, err := parseTimeRangeParams(c, now)
		require.NoError(t, err)
		require.Equal(t, TimeRangePreset24h, preset)
		require.True(t, end.After(start))
	})

	t.Run("execution trends for range", func(t *testing.T) {
		trends := buildExecutionTrendsForRange(executions, now.Add(-24*time.Hour), now, TimeRangePreset24h)
		require.Equal(t, 4, trends.Last24h.Total)
		require.Equal(t, 1, trends.Last24h.Succeeded)
		require.NotEmpty(t, trends.Last7Days)

		custom := buildExecutionTrendsForRange(executions, now.Add(-12*time.Hour), now, TimeRangePresetCustom)
		require.NotEmpty(t, custom.Last7Days)
	})

	t.Run("comparison and medians", func(t *testing.T) {
		comparison := buildComparisonData(
			EnhancedOverview{ExecutionsLast24h: 4, SuccessRate24h: 50, AverageDurationMs24h: 3000},
			EnhancedOverview{ExecutionsLast24h: 2, SuccessRate24h: 25, AverageDurationMs24h: 1500},
			now.Add(-48*time.Hour),
			now.Add(-24*time.Hour),
		)
		require.Equal(t, 2, comparison.OverviewDelta.ExecutionsDelta)
		require.Equal(t, 25.0, comparison.OverviewDelta.SuccessRateDelta)
		require.Equal(t, 250.0, computeMedian([]int64{100, 200, 300, 400}))
		require.Equal(t, 300.0, computeMedian([]int64{100, 300, 500}))
	})

		t.Run("hotspots activity workflows and incidents", func(t *testing.T) {
		hotspots := buildHotspotSummary(executions)
		require.Len(t, hotspots.TopFailingReasoners, 2)
		foundTopErrors := false
		for _, item := range hotspots.TopFailingReasoners {
			if len(item.TopErrors) > 0 {
				foundTopErrors = true
			}
		}
		require.True(t, foundTopErrors)

		patterns := buildActivityPatterns(executions)
		require.Len(t, patterns.HourlyHeatmap, 7)

		insights := buildWorkflowInsights(executions, executions[2:4])
		require.NotEmpty(t, insights.TopWorkflows)
		require.NotEmpty(t, insights.ActiveRuns)

		incidents := buildIncidentItems(executions, 2)
		require.Len(t, incidents, 2)
		require.Equal(t, "2", incidents[0].ExecutionID)
	})

	t.Run("misc helpers", func(t *testing.T) {
		require.Equal(t, now, maxTime(time.Time{}, now))
		require.Equal(t, now.Add(time.Minute), maxTime(now, now.Add(time.Minute)))
		require.Equal(t, 50.0, (&DashboardHandler{}).calculateSuccessRate([]*types.Execution{
			{Status: string(types.ExecutionStatusSucceeded)},
			{Status: string(types.ExecutionStatusFailed)},
		}))
	})
}

func timePtr(v time.Time) *time.Time { return &v }
