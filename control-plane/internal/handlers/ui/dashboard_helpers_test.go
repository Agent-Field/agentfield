package ui

import (
	"context"
	"errors"
	"fmt"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type stubDashboardAgentService struct {
	statuses map[string]*domain.AgentStatus
	errs     map[string]error
}

func (s *stubDashboardAgentService) RunAgent(name string, options domain.RunOptions) (*domain.RunningAgent, error) {
	return nil, nil
}

func (s *stubDashboardAgentService) StopAgent(name string) error {
	return nil
}

func (s *stubDashboardAgentService) GetAgentStatus(name string) (*domain.AgentStatus, error) {
	if err := s.errs[name]; err != nil {
		return nil, err
	}
	if status, ok := s.statuses[name]; ok {
		return status, nil
	}
	return nil, errors.New("missing agent status")
}

func (s *stubDashboardAgentService) ListRunningAgents() ([]domain.RunningAgent, error) {
	return nil, nil
}

func TestDashboardCacheLifecycle(t *testing.T) {
	cache := NewDashboardCache()

	_, found := cache.Get()
	require.False(t, found)
	require.Equal(t, 30*time.Second, cache.ttl)

	response := &DashboardSummaryResponse{SuccessRate: 99.5}
	cache.Set(response)

	got, found := cache.Get()
	require.True(t, found)
	require.Same(t, response, got)

	cache.timestamp = time.Now().Add(-cache.ttl - time.Second)
	_, found = cache.Get()
	require.False(t, found)
}

func TestEnhancedDashboardCacheTTLAndEviction(t *testing.T) {
	cache := NewEnhancedDashboardCache()
	start := time.Date(2026, 4, 7, 12, 34, 0, 0, time.UTC)
	end := start.Add(24 * time.Hour)
	require.Equal(t, 10, cache.maxSize)
	require.Equal(t, 30*time.Second, getTTLForPreset(TimeRangePreset1h))
	require.Equal(t, 60*time.Second, getTTLForPreset(TimeRangePreset24h))
	require.Equal(t, 2*time.Minute, getTTLForPreset(TimeRangePreset7d))
	require.Equal(t, 5*time.Minute, getTTLForPreset(TimeRangePreset30d))
	require.Equal(t, 60*time.Second, getTTLForPreset(TimeRangePresetCustom))

	key := generateCacheKey(start, end, true)
	cache.Set(key, &EnhancedDashboardResponse{Overview: EnhancedOverview{ExecutionsLast24h: 10}})
	cache.entries[key].timestamp = time.Now().Add(-29 * time.Second)

	got, found := cache.Get(key, TimeRangePreset1h)
	require.True(t, found)
	require.Equal(t, 10, got.Overview.ExecutionsLast24h)

	cache.entries[key].timestamp = time.Now().Add(-31 * time.Second)
	_, found = cache.Get(key, TimeRangePreset1h)
	require.False(t, found)

	for i := 0; i < cache.maxSize; i++ {
		entryKey := fmt.Sprintf("entry-%d", i)
		cache.Set(entryKey, &EnhancedDashboardResponse{Overview: EnhancedOverview{ExecutionsLast24h: i}})
		cache.entries[entryKey].timestamp = time.Date(2026, 4, 7, 0, i, 0, 0, time.UTC)
	}

	cache.Set("entry-new", &EnhancedDashboardResponse{Overview: EnhancedOverview{ExecutionsLast24h: 99}})
	require.Len(t, cache.entries, cache.maxSize)
	_, exists := cache.entries["entry-0"]
	require.False(t, exists)
	_, exists = cache.entries["entry-new"]
	require.True(t, exists)
	require.Equal(t, fmt.Sprintf("%d-%d-1", start.Truncate(time.Hour).Unix(), end.Truncate(time.Hour).Unix()), key)
	require.Equal(t, fmt.Sprintf("%d-%d-0", start.Truncate(time.Hour).Unix(), end.Truncate(time.Hour).Unix()), generateCacheKey(start, end, false))
}

func TestParseTimeRangeParams(t *testing.T) {
	now := time.Date(2026, 4, 7, 12, 34, 56, 0, time.UTC)

	gin.SetMode(gin.TestMode)

	t.Run("default preset rounds to the next hour", func(t *testing.T) {
		ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
		ctx.Request = httptest.NewRequest("GET", "/dashboard", nil)

		start, end, preset, err := parseTimeRangeParams(ctx, now)
		require.NoError(t, err)
		require.Equal(t, TimeRangePreset24h, preset)
		require.Equal(t, time.Date(2026, 4, 6, 13, 0, 0, 0, time.UTC), start)
		require.Equal(t, time.Date(2026, 4, 7, 13, 0, 0, 0, time.UTC), end)
	})

	t.Run("custom range honors explicit timestamps", func(t *testing.T) {
		ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
		ctx.Request = httptest.NewRequest(
			"GET",
			"/dashboard?preset=custom&start_time=2026-04-01T00:00:00Z&end_time=2026-04-03T12:00:00Z",
			nil,
		)

		start, end, preset, err := parseTimeRangeParams(ctx, now)
		require.NoError(t, err)
		require.Equal(t, TimeRangePresetCustom, preset)
		require.Equal(t, time.Date(2026, 4, 1, 0, 0, 0, 0, time.UTC), start)
		require.Equal(t, time.Date(2026, 4, 3, 12, 0, 0, 0, time.UTC), end)
	})

	t.Run("invalid custom values fall back to raw 24h window", func(t *testing.T) {
		ctx, _ := gin.CreateTestContext(httptest.NewRecorder())
		ctx.Request = httptest.NewRequest(
			"GET",
			"/dashboard?preset=custom&start_time=not-a-time",
			nil,
		)

		start, end, preset, err := parseTimeRangeParams(ctx, now)
		require.NoError(t, err)
		require.Equal(t, TimeRangePreset24h, preset)
		require.Equal(t, now.Add(-24*time.Hour), start)
		require.Equal(t, now, end)
	})

	prevStart, prevEnd := calculateComparisonPeriod(
		time.Date(2026, 4, 6, 13, 0, 0, 0, time.UTC),
		time.Date(2026, 4, 7, 13, 0, 0, 0, time.UTC),
	)
	require.Equal(t, time.Date(2026, 4, 5, 13, 0, 0, 0, time.UTC), prevStart)
	require.Equal(t, time.Date(2026, 4, 6, 13, 0, 0, 0, time.UTC), prevEnd)
}

func TestBuildExecutionTrendsForRangeAndComparisonData(t *testing.T) {
	end := time.Date(2026, 4, 8, 11, 0, 0, 0, time.UTC)
	start := end.Add(-24 * time.Hour)
	trend := buildExecutionTrendsForRange([]*types.Execution{
		testExecution("exec-1", "run-a", "reasoner-a", string(types.ExecutionStatusSucceeded), end.Add(-23*time.Hour+30*time.Minute), 200, nil),
		testExecution("exec-2", "run-b", "reasoner-b", string(types.ExecutionStatusFailed), end.Add(-22*time.Hour+5*time.Minute), 400, dashboardStringPtr("failed to invoke")),
		testExecution("exec-3", "run-c", "reasoner-c", string(types.ExecutionStatusCancelled), end.Add(-30*time.Minute), 0, nil),
	}, start, end, TimeRangePreset24h)

	require.Len(t, trend.Last7Days, 24)
	require.Equal(t, 3, trend.Last24h.Total)
	require.Equal(t, 1, trend.Last24h.Succeeded)
	require.Equal(t, 2, trend.Last24h.Failed)
	require.InDelta(t, 33.33, trend.Last24h.SuccessRate, 0.02)
	require.InDelta(t, 300.0, trend.Last24h.AverageDurationMs, 0.01)
	require.InDelta(t, 0.125, trend.Last24h.ThroughputPerHour, 0.0001)
	require.Equal(t, 1, trend.Last7Days[0].Total)
	require.Equal(t, 1, trend.Last7Days[1].Failed)
	require.Equal(t, 1, trend.Last7Days[len(trend.Last7Days)-2].Failed)

	comparison := buildComparisonData(
		EnhancedOverview{ExecutionsLast24h: 12, SuccessRate24h: 75, AverageDurationMs24h: 250},
		EnhancedOverview{ExecutionsLast24h: 8, SuccessRate24h: 50, AverageDurationMs24h: 200},
		start.Add(-24*time.Hour),
		start,
	)
	require.Equal(t, 4, comparison.OverviewDelta.ExecutionsDelta)
	require.InDelta(t, 50.0, comparison.OverviewDelta.ExecutionsDeltaPct, 0.01)
	require.InDelta(t, 25.0, comparison.OverviewDelta.SuccessRateDelta, 0.01)
	require.InDelta(t, 50.0, comparison.OverviewDelta.AvgDurationDeltaMs, 0.01)
	require.InDelta(t, 25.0, comparison.OverviewDelta.AvgDurationDeltaPct, 0.01)
}

func TestBuildHotspotsAndActivityPatterns(t *testing.T) {
	longError := "this is a very long error message that should be truncated to keep the hotspot summary compact for operators reviewing failures"
	start := time.Date(2026, 4, 6, 9, 15, 0, 0, time.UTC)
	executions := []*types.Execution{
		testExecution("exec-1", "run-a", "reasoner-a", string(types.ExecutionStatusFailed), start, 100, dashboardStringPtr(longError)),
		testExecution("exec-2", "run-a", "reasoner-a", string(types.ExecutionStatusTimeout), start.Add(30*time.Minute), 120, dashboardStringPtr("network timeout")),
		testExecution("exec-3", "run-b", "reasoner-b", string(types.ExecutionStatusSucceeded), start.Add(time.Hour), 90, nil),
		testExecution("exec-4", "run-b", "reasoner-b", string(types.ExecutionStatusCancelled), start.Add(2*time.Hour), 90, dashboardStringPtr("network timeout")),
		{ExecutionID: "ignored", StartedAt: start, Status: string(types.ExecutionStatusFailed)},
	}

	hotspots := buildHotspotSummary(executions)
	require.Len(t, hotspots.TopFailingReasoners, 2)
	require.Equal(t, "reasoner-a", hotspots.TopFailingReasoners[0].ReasonerID)
	require.Equal(t, 2, hotspots.TopFailingReasoners[0].FailedExecutions)
	require.InDelta(t, 66.66, hotspots.TopFailingReasoners[0].ContributionPct, 0.05)
	require.True(t, len(hotspots.TopFailingReasoners[0].TopErrors[0].Message) <= 103)
	require.Equal(t, "network timeout", hotspots.TopFailingReasoners[1].TopErrors[0].Message)

	patterns := buildActivityPatterns(executions)
	require.Len(t, patterns.HourlyHeatmap, 7)
	require.Len(t, patterns.HourlyHeatmap[0], 24)
	cell := patterns.HourlyHeatmap[int(start.Weekday())][start.Hour()]
	require.Equal(t, 3, cell.Total)
	require.Equal(t, 3, cell.Failed)
	require.InDelta(t, 100.0, cell.ErrorRate, 0.01)
	secondCell := patterns.HourlyHeatmap[int(start.Add(2*time.Hour).Weekday())][start.Add(2*time.Hour).Hour()]
	require.Equal(t, 1, secondCell.Failed)
	thirdCell := patterns.HourlyHeatmap[int(start.Add(time.Hour).Weekday())][start.Add(time.Hour).Hour()]
	require.Equal(t, 1, thirdCell.Total)
	require.Equal(t, 0, thirdCell.Failed)
	require.InDelta(t, 0.0, thirdCell.ErrorRate, 0.01)
}

func TestBuildOverviewHealthInsightsIncidentsAndStats(t *testing.T) {
	now := time.Date(2026, 4, 8, 15, 0, 0, 0, time.UTC)
	handler := &DashboardHandler{agentService: &stubDashboardAgentService{
		statuses: map[string]*domain.AgentStatus{
			"agent-running": {Name: "agent-running", IsRunning: true, Uptime: "4h"},
			"agent-idle":    {Name: "agent-idle", IsRunning: false},
		},
		errs: map[string]error{
			"agent-missing": errors.New("offline"),
		},
	}}

	agents := []*types.AgentNode{
		{
			ID:              "agent-running",
			TeamID:          "team-a",
			Version:         "1.0.0",
			Reasoners:       []types.ReasonerDefinition{{ID: "r1"}, {ID: "r2"}},
			Skills:          []types.SkillDefinition{{ID: "s1"}},
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
			LastHeartbeat:   now.Add(-2 * time.Minute),
		},
		{
			ID:              "agent-degraded",
			TeamID:          "team-a",
			Version:         "1.0.1",
			Reasoners:       []types.ReasonerDefinition{{ID: "r3"}},
			Skills:          []types.SkillDefinition{{ID: "s2"}, {ID: "s3"}},
			HealthStatus:    types.HealthStatusInactive,
			LifecycleStatus: types.AgentStatusDegraded,
			LastHeartbeat:   now.Add(-5 * time.Minute),
		},
		{
			ID:              "agent-missing",
			TeamID:          "team-b",
			Version:         "2.0.0",
			Reasoners:       []types.ReasonerDefinition{{ID: "r4"}},
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
			LastHeartbeat:   now.Add(-15 * time.Minute),
		},
		{
			ID:              "agent-idle",
			TeamID:          "team-c",
			Version:         "2.1.0",
			Skills:          []types.SkillDefinition{{ID: "s4"}},
			HealthStatus:    types.HealthStatusActive,
			LifecycleStatus: types.AgentStatusReady,
			LastHeartbeat:   now.Add(-30 * time.Minute),
		},
	}

	executions := []*types.Execution{
		testExecutionAt("exec-1", "run-a", "flow-a", string(types.ExecutionStatusSucceeded), now.Add(-2*time.Hour), dashboardTimePtr(now.Add(-110*time.Minute)), dashboardInt64Ptr(600), nil),
		testExecutionAt("exec-2", "run-a", "flow-a", string(types.ExecutionStatusFailed), now.Add(-90*time.Minute), dashboardTimePtr(now.Add(-80*time.Minute)), dashboardInt64Ptr(1200), dashboardStringPtr("failed")),
		testExecutionAt("exec-3", "run-b", "flow-b", string(types.ExecutionStatusSucceeded), now.Add(-80*time.Minute), dashboardTimePtr(now.Add(-70*time.Minute)), dashboardInt64Ptr(300), nil),
		testExecutionAt("exec-4", "run-b", "flow-b", string(types.ExecutionStatusCancelled), now.Add(-70*time.Minute), dashboardTimePtr(now.Add(-65*time.Minute)), dashboardInt64Ptr(200), dashboardStringPtr("cancelled")),
		testExecutionAt("exec-5", "run-c", "flow-c", string(types.ExecutionStatusTimeout), now.Add(-60*time.Minute), dashboardTimePtr(now.Add(-58*time.Minute)), dashboardInt64Ptr(700), dashboardStringPtr("timed out")),
		testExecutionAt("exec-6", "run-d", "flow-d", string(types.ExecutionStatusSucceeded), now.Add(-50*time.Minute), dashboardTimePtr(now.Add(-47*time.Minute)), dashboardInt64Ptr(1500), nil),
		testExecutionAt("exec-7", "run-e", "flow-e", string(types.ExecutionStatusSucceeded), now.Add(-40*time.Minute), dashboardTimePtr(now.Add(-35*time.Minute)), dashboardInt64Ptr(100), nil),
		testExecutionAt("exec-8", "run-f", "flow-f", string(types.ExecutionStatusSucceeded), now.Add(-5*24*time.Hour), dashboardTimePtr(now.Add(-5*24*time.Hour+5*time.Minute)), dashboardInt64Ptr(400), nil),
	}

	overview := handler.buildEnhancedOverview(now, agents, executions)
	require.Equal(t, 4, overview.TotalAgents)
	require.Equal(t, 1, overview.ActiveAgents)
	require.Equal(t, 1, overview.DegradedAgents)
	require.Equal(t, 2, overview.OfflineAgents)
	require.Equal(t, 4, overview.TotalReasoners)
	require.Equal(t, 4, overview.TotalSkills)
	require.Equal(t, 7, overview.ExecutionsLast24h)
	require.Equal(t, 8, overview.ExecutionsLast7d)
	require.InDelta(t, 57.14, overview.SuccessRate24h, 0.02)
	require.InDelta(t, 657.14, overview.AverageDurationMs24h, 0.02)
	require.InDelta(t, 600.0, overview.MedianDurationMs24h, 0.02)

	health := handler.buildAgentHealthSummary(context.Background(), agents)
	require.Equal(t, 4, health.Total)
	require.Equal(t, 1, health.Active)
	require.Equal(t, 1, health.Degraded)
	require.Equal(t, 2, health.Offline)
	require.Len(t, health.Agents, 4)
	require.Equal(t, "agent-degraded", health.Agents[0].ID)
	require.Equal(t, "degraded", health.Agents[0].Status)
	require.Equal(t, "running", health.Agents[len(health.Agents)-1].Status)
	require.Equal(t, "4h", health.Agents[len(health.Agents)-1].Uptime)

	insights := buildWorkflowInsights(executions[:7], []*types.Execution{
		testExecution("run-1", "run-1", "flow-a", string(types.ExecutionStatusRunning), now.Add(-2*time.Minute), 0, nil),
		testExecution("run-2", "run-2", "flow-b", string(types.ExecutionStatusRunning), now.Add(-4*time.Minute), 0, nil),
		testExecution("run-3", "run-3", "flow-c", string(types.ExecutionStatusRunning), now.Add(-6*time.Minute), 0, nil),
		testExecution("run-4", "run-4", "flow-d", string(types.ExecutionStatusRunning), now.Add(-8*time.Minute), 0, nil),
		testExecution("run-5", "run-5", "flow-e", string(types.ExecutionStatusRunning), now.Add(-10*time.Minute), 0, nil),
		testExecution("run-6", "run-6", "flow-f", string(types.ExecutionStatusRunning), now.Add(-12*time.Minute), 0, nil),
		testExecution("run-7", "run-7", "flow-g", string(types.ExecutionStatusRunning), now.Add(-14*time.Minute), 0, nil),
	})
	require.Len(t, insights.TopWorkflows, 5)
	require.Equal(t, "run-b", insights.TopWorkflows[0].WorkflowID)
	require.InDelta(t, 50.0, insights.TopWorkflows[0].SuccessRate, 0.01)
	require.Len(t, insights.ActiveRuns, 6)
	require.True(t, insights.ActiveRuns[0].ElapsedMs >= insights.ActiveRuns[len(insights.ActiveRuns)-1].ElapsedMs)
	require.Len(t, insights.LongestExecutions, 5)
	require.Equal(t, int64(1500), insights.LongestExecutions[0].DurationMs)

	incidents := buildIncidentItems(executions, 2)
	require.Len(t, incidents, 2)
	require.Equal(t, "exec-5", incidents[0].ExecutionID)
	require.Equal(t, "timed out", incidents[0].Error)
	require.Equal(t, "exec-4", incidents[1].ExecutionID)

	require.Equal(t, 0.0, computeMedian(nil))
	require.Equal(t, 3.0, computeMedian([]int64{1, 3, 5}))
	require.Equal(t, 4.0, computeMedian([]int64{7, 1, 5, 3}))
	require.Equal(t, now, maxTime(time.Time{}, now))
	require.Equal(t, now, maxTime(now.Add(-time.Minute), now))
}

func testExecution(id, runID, reasonerID, status string, startedAt time.Time, durationMS int64, errorMessage *string) *types.Execution {
	var completedAt *time.Time
	var duration *int64
	if durationMS > 0 {
		completedAt = dashboardTimePtr(startedAt.Add(time.Duration(durationMS) * time.Millisecond))
		duration = dashboardInt64Ptr(durationMS)
	}

	return testExecutionAt(id, runID, reasonerID, status, startedAt, completedAt, duration, errorMessage)
}

func testExecutionAt(id, runID, reasonerID, status string, startedAt time.Time, completedAt *time.Time, durationMS *int64, errorMessage *string) *types.Execution {
	return &types.Execution{
		ExecutionID:  id,
		RunID:        runID,
		ReasonerID:   reasonerID,
		AgentNodeID:  reasonerID + "-agent",
		Status:       status,
		StartedAt:    startedAt,
		CompletedAt:  completedAt,
		DurationMS:   durationMS,
		ErrorMessage: errorMessage,
	}
}

func dashboardInt64Ptr(v int64) *int64 {
	return &v
}

func dashboardStringPtr(v string) *string {
	return &v
}

func dashboardTimePtr(v time.Time) *time.Time {
	return &v
}
