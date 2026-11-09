package ui

import (
	"context"
	"net/http"
	"sort"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// DashboardHandler provides handlers for dashboard summary operations.
type DashboardHandler struct {
	storage       storage.StorageProvider
	store         executionRecordStore
	agentService  interfaces.AgentService
	cache         *DashboardCache
	enhancedCache *EnhancedDashboardCache
	homeCache     homeCache
	// now overrides the clock in tests.
	now func() time.Time
}

// NewDashboardHandler creates a new DashboardHandler.
func NewDashboardHandler(storage storage.StorageProvider, agentService interfaces.AgentService) *DashboardHandler {
	return &DashboardHandler{
		storage:       storage,
		store:         storage,
		agentService:  agentService,
		cache:         NewDashboardCache(),
		enhancedCache: NewEnhancedDashboardCache(),
	}
}

// DashboardSummaryResponse represents the dashboard summary response
type DashboardSummaryResponse struct {
	Agents     AgentsSummary     `json:"agents"`
	Executions ExecutionsSummary `json:"executions"`
	// SuccessRate is the percentage (0-100) of terminal executions started in
	// the last 24 hours that succeeded; 100 when none have finished.
	SuccessRate float64         `json:"success_rate"`
	Packages    PackagesSummary `json:"packages"`
}

// AgentsSummary represents agent statistics
type AgentsSummary struct {
	Running int `json:"running"`
	Total   int `json:"total"`
}

// ExecutionsSummary represents execution statistics
type ExecutionsSummary struct {
	Today     int `json:"today"`
	Yesterday int `json:"yesterday"`
}

// PackagesSummary represents package statistics
type PackagesSummary struct {
	Available int `json:"available"`
	Installed int `json:"installed"`
}

// GetDashboardSummaryHandler handles dashboard summary requests
// GET /api/ui/v1/dashboard/summary
func (h *DashboardHandler) GetDashboardSummaryHandler(c *gin.Context) {
	ctx := c.Request.Context()
	now := time.Now().UTC()

	// Check cache first
	if cachedData, found := h.cache.Get(); found {
		logger.Logger.Debug().Msg("Returning cached dashboard summary")
		c.JSON(http.StatusOK, cachedData)
		return
	}

	logger.Logger.Debug().Msg("Generating fresh dashboard summary")

	// Collect all data concurrently for better performance
	var wg sync.WaitGroup
	var agentsSummary AgentsSummary
	var executionsSummary ExecutionsSummary
	var packagesSummary PackagesSummary
	var successRate float64
	var errors []error
	var errorsMutex sync.Mutex

	// Helper function to handle errors
	addError := func(err error) {
		if err != nil {
			errorsMutex.Lock()
			errors = append(errors, err)
			errorsMutex.Unlock()
		}
	}

	// Collect agents data
	wg.Add(1)
	go func() {
		defer wg.Done()
		summary, err := h.getAgentsSummary(ctx)
		if err != nil {
			addError(err)
			return
		}
		agentsSummary = summary
	}()

	// Collect executions data and success rate
	wg.Add(1)
	go func() {
		defer wg.Done()
		summary, rate, err := h.getExecutionsSummaryAndSuccessRate(ctx, now)
		if err != nil {
			addError(err)
			return
		}
		executionsSummary = summary
		successRate = rate
	}()

	// Collect packages data
	wg.Add(1)
	go func() {
		defer wg.Done()
		summary, err := h.getPackagesSummary(ctx)
		if err != nil {
			addError(err)
			return
		}
		packagesSummary = summary
	}()

	// Wait for all goroutines to complete
	wg.Wait()

	// Check for errors
	if len(errors) > 0 {
		logger.Logger.Error().Errs("errors", errors).Msg("Errors occurred while collecting dashboard data")
		RespondInternalError(c, "failed to collect dashboard data")
		return
	}

	// Build response
	response := &DashboardSummaryResponse{
		Agents:      agentsSummary,
		Executions:  executionsSummary,
		SuccessRate: successRate,
		Packages:    packagesSummary,
	}

	// Cache the response
	h.cache.Set(response)

	c.JSON(http.StatusOK, response)
}

// parseTimeRangeParams extracts time range from query parameters
func parseTimeRangeParams(c *gin.Context, now time.Time) (startTime, endTime time.Time, preset TimeRangePreset, err error) {
	presetStr := c.DefaultQuery("preset", "24h")
	preset = TimeRangePreset(presetStr)

	// Round to hour for consistent cache behavior
	roundedNow := now.Truncate(time.Hour).Add(time.Hour) // Include current hour

	switch preset {
	case TimeRangePreset1h:
		startTime = roundedNow.Add(-1 * time.Hour)
		endTime = roundedNow
	case TimeRangePreset24h:
		startTime = roundedNow.Add(-24 * time.Hour)
		endTime = roundedNow
	case TimeRangePreset7d:
		startTime = roundedNow.AddDate(0, 0, -7)
		endTime = roundedNow
	case TimeRangePreset30d:
		startTime = roundedNow.AddDate(0, 0, -30)
		endTime = roundedNow
	case TimeRangePresetCustom:
		startStr := c.Query("start_time")
		endStr := c.Query("end_time")
		if startStr == "" || endStr == "" {
			logger.Logger.Warn().Msg("start_time and end_time required for custom range, falling back to 24h")
			return now.Add(-24 * time.Hour), now, TimeRangePreset24h, nil
		}
		startTime, err = time.Parse(time.RFC3339, startStr)
		if err != nil {
			logger.Logger.Warn().Err(err).Msg("invalid start_time format, falling back to 24h")
			return now.Add(-24 * time.Hour), now, TimeRangePreset24h, nil
		}
		endTime, err = time.Parse(time.RFC3339, endStr)
		if err != nil {
			logger.Logger.Warn().Err(err).Msg("invalid end_time format, falling back to 24h")
			return now.Add(-24 * time.Hour), now, TimeRangePreset24h, nil
		}
	default:
		// Default to 24h
		preset = TimeRangePreset24h
		startTime = roundedNow.Add(-24 * time.Hour)
		endTime = roundedNow
	}

	return startTime, endTime, preset, nil
}

// calculateComparisonPeriod returns the previous period for comparison
func calculateComparisonPeriod(startTime, endTime time.Time) (prevStart, prevEnd time.Time) {
	duration := endTime.Sub(startTime)
	return startTime.Add(-duration), startTime
}

// GetEnhancedDashboardSummaryHandler handles requests for the enhanced dashboard view
// GET /api/ui/v1/dashboard/enhanced
// Query params:
//   - preset: "1h", "24h", "7d", "30d", "custom" (default: "24h")
//   - start_time: RFC3339 timestamp (required if preset=custom)
//   - end_time: RFC3339 timestamp (required if preset=custom)
//   - compare: "true" to include comparison data (default: "false")
func (h *DashboardHandler) GetEnhancedDashboardSummaryHandler(c *gin.Context) {
	ctx := c.Request.Context()
	now := time.Now().UTC()

	// Parse time range from query params
	startTime, endTime, preset, err := parseTimeRangeParams(c, now)
	if err != nil {
		RespondBadRequest(c, "invalid time range parameters")
		return
	}

	// Check if comparison is requested
	enableComparison := c.Query("compare") == "true"

	// Generate cache key and check cache
	cacheKey := generateCacheKey(startTime, endTime, enableComparison)
	if cached, found := h.enhancedCache.Get(cacheKey, preset); found {
		logger.Logger.Debug().Str("key", cacheKey).Msg("Returning cached enhanced dashboard summary")
		c.JSON(http.StatusOK, cached)
		return
	}

	// The per-reasoner breakdowns (trends, hotspots, heatmap, top workflows)
	// are built from a sample of the window's executions. The sample is the
	// NEWEST rows, so a window with more executions than the sample holds
	// still describes the latest part of the window, and detail_sampled says
	// so. The counts and incidents below never come from the sample.
	executions, sampled, err := h.queryDashboardSample(ctx, startTime, endTime)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to query workflow executions for enhanced dashboard")
		RespondInternalError(c, "failed to load workflow execution data")
		return
	}

	rangeCounts, err := h.storage.CountExecutionsByStatus(ctx, startTime, endTime)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to count executions for enhanced dashboard")
		RespondInternalError(c, "failed to load workflow execution data")
		return
	}
	weekCounts, err := h.storage.CountExecutionsByStatus(ctx, endTime.AddDate(0, 0, -7), endTime)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to count executions for enhanced dashboard")
		RespondInternalError(c, "failed to load workflow execution data")
		return
	}
	recentFailures, err := h.storage.QueryRecentFailures(ctx, startTime, endTime, 10)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to query incidents for enhanced dashboard")
		RespondInternalError(c, "failed to load workflow execution data")
		return
	}

	agents, err := h.storage.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to list agents for enhanced dashboard")
		RespondInternalError(c, "failed to load agent data")
		return
	}

	statusRunning := string(types.ExecutionStatusRunning)
	runningExecutions, err := h.store.QueryExecutionRecords(ctx, types.ExecutionFilter{
		Status:          &statusRunning,
		Limit:           12,
		SortBy:          "started_at",
		SortDescending:  true,
		ExcludePayloads: true,
	})
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to query running executions for enhanced dashboard")
		RespondInternalError(c, "failed to load active workflow data")
		return
	}

	statusWaiting := string(types.ExecutionStatusWaiting)
	waitingExecutions, err := h.store.QueryExecutionRecords(ctx, types.ExecutionFilter{
		Status:          &statusWaiting,
		Limit:           12,
		SortBy:          "started_at",
		SortDescending:  true,
		ExcludePayloads: true,
	})
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to query waiting executions for enhanced dashboard")
		RespondInternalError(c, "failed to load active workflow data")
		return
	}

	statusQueued := string(types.ExecutionStatusQueued)
	queuedExecutions, err := h.store.QueryExecutionRecords(ctx, types.ExecutionFilter{
		Status:          &statusQueued,
		Limit:           12,
		SortBy:          "started_at",
		SortDescending:  true,
		ExcludePayloads: true,
	})
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to query queued executions for enhanced dashboard")
		RespondInternalError(c, "failed to load active workflow data")
		return
	}

	activeExecutions := make([]*types.Execution, 0, len(runningExecutions)+len(waitingExecutions)+len(queuedExecutions))
	activeExecutions = append(activeExecutions, runningExecutions...)
	activeExecutions = append(activeExecutions, waitingExecutions...)
	activeExecutions = append(activeExecutions, queuedExecutions...)
	sort.Slice(activeExecutions, func(i, j int) bool {
		return activeExecutions[i].StartedAt.After(activeExecutions[j].StartedAt)
	})
	if len(activeExecutions) > 12 {
		activeExecutions = activeExecutions[:12]
	}

	// Build time range info
	timeRange := TimeRangeInfo{
		StartTime: startTime,
		EndTime:   endTime,
		Preset:    preset,
	}

	overview := h.buildEnhancedOverviewForRange(agents, executions, startTime, endTime)
	applyExactCounts(&overview, rangeCounts, weekCounts)
	trends := buildExecutionTrendsForRange(executions, startTime, endTime, preset)
	agentHealth := h.buildAgentHealthSummary(ctx, agents)
	workflows := buildWorkflowInsights(executions, activeExecutions)
	incidents := buildIncidentItems(recentFailures, 10)
	hotspots := buildHotspotSummary(executions)
	activityPatterns := buildActivityPatterns(executions)

	response := &EnhancedDashboardResponse{
		GeneratedAt:      now,
		TimeRange:        timeRange,
		Overview:         overview,
		ExecutionTrends:  trends,
		AgentHealth:      agentHealth,
		Workflows:        workflows,
		Incidents:        incidents,
		Hotspots:         hotspots,
		ActivityPatterns: activityPatterns,
		DetailSampled:    sampled,
	}

	// Calculate comparison data if requested
	if enableComparison {
		prevStart, prevEnd := calculateComparisonPeriod(startTime, endTime)
		prevExecutions, _, err := h.queryDashboardSample(ctx, prevStart, prevEnd)
		if err == nil {
			prevOverview := h.buildEnhancedOverviewForRange(agents, prevExecutions, prevStart, prevEnd)
			prevCounts, countErr := h.storage.CountExecutionsByStatus(ctx, prevStart, prevEnd)
			prevWeek, weekErr := h.storage.CountExecutionsByStatus(ctx, prevEnd.AddDate(0, 0, -7), prevEnd)
			if countErr == nil && weekErr == nil {
				applyExactCounts(&prevOverview, prevCounts, prevWeek)
				response.Comparison = buildComparisonData(overview, prevOverview, prevStart, prevEnd)
			}
		}
	}

	h.enhancedCache.Set(cacheKey, response)
	c.JSON(http.StatusOK, response)
}

// dashboardSampleLimit bounds how many executions the enhanced dashboard
// loads to build its per-reasoner breakdowns.
const dashboardSampleLimit = 50000

// queryDashboardSample loads the newest executions of a window, oldest first,
// and reports whether the window held more than the sample.
func (h *DashboardHandler) queryDashboardSample(ctx context.Context, start, end time.Time) ([]*types.Execution, bool, error) {
	executions, err := h.store.QueryExecutionRecords(ctx, types.ExecutionFilter{
		StartTime:       &start,
		EndTime:         &end,
		Limit:           dashboardSampleLimit,
		SortBy:          "started_at",
		SortDescending:  true,
		ExcludePayloads: true,
	})
	if err != nil {
		return nil, false, err
	}
	for i, j := 0, len(executions)-1; i < j; i, j = i+1, j-1 {
		executions[i], executions[j] = executions[j], executions[i]
	}
	return executions, len(executions) >= dashboardSampleLimit, nil
}

// applyExactCounts replaces the sample-derived counts in an overview with
// COUNT results: executions in the requested range, executions in the seven
// days ending with it, and the range's success rate.
func applyExactCounts(overview *EnhancedOverview, inRange, lastWeek types.ExecutionStatusCounts) {
	overview.ExecutionsLast24h = int(inRange.Total)
	overview.ExecutionsLast7d = int(lastWeek.Total)
	overview.SuccessRate24h = 0
	if inRange.Total > 0 {
		overview.SuccessRate24h = float64(inRange.Succeeded) / float64(inRange.Total) * 100
	}
}
