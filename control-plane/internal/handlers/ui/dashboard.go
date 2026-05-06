package ui

import (
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
	Agents      AgentsSummary     `json:"agents"`
	Executions  ExecutionsSummary `json:"executions"`
	SuccessRate float64           `json:"success_rate"`
	Packages    PackagesSummary   `json:"packages"`
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

	// Query executions for the specified time range
	filters := types.ExecutionFilter{
		StartTime:       &startTime,
		EndTime:         &endTime,
		Limit:           50000,
		SortBy:          "started_at",
		SortDescending:  false,
		ExcludePayloads: true,
	}

	executions, err := h.store.QueryExecutionRecords(ctx, filters)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to query workflow executions for enhanced dashboard")
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

	activeExecutions := append(runningExecutions, waitingExecutions...)
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
	trends := buildExecutionTrendsForRange(executions, startTime, endTime, preset)
	agentHealth := h.buildAgentHealthSummary(ctx, agents)
	workflows := buildWorkflowInsights(executions, activeExecutions)
	incidents := buildIncidentItems(executions, 10)
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
	}

	// Calculate comparison data if requested
	if enableComparison {
		prevStart, prevEnd := calculateComparisonPeriod(startTime, endTime)
		prevFilters := types.ExecutionFilter{
			StartTime:       &prevStart,
			EndTime:         &prevEnd,
			Limit:           50000,
			SortBy:          "started_at",
			SortDescending:  false,
			ExcludePayloads: true,
		}

		prevExecutions, err := h.store.QueryExecutionRecords(ctx, prevFilters)
		if err == nil {
			prevOverview := h.buildEnhancedOverviewForRange(agents, prevExecutions, prevStart, prevEnd)
			response.Comparison = buildComparisonData(overview, prevOverview, prevStart, prevEnd)
		}
	}

	h.enhancedCache.Set(cacheKey, response)
	c.JSON(http.StatusOK, response)
}
