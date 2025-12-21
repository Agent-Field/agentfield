package ui

import (
	"context"
	"fmt"
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

// DashboardCache provides 30-second caching for dashboard data
type DashboardCache struct {
	data      *DashboardSummaryResponse
	timestamp time.Time
	mutex     sync.RWMutex
	ttl       time.Duration
}

// NewDashboardCache creates a new dashboard cache with 30-second TTL
func NewDashboardCache() *DashboardCache {
	return &DashboardCache{
		ttl: 30 * time.Second,
	}
}

// Get retrieves cached data if still valid
func (c *DashboardCache) Get() (*DashboardSummaryResponse, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if c.data != nil && time.Since(c.timestamp) < c.ttl {
		return c.data, true
	}
	return nil, false
}

// Set stores data in cache with current timestamp
func (c *DashboardCache) Set(data *DashboardSummaryResponse) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.data = data
	c.timestamp = time.Now()
}

// TimeRangePreset represents standard time range options
type TimeRangePreset string

const (
	TimeRangePreset1h  TimeRangePreset = "1h"
	TimeRangePreset24h TimeRangePreset = "24h"
	TimeRangePreset7d  TimeRangePreset = "7d"
	TimeRangePreset30d TimeRangePreset = "30d"
	TimeRangePresetCustom TimeRangePreset = "custom"
)

// TimeRangeInfo describes the time range used for the dashboard query
type TimeRangeInfo struct {
	StartTime time.Time       `json:"start_time"`
	EndTime   time.Time       `json:"end_time"`
	Preset    TimeRangePreset `json:"preset,omitempty"`
}

// ComparisonData contains delta information comparing current to previous period
type ComparisonData struct {
	PreviousPeriod TimeRangeInfo           `json:"previous_period"`
	OverviewDelta  EnhancedOverviewDelta   `json:"overview_delta"`
}

// EnhancedOverviewDelta contains changes compared to the previous period
type EnhancedOverviewDelta struct {
	ExecutionsDelta     int     `json:"executions_delta"`
	ExecutionsDeltaPct  float64 `json:"executions_delta_pct"`
	SuccessRateDelta    float64 `json:"success_rate_delta"`
	AvgDurationDeltaMs  float64 `json:"avg_duration_delta_ms"`
	AvgDurationDeltaPct float64 `json:"avg_duration_delta_pct"`
}

// HotspotSummary contains top error contributors by reasoner
type HotspotSummary struct {
	TopFailingReasoners []HotspotItem `json:"top_failing_reasoners"`
}

// HotspotItem represents a single reasoner's failure statistics
type HotspotItem struct {
	ReasonerID        string       `json:"reasoner_id"`
	TotalExecutions   int          `json:"total_executions"`
	FailedExecutions  int          `json:"failed_executions"`
	ErrorRate         float64      `json:"error_rate"`
	ContributionPct   float64      `json:"contribution_pct"`
	TopErrors         []ErrorCount `json:"top_errors"`
}

// ErrorCount tracks error message frequency
type ErrorCount struct {
	Message string `json:"message"`
	Count   int    `json:"count"`
}

// ActivityPatterns contains temporal patterns for failures and usage
type ActivityPatterns struct {
	// HourlyHeatmap is a 7x24 matrix [dayOfWeek][hourOfDay]
	HourlyHeatmap [][]HeatmapCell `json:"hourly_heatmap"`
}

// HeatmapCell contains execution statistics for a specific day/hour combination
type HeatmapCell struct {
	Total     int     `json:"total"`
	Failed    int     `json:"failed"`
	ErrorRate float64 `json:"error_rate"`
}

// Enhanced dashboard response structures
type EnhancedDashboardResponse struct {
	GeneratedAt      time.Time           `json:"generated_at"`
	TimeRange        TimeRangeInfo       `json:"time_range"`
	Overview         EnhancedOverview    `json:"overview"`
	ExecutionTrends  ExecutionTrends     `json:"execution_trends"`
	AgentHealth      AgentHealthSummary  `json:"agent_health"`
	Workflows        WorkflowInsights    `json:"workflows"`
	Incidents        []IncidentItem      `json:"incidents"`
	Comparison       *ComparisonData     `json:"comparison,omitempty"`
	Hotspots         HotspotSummary      `json:"hotspots"`
	ActivityPatterns ActivityPatterns    `json:"activity_patterns"`
}

type EnhancedOverview struct {
	TotalAgents          int     `json:"total_agents"`
	ActiveAgents         int     `json:"active_agents"`
	DegradedAgents       int     `json:"degraded_agents"`
	OfflineAgents        int     `json:"offline_agents"`
	TotalReasoners       int     `json:"total_reasoners"`
	TotalSkills          int     `json:"total_skills"`
	ExecutionsLast24h    int     `json:"executions_last_24h"`
	ExecutionsLast7d     int     `json:"executions_last_7d"`
	SuccessRate24h       float64 `json:"success_rate_24h"`
	AverageDurationMs24h float64 `json:"average_duration_ms_24h"`
	MedianDurationMs24h  float64 `json:"median_duration_ms_24h"`
}

type ExecutionTrends struct {
	Last24h   ExecutionWindowMetrics `json:"last_24h"`
	Last7Days []ExecutionTrendPoint  `json:"last_7_days"`
}

type ExecutionWindowMetrics struct {
	Total             int     `json:"total"`
	Succeeded         int     `json:"succeeded"`
	Failed            int     `json:"failed"`
	SuccessRate       float64 `json:"success_rate"`
	AverageDurationMs float64 `json:"average_duration_ms"`
	ThroughputPerHour float64 `json:"throughput_per_hour"`
}

type ExecutionTrendPoint struct {
	Date      string `json:"date"`
	Total     int    `json:"total"`
	Succeeded int    `json:"succeeded"`
	Failed    int    `json:"failed"`
}

type AgentHealthSummary struct {
	Total    int               `json:"total"`
	Active   int               `json:"active"`
	Degraded int               `json:"degraded"`
	Offline  int               `json:"offline"`
	Agents   []AgentHealthItem `json:"agents"`
}

type AgentHealthItem struct {
	ID            string    `json:"id"`
	TeamID        string    `json:"team_id"`
	Version       string    `json:"version"`
	Status        string    `json:"status"`
	Health        string    `json:"health"`
	Lifecycle     string    `json:"lifecycle"`
	LastHeartbeat time.Time `json:"last_heartbeat"`
	Reasoners     int       `json:"reasoners"`
	Skills        int       `json:"skills"`
	Uptime        string    `json:"uptime,omitempty"`
}

type WorkflowInsights struct {
	TopWorkflows      []WorkflowStat           `json:"top_workflows"`
	ActiveRuns        []ActiveWorkflowRun      `json:"active_runs"`
	LongestExecutions []CompletedExecutionStat `json:"longest_executions"`
}

type WorkflowStat struct {
	WorkflowID       string    `json:"workflow_id"`
	Name             string    `json:"name,omitempty"`
	TotalExecutions  int       `json:"total_executions"`
	SuccessRate      float64   `json:"success_rate"`
	FailedExecutions int       `json:"failed_executions"`
	AverageDuration  float64   `json:"average_duration_ms"`
	LastActivity     time.Time `json:"last_activity"`
}

type ActiveWorkflowRun struct {
	ExecutionID string    `json:"execution_id"`
	WorkflowID  string    `json:"workflow_id"`
	Name        string    `json:"name,omitempty"`
	StartedAt   time.Time `json:"started_at"`
	ElapsedMs   int64     `json:"elapsed_ms"`
	AgentNodeID string    `json:"agent_node_id"`
	ReasonerID  string    `json:"reasoner_id"`
	Status      string    `json:"status"`
}

type CompletedExecutionStat struct {
	ExecutionID string     `json:"execution_id"`
	WorkflowID  string     `json:"workflow_id"`
	Name        string     `json:"name,omitempty"`
	DurationMs  int64      `json:"duration_ms"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
	Status      string     `json:"status"`
}

type IncidentItem struct {
	ExecutionID string     `json:"execution_id"`
	WorkflowID  string     `json:"workflow_id"`
	Name        string     `json:"name,omitempty"`
	Status      string     `json:"status"`
	StartedAt   time.Time  `json:"started_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty"`
	AgentNodeID string     `json:"agent_node_id"`
	ReasonerID  string     `json:"reasoner_id"`
	Error       string     `json:"error,omitempty"`
}

// enhancedCacheEntry holds cached data with timestamp
type enhancedCacheEntry struct {
	data      *EnhancedDashboardResponse
	timestamp time.Time
}

// EnhancedDashboardCache provides time-range-aware caching for the enhanced dashboard response
type EnhancedDashboardCache struct {
	entries  map[string]*enhancedCacheEntry
	mutex    sync.RWMutex
	maxSize  int
}

// NewEnhancedDashboardCache creates a new cache instance for enhanced dashboard data
func NewEnhancedDashboardCache() *EnhancedDashboardCache {
	return &EnhancedDashboardCache{
		entries: make(map[string]*enhancedCacheEntry),
		maxSize: 10, // LRU limit
	}
}

// getTTLForPreset returns the appropriate cache TTL based on time range
func getTTLForPreset(preset TimeRangePreset) time.Duration {
	switch preset {
	case TimeRangePreset1h:
		return 30 * time.Second
	case TimeRangePreset24h:
		return 60 * time.Second
	case TimeRangePreset7d:
		return 2 * time.Minute
	case TimeRangePreset30d:
		return 5 * time.Minute
	default:
		return 60 * time.Second
	}
}

// generateCacheKey creates a cache key from time range parameters
func generateCacheKey(startTime, endTime time.Time, compare bool) string {
	// Round to hour for better cache reuse
	startHour := startTime.Truncate(time.Hour).Unix()
	endHour := endTime.Truncate(time.Hour).Unix()
	compareStr := "0"
	if compare {
		compareStr = "1"
	}
	return fmt.Sprintf("%d-%d-%s", startHour, endHour, compareStr)
}

// Get retrieves cached enhanced dashboard data if still valid
func (c *EnhancedDashboardCache) Get(key string, preset TimeRangePreset) (*EnhancedDashboardResponse, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	entry, exists := c.entries[key]
	if !exists {
		return nil, false
	}

	ttl := getTTLForPreset(preset)
	if time.Since(entry.timestamp) >= ttl {
		return nil, false
	}

	return entry.data, true
}

// Set stores enhanced dashboard data in the cache with LRU eviction
func (c *EnhancedDashboardCache) Set(key string, data *EnhancedDashboardResponse) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	// Simple LRU: if at capacity, remove oldest entry
	if len(c.entries) >= c.maxSize {
		var oldestKey string
		var oldestTime time.Time
		for k, entry := range c.entries {
			if oldestKey == "" || entry.timestamp.Before(oldestTime) {
				oldestKey = k
				oldestTime = entry.timestamp
			}
		}
		if oldestKey != "" {
			delete(c.entries, oldestKey)
		}
	}

	c.entries[key] = &enhancedCacheEntry{
		data:      data,
		timestamp: time.Now(),
	}
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
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to collect dashboard data"})
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
			return now.Add(-24*time.Hour), now, TimeRangePreset24h, nil
		}
		startTime, err = time.Parse(time.RFC3339, startStr)
		if err != nil {
			logger.Logger.Warn().Err(err).Msg("invalid start_time format, falling back to 24h")
			return now.Add(-24*time.Hour), now, TimeRangePreset24h, nil
		}
		endTime, err = time.Parse(time.RFC3339, endStr)
		if err != nil {
			logger.Logger.Warn().Err(err).Msg("invalid end_time format, falling back to 24h")
			return now.Add(-24*time.Hour), now, TimeRangePreset24h, nil
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
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "invalid time range parameters"})
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
		StartTime:      &startTime,
		EndTime:        &endTime,
		Limit:          50000,
		SortBy:         "started_at",
		SortDescending: false,
	}

	executions, err := h.store.QueryExecutionRecords(ctx, filters)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to query workflow executions for enhanced dashboard")
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to load workflow execution data"})
		return
	}

	agents, err := h.storage.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to list agents for enhanced dashboard")
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to load agent data"})
		return
	}

	statusRunning := string(types.ExecutionStatusRunning)
	runningExecutions, err := h.store.QueryExecutionRecords(ctx, types.ExecutionFilter{
		Status:         &statusRunning,
		Limit:          12,
		SortBy:         "started_at",
		SortDescending: true,
	})
	if err != nil {
		logger.Logger.Error().Err(err).Msg("failed to query running executions for enhanced dashboard")
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to load active workflow data"})
		return
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
	workflows := buildWorkflowInsights(executions, runningExecutions)
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
			StartTime:      &prevStart,
			EndTime:        &prevEnd,
			Limit:          50000,
			SortBy:         "started_at",
			SortDescending: false,
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

// buildEnhancedOverviewForRange builds overview metrics for a specific time range
func (h *DashboardHandler) buildEnhancedOverviewForRange(agents []*types.AgentNode, executions []*types.Execution, startTime, endTime time.Time) EnhancedOverview {
	overview := EnhancedOverview{TotalAgents: len(agents)}

	for _, agent := range agents {
		overview.TotalReasoners += len(agent.Reasoners)
		overview.TotalSkills += len(agent.Skills)

		isDegraded := agent.LifecycleStatus == types.AgentStatusDegraded || agent.HealthStatus == types.HealthStatusInactive
		if isDegraded {
			overview.DegradedAgents++
			continue
		}

		status, err := h.agentService.GetAgentStatus(agent.ID)
		if err != nil {
			overview.OfflineAgents++
			continue
		}

		if status != nil && status.IsRunning {
			overview.ActiveAgents++
		} else {
			overview.OfflineAgents++
		}
	}

	if overview.OfflineAgents < 0 {
		overview.OfflineAgents = 0
	}

	var durationSamples []int64
	var durationSum float64
	var durationCount float64
	var successCount, failedCount int

	for _, exec := range executions {
		overview.ExecutionsLast24h++ // Repurposed as "executions in range"

		normalized := types.NormalizeExecutionStatus(exec.Status)
		switch normalized {
		case string(types.ExecutionStatusSucceeded):
			successCount++
		case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
			failedCount++
		}

		if exec.DurationMS != nil {
			d := *exec.DurationMS
			durationSamples = append(durationSamples, d)
			durationSum += float64(d)
			durationCount++
		}
	}

	overview.ExecutionsLast7d = len(executions)
	if len(executions) > 0 {
		overview.SuccessRate24h = (float64(successCount) / float64(len(executions))) * 100
	}
	if durationCount > 0 {
		overview.AverageDurationMs24h = durationSum / durationCount
	}
	overview.MedianDurationMs24h = computeMedian(durationSamples)

	return overview
}

// buildExecutionTrendsForRange builds trends for the specified time range
func buildExecutionTrendsForRange(executions []*types.Execution, startTime, endTime time.Time, preset TimeRangePreset) ExecutionTrends {
	trend := ExecutionTrends{}
	duration := endTime.Sub(startTime)

	// Determine bucket size based on preset
	var bucketDuration time.Duration
	var numBuckets int
	switch preset {
	case TimeRangePreset1h:
		bucketDuration = 5 * time.Minute
		numBuckets = 12
	case TimeRangePreset24h:
		bucketDuration = time.Hour
		numBuckets = 24
	case TimeRangePreset7d:
		bucketDuration = 24 * time.Hour
		numBuckets = 7
	case TimeRangePreset30d:
		bucketDuration = 24 * time.Hour
		numBuckets = 30
	default:
		// For custom, use day buckets capped at 30
		bucketDuration = 24 * time.Hour
		numBuckets = int(duration.Hours() / 24)
		if numBuckets > 30 {
			numBuckets = 30
		}
		if numBuckets < 1 {
			numBuckets = 1
		}
	}

	// Create buckets
	dayBuckets := make(map[string]*ExecutionTrendPoint)
	orderedKeys := make([]string, 0, numBuckets)

	for i := numBuckets - 1; i >= 0; i-- {
		bucketTime := endTime.Add(-time.Duration(i) * bucketDuration)
		var key string
		if bucketDuration >= 24*time.Hour {
			key = bucketTime.Format("2006-01-02")
		} else {
			key = bucketTime.Format("2006-01-02T15:04")
		}
		orderedKeys = append(orderedKeys, key)
		dayBuckets[key] = &ExecutionTrendPoint{Date: key}
	}

	var totalInRange, successInRange, failedInRange int
	var durationSum float64
	var durationCount float64

	for _, exec := range executions {
		var bucketKey string
		if bucketDuration >= 24*time.Hour {
			bucketKey = exec.StartedAt.Format("2006-01-02")
		} else {
			// Round to bucket
			bucketKey = exec.StartedAt.Truncate(bucketDuration).Format("2006-01-02T15:04")
		}

		if point, ok := dayBuckets[bucketKey]; ok {
			point.Total++
			normalized := types.NormalizeExecutionStatus(exec.Status)
			switch normalized {
			case string(types.ExecutionStatusSucceeded):
				point.Succeeded++
			case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
				point.Failed++
			}
		}

		totalInRange++
		normalized := types.NormalizeExecutionStatus(exec.Status)
		switch normalized {
		case string(types.ExecutionStatusSucceeded):
			successInRange++
		case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
			failedInRange++
		}

		if exec.DurationMS != nil {
			durationSum += float64(*exec.DurationMS)
			durationCount++
		}
	}

	trend.Last7Days = make([]ExecutionTrendPoint, 0, len(orderedKeys))
	for _, key := range orderedKeys {
		trend.Last7Days = append(trend.Last7Days, *dayBuckets[key])
	}

	trend.Last24h.Total = totalInRange
	trend.Last24h.Succeeded = successInRange
	trend.Last24h.Failed = failedInRange
	if totalInRange > 0 {
		trend.Last24h.SuccessRate = (float64(successInRange) / float64(totalInRange)) * 100
		hours := duration.Hours()
		if hours > 0 {
			trend.Last24h.ThroughputPerHour = float64(totalInRange) / hours
		}
	}
	if durationCount > 0 {
		trend.Last24h.AverageDurationMs = durationSum / durationCount
	}

	return trend
}

// buildComparisonData creates comparison metrics between current and previous periods
func buildComparisonData(current, previous EnhancedOverview, prevStart, prevEnd time.Time) *ComparisonData {
	delta := EnhancedOverviewDelta{}

	// Executions delta
	delta.ExecutionsDelta = current.ExecutionsLast24h - previous.ExecutionsLast24h
	if previous.ExecutionsLast24h > 0 {
		delta.ExecutionsDeltaPct = (float64(delta.ExecutionsDelta) / float64(previous.ExecutionsLast24h)) * 100
	}

	// Success rate delta
	delta.SuccessRateDelta = current.SuccessRate24h - previous.SuccessRate24h

	// Duration delta
	delta.AvgDurationDeltaMs = current.AverageDurationMs24h - previous.AverageDurationMs24h
	if previous.AverageDurationMs24h > 0 {
		delta.AvgDurationDeltaPct = (delta.AvgDurationDeltaMs / previous.AverageDurationMs24h) * 100
	}

	return &ComparisonData{
		PreviousPeriod: TimeRangeInfo{
			StartTime: prevStart,
			EndTime:   prevEnd,
		},
		OverviewDelta: delta,
	}
}

// buildHotspotSummary aggregates failures by reasoner
func buildHotspotSummary(executions []*types.Execution) HotspotSummary {
	type reasonerStats struct {
		total      int
		failed     int
		errorMsgs  map[string]int
	}

	statsMap := make(map[string]*reasonerStats)
	totalFailures := 0

	for _, exec := range executions {
		if exec.ReasonerID == "" {
			continue
		}

		stats, ok := statsMap[exec.ReasonerID]
		if !ok {
			stats = &reasonerStats{errorMsgs: make(map[string]int)}
			statsMap[exec.ReasonerID] = stats
		}

		stats.total++

		normalized := types.NormalizeExecutionStatus(exec.Status)
		if normalized == string(types.ExecutionStatusFailed) ||
			normalized == string(types.ExecutionStatusCancelled) ||
			normalized == string(types.ExecutionStatusTimeout) {
			stats.failed++
			totalFailures++

			if exec.ErrorMessage != nil && *exec.ErrorMessage != "" {
				// Truncate long error messages
				errMsg := *exec.ErrorMessage
				if len(errMsg) > 100 {
					errMsg = errMsg[:100] + "..."
				}
				stats.errorMsgs[errMsg]++
			}
		}
	}

	// Convert to slice and sort by failure count
	items := make([]HotspotItem, 0, len(statsMap))
	for reasonerID, stats := range statsMap {
		if stats.failed == 0 {
			continue
		}

		item := HotspotItem{
			ReasonerID:       reasonerID,
			TotalExecutions:  stats.total,
			FailedExecutions: stats.failed,
		}

		if stats.total > 0 {
			item.ErrorRate = (float64(stats.failed) / float64(stats.total)) * 100
		}
		if totalFailures > 0 {
			item.ContributionPct = (float64(stats.failed) / float64(totalFailures)) * 100
		}

		// Get top errors (up to 3)
		topErrors := make([]ErrorCount, 0, 3)
		for msg, count := range stats.errorMsgs {
			topErrors = append(topErrors, ErrorCount{Message: msg, Count: count})
		}
		sort.Slice(topErrors, func(i, j int) bool {
			return topErrors[i].Count > topErrors[j].Count
		})
		if len(topErrors) > 3 {
			topErrors = topErrors[:3]
		}
		item.TopErrors = topErrors

		items = append(items, item)
	}

	// Sort by failure count descending
	sort.Slice(items, func(i, j int) bool {
		return items[i].FailedExecutions > items[j].FailedExecutions
	})

	// Limit to top 10
	if len(items) > 10 {
		items = items[:10]
	}

	return HotspotSummary{TopFailingReasoners: items}
}

// buildActivityPatterns creates a 7x24 heatmap of execution activity
func buildActivityPatterns(executions []*types.Execution) ActivityPatterns {
	// Initialize 7x24 grid (Sunday=0 through Saturday=6)
	heatmap := make([][]HeatmapCell, 7)
	for day := 0; day < 7; day++ {
		heatmap[day] = make([]HeatmapCell, 24)
	}

	for _, exec := range executions {
		dayOfWeek := int(exec.StartedAt.Weekday())
		hourOfDay := exec.StartedAt.Hour()

		heatmap[dayOfWeek][hourOfDay].Total++

		normalized := types.NormalizeExecutionStatus(exec.Status)
		if normalized == string(types.ExecutionStatusFailed) ||
			normalized == string(types.ExecutionStatusCancelled) ||
			normalized == string(types.ExecutionStatusTimeout) {
			heatmap[dayOfWeek][hourOfDay].Failed++
		}
	}

	// Calculate error rates
	for day := 0; day < 7; day++ {
		for hour := 0; hour < 24; hour++ {
			if heatmap[day][hour].Total > 0 {
				heatmap[day][hour].ErrorRate = (float64(heatmap[day][hour].Failed) / float64(heatmap[day][hour].Total)) * 100
			}
		}
	}

	return ActivityPatterns{HourlyHeatmap: heatmap}
}

func (h *DashboardHandler) buildEnhancedOverview(now time.Time, agents []*types.AgentNode, executions []*types.Execution) EnhancedOverview {
	overview := EnhancedOverview{TotalAgents: len(agents)}

	for _, agent := range agents {
		// Count reasoners and skills
		overview.TotalReasoners += len(agent.Reasoners)
		overview.TotalSkills += len(agent.Skills)

		isDegraded := agent.LifecycleStatus == types.AgentStatusDegraded || agent.HealthStatus == types.HealthStatusInactive
		if isDegraded {
			overview.DegradedAgents++
			continue
		}

		status, err := h.agentService.GetAgentStatus(agent.ID)
		if err != nil {
			overview.OfflineAgents++
			continue
		}

		if status != nil && status.IsRunning {
			overview.ActiveAgents++
		} else {
			overview.OfflineAgents++
		}
	}

	// Ensure offline count is consistent
	if overview.OfflineAgents < 0 {
		overview.OfflineAgents = 0
	}

	last24h := now.Add(-24 * time.Hour)
	var durationSamples []int64
	var durationSum float64
	var durationCount float64
	var success24h, failed24h int

	for _, exec := range executions {
		if exec.StartedAt.After(last24h) || exec.StartedAt.Equal(last24h) {
			overview.ExecutionsLast24h++

			normalized := types.NormalizeExecutionStatus(exec.Status)
			switch normalized {
			case string(types.ExecutionStatusSucceeded):
				success24h++
			case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
				failed24h++
			}

			if exec.DurationMS != nil {
				d := *exec.DurationMS
				durationSamples = append(durationSamples, d)
				durationSum += float64(d)
				durationCount++
			}
		}
	}

	overview.ExecutionsLast7d = len(executions)
	if overview.ExecutionsLast24h > 0 {
		overview.SuccessRate24h = (float64(success24h) / float64(overview.ExecutionsLast24h)) * 100
	}
	if durationCount > 0 {
		overview.AverageDurationMs24h = durationSum / durationCount
	}
	overview.MedianDurationMs24h = computeMedian(durationSamples)

	return overview
}

func buildExecutionTrends(now time.Time, executions []*types.Execution) ExecutionTrends {
	trend := ExecutionTrends{}
	last24h := now.Add(-24 * time.Hour)
	var total24h, success24h, failed24h int
	var durationSum float64
	var durationCount float64

	// Prepare day buckets for the last 7 days (including today)
	dayBuckets := make(map[string]*ExecutionTrendPoint)
	orderedDays := make([]string, 0, 7)
	for i := 6; i >= 0; i-- {
		day := now.AddDate(0, 0, -i)
		key := day.Format("2006-01-02")
		orderedDays = append(orderedDays, key)
		dayBuckets[key] = &ExecutionTrendPoint{Date: key}
	}

	for _, exec := range executions {
		dayKey := exec.StartedAt.Format("2006-01-02")
		point, ok := dayBuckets[dayKey]
		if ok {
			point.Total++
			normalized := types.NormalizeExecutionStatus(exec.Status)
			switch normalized {
			case string(types.ExecutionStatusSucceeded):
				point.Succeeded++
			case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
				point.Failed++
			}
		}

		if exec.StartedAt.After(last24h) || exec.StartedAt.Equal(last24h) {
			total24h++
			normalized := types.NormalizeExecutionStatus(exec.Status)
			switch normalized {
			case string(types.ExecutionStatusSucceeded):
				success24h++
			case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
				failed24h++
			}

			if exec.DurationMS != nil {
				durationSum += float64(*exec.DurationMS)
				durationCount++
			}
		}
	}

	trend.Last7Days = make([]ExecutionTrendPoint, 0, len(orderedDays))
	for _, key := range orderedDays {
		trend.Last7Days = append(trend.Last7Days, *dayBuckets[key])
	}

	trend.Last24h.Total = total24h
	trend.Last24h.Succeeded = success24h
	trend.Last24h.Failed = failed24h
	if total24h > 0 {
		trend.Last24h.SuccessRate = (float64(success24h) / float64(total24h)) * 100
		trend.Last24h.ThroughputPerHour = float64(total24h) / 24.0
	}
	if durationCount > 0 {
		trend.Last24h.AverageDurationMs = durationSum / durationCount
	}

	return trend
}

func (h *DashboardHandler) buildAgentHealthSummary(ctx context.Context, agents []*types.AgentNode) AgentHealthSummary {
	summary := AgentHealthSummary{Total: len(agents)}
	items := make([]AgentHealthItem, 0, len(agents))

	for _, agent := range agents {
		item := AgentHealthItem{
			ID:            agent.ID,
			TeamID:        agent.TeamID,
			Version:       agent.Version,
			Health:        string(agent.HealthStatus),
			Lifecycle:     string(agent.LifecycleStatus),
			LastHeartbeat: agent.LastHeartbeat,
			Reasoners:     len(agent.Reasoners),
			Skills:        len(agent.Skills),
		}

		isDegraded := agent.LifecycleStatus == types.AgentStatusDegraded || agent.HealthStatus == types.HealthStatusInactive
		if isDegraded {
			summary.Degraded++
			item.Status = "degraded"
			items = append(items, item)
			continue
		}

		status, err := h.agentService.GetAgentStatus(agent.ID)
		if err != nil {
			summary.Offline++
			item.Status = "offline"
			items = append(items, item)
			continue
		}

		if status != nil {
			item.Uptime = status.Uptime
			if status.IsRunning {
				summary.Active++
				item.Status = "running"
			} else {
				summary.Offline++
				item.Status = "offline"
			}
		} else {
			summary.Offline++
			item.Status = "offline"
		}

		items = append(items, item)
	}

	// Derive offline count if we encountered transient errors
	if summary.Offline < 0 {
		summary.Offline = 0
	}

	priority := map[string]int{
		"degraded": 0,
		"offline":  1,
		"running":  2,
		"unknown":  3,
	}

	sort.Slice(items, func(i, j int) bool {
		pi := priority[items[i].Status]
		pj := priority[items[j].Status]
		if pi == pj {
			return items[i].LastHeartbeat.After(items[j].LastHeartbeat)
		}
		return pi < pj
	})

	if len(items) > 12 {
		items = items[:12]
	}

	summary.Agents = items
	return summary
}

func buildWorkflowInsights(executions []*types.Execution, running []*types.Execution) WorkflowInsights {
	insights := WorkflowInsights{}
	workflowAggregates := make(map[string]*WorkflowStat)

	for _, exec := range executions {
		id := exec.RunID
		aggregate, ok := workflowAggregates[id]
		if !ok {
			aggregate = &WorkflowStat{
				WorkflowID: id,
				Name:       exec.ReasonerID,
			}
			workflowAggregates[id] = aggregate
		}

		aggregate.TotalExecutions++
		aggregate.LastActivity = maxTime(aggregate.LastActivity, exec.StartedAt)
		if exec.DurationMS != nil {
			aggregate.AverageDuration += float64(*exec.DurationMS)
		}

		normalized := types.NormalizeExecutionStatus(exec.Status)
		switch normalized {
		case string(types.ExecutionStatusSucceeded):
			aggregate.SuccessRate++
		case string(types.ExecutionStatusFailed), string(types.ExecutionStatusCancelled), string(types.ExecutionStatusTimeout):
			aggregate.FailedExecutions++
		}
	}

	topWorkflows := make([]WorkflowStat, 0, len(workflowAggregates))
	for _, aggregate := range workflowAggregates {
		if aggregate.TotalExecutions > 0 {
			aggregate.AverageDuration = aggregate.AverageDuration / float64(aggregate.TotalExecutions)
			aggregate.SuccessRate = (aggregate.SuccessRate / float64(aggregate.TotalExecutions)) * 100
		}
		topWorkflows = append(topWorkflows, *aggregate)
	}

	sort.Slice(topWorkflows, func(i, j int) bool {
		if topWorkflows[i].TotalExecutions == topWorkflows[j].TotalExecutions {
			return topWorkflows[i].LastActivity.After(topWorkflows[j].LastActivity)
		}
		return topWorkflows[i].TotalExecutions > topWorkflows[j].TotalExecutions
	})

	if len(topWorkflows) > 5 {
		topWorkflows = topWorkflows[:5]
	}

	insights.TopWorkflows = topWorkflows

	activeRuns := make([]ActiveWorkflowRun, 0, len(running))
	for _, exec := range running {
		elapsed := time.Since(exec.StartedAt).Milliseconds()
		activeRuns = append(activeRuns, ActiveWorkflowRun{
			ExecutionID: exec.ExecutionID,
			WorkflowID:  exec.RunID,
			Name:        exec.ReasonerID,
			StartedAt:   exec.StartedAt,
			ElapsedMs:   elapsed,
			AgentNodeID: exec.AgentNodeID,
			ReasonerID:  exec.ReasonerID,
			Status:      exec.Status,
		})
	}

	sort.Slice(activeRuns, func(i, j int) bool {
		return activeRuns[i].ElapsedMs > activeRuns[j].ElapsedMs
	})
	if len(activeRuns) > 6 {
		activeRuns = activeRuns[:6]
	}
	insights.ActiveRuns = activeRuns

	completed := make([]CompletedExecutionStat, 0, len(executions))
	for _, exec := range executions {
		if exec.DurationMS == nil || exec.CompletedAt == nil {
			continue
		}
		completed = append(completed, CompletedExecutionStat{
			ExecutionID: exec.ExecutionID,
			WorkflowID:  exec.RunID,
			Name:        exec.ReasonerID,
			DurationMs:  *exec.DurationMS,
			CompletedAt: exec.CompletedAt,
			Status:      exec.Status,
		})
	}

	sort.Slice(completed, func(i, j int) bool {
		if completed[i].DurationMs == completed[j].DurationMs {
			return completed[i].CompletedAt.After(*completed[j].CompletedAt)
		}
		return completed[i].DurationMs > completed[j].DurationMs
	})
	if len(completed) > 5 {
		completed = completed[:5]
	}

	insights.LongestExecutions = completed
	return insights
}

func buildIncidentItems(executions []*types.Execution, limit int) []IncidentItem {
	incidents := make([]IncidentItem, 0, limit)

	for _, exec := range executions {
		normalized := types.NormalizeExecutionStatus(exec.Status)
		if normalized != string(types.ExecutionStatusFailed) &&
			normalized != string(types.ExecutionStatusTimeout) &&
			normalized != string(types.ExecutionStatusCancelled) {
			continue
		}

		errorMessage := ""
		if exec.ErrorMessage != nil {
			errorMessage = *exec.ErrorMessage
		}

		incidents = append(incidents, IncidentItem{
			ExecutionID: exec.ExecutionID,
			WorkflowID:  exec.RunID,
			Name:        exec.ReasonerID,
			Status:      exec.Status,
			StartedAt:   exec.StartedAt,
			CompletedAt: exec.CompletedAt,
			AgentNodeID: exec.AgentNodeID,
			ReasonerID:  exec.ReasonerID,
			Error:       errorMessage,
		})
	}

	sort.Slice(incidents, func(i, j int) bool {
		return incidents[i].StartedAt.After(incidents[j].StartedAt)
	})

	if len(incidents) > limit {
		incidents = incidents[:limit]
	}

	return incidents
}

func computeMedian(values []int64) float64 {
	if len(values) == 0 {
		return 0
	}

	sort.Slice(values, func(i, j int) bool { return values[i] < values[j] })
	mid := len(values) / 2
	if len(values)%2 == 1 {
		return float64(values[mid])
	}
	return float64(values[mid-1]+values[mid]) / 2.0
}

func maxTime(current time.Time, candidate time.Time) time.Time {
	if current.IsZero() {
		return candidate
	}
	if candidate.After(current) {
		return candidate
	}
	return current
}

// getAgentsSummary collects agent statistics
func (h *DashboardHandler) getAgentsSummary(ctx context.Context) (AgentsSummary, error) {
	// Get all registered agents
	agents, err := h.storage.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		return AgentsSummary{}, err
	}

	total := len(agents)
	running := 0

	// Count running agents using the agent service
	for _, agent := range agents {
		if status, err := h.agentService.GetAgentStatus(agent.ID); err == nil && status.IsRunning {
			running++
		}
	}

	return AgentsSummary{
		Running: running,
		Total:   total,
	}, nil
}

// getExecutionsSummaryAndSuccessRate collects execution statistics and calculates success rate
func (h *DashboardHandler) getExecutionsSummaryAndSuccessRate(ctx context.Context, now time.Time) (ExecutionsSummary, float64, error) {
	today := time.Date(now.Year(), now.Month(), now.Day(), 0, 0, 0, 0, now.Location())
	yesterday := today.AddDate(0, 0, -1)
	tomorrow := today.AddDate(0, 0, 1)

	// Get today's executions
	todayFilters := types.ExecutionFilter{
		StartTime:      &today,
		EndTime:        &tomorrow,
		Limit:          10000,
		SortBy:         "started_at",
		SortDescending: false,
	}
	todayExecutions, err := h.store.QueryExecutionRecords(ctx, todayFilters)
	if err != nil {
		return ExecutionsSummary{}, 0, err
	}

	// Get yesterday's executions
	yesterdayFilters := types.ExecutionFilter{
		StartTime:      &yesterday,
		EndTime:        &today,
		Limit:          10000,
		SortBy:         "started_at",
		SortDescending: false,
	}
	yesterdayExecutions, err := h.store.QueryExecutionRecords(ctx, yesterdayFilters)
	if err != nil {
		return ExecutionsSummary{}, 0, err
	}

	// Calculate success rate from today's executions
	successRate := h.calculateSuccessRate(todayExecutions)

	return ExecutionsSummary{
		Today:     len(todayExecutions),
		Yesterday: len(yesterdayExecutions),
	}, successRate, nil
}

// calculateSuccessRate calculates the success rate from executions
func (h *DashboardHandler) calculateSuccessRate(executions []*types.Execution) float64 {
	if len(executions) == 0 {
		return 0.0
	}

	successCount := 0
	for _, exec := range executions {
		if types.NormalizeExecutionStatus(exec.Status) == types.ExecutionStatusSucceeded {
			successCount++
		}
	}

	return float64(successCount) / float64(len(executions)) * 100.0
}

// getPackagesSummary collects package statistics
func (h *DashboardHandler) getPackagesSummary(ctx context.Context) (PackagesSummary, error) {
	// Get all agent packages
	packages, err := h.storage.QueryAgentPackages(ctx, types.PackageFilters{})
	if err != nil {
		return PackagesSummary{}, err
	}

	available := len(packages)
	installed := 0

	// Count installed packages (packages with configuration or no configuration required)
	for _, pkg := range packages {
		configRequired := len(pkg.ConfigurationSchema) > 0

		if !configRequired {
			// No configuration required means it's installed
			installed++
		} else {
			// Check if configuration exists and is active
			if config, err := h.storage.GetAgentConfiguration(ctx, pkg.ID, pkg.ID); err == nil {
				if config.Status == types.ConfigurationStatusActive || config.Status == types.ConfigurationStatusDraft {
					installed++
				}
			}
		}
	}

	return PackagesSummary{
		Available: available,
		Installed: installed,
	}, nil
}
