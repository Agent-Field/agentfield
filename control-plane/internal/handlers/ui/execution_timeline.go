package ui

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/your-org/agentfield/control-plane/internal/logger"
	"github.com/your-org/agentfield/control-plane/internal/storage"
	"github.com/your-org/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// ExecutionTimelineHandler provides handlers for execution timeline operations.
type ExecutionTimelineHandler struct {
	storage storage.StorageProvider
	store   executionRecordStore
	cache   *ExecutionTimelineCache
}

// NewExecutionTimelineHandler creates a new ExecutionTimelineHandler.
func NewExecutionTimelineHandler(storage storage.StorageProvider) *ExecutionTimelineHandler {
	return &ExecutionTimelineHandler{
		storage: storage,
		store:   storage,
		cache:   NewExecutionTimelineCache(),
	}
}

// ExecutionTimelineResponse represents the execution timeline response
type ExecutionTimelineResponse struct {
	TimelineData   []TimelineDataPoint `json:"timeline_data"`
	CacheTimestamp string              `json:"cache_timestamp"`
	Summary        TimelineSummary     `json:"summary"`
}

// TimelineDataPoint represents a single hour's execution data
type TimelineDataPoint struct {
	Timestamp           string  `json:"timestamp"`             // ISO timestamp for the hour
	Hour                string  `json:"hour"`                  // "14:00" format for display
	Executions          int     `json:"executions"`            // Total executions in this hour
	Successful          int     `json:"successful"`            // Successful executions
	Failed              int     `json:"failed"`                // Failed executions
	Running             int     `json:"running"`               // Currently running executions
	SuccessRate         float64 `json:"success_rate"`          // Percentage (0-100)
	AvgDurationMS       int64   `json:"avg_duration_ms"`       // Average execution duration
	TotalDurationMS     int64   `json:"total_duration_ms"`     // Total duration for all executions
	TotalNotes          int     `json:"total_notes"`           // Total notes across all executions in this hour
	ExecutionsWithNotes int     `json:"executions_with_notes"` // Number of executions that have notes
}

// TimelineSummary represents summary statistics for the timeline
type TimelineSummary struct {
	TotalExecutions int     `json:"total_executions"`
	AvgSuccessRate  float64 `json:"avg_success_rate"`
	TotalErrors     int     `json:"total_errors"`
	PeakHour        string  `json:"peak_hour"`
	PeakExecutions  int     `json:"peak_executions"`
}

// ExecutionTimelineCache provides 5-minute caching for timeline data
type ExecutionTimelineCache struct {
	data      *ExecutionTimelineResponse
	timestamp time.Time
	mutex     sync.RWMutex
	ttl       time.Duration
}

// NewExecutionTimelineCache creates a new execution timeline cache with 5-minute TTL
func NewExecutionTimelineCache() *ExecutionTimelineCache {
	return &ExecutionTimelineCache{
		ttl: 5 * time.Minute, // 5 minutes as specified
	}
}

// Get retrieves cached data if still valid
func (c *ExecutionTimelineCache) Get() (*ExecutionTimelineResponse, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if c.data != nil && time.Since(c.timestamp) < c.ttl {
		return c.data, true
	}
	return nil, false
}

// Set stores data in cache with current timestamp
func (c *ExecutionTimelineCache) Set(data *ExecutionTimelineResponse) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.data = data
	c.timestamp = time.Now()
}

// GetExecutionTimelineHandler handles execution timeline requests
// GET /api/ui/v1/executions/timeline
func (h *ExecutionTimelineHandler) GetExecutionTimelineHandler(c *gin.Context) {
	ctx := c.Request.Context()

	// Check cache first
	if cachedData, found := h.cache.Get(); found {
		logger.Logger.Debug().Msg("Returning cached execution timeline data")
		c.JSON(http.StatusOK, cachedData)
		return
	}

	logger.Logger.Debug().Msg("Generating fresh execution timeline data")

	// Generate timeline data
	timelineData, summary, err := h.generateTimelineData(ctx)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to generate timeline data")
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to generate timeline data"})
		return
	}

	// Build response
	response := &ExecutionTimelineResponse{
		TimelineData:   timelineData,
		CacheTimestamp: time.Now().Format(time.RFC3339),
		Summary:        summary,
	}

	// Cache the response
	h.cache.Set(response)

	c.JSON(http.StatusOK, response)
}

// generateTimelineData creates hourly aggregated execution data for the last 24 hours
func (h *ExecutionTimelineHandler) generateTimelineData(ctx context.Context) ([]TimelineDataPoint, TimelineSummary, error) {
	now := time.Now()

	// Calculate 24 hours ago, rounded to the hour
	startTime := now.Add(-24 * time.Hour).Truncate(time.Hour)
	endTime := now.Truncate(time.Hour).Add(time.Hour) // Include current hour

	// Get all executions for the last 24 hours
	filters := types.ExecutionFilter{
		StartTime:      &startTime,
		EndTime:        &endTime,
		Limit:          50000,
		SortBy:         "started_at",
		SortDescending: false,
	}

	executions, err := h.store.QueryExecutionRecords(ctx, filters)
	if err != nil {
		return nil, TimelineSummary{}, fmt.Errorf("failed to query executions: %w", err)
	}

	// Create 24 hourly buckets
	timelineData := make([]TimelineDataPoint, 24)
	hourlyData := make(map[string]*TimelineDataPoint)

	// Initialize all 24 hours
	for i := 0; i < 24; i++ {
		hourTime := startTime.Add(time.Duration(i) * time.Hour)
		hourKey := hourTime.Format("2006-01-02T15")

		dataPoint := &TimelineDataPoint{
			Timestamp:           hourTime.Format(time.RFC3339),
			Hour:                hourTime.Format("15:04"),
			Executions:          0,
			Successful:          0,
			Failed:              0,
			Running:             0,
			SuccessRate:         0.0,
			AvgDurationMS:       0,
			TotalDurationMS:     0,
			TotalNotes:          0,
			ExecutionsWithNotes: 0,
		}

		timelineData[i] = *dataPoint
		hourlyData[hourKey] = &timelineData[i]
	}

	// Aggregate executions into hourly buckets
	for _, exec := range executions {
		hourKey := exec.StartedAt.Format("2006-01-02T15")

		if dataPoint, exists := hourlyData[hourKey]; exists {
			dataPoint.Executions++

			// Categorize by status
			switch types.NormalizeExecutionStatus(exec.Status) {
			case string(types.ExecutionStatusSucceeded):
				dataPoint.Successful++
			case string(types.ExecutionStatusFailed):
				dataPoint.Failed++
			case string(types.ExecutionStatusRunning), string(types.ExecutionStatusPending), string(types.ExecutionStatusQueued):
				dataPoint.Running++
			}

			// Add duration if available
			if exec.DurationMS != nil {
				dataPoint.TotalDurationMS += *exec.DurationMS
			}

			// Notes are no longer tracked in the simplified execution model
		}
	}

	// Calculate success rates and averages
	totalExecutions := 0
	totalErrors := 0
	peakExecutions := 0
	peakHour := ""
	totalSuccessRate := 0.0
	hoursWithData := 0

	for i := range timelineData {
		dataPoint := &timelineData[i]

		if dataPoint.Executions > 0 {
			// Calculate success rate
			dataPoint.SuccessRate = float64(dataPoint.Successful) / float64(dataPoint.Executions) * 100.0

			// Calculate average duration
			if dataPoint.TotalDurationMS > 0 {
				dataPoint.AvgDurationMS = dataPoint.TotalDurationMS / int64(dataPoint.Executions)
			}

			// Track summary statistics
			totalExecutions += dataPoint.Executions
			totalErrors += dataPoint.Failed
			totalSuccessRate += dataPoint.SuccessRate
			hoursWithData++

			// Track peak hour
			if dataPoint.Executions > peakExecutions {
				peakExecutions = dataPoint.Executions
				peakHour = dataPoint.Hour
			}
		}
	}

	// Calculate average success rate
	avgSuccessRate := 0.0
	if hoursWithData > 0 {
		avgSuccessRate = totalSuccessRate / float64(hoursWithData)
	}

	summary := TimelineSummary{
		TotalExecutions: totalExecutions,
		AvgSuccessRate:  avgSuccessRate,
		TotalErrors:     totalErrors,
		PeakHour:        peakHour,
		PeakExecutions:  peakExecutions,
	}

	return timelineData, summary, nil
}
