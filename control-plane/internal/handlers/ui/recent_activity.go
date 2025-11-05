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

// RecentActivityHandler provides handlers for recent activity operations.
type RecentActivityHandler struct {
	storage storage.StorageProvider
	store   executionRecordStore
	cache   *RecentActivityCache
}

// NewRecentActivityHandler creates a new RecentActivityHandler.
func NewRecentActivityHandler(storage storage.StorageProvider) *RecentActivityHandler {
	return &RecentActivityHandler{
		storage: storage,
		store:   storage,
		cache:   NewRecentActivityCache(),
	}
}

// RecentActivityResponse represents the recent activity response
type RecentActivityResponse struct {
	Executions     []ActivityExecution `json:"executions"`
	CacheTimestamp time.Time           `json:"cache_timestamp"`
}

// ActivityExecution represents a recent execution with formatted data
type ActivityExecution struct {
	ExecutionID  string               `json:"execution_id"`
	AgentName    string               `json:"agent_name"`
	ReasonerName string               `json:"reasoner_name"`
	Status       string               `json:"status"`
	StartedAt    string               `json:"started_at"`
	DurationMs   *int64               `json:"duration_ms,omitempty"`
	RelativeTime string               `json:"relative_time"`
	NotesCount   int                  `json:"notes_count"`
	LatestNote   *types.ExecutionNote `json:"latest_note,omitempty"`
}

// RecentActivityCache provides 15-second caching for recent activity data
type RecentActivityCache struct {
	data      *RecentActivityResponse
	timestamp time.Time
	mutex     sync.RWMutex
	ttl       time.Duration
}

// NewRecentActivityCache creates a new recent activity cache with 15-second TTL
func NewRecentActivityCache() *RecentActivityCache {
	return &RecentActivityCache{
		ttl: 15 * time.Second,
	}
}

// Get retrieves cached data if still valid
func (c *RecentActivityCache) Get() (*RecentActivityResponse, bool) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	if c.data != nil && time.Since(c.timestamp) < c.ttl {
		return c.data, true
	}
	return nil, false
}

// Set stores data in cache with current timestamp
func (c *RecentActivityCache) Set(data *RecentActivityResponse) {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.data = data
	c.timestamp = time.Now()
}

// GetRecentActivityHandler handles recent activity requests
// GET /api/ui/v1/executions/recent
func (h *RecentActivityHandler) GetRecentActivityHandler(c *gin.Context) {
	ctx := c.Request.Context()

	// Check cache first
	if cachedData, found := h.cache.Get(); found {
		logger.Logger.Debug().Msg("Returning cached recent activity data")
		c.JSON(http.StatusOK, cachedData)
		return
	}

	logger.Logger.Debug().Msg("Generating fresh recent activity data")

	// Get recent executions with agent information
	recentExecutions, err := h.getRecentExecutions(ctx)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to get recent executions")
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to get recent executions"})
		return
	}

	// Build response
	response := &RecentActivityResponse{
		Executions:     recentExecutions,
		CacheTimestamp: time.Now(),
	}

	// Cache the response
	h.cache.Set(response)

	c.JSON(http.StatusOK, response)
}

// getRecentExecutions retrieves the last 20 executions with agent information
func (h *RecentActivityHandler) getRecentExecutions(ctx context.Context) ([]ActivityExecution, error) {
	// Query for the last 20 executions to support dashboard display
	filters := types.ExecutionFilter{
		Limit:          20,
		Offset:         0,
		SortBy:         "started_at",
		SortDescending: true,
	}

	executions, err := h.store.QueryExecutionRecords(ctx, filters)
	if err != nil {
		return nil, err
	}

	// Get all unique agent node IDs for batch lookup
	agentNodeIDs := make(map[string]bool)
	for _, exec := range executions {
		agentNodeIDs[exec.AgentNodeID] = true
	}

	// Batch fetch agent information
	agentMap := make(map[string]string) // agent_node_id -> agent_name
	for agentNodeID := range agentNodeIDs {
		_, err := h.storage.GetAgent(ctx, agentNodeID)
		if err != nil {
			// If agent not found, use the ID as fallback
			agentMap[agentNodeID] = agentNodeID
			continue
		}
		agentMap[agentNodeID] = agentNodeID // Use ID as name for now, can be enhanced later
	}

	// Convert to response format
	recentExecutions := make([]ActivityExecution, 0, len(executions))
	for _, exec := range executions {
		agentName := agentMap[exec.AgentNodeID]
		if agentName == "" {
			agentName = exec.AgentNodeID
		}

		// Format reasoner name (remove agent prefix if present)
		reasonerName := exec.ReasonerID
		if len(reasonerName) > len(agentName)+1 && reasonerName[:len(agentName)+1] == agentName+"." {
			reasonerName = reasonerName[len(agentName)+1:]
		}

		// Calculate relative time
		relativeTime := formatRelativeTime(exec.StartedAt)

		// Ensure notes is not nil
		recentExec := ActivityExecution{
			ExecutionID:  exec.ExecutionID,
			AgentName:    agentName,
			ReasonerName: reasonerName,
			Status:       types.NormalizeExecutionStatus(exec.Status),
			StartedAt:    exec.StartedAt.Format(time.RFC3339),
			RelativeTime: relativeTime,
			NotesCount:   0,
			LatestNote:   nil,
		}

		// Add duration if execution is completed
		if exec.DurationMS != nil {
			recentExec.DurationMs = exec.DurationMS
		}

		recentExecutions = append(recentExecutions, recentExec)
	}

	return recentExecutions, nil
}

// formatRelativeTime formats a timestamp as relative time (e.g., "2m ago")
func formatRelativeTime(timestamp time.Time) string {
	now := time.Now()
	diff := now.Sub(timestamp)

	if diff < time.Minute {
		seconds := int(diff.Seconds())
		if seconds <= 1 {
			return "just now"
		}
		return fmt.Sprintf("%ds ago", seconds)
	}

	if diff < time.Hour {
		minutes := int(diff.Minutes())
		return fmt.Sprintf("%dm ago", minutes)
	}

	if diff < 24*time.Hour {
		hours := int(diff.Hours())
		return fmt.Sprintf("%dh ago", hours)
	}

	days := int(diff.Hours() / 24)
	return fmt.Sprintf("%dd ago", days)
}
