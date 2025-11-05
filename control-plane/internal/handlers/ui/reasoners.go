package ui

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/your-org/agentfield/control-plane/internal/events"
	"github.com/your-org/agentfield/control-plane/internal/storage"
	"github.com/your-org/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// ReasonersHandler provides handlers for UI-related reasoner operations.
type ReasonersHandler struct {
	storage storage.StorageProvider
}

// NewReasonersHandler creates a new ReasonersHandler.
func NewReasonersHandler(storageProvider storage.StorageProvider) *ReasonersHandler {
	return &ReasonersHandler{storage: storageProvider}
}

// ReasonerWithNode represents a reasoner with its associated node information.
type ReasonerWithNode struct {
	// Reasoner identification
	ReasonerID  string `json:"reasoner_id"` // Format: "node_id.reasoner_id"
	Name        string `json:"name"`        // Human-readable name
	Description string `json:"description"` // Reasoner description

	// Node context
	NodeID      string             `json:"node_id"`
	NodeStatus  types.HealthStatus `json:"node_status"`
	NodeVersion string             `json:"node_version"`

	// Reasoner details
	InputSchema  interface{}        `json:"input_schema"`
	OutputSchema interface{}        `json:"output_schema"`
	MemoryConfig types.MemoryConfig `json:"memory_config"`

	// Performance metrics (placeholder for future implementation)
	AvgResponseTime *int       `json:"avg_response_time_ms,omitempty"`
	SuccessRate     *float64   `json:"success_rate,omitempty"`
	TotalRuns       *int       `json:"total_runs,omitempty"`
	LastExecuted    *time.Time `json:"last_executed,omitempty"`

	// Timestamps
	LastUpdated time.Time `json:"last_updated"`
}

// ReasonersResponse represents the response for the all reasoners endpoint.
type ReasonersResponse struct {
	Reasoners    []ReasonerWithNode `json:"reasoners"`
	Total        int                `json:"total"`
	OnlineCount  int                `json:"online_count"`
	OfflineCount int                `json:"offline_count"`
	NodesCount   int                `json:"nodes_count"`
}

// GetAllReasonersHandler handles requests for all reasoners across all nodes.
func (h *ReasonersHandler) GetAllReasonersHandler(c *gin.Context) {
	// Parse query parameters
	statusFilter := c.Query("status") // "online", "offline", "all" (default: "all")
	searchTerm := c.Query("search")   // Search in reasoner names/descriptions
	limitStr := c.Query("limit")      // Pagination limit
	offsetStr := c.Query("offset")    // Pagination offset

	// Set defaults
	if statusFilter == "" {
		statusFilter = "all"
	}

	limit := 50 // Default limit
	if limitStr != "" {
		if parsedLimit, err := strconv.Atoi(limitStr); err == nil && parsedLimit > 0 {
			limit = parsedLimit
		}
	}

	offset := 0 // Default offset
	if offsetStr != "" {
		if parsedOffset, err := strconv.Atoi(offsetStr); err == nil && parsedOffset >= 0 {
			offset = parsedOffset
		}
	}

	// Get all nodes based on status filter
	var filters types.AgentFilters
	if statusFilter == "online" {
		activeStatus := types.HealthStatusActive
		filters.HealthStatus = &activeStatus
	}

	ctx := c.Request.Context()
	nodes, err := h.storage.ListAgents(ctx, filters)
	if err != nil {
		fmt.Printf("❌ Error listing agents for reasoners: %v\n", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to retrieve nodes"})
		return
	}

	fmt.Printf("📊 Found %d nodes for reasoner aggregation\n", len(nodes))

	// Aggregate reasoners from all nodes
	var allReasoners []ReasonerWithNode
	onlineCount := 0
	offlineCount := 0

	// Ensure we always have a valid slice
	if allReasoners == nil {
		allReasoners = make([]ReasonerWithNode, 0)
	}

	for _, node := range nodes {
		fmt.Printf("  Processing node %s with %d reasoners (status: %s)\n",
			node.ID, len(node.Reasoners), node.HealthStatus)

		for _, reasoner := range node.Reasoners {
			// Create full reasoner ID
			fullReasonerID := fmt.Sprintf("%s.%s", node.ID, reasoner.ID)

			// Extract name from reasoner ID (use ID as name for now)
			name := reasoner.ID
			description := fmt.Sprintf("Reasoner %s from node %s", reasoner.ID, node.ID)

			// DIAGNOSTIC LOG: Track reasoner status determination
			fmt.Printf("🔍 REASONER_STATUS_DEBUG: Reasoner %s - NodeHealth: %s, NodeLifecycle: %s, LastHeartbeat: %s\n",
				fullReasonerID, node.HealthStatus, node.LifecycleStatus, node.LastHeartbeat.Format(time.RFC3339))

			reasonerWithNode := ReasonerWithNode{
				ReasonerID:   fullReasonerID,
				Name:         name,
				Description:  description,
				NodeID:       node.ID,
				NodeStatus:   node.HealthStatus,
				NodeVersion:  node.Version,
				InputSchema:  reasoner.InputSchema,
				OutputSchema: reasoner.OutputSchema,
				MemoryConfig: reasoner.MemoryConfig,
				LastUpdated:  node.LastHeartbeat,
			}

			// Apply search filter
			if searchTerm != "" {
				searchLower := strings.ToLower(searchTerm)
				if !strings.Contains(strings.ToLower(name), searchLower) &&
					!strings.Contains(strings.ToLower(description), searchLower) &&
					!strings.Contains(strings.ToLower(reasoner.ID), searchLower) {
					continue
				}
			}

			// Count by status
			if node.HealthStatus == types.HealthStatusActive {
				onlineCount++
			} else {
				offlineCount++
			}

			allReasoners = append(allReasoners, reasonerWithNode)
		}
	}

	// Apply status filter after aggregation (for accurate counts)
	var filteredReasoners []ReasonerWithNode
	if statusFilter == "online" {
		for _, reasoner := range allReasoners {
			if reasoner.NodeStatus == types.HealthStatusActive {
				filteredReasoners = append(filteredReasoners, reasoner)
			}
		}
	} else if statusFilter == "offline" {
		for _, reasoner := range allReasoners {
			if reasoner.NodeStatus != types.HealthStatusActive {
				filteredReasoners = append(filteredReasoners, reasoner)
			}
		}
	} else {
		filteredReasoners = allReasoners
	}

	// Apply pagination
	total := len(filteredReasoners)
	start := offset
	end := offset + limit

	if start > total {
		start = total
	}
	if end > total {
		end = total
	}

	paginatedReasoners := filteredReasoners[start:end]

	fmt.Printf("📋 Returning %d reasoners (total: %d, online: %d, offline: %d) from %d nodes\n",
		len(paginatedReasoners), total, onlineCount, offlineCount, len(nodes))

	response := ReasonersResponse{
		Reasoners:    paginatedReasoners,
		Total:        total,
		OnlineCount:  onlineCount,
		OfflineCount: offlineCount,
		NodesCount:   len(nodes),
	}

	c.JSON(http.StatusOK, response)
}

// GetReasonerDetailsHandler handles requests for detailed information about a specific reasoner.
func (h *ReasonersHandler) GetReasonerDetailsHandler(c *gin.Context) {
	reasonerID := c.Param("reasonerId")
	if reasonerID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "reasoner_id is required"})
		return
	}

	// Parse reasoner ID (format: "node_id.reasoner_id")
	parts := strings.SplitN(reasonerID, ".", 2)
	if len(parts) != 2 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid reasoner_id format, expected 'node_id.reasoner_id'"})
		return
	}

	nodeID := parts[0]
	localReasonerID := parts[1]

	// Get the node
	ctx := c.Request.Context()
	node, err := h.storage.GetAgent(ctx, nodeID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "node not found"})
		return
	}

	// Find the reasoner
	var foundReasoner *types.ReasonerDefinition
	for _, reasoner := range node.Reasoners {
		if reasoner.ID == localReasonerID {
			foundReasoner = &reasoner
			break
		}
	}

	if foundReasoner == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "reasoner not found"})
		return
	}

	// Create detailed response
	reasonerDetails := ReasonerWithNode{
		ReasonerID:   reasonerID,
		Name:         foundReasoner.ID,
		Description:  fmt.Sprintf("Reasoner %s from node %s", foundReasoner.ID, nodeID),
		NodeID:       nodeID,
		NodeStatus:   node.HealthStatus,
		NodeVersion:  node.Version,
		InputSchema:  foundReasoner.InputSchema,
		OutputSchema: foundReasoner.OutputSchema,
		MemoryConfig: foundReasoner.MemoryConfig,
		LastUpdated:  node.LastHeartbeat,
	}

	fmt.Printf("📋 Retrieved details for reasoner %s\n", reasonerID)

	c.JSON(http.StatusOK, reasonerDetails)
}

// PerformanceMetrics represents performance data for a reasoner
type PerformanceMetrics struct {
	AvgResponseTimeMs int               `json:"avg_response_time_ms"`
	SuccessRate       float64           `json:"success_rate"`
	TotalExecutions   int               `json:"total_executions"`
	ExecutionsLast24h int               `json:"executions_last_24h"`
	RecentExecutions  []RecentExecution `json:"recent_executions"`
}

// RecentExecution represents a recent execution for metrics
type RecentExecution struct {
	ExecutionID string    `json:"execution_id"`
	Status      string    `json:"status"`
	DurationMs  int       `json:"duration_ms"`
	Timestamp   time.Time `json:"timestamp"`
}

// ExecutionHistory represents paginated execution history
type ExecutionHistory struct {
	Executions []ExecutionRecord `json:"executions"`
	Total      int               `json:"total"`
	Page       int               `json:"page"`
	Limit      int               `json:"limit"`
	HasMore    bool              `json:"has_more"`
}

// ExecutionRecord represents a single execution record
type ExecutionRecord struct {
	ExecutionID string                 `json:"execution_id"`
	Status      string                 `json:"status"`
	Input       map[string]interface{} `json:"input"`
	Output      map[string]interface{} `json:"output,omitempty"`
	Error       string                 `json:"error,omitempty"`
	DurationMs  int                    `json:"duration_ms"`
	Timestamp   time.Time              `json:"timestamp"`
}

// ExecutionTemplate represents a saved execution template
type ExecutionTemplate struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Input       map[string]interface{} `json:"input"`
	CreatedAt   time.Time              `json:"created_at"`
}

// GetPerformanceMetricsHandler handles requests for reasoner performance metrics
func (h *ReasonersHandler) GetPerformanceMetricsHandler(c *gin.Context) {
	reasonerID := c.Param("reasonerId")
	if reasonerID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "reasoner_id is required"})
		return
	}

	// Parse reasoner ID (format: "node_id.reasoner_id")
	parts := strings.SplitN(reasonerID, ".", 2)
	if len(parts) != 2 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid reasoner_id format, expected 'node_id.reasoner_id'"})
		return
	}

	// Get real performance metrics from storage
	ctx := c.Request.Context()
	metrics, err := h.storage.GetReasonerPerformanceMetrics(ctx, reasonerID)
	if err != nil {
		fmt.Printf("❌ Error getting performance metrics for reasoner %s: %v\n", reasonerID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to retrieve performance metrics"})
		return
	}

	fmt.Printf("📊 Retrieved performance metrics for reasoner %s: %d executions, %.2f%% success rate\n",
		reasonerID, metrics.TotalExecutions, metrics.SuccessRate*100)

	c.JSON(http.StatusOK, metrics)
}

// GetExecutionHistoryHandler handles requests for reasoner execution history
func (h *ReasonersHandler) GetExecutionHistoryHandler(c *gin.Context) {
	reasonerID := c.Param("reasonerId")
	if reasonerID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "reasoner_id is required"})
		return
	}

	// Parse pagination parameters
	pageStr := c.DefaultQuery("page", "1")
	limitStr := c.DefaultQuery("limit", "20")

	page, err := strconv.Atoi(pageStr)
	if err != nil || page < 1 {
		page = 1
	}

	limit, err := strconv.Atoi(limitStr)
	if err != nil || limit < 1 || limit > 100 {
		limit = 20
	}

	// Parse reasoner ID (format: "node_id.reasoner_id")
	parts := strings.SplitN(reasonerID, ".", 2)
	if len(parts) != 2 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid reasoner_id format, expected 'node_id.reasoner_id'"})
		return
	}

	// Get real execution history from storage
	ctx := c.Request.Context()
	history, err := h.storage.GetReasonerExecutionHistory(ctx, reasonerID, page, limit)
	if err != nil {
		fmt.Printf("❌ Error getting execution history for reasoner %s: %v\n", reasonerID, err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to retrieve execution history"})
		return
	}

	fmt.Printf("📋 Retrieved execution history for reasoner %s: %d executions (page %d, limit %d)\n",
		reasonerID, len(history.Executions), page, limit)

	c.JSON(http.StatusOK, history)
}

// GetExecutionTemplatesHandler handles requests for reasoner execution templates
func (h *ReasonersHandler) GetExecutionTemplatesHandler(c *gin.Context) {
	reasonerID := c.Param("reasonerId")
	if reasonerID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "reasoner_id is required"})
		return
	}

	// Parse reasoner ID (format: "node_id.reasoner_id")
	parts := strings.SplitN(reasonerID, ".", 2)
	if len(parts) != 2 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid reasoner_id format, expected 'node_id.reasoner_id'"})
		return
	}

	// For now, return mock data since we don't have template storage yet
	// TODO: Implement actual template storage and retrieval
	templates := []ExecutionTemplate{
		{
			ID:          "template_001",
			Name:        "NVDA Analysis",
			Description: "Standard NVIDIA stock analysis",
			Input:       map[string]interface{}{"ticker": "NVDA"},
			CreatedAt:   time.Now().Add(-24 * time.Hour),
		},
		{
			ID:          "template_002",
			Name:        "Tech Stock Analysis",
			Description: "General tech stock analysis template",
			Input:       map[string]interface{}{"ticker": "AAPL", "sector": "technology"},
			CreatedAt:   time.Now().Add(-48 * time.Hour),
		},
	}

	c.JSON(http.StatusOK, templates)
}

// SaveExecutionTemplateHandler handles saving new execution templates
func (h *ReasonersHandler) SaveExecutionTemplateHandler(c *gin.Context) {
	reasonerID := c.Param("reasonerId")
	if reasonerID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "reasoner_id is required"})
		return
	}

	// Parse reasoner ID (format: "node_id.reasoner_id")
	parts := strings.SplitN(reasonerID, ".", 2)
	if len(parts) != 2 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid reasoner_id format, expected 'node_id.reasoner_id'"})
		return
	}

	var template struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		Input       map[string]interface{} `json:"input"`
	}

	if err := c.ShouldBindJSON(&template); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	// For now, return mock response since we don't have template storage yet
	// TODO: Implement actual template storage
	savedTemplate := ExecutionTemplate{
		ID:          fmt.Sprintf("template_%d", time.Now().Unix()),
		Name:        template.Name,
		Description: template.Description,
		Input:       template.Input,
		CreatedAt:   time.Now(),
	}

	c.JSON(http.StatusCreated, savedTemplate)
}

// StreamReasonerEventsHandler handles reasoner event streaming
// GET /api/ui/v1/reasoners/events
func (h *ReasonersHandler) StreamReasonerEventsHandler(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("Access-Control-Allow-Headers", "Cache-Control")

	// Generate unique subscriber ID
	subscriberID := fmt.Sprintf("reasoner_sse_%d_%s", time.Now().UnixNano(), c.ClientIP())

	// Subscribe to reasoner events
	eventChan := events.GlobalReasonerEventBus.Subscribe(subscriberID)
	defer events.GlobalReasonerEventBus.Unsubscribe(subscriberID)

	// Send initial connection confirmation
	initialEvent := map[string]interface{}{
		"type":      "connected",
		"message":   "Reasoner events stream connected",
		"timestamp": time.Now().Format(time.RFC3339),
	}

	if eventJSON, err := json.Marshal(initialEvent); err == nil {
		if !writeSSE(c, eventJSON) {
			return
		}
	}

	// Set up context for handling client disconnection
	ctx := c.Request.Context()

	// Send periodic heartbeat to keep connection alive
	heartbeatTicker := time.NewTicker(30 * time.Second)
	defer heartbeatTicker.Stop()

	fmt.Printf("🔄 Reasoner SSE client connected: %s\n", subscriberID)

	for {
		select {
		case <-ctx.Done():
			// Client disconnected
			fmt.Printf("🔌 Reasoner SSE client disconnected: %s\n", subscriberID)
			return
		case <-heartbeatTicker.C:
			// Send heartbeat
			heartbeat := map[string]interface{}{
				"type":      "heartbeat",
				"timestamp": time.Now().Format(time.RFC3339),
			}
			if heartbeatJSON, err := json.Marshal(heartbeat); err == nil {
				if !writeSSE(c, heartbeatJSON) {
					return
				}
			}
		case event, ok := <-eventChan:
			if !ok {
				// Channel closed
				fmt.Printf("📡 Reasoner SSE channel closed for: %s\n", subscriberID)
				return
			}

			// Convert event to JSON and send
			if eventJSON, err := event.ToJSON(); err == nil {
				if !writeSSE(c, []byte(eventJSON)) {
					return
				}
				fmt.Printf("📤 Sent reasoner event %s to client %s\n", event.Type, subscriberID)
			}
		}
	}
}
