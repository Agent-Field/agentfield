package ui

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/your-org/haxen/control-plane/internal/events"
	"github.com/your-org/haxen/control-plane/internal/logger"
	"github.com/your-org/haxen/control-plane/internal/services"

	"github.com/gin-gonic/gin"
)

// NodesHandler provides handlers for UI-related node operations.
type NodesHandler struct {
	service *services.UIService
}

// NewNodesHandler creates a new NodesHandler.
func NewNodesHandler(uiService *services.UIService) *NodesHandler {
	return &NodesHandler{service: uiService}
}

// GetNodesSummaryHandler handles requests for a summary list of nodes.
func (h *NodesHandler) GetNodesSummaryHandler(c *gin.Context) {
	ctx := c.Request.Context()
	summaries, count, err := h.service.GetNodesSummary(ctx)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get nodes summary"})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"nodes": summaries,
		"count": count,
	})
}

// GetNodeDetailsHandler handles requests for detailed information about a specific node.
func (h *NodesHandler) GetNodeDetailsHandler(c *gin.Context) {
	nodeID := c.Param("nodeId")
	if nodeID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "nodeId is required"})
		return
	}

	ctx := c.Request.Context()
	details, err := h.service.GetNodeDetailsWithPackageInfo(ctx, nodeID)
	if err != nil {
		// TODO: Differentiate between not found and other errors
		c.JSON(http.StatusNotFound, gin.H{"error": "node not found or failed to retrieve details"})
		return
	}
	c.JSON(http.StatusOK, details)
}

// StreamNodeEventsHandler handles SSE connections for real-time node events.
func (h *NodesHandler) StreamNodeEventsHandler(c *gin.Context) {
	// Set headers for SSE
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("Access-Control-Allow-Headers", "Cache-Control")
	c.Header("X-Accel-Buffering", "no") // Disable buffering for Nginx

	// Generate unique subscriber ID
	subscriberID := fmt.Sprintf("node_sse_%d_%s", time.Now().UnixNano(), c.ClientIP())
	
	// Subscribe to node events using the dedicated event bus
	eventChan := events.GlobalNodeEventBus.Subscribe(subscriberID)
	defer events.GlobalNodeEventBus.Unsubscribe(subscriberID)

	// Send initial connection confirmation
	initialEvent := map[string]interface{}{
		"type":      "connected",
		"message":   "Node events stream connected",
		"timestamp": time.Now().Format(time.RFC3339),
	}
	
	if eventJSON, err := json.Marshal(initialEvent); err == nil {
		c.Writer.WriteString("data: " + string(eventJSON) + "\n\n")
		c.Writer.Flush()
	}

	// Set up context for handling client disconnection
	ctx := c.Request.Context()
	
	// Send periodic heartbeat to keep connection alive
	heartbeatTicker := time.NewTicker(30 * time.Second)
	defer heartbeatTicker.Stop()

	logger.Logger.Debug().Msgf("🔄 Node SSE client connected: %s", subscriberID)

	// Keep the connection open
	for {
		select {
		case event := <-eventChan:
			// Marshal event to JSON
			eventData, err := json.Marshal(event)
			if err != nil {
				logger.Logger.Error().Err(err).Msg("❌ Error marshalling node event")
				continue
			}
			
			// Send event to client using SSE format
			c.Writer.WriteString("data: " + string(eventData) + "\n\n")
			c.Writer.Flush()
			
			logger.Logger.Debug().Msgf("📡 Sent node event to client %s: %s", subscriberID, event.Type)
			
		case <-heartbeatTicker.C:
			// Send heartbeat to keep connection alive
			heartbeatEvent := map[string]interface{}{
				"type":      "heartbeat",
				"timestamp": time.Now().Format(time.RFC3339),
			}
			if heartbeatJSON, err := json.Marshal(heartbeatEvent); err == nil {
				c.Writer.WriteString("data: " + string(heartbeatJSON) + "\n\n")
				c.Writer.Flush()
			}
			
		case <-ctx.Done():
			// Client disconnected
			logger.Logger.Debug().Msgf("🔌 Node SSE client disconnected: %s", subscriberID)
			return
		}
	}
}

// GetNodeStatusHandler handles requests for getting a specific node's unified status
// GET /api/ui/v1/nodes/:nodeId/status
func (h *NodesHandler) GetNodeStatusHandler(c *gin.Context) {
	nodeID := c.Param("nodeId")
	if nodeID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "nodeId is required"})
		return
	}

	ctx := c.Request.Context()
	status, err := h.service.GetNodeUnifiedStatus(ctx, nodeID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get node status"})
		return
	}

	c.JSON(http.StatusOK, status)
}

// RefreshNodeStatusHandler handles requests for refreshing a specific node's status
// POST /api/ui/v1/nodes/:nodeId/status/refresh
func (h *NodesHandler) RefreshNodeStatusHandler(c *gin.Context) {
	nodeID := c.Param("nodeId")
	if nodeID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "nodeId is required"})
		return
	}

	ctx := c.Request.Context()
	err := h.service.RefreshNodeStatus(ctx, nodeID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to refresh node status"})
		return
	}

	// Get the refreshed status
	status, err := h.service.GetNodeUnifiedStatus(ctx, nodeID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get refreshed node status"})
		return
	}

	c.JSON(http.StatusOK, status)
}

// BulkNodeStatusHandler handles requests for bulk status operations
// POST /api/ui/v1/nodes/status/bulk
func (h *NodesHandler) BulkNodeStatusHandler(c *gin.Context) {
	var request struct {
		NodeIDs []string `json:"node_ids" binding:"required"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	ctx := c.Request.Context()
	statuses, err := h.service.BulkNodeStatus(ctx, request.NodeIDs)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get bulk node status"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"statuses": statuses})
}

// RefreshAllNodeStatusHandler handles requests for refreshing all node statuses
// POST /api/ui/v1/nodes/status/refresh
func (h *NodesHandler) RefreshAllNodeStatusHandler(c *gin.Context) {
	ctx := c.Request.Context()
	statuses, err := h.service.RefreshAllNodeStatus(ctx)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to refresh all node statuses"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"statuses": statuses})
}
