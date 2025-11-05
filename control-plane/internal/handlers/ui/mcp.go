package ui

import (
	"net/http"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/core/domain"
	"github.com/Agent-Field/agentfield/control-plane/internal/core/interfaces"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"

	"github.com/gin-gonic/gin"
)

// MCPHandler provides handlers for MCP-related operations
type MCPHandler struct {
	uiService   *services.UIService
	agentClient interfaces.AgentClient
}

// NewMCPHandler creates a new MCPHandler
func NewMCPHandler(uiService *services.UIService, agentClient interfaces.AgentClient) *MCPHandler {
	return &MCPHandler{
		uiService:   uiService,
		agentClient: agentClient,
	}
}

// GetMCPHealthHandler handles requests for detailed MCP health information
// GET /api/ui/v1/nodes/{nodeId}/mcp/health?mode=developer|user
func (h *MCPHandler) GetMCPHealthHandler(c *gin.Context) {
	ctx := c.Request.Context()
	nodeID := c.Param("nodeId")
	if nodeID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "nodeId is required"})
		return
	}

	// Get mode parameter (default to developer)
	modeParam := c.DefaultQuery("mode", "developer")
	var mode domain.MCPHealthMode
	switch modeParam {
	case "user":
		mode = domain.MCPHealthModeUser
	case "developer":
		mode = domain.MCPHealthModeDeveloper
	default:
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "mode must be 'user' or 'developer'"})
		return
	}

	// Get detailed node information with MCP data
	nodeDetails, err := h.uiService.GetNodeDetailsWithMCP(ctx, nodeID, mode)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "node not found or failed to retrieve MCP health"})
		return
	}

	// Return the appropriate response based on mode
	if mode == domain.MCPHealthModeUser {
		// User mode: return only summary
		response := map[string]interface{}{
			"node_id":     nodeID,
			"mcp_summary": nodeDetails.MCPSummary,
			"timestamp":   time.Now(),
		}
		c.JSON(http.StatusOK, response)
	} else {
		// Developer mode: return full details
		response := map[string]interface{}{
			"node_id":     nodeID,
			"mcp_summary": nodeDetails.MCPSummary,
			"mcp_servers": nodeDetails.MCPServers,
			"timestamp":   time.Now(),
		}
		c.JSON(http.StatusOK, response)
	}
}

// RestartMCPServerHandler handles requests to restart a specific MCP server
// POST /api/ui/v1/nodes/{nodeId}/mcp/servers/{alias}/restart
func (h *MCPHandler) RestartMCPServerHandler(c *gin.Context) {
	nodeID := c.Param("nodeId")
	alias := c.Param("alias")

	if nodeID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "nodeId is required"})
		return
	}

	if alias == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "alias is required"})
		return
	}

	// Check mode - only allow in developer mode
	mode := c.DefaultQuery("mode", "developer")
	if mode != "developer" {
		c.JSON(http.StatusForbidden, ErrorResponse{Error: "MCP server restart is only available in developer mode"})
		return
	}

	// Create context with timeout
	ctx := c.Request.Context()

	// Call agent client to restart the MCP server
	err := h.agentClient.RestartMCPServer(ctx, nodeID, alias)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to restart MCP server: " + err.Error()})
		return
	}

	// Return success response
	response := map[string]interface{}{
		"success":   true,
		"message":   "MCP server restart initiated successfully",
		"node_id":   nodeID,
		"alias":     alias,
		"timestamp": time.Now(),
	}

	c.JSON(http.StatusOK, response)
}

// GetMCPToolsHandler handles requests to get tools from a specific MCP server
// GET /api/ui/v1/nodes/{nodeId}/mcp/servers/{alias}/tools
func (h *MCPHandler) GetMCPToolsHandler(c *gin.Context) {
	nodeID := c.Param("nodeId")
	alias := c.Param("alias")

	if nodeID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "nodeId is required"})
		return
	}

	if alias == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "alias is required"})
		return
	}

	// Check mode - only allow in developer mode
	mode := c.DefaultQuery("mode", "developer")
	if mode != "developer" {
		c.JSON(http.StatusForbidden, ErrorResponse{Error: "MCP tools listing is only available in developer mode"})
		return
	}

	// Create context with timeout
	ctx := c.Request.Context()

	// Call agent client to get MCP tools
	toolsResponse, err := h.agentClient.GetMCPTools(ctx, nodeID, alias)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to get MCP tools: " + err.Error()})
		return
	}

	// Return tools response
	response := map[string]interface{}{
		"node_id":   nodeID,
		"alias":     alias,
		"tools":     toolsResponse.Tools,
		"count":     len(toolsResponse.Tools),
		"timestamp": time.Now(),
	}

	c.JSON(http.StatusOK, response)
}

// GetMCPStatusHandler handles requests for overall MCP status across all nodes
// GET /api/ui/v1/mcp/status
func (h *MCPHandler) GetMCPStatusHandler(c *gin.Context) {
	ctx := c.Request.Context()
	// Get mode parameter
	modeParam := c.DefaultQuery("mode", "user")
	var mode domain.MCPHealthMode
	switch modeParam {
	case "user":
		mode = domain.MCPHealthModeUser
	case "developer":
		mode = domain.MCPHealthModeDeveloper
	default:
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "mode must be 'user' or 'developer'"})
		return
	}

	// Get all node summaries (which now include MCP data)
	summaries, _, err := h.uiService.GetNodesSummary(ctx)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to get nodes summary"})
		return
	}

	// Aggregate MCP status across all nodes
	totalNodes := 0
	nodesWithMCP := 0
	totalMCPServers := 0
	runningMCPServers := 0
	totalTools := 0
	var overallHealth float64 = 1.0

	for _, summary := range summaries {
		totalNodes++
		if summary.MCPSummary != nil {
			nodesWithMCP++
			totalMCPServers += summary.MCPSummary.TotalServers
			runningMCPServers += summary.MCPSummary.RunningServers
			totalTools += summary.MCPSummary.TotalTools

			// Calculate weighted average health
			if summary.MCPSummary.TotalServers > 0 {
				overallHealth = (overallHealth + summary.MCPSummary.OverallHealth) / 2
			}
		}
	}

	// Build response
	response := map[string]interface{}{
		"total_nodes":         totalNodes,
		"nodes_with_mcp":      nodesWithMCP,
		"total_mcp_servers":   totalMCPServers,
		"running_mcp_servers": runningMCPServers,
		"total_tools":         totalTools,
		"overall_health":      overallHealth,
		"timestamp":           time.Now(),
		"mode":                mode,
	}

	// Add detailed node data for developer mode
	if mode == domain.MCPHealthModeDeveloper {
		nodeDetails := make([]map[string]interface{}, 0)
		for _, summary := range summaries {
			nodeDetail := map[string]interface{}{
				"node_id":     summary.ID,
				"team_id":     summary.TeamID,
				"version":     summary.Version,
				"health":      summary.HealthStatus,
				"mcp_summary": summary.MCPSummary,
			}
			nodeDetails = append(nodeDetails, nodeDetail)
		}
		response["nodes"] = nodeDetails
	}

	c.JSON(http.StatusOK, response)
}

// GetMCPEventsHandler handles requests for MCP events (SSE endpoint)
// GET /api/ui/v1/nodes/{nodeId}/mcp/events
func (h *MCPHandler) GetMCPEventsHandler(c *gin.Context) {
	nodeID := c.Param("nodeId")
	if nodeID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "nodeId is required"})
		return
	}

	// Set SSE headers
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("Access-Control-Allow-Origin", "*")
	c.Header("Access-Control-Allow-Headers", "Cache-Control")

	// Get the response writer
	w := c.Writer

	// Send initial connection event
	initialEvent := map[string]interface{}{
		"type":      "connection",
		"node_id":   nodeID,
		"timestamp": time.Now().Format(time.RFC3339),
		"message":   "Connected to MCP events stream",
	}

	// Write SSE formatted data
	c.SSEvent("mcp-event", initialEvent)
	w.Flush()

	// Keep connection alive with periodic heartbeat
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	// Create a channel to handle client disconnect
	clientGone := c.Request.Context().Done()

	for {
		select {
		case <-clientGone:
			// Client disconnected
			return
		case <-ticker.C:
			// Send heartbeat
			heartbeat := map[string]interface{}{
				"type":      "heartbeat",
				"node_id":   nodeID,
				"timestamp": time.Now().Format(time.RFC3339),
			}
			c.SSEvent("heartbeat", heartbeat)
			w.Flush()
		}
	}
}

// GetMCPMetricsHandler handles requests for MCP metrics
// GET /api/ui/v1/nodes/{nodeId}/mcp/metrics
func (h *MCPHandler) GetMCPMetricsHandler(c *gin.Context) {
	nodeID := c.Param("nodeId")
	if nodeID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "nodeId is required"})
		return
	}

	// For now, return empty metrics as this endpoint is not fully implemented
	// TODO: Implement real MCP metrics collection
	response := map[string]interface{}{
		"metrics":   map[string]interface{}{},
		"node_id":   nodeID,
		"timestamp": time.Now(),
	}

	c.JSON(http.StatusOK, response)
}
