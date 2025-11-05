package ui

import (
	"context"
	"net/http"
	"strconv"
	"strings"

	"github.com/your-org/agentfield/control-plane/internal/core/domain"
	"github.com/your-org/agentfield/control-plane/internal/core/interfaces"
	"github.com/your-org/agentfield/control-plane/internal/storage"
	"github.com/your-org/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// LifecycleHandler provides handlers for agent lifecycle management operations.
type LifecycleHandler struct {
	storage      storage.StorageProvider
	agentService interfaces.AgentService
}

// NewLifecycleHandler creates a new LifecycleHandler.
func NewLifecycleHandler(storage storage.StorageProvider, agentService interfaces.AgentService) *LifecycleHandler {
	return &LifecycleHandler{
		storage:      storage,
		agentService: agentService,
	}
}

// getAgentBaseURL attempts to get the stored base_url for an agent, falling back to localhost construction
func (h *LifecycleHandler) getAgentBaseURL(ctx context.Context, agentID string, port int) string {
	// First try to get the registered agent's base_url from storage
	if registeredAgent, err := h.storage.GetAgent(ctx, agentID); err == nil && registeredAgent != nil && registeredAgent.BaseURL != "" {
		return registeredAgent.BaseURL
	}

	// Fallback to localhost construction for locally running agents
	return "http://localhost:" + strconv.Itoa(port)
}

// buildEndpoints creates endpoint URLs using the appropriate base URL
func (h *LifecycleHandler) buildEndpoints(ctx context.Context, agentID string, port int) map[string]string {
	baseURL := h.getAgentBaseURL(ctx, agentID, port)
	return map[string]string{
		"health":    baseURL + "/health",
		"reasoners": baseURL + "/reasoners",
		"skills":    baseURL + "/skills",
	}
}

// StartAgentRequest represents the request body for starting an agent
type StartAgentRequest struct {
	Port   *int  `json:"port,omitempty"`
	Detach *bool `json:"detach,omitempty"`
}

// StartAgentHandler handles requests for starting an agent with configuration
// POST /api/ui/v1/agents/:agentId/start
func (h *LifecycleHandler) StartAgentHandler(c *gin.Context) {
	ctx := c.Request.Context()
	agentID := c.Param("agentId")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "agentId is required"})
		return
	}

	// Parse request body (optional)
	var req StartAgentRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		// If JSON parsing fails, continue with defaults
		req = StartAgentRequest{}
	}

	// Set default values
	port := 0
	if req.Port != nil {
		port = *req.Port
	}

	detach := true
	if req.Detach != nil {
		detach = *req.Detach
	}

	// Create run options
	runOptions := domain.RunOptions{
		Port:   port,
		Detach: detach,
	}

	// Start the agent using the agent service
	// The AgentService will handle validation of agent existence and configuration
	runningAgent, err := h.agentService.RunAgent(agentID, runOptions)
	if err != nil {
		// Check if it's a "not found" error and return appropriate status
		if strings.Contains(err.Error(), "not installed") || strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, ErrorResponse{Error: err.Error()})
			return
		}
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to start agent: " + err.Error()})
		return
	}

	// Return success response
	response := map[string]interface{}{
		"agent_id":   agentID,
		"status":     "started",
		"pid":        runningAgent.PID,
		"port":       runningAgent.Port,
		"started_at": runningAgent.StartedAt,
		"log_file":   runningAgent.LogFile,
		"message":    "agent started successfully",
		"endpoints":  h.buildEndpoints(ctx, agentID, runningAgent.Port),
	}

	c.JSON(http.StatusOK, response)
}

// StopAgentHandler handles requests for stopping a running agent
// POST /api/ui/v1/agents/:agentId/stop
func (h *LifecycleHandler) StopAgentHandler(c *gin.Context) {
	agentID := c.Param("agentId")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "agentId is required"})
		return
	}

	// Get current agent status
	agentStatus, err := h.agentService.GetAgentStatus(agentID)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "agent not found or not installed"})
		return
	}

	if !agentStatus.IsRunning {
		c.JSON(http.StatusBadRequest, map[string]interface{}{
			"error":   "agent not running",
			"status":  "stopped",
			"message": "agent is already stopped",
		})
		return
	}

	// Stop the agent using the agent service
	if err := h.agentService.StopAgent(agentID); err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to stop agent: " + err.Error()})
		return
	}

	// Return success response
	response := map[string]interface{}{
		"agent_id": agentID,
		"status":   "stopped",
		"message":  "agent stopped successfully",
	}

	c.JSON(http.StatusOK, response)
}

// GetAgentStatusHandler handles requests for getting agent status
// GET /api/ui/v1/agents/:agentId/status
func (h *LifecycleHandler) GetAgentStatusHandler(c *gin.Context) {
	ctx := c.Request.Context()
	agentID := c.Param("agentId")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "agentId is required"})
		return
	}

	// Check if agent package exists
	agentPackage, err := h.storage.GetAgentPackage(ctx, agentID)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "agent package not found"})
		return
	}

	// Get agent status from service
	agentStatus, err := h.agentService.GetAgentStatus(agentID)
	if err != nil {
		// Agent not installed, return basic status
		response := map[string]interface{}{
			"agent_id":   agentID,
			"name":       agentPackage.Name,
			"is_running": false,
			"status":     "not_installed",
			"message":    "agent package found but not installed",
		}
		c.JSON(http.StatusOK, response)
		return
	}

	// Check configuration status
	var configurationStatus string
	var configurationRequired bool
	if len(agentPackage.ConfigurationSchema) > 0 {
		configurationRequired = true
		config, err := h.storage.GetAgentConfiguration(ctx, agentID, agentID)
		configurationStatus = getConfigurationStatus(config)
		if err != nil {
			configurationStatus = "not_configured"
		}
	} else {
		configurationRequired = false
		configurationStatus = "not_required"
	}

	// Build response
	response := map[string]interface{}{
		"agent_id":               agentID,
		"name":                   agentStatus.Name,
		"is_running":             agentStatus.IsRunning,
		"status":                 getAgentLifecycleStatus(agentStatus, configurationStatus, configurationRequired),
		"pid":                    agentStatus.PID,
		"port":                   agentStatus.Port,
		"uptime":                 agentStatus.Uptime,
		"last_seen":              agentStatus.LastSeen,
		"configuration_required": configurationRequired,
		"configuration_status":   configurationStatus,
	}

	// Add endpoints if running
	if agentStatus.IsRunning && agentStatus.Port > 0 {
		response["endpoints"] = h.buildEndpoints(ctx, agentID, agentStatus.Port)
	}

	c.JSON(http.StatusOK, response)
}

// ListRunningAgentsHandler handles requests for listing all running agents
// GET /api/ui/v1/agents/running
func (h *LifecycleHandler) ListRunningAgentsHandler(c *gin.Context) {
	ctx := c.Request.Context()
	// Get all running agents from service
	runningAgents, err := h.agentService.ListRunningAgents()
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to list running agents: " + err.Error()})
		return
	}

	// Build response with additional metadata
	agents := make([]map[string]interface{}, 0) // Initialize with an empty slice
	for _, agent := range runningAgents {
		// Get package info for additional metadata
		agentPackage, err := h.storage.GetAgentPackage(ctx, agent.Name)
		var packageInfo map[string]interface{}
		if err == nil {
			packageInfo = map[string]interface{}{
				"name":        agentPackage.Name,
				"version":     agentPackage.Version,
				"description": agentPackage.Description,
				"author":      agentPackage.Author,
			}
		}

		agentInfo := map[string]interface{}{
			"agent_id":   agent.Name,
			"name":       agent.Name,
			"status":     agent.Status,
			"pid":        agent.PID,
			"port":       agent.Port,
			"started_at": agent.StartedAt,
			"log_file":   agent.LogFile,
			"package":    packageInfo,
		}

		// Add endpoints if port is available
		if agent.Port > 0 {
			agentInfo["endpoints"] = h.buildEndpoints(ctx, agent.Name, agent.Port)
		}

		agents = append(agents, agentInfo)
	}

	response := map[string]interface{}{
		"running_agents": agents,
		"total_count":    len(agents),
	}

	c.JSON(http.StatusOK, response)
}

// Helper function to determine configuration status
func getConfigurationStatus(config *types.AgentConfiguration) string {
	if config == nil {
		return "not_configured"
	}

	switch config.Status {
	case "active":
		return "configured"
	case "draft":
		return "partially_configured"
	default:
		return "not_configured"
	}
}

// ReconcileAgentHandler forces reconciliation of agent state with actual process state
// POST /api/ui/v1/agents/:agentId/reconcile
func (h *LifecycleHandler) ReconcileAgentHandler(c *gin.Context) {
	agentID := c.Param("agentId")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "agentId is required"})
		return
	}

	// Get current status which will trigger reconciliation
	status, err := h.agentService.GetAgentStatus(agentID)
	if err != nil {
		if strings.Contains(err.Error(), "not installed") || strings.Contains(err.Error(), "not found") {
			c.JSON(http.StatusNotFound, ErrorResponse{Error: err.Error()})
			return
		}
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "failed to reconcile agent state: " + err.Error()})
		return
	}

	// Return reconciled status
	response := map[string]interface{}{
		"agent_id":   agentID,
		"status":     "reconciled",
		"is_running": status.IsRunning,
		"pid":        status.PID,
		"port":       status.Port,
		"last_seen":  status.LastSeen,
		"uptime":     status.Uptime,
		"message":    "agent state reconciled with actual process state",
	}

	c.JSON(http.StatusOK, response)
}

// Helper function to determine overall agent lifecycle status
func getAgentLifecycleStatus(agentStatus *domain.AgentStatus, configStatus string, configRequired bool) string {
	if agentStatus.IsRunning {
		return "running"
	}

	if configRequired && configStatus != "configured" {
		return "not_configured"
	}

	return "stopped"
}
