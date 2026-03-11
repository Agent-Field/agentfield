package connector

import (
	"fmt"
	"net/http"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers/admin"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
)

// Handlers provides connector-specific HTTP handlers for the control plane.
type Handlers struct {
	connectorConfig     config.ConnectorConfig
	storage             storage.StorageProvider
	statusManager       *services.StatusManager
	accessPolicyService *services.AccessPolicyService
	tagApprovalService  *services.TagApprovalService
	didService          *services.DIDService
}

// NewHandlers creates connector handlers with injected dependencies.
func NewHandlers(
	cfg config.ConnectorConfig,
	store storage.StorageProvider,
	statusManager *services.StatusManager,
	accessPolicyService *services.AccessPolicyService,
	tagApprovalService *services.TagApprovalService,
	didService *services.DIDService,
) *Handlers {
	return &Handlers{
		connectorConfig:     cfg,
		storage:             store,
		statusManager:       statusManager,
		accessPolicyService: accessPolicyService,
		tagApprovalService:  tagApprovalService,
		didService:          didService,
	}
}

// RegisterRoutes registers all connector routes on the given router group.
// Each route group is gated by its corresponding capability — the CP is the
// sole authority for what the connector token is allowed to access.
// The /manifest endpoint is always accessible so the connector can learn
// its granted capabilities on startup.
func (h *Handlers) RegisterRoutes(group *gin.RouterGroup) {
	caps := h.connectorConfig.Capabilities

	// Manifest endpoint — always accessible (connector needs this to learn capabilities)
	group.GET("/manifest", h.GetManifest)

	// Reasoner management routes
	reasonerGroup := group.Group("")
	reasonerGroup.Use(middleware.ConnectorCapabilityCheck("reasoner_management", caps))
	{
		reasonerGroup.GET("/reasoners", h.ListReasoners)
		reasonerGroup.GET("/reasoners/:id", h.GetReasoner)
		reasonerGroup.PUT("/reasoners/:id/version", h.SetReasonerVersion)
		reasonerGroup.POST("/reasoners/:id/restart", h.RestartReasoner)
		reasonerGroup.GET("/groups", h.ListAgentGroups)
		reasonerGroup.GET("/groups/:group_id/nodes", h.ListGroupNodes)

		// Version-aware routes (Phase 2)
		reasonerGroup.GET("/reasoners/:id/versions", h.ListReasonerVersions)
		reasonerGroup.GET("/reasoners/:id/versions/:version", h.GetReasonerVersion)
		reasonerGroup.PUT("/reasoners/:id/versions/:version/weight", h.SetReasonerTrafficWeight)
		reasonerGroup.POST("/reasoners/:id/versions/:version/restart", h.RestartReasonerVersion)
	}

	// Status routes (read-only node information)
	statusGroup := group.Group("")
	statusGroup.Use(middleware.ConnectorCapabilityCheck("status_read", caps))
	{
		statusGroup.GET("/nodes", h.ListNodes)
		statusGroup.GET("/nodes/:id/status", h.GetNodeStatus)
	}

	// Policy management routes (proxied admin endpoints)
	if h.accessPolicyService != nil {
		policyGroup := group.Group("")
		policyGroup.Use(middleware.ConnectorCapabilityCheck("policy_management", caps))
		policyHandlers := admin.NewAccessPolicyHandlers(h.accessPolicyService)
		policyHandlers.RegisterRoutes(policyGroup)
	}

	// Tag management routes (proxied admin endpoints)
	if h.tagApprovalService != nil {
		tagGroup := group.Group("")
		tagGroup.Use(middleware.ConnectorCapabilityCheck("tag_management", caps))
		tagHandlers := admin.NewTagApprovalHandlers(h.tagApprovalService, h.storage)
		tagHandlers.RegisterRoutes(tagGroup)
	}
}

// ListNodes returns all registered agent nodes for the connector.
func (h *Handlers) ListNodes(c *gin.Context) {
	ctx := c.Request.Context()
	agents, err := h.storage.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"agents": agents, "total": len(agents)})
}

// GetNodeStatus returns status info for a specific agent node.
func (h *Handlers) GetNodeStatus(c *gin.Context) {
	ctx := c.Request.Context()
	id := c.Param("id")

	agent, err := h.storage.GetAgent(ctx, id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if agent == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "agent node not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"id":               agent.ID,
		"group_id":         agent.GroupID,
		"health_status":    agent.HealthStatus,
		"lifecycle_status": agent.LifecycleStatus,
		"version":          agent.Version,
	})
}

// GetManifest returns the server-side capability manifest showing what
// this control plane supports and what the connector is configured to access.
func (h *Handlers) GetManifest(c *gin.Context) {
	capabilities := make(map[string]map[string]interface{})
	for name, cap := range h.connectorConfig.Capabilities {
		capabilities[name] = map[string]interface{}{
			"enabled":   cap.Enabled,
			"read_only": cap.ReadOnly,
		}
	}

	manifest := gin.H{
		"connector_enabled": h.connectorConfig.Enabled,
		"capabilities":      capabilities,
		"features": gin.H{
			"did_enabled":           h.didService != nil,
			"authorization_enabled": h.accessPolicyService != nil,
		},
	}

	c.JSON(http.StatusOK, manifest)
}

// ListReasoners returns all registered agent nodes with their reasoner info.
func (h *Handlers) ListReasoners(c *gin.Context) {
	ctx := c.Request.Context()
	agents, err := h.storage.ListAgents(ctx, types.AgentFilters{})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	type nodeInfo struct {
		NodeID       string                    `json:"node_id"`
		GroupID      string                    `json:"group_id"`
		TeamID       string                    `json:"team_id"`
		Version      string                    `json:"version"`
		HealthStatus types.HealthStatus        `json:"health_status"`
		Reasoners    []types.ReasonerDefinition `json:"reasoners"`
		Skills       []types.SkillDefinition    `json:"skills"`
	}

	var result []nodeInfo
	for _, agent := range agents {
		result = append(result, nodeInfo{
			NodeID:       agent.ID,
			GroupID:      agent.GroupID,
			TeamID:       agent.TeamID,
			Version:      agent.Version,
			HealthStatus: agent.HealthStatus,
			Reasoners:    agent.Reasoners,
			Skills:       agent.Skills,
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"reasoners": result,
		"total":     len(result),
	})
}

// GetReasoner returns detailed info for a specific agent node.
func (h *Handlers) GetReasoner(c *gin.Context) {
	ctx := c.Request.Context()
	id := c.Param("id")

	agent, err := h.storage.GetAgent(ctx, id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if agent == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "agent node not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"id":               agent.ID,
		"group_id":         agent.GroupID,
		"team_id":          agent.TeamID,
		"version":          agent.Version,
		"health_status":    agent.HealthStatus,
		"lifecycle_status": agent.LifecycleStatus,
		"reasoners":        agent.Reasoners,
		"skills":           agent.Skills,
		"base_url":         agent.BaseURL,
	})
}

// SetReasonerVersion updates the version for a specific agent node.
func (h *Handlers) SetReasonerVersion(c *gin.Context) {
	ctx := c.Request.Context()
	id := c.Param("id")

	var body struct {
		Version string `json:"version" binding:"required"`
	}
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "version is required"})
		return
	}

	agent, err := h.storage.GetAgent(ctx, id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if agent == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "agent node not found"})
		return
	}

	previousVersion := agent.Version

	if err := h.storage.UpdateAgentVersion(ctx, id, body.Version); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success":          true,
		"previous_version": previousVersion,
	})
}

// RestartReasoner initiates a restart for a specific agent node by transitioning
// its lifecycle status to "starting".
func (h *Handlers) RestartReasoner(c *gin.Context) {
	ctx := c.Request.Context()
	id := c.Param("id")

	agent, err := h.storage.GetAgent(ctx, id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if agent == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "agent node not found"})
		return
	}

	if h.statusManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "status manager not available"})
		return
	}

	startingState := types.AgentStateStarting
	update := &types.AgentStatusUpdate{
		State:  &startingState,
		Source: types.StatusSourceManual,
		Reason: "connector restart request",
	}

	if err := h.statusManager.UpdateAgentStatus(ctx, id, update); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success":      true,
		"restarted_at": time.Now().UTC().Format(time.RFC3339),
	})
}

// ListAgentGroups returns distinct agent groups with summary info.
func (h *Handlers) ListAgentGroups(c *gin.Context) {
	ctx := c.Request.Context()
	teamID := c.Query("team_id")
	if teamID == "" {
		teamID = "default"
	}

	groups, err := h.storage.ListAgentGroups(ctx, teamID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"groups": groups,
		"total":  len(groups),
	})
}

// ListGroupNodes returns all nodes belonging to a specific group.
func (h *Handlers) ListGroupNodes(c *gin.Context) {
	ctx := c.Request.Context()
	groupID := c.Param("group_id")

	agents, err := h.storage.ListAgentsByGroup(ctx, groupID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	type nodeInfo struct {
		NodeID          string                     `json:"node_id"`
		GroupID         string                     `json:"group_id"`
		TeamID          string                     `json:"team_id"`
		Version         string                     `json:"version"`
		HealthStatus    types.HealthStatus         `json:"health_status"`
		LifecycleStatus types.AgentLifecycleStatus `json:"lifecycle_status"`
		Reasoners       []types.ReasonerDefinition `json:"reasoners"`
		Skills          []types.SkillDefinition    `json:"skills"`
	}

	var result []nodeInfo
	for _, agent := range agents {
		result = append(result, nodeInfo{
			NodeID:          agent.ID,
			GroupID:         agent.GroupID,
			TeamID:          agent.TeamID,
			Version:         agent.Version,
			HealthStatus:    agent.HealthStatus,
			LifecycleStatus: agent.LifecycleStatus,
			Reasoners:       agent.Reasoners,
			Skills:          agent.Skills,
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"nodes": result,
		"total": len(result),
	})
}

// ListReasonerVersions returns all versions of a specific agent.
func (h *Handlers) ListReasonerVersions(c *gin.Context) {
	ctx := c.Request.Context()
	id := c.Param("id")

	versions, err := h.storage.ListAgentVersions(ctx, id)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Also check for the default (unversioned) agent
	defaultAgent, _ := h.storage.GetAgent(ctx, id)

	type versionInfo struct {
		Version         string                     `json:"version"`
		TrafficWeight   int                        `json:"traffic_weight"`
		HealthStatus    types.HealthStatus         `json:"health_status"`
		LifecycleStatus types.AgentLifecycleStatus `json:"lifecycle_status"`
		BaseURL         string                     `json:"base_url"`
		LastHeartbeat   time.Time                  `json:"last_heartbeat"`
	}

	var result []versionInfo
	if defaultAgent != nil {
		result = append(result, versionInfo{
			Version:         defaultAgent.Version,
			TrafficWeight:   defaultAgent.TrafficWeight,
			HealthStatus:    defaultAgent.HealthStatus,
			LifecycleStatus: defaultAgent.LifecycleStatus,
			BaseURL:         defaultAgent.BaseURL,
			LastHeartbeat:   defaultAgent.LastHeartbeat,
		})
	}
	for _, v := range versions {
		result = append(result, versionInfo{
			Version:         v.Version,
			TrafficWeight:   v.TrafficWeight,
			HealthStatus:    v.HealthStatus,
			LifecycleStatus: v.LifecycleStatus,
			BaseURL:         v.BaseURL,
			LastHeartbeat:   v.LastHeartbeat,
		})
	}

	if len(result) == 0 {
		c.JSON(http.StatusNotFound, gin.H{"error": "agent not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"id":       id,
		"versions": result,
		"total":    len(result),
	})
}

// GetReasonerVersion returns detailed info for a specific (id, version) pair.
func (h *Handlers) GetReasonerVersion(c *gin.Context) {
	ctx := c.Request.Context()
	id := c.Param("id")
	version := c.Param("version")

	agent, err := h.storage.GetAgentVersion(ctx, id, version)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if agent == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "agent version not found"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"id":               agent.ID,
		"version":          agent.Version,
		"traffic_weight":   agent.TrafficWeight,
		"health_status":    agent.HealthStatus,
		"lifecycle_status": agent.LifecycleStatus,
		"reasoners":        agent.Reasoners,
		"skills":           agent.Skills,
		"base_url":         agent.BaseURL,
	})
}

// SetReasonerTrafficWeight updates the traffic_weight for a specific (id, version) pair.
func (h *Handlers) SetReasonerTrafficWeight(c *gin.Context) {
	ctx := c.Request.Context()
	id := c.Param("id")
	version := c.Param("version")

	var body struct {
		Weight int `json:"weight"`
	}
	if err := c.ShouldBindJSON(&body); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "weight is required"})
		return
	}
	if body.Weight < 0 || body.Weight > 10000 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "weight must be between 0 and 10000"})
		return
	}

	// Verify the version exists and get previous weight
	agent, err := h.storage.GetAgentVersion(ctx, id, version)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if agent == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "agent version not found"})
		return
	}

	previousWeight := agent.TrafficWeight

	if err := h.storage.UpdateAgentTrafficWeight(ctx, id, version, body.Weight); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success":         true,
		"id":              id,
		"version":         version,
		"previous_weight": previousWeight,
		"new_weight":      body.Weight,
	})
}

// RestartReasonerVersion initiates a restart for a specific agent version.
func (h *Handlers) RestartReasonerVersion(c *gin.Context) {
	ctx := c.Request.Context()
	id := c.Param("id")
	version := c.Param("version")

	agent, err := h.storage.GetAgentVersion(ctx, id, version)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if agent == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "agent version not found"})
		return
	}

	if h.statusManager == nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "status manager not available"})
		return
	}

	startingState := types.AgentStateStarting
	update := &types.AgentStatusUpdate{
		State:   &startingState,
		Source:  types.StatusSourceManual,
		Reason:  fmt.Sprintf("connector restart request (version: %s)", version),
		Version: version,
	}

	if err := h.statusManager.UpdateAgentStatus(ctx, id, update); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success":      true,
		"id":           id,
		"version":      version,
		"restarted_at": time.Now().UTC().Format(time.RFC3339),
	})
}

