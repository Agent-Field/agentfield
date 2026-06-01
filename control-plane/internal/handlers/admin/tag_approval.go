package admin

import (
	"errors"
	"net/http"
	"sort"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// TagApprovalHandlers handles admin tag approval HTTP requests.
type TagApprovalHandlers struct {
	tagApprovalService *services.TagApprovalService
	storage            storage.StorageProvider
}

// NewTagApprovalHandlers creates a new tag approval admin handlers instance.
func NewTagApprovalHandlers(tagApprovalService *services.TagApprovalService, storage storage.StorageProvider) *TagApprovalHandlers {
	return &TagApprovalHandlers{
		tagApprovalService: tagApprovalService,
		storage:            storage,
	}
}

// ListPendingAgents returns all agents in pending_approval status.
// GET /api/v1/admin/agents/pending
func (h *TagApprovalHandlers) ListPendingAgents(c *gin.Context) {
	agents, err := h.tagApprovalService.ListPendingAgents(c.Request.Context())
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to list pending agents")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "list_failed",
			"message": "Failed to list pending agents",
		})
		return
	}

	// Convert to response format
	responses := make([]types.PendingAgentResponse, 0, len(agents))
	for _, agent := range agents {
		responses = append(responses, types.PendingAgentResponse{
			AgentID:      agent.ID,
			ProposedTags: agent.ProposedTags,
			ApprovedTags: agent.ApprovedTags,
			Status:       string(agent.LifecycleStatus),
			RegisteredAt: agent.RegisteredAt.Format("2006-01-02T15:04:05Z"),
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"agents": responses,
		"total":  len(responses),
	})
}

// ListApprovedAgents returns all agents that have approved tags (not pending).
// GET /api/v1/admin/agents/approved
func (h *TagApprovalHandlers) ListApprovedAgents(c *gin.Context) {
	agents, err := h.storage.ListAgents(c.Request.Context(), types.AgentFilters{})
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to list agents for approved tags")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "list_failed",
			"message": "Failed to list approved agents",
		})
		return
	}

	responses := make([]types.PendingAgentResponse, 0)
	for _, agent := range agents {
		if agent.LifecycleStatus == types.AgentStatusPendingApproval || len(agent.ApprovedTags) == 0 {
			continue
		}
		responses = append(responses, types.PendingAgentResponse{
			AgentID:      agent.ID,
			ProposedTags: agent.ProposedTags,
			ApprovedTags: agent.ApprovedTags,
			Status:       string(agent.LifecycleStatus),
			RegisteredAt: agent.RegisteredAt.Format("2006-01-02T15:04:05Z"),
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"agents": responses,
		"total":  len(responses),
	})
}

// ApproveAgentTags approves an agent's proposed tags.
// POST /api/v1/admin/agents/:agent_id/approve-tags
func (h *TagApprovalHandlers) ApproveAgentTags(c *gin.Context) {
	agentID := c.Param("agent_id")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "missing_agent_id",
			"message": "agent_id is required",
		})
		return
	}

	var req types.TagApprovalRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "invalid_request",
			"message": "Invalid JSON: " + err.Error(),
		})
		return
	}

	// If per-skill/per-reasoner tags are provided, use per-skill approval
	if len(req.SkillTags) > 0 || len(req.ReasonerTags) > 0 {
		if err := h.tagApprovalService.ApproveAgentTagsPerSkill(c.Request.Context(), agentID, req.SkillTags, req.ReasonerTags, "admin"); err != nil {
			if errors.Is(err, services.ErrNotPendingApproval) {
				c.JSON(http.StatusConflict, gin.H{
					"error":   "not_pending_approval",
					"message": err.Error(),
				})
				return
			}
			logger.Logger.Error().Err(err).Str("agent_id", agentID).Msg("Failed to approve agent tags (per-skill)")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "approval_failed",
				"message": "Failed to approve agent tags: " + err.Error(),
			})
			return
		}
	} else {
		if err := h.tagApprovalService.ApproveAgentTags(c.Request.Context(), agentID, req.ApprovedTags, "admin"); err != nil {
			if errors.Is(err, services.ErrNotPendingApproval) {
				c.JSON(http.StatusConflict, gin.H{
					"error":   "not_pending_approval",
					"message": err.Error(),
				})
				return
			}
			logger.Logger.Error().Err(err).Str("agent_id", agentID).Msg("Failed to approve agent tags")
			c.JSON(http.StatusInternalServerError, gin.H{
				"error":   "approval_failed",
				"message": "Failed to approve agent tags: " + err.Error(),
			})
			return
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"success":       true,
		"message":       "Agent tags approved",
		"agent_id":      agentID,
		"approved_tags": req.ApprovedTags,
	})
}

// RejectAgentTags rejects an agent's proposed tags.
// POST /api/v1/admin/agents/:agent_id/reject-tags
func (h *TagApprovalHandlers) RejectAgentTags(c *gin.Context) {
	agentID := c.Param("agent_id")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "missing_agent_id",
			"message": "agent_id is required",
		})
		return
	}

	var req types.TagRejectionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		// Allow empty body for rejection (reason is optional)
		req = types.TagRejectionRequest{}
	}

	if err := h.tagApprovalService.RejectAgentTags(c.Request.Context(), agentID, "admin", req.Reason); err != nil {
		if errors.Is(err, services.ErrNotPendingApproval) {
			c.JSON(http.StatusConflict, gin.H{
				"error":   "not_pending_approval",
				"message": err.Error(),
			})
			return
		}
		logger.Logger.Error().Err(err).Str("agent_id", agentID).Msg("Failed to reject agent tags")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "rejection_failed",
			"message": "Failed to reject agent tags: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success":  true,
		"message":  "Agent tags rejected",
		"agent_id": agentID,
	})
}

// RevokeAgentTags revokes an agent's approved tags, transitioning it back to pending_approval.
// POST /api/v1/admin/agents/:agent_id/revoke-tags
func (h *TagApprovalHandlers) RevokeAgentTags(c *gin.Context) {
	agentID := c.Param("agent_id")
	if agentID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "missing_agent_id",
			"message": "agent_id is required",
		})
		return
	}

	var req struct {
		Reason string `json:"reason"`
	}
	_ = c.ShouldBindJSON(&req) // reason is optional

	if err := h.tagApprovalService.RevokeAgentTags(c.Request.Context(), agentID, "admin", req.Reason); err != nil {
		if strings.HasPrefix(err.Error(), "already_revoked:") {
			c.JSON(http.StatusConflict, gin.H{
				"error":   "already_revoked",
				"message": "Agent tags are already revoked",
			})
			return
		}
		logger.Logger.Error().Err(err).Str("agent_id", agentID).Msg("Failed to revoke agent tags")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "revocation_failed",
			"message": "Failed to revoke agent tags: " + err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success":  true,
		"message":  "Agent tags revoked",
		"agent_id": agentID,
	})
}

// ListKnownTags returns all unique tags known to the system from agents and policies.
// GET /api/v1/admin/tags
func (h *TagApprovalHandlers) ListKnownTags(c *gin.Context) {
	tagSet := make(map[string]struct{})

	// Collect tags from all agents
	agents, err := h.storage.ListAgents(c.Request.Context(), types.AgentFilters{})
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to list agents for known tags")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "list_failed",
			"message": "Failed to collect known tags",
		})
		return
	}

	for _, agent := range agents {
		for _, t := range agent.ProposedTags {
			tagSet[t] = struct{}{}
		}
		for _, t := range agent.ApprovedTags {
			tagSet[t] = struct{}{}
		}
		for _, r := range agent.Reasoners {
			for _, t := range r.Tags {
				tagSet[t] = struct{}{}
			}
			for _, t := range r.ProposedTags {
				tagSet[t] = struct{}{}
			}
		}
		for _, s := range agent.Skills {
			for _, t := range s.Tags {
				tagSet[t] = struct{}{}
			}
			for _, t := range s.ProposedTags {
				tagSet[t] = struct{}{}
			}
		}
	}

	tags := make([]string, 0, len(tagSet))
	for t := range tagSet {
		tags = append(tags, t)
	}
	sort.Strings(tags)

	c.JSON(http.StatusOK, gin.H{
		"tags":  tags,
		"total": len(tags),
	})
}

// RegisterRoutes registers the tag approval admin routes.
func (h *TagApprovalHandlers) RegisterRoutes(router *gin.RouterGroup) {
	adminGroup := router.Group("/admin")
	{
		agentsGroup := adminGroup.Group("/agents")
		{
			agentsGroup.GET("/pending", h.ListPendingAgents)
			agentsGroup.GET("/approved", h.ListApprovedAgents)
			agentsGroup.POST("/:agent_id/approve-tags", h.ApproveAgentTags)
			agentsGroup.POST("/:agent_id/reject-tags", h.RejectAgentTags)
			agentsGroup.POST("/:agent_id/revoke-tags", h.RevokeAgentTags)
		}
		adminGroup.GET("/tags", h.ListKnownTags)
	}
}
