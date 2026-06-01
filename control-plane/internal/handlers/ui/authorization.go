package ui

import (
	"net/http"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// AuthorizationHandler handles authorization-related UI endpoints.
type AuthorizationHandler struct {
	storage storage.StorageProvider
}

// NewAuthorizationHandler creates a new authorization handler.
func NewAuthorizationHandler(storage storage.StorageProvider) *AuthorizationHandler {
	return &AuthorizationHandler{storage: storage}
}

// AgentTagSummaryResponse is the per-agent response for the authorization agents list.
type AgentTagSummaryResponse struct {
	AgentID         string   `json:"agent_id"`
	ProposedTags    []string `json:"proposed_tags"`
	ApprovedTags    []string `json:"approved_tags"`
	LifecycleStatus string   `json:"lifecycle_status"`
	RegisteredAt    string   `json:"registered_at"`
}

// GetAgentsWithTagsHandler returns all agents with their tag data.
// GET /api/ui/v1/authorization/agents
func (h *AuthorizationHandler) GetAgentsWithTagsHandler(c *gin.Context) {
	agents, err := h.storage.ListAgents(c.Request.Context(), types.AgentFilters{})
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to list agents for authorization view")
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "list_failed",
			"message": "Failed to list agents",
		})
		return
	}

	responses := make([]AgentTagSummaryResponse, 0, len(agents))
	for _, agent := range agents {
		proposed := agent.ProposedTags
		if proposed == nil {
			proposed = []string{}
		}
		approved := agent.ApprovedTags
		if approved == nil {
			approved = []string{}
		}

		responses = append(responses, AgentTagSummaryResponse{
			AgentID:         agent.ID,
			ProposedTags:    proposed,
			ApprovedTags:    approved,
			LifecycleStatus: string(agent.LifecycleStatus),
			RegisteredAt:    agent.RegisteredAt.Format("2006-01-02T15:04:05Z"),
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"agents": responses,
		"total":  len(responses),
	})
}
