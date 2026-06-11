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
	AgentID         string              `json:"agent_id"`
	ProposedTags    []string            `json:"proposed_tags"`
	ApprovedTags    []string            `json:"approved_tags"`
	Components      ComponentTagSummary `json:"components"`
	LifecycleStatus string              `json:"lifecycle_status"`
	RegisteredAt    string              `json:"registered_at"`
}

type ComponentTagSummary struct {
	Reasoners []ComponentTagRow `json:"reasoners"`
	Skills    []ComponentTagRow `json:"skills"`
	Sessions  []ComponentTagRow `json:"sessions"`
}

type ComponentTagRow struct {
	ID           string   `json:"id"`
	Kind         string   `json:"kind"`
	ProposedTags []string `json:"proposed_tags"`
	ApprovedTags []string `json:"approved_tags"`
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
		types.HydrateAgentSessions(agent)
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
			Components:      buildComponentTagSummary(agent),
			LifecycleStatus: string(agent.LifecycleStatus),
			RegisteredAt:    agent.RegisteredAt.Format("2006-01-02T15:04:05Z"),
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"agents": responses,
		"total":  len(responses),
	})
}

func buildComponentTagSummary(agent *types.AgentNode) ComponentTagSummary {
	summary := ComponentTagSummary{
		Reasoners: []ComponentTagRow{},
		Skills:    []ComponentTagRow{},
		Sessions:  []ComponentTagRow{},
	}
	for _, reasoner := range agent.Reasoners {
		summary.Reasoners = append(summary.Reasoners, ComponentTagRow{
			ID:           reasoner.ID,
			Kind:         "reasoner",
			ProposedTags: nonNilTags(firstTagList(reasoner.ProposedTags, reasoner.Tags)),
			ApprovedTags: nonNilTags(reasoner.ApprovedTags),
		})
	}
	for _, skill := range agent.Skills {
		summary.Skills = append(summary.Skills, ComponentTagRow{
			ID:           skill.ID,
			Kind:         "skill",
			ProposedTags: nonNilTags(firstTagList(skill.ProposedTags, skill.Tags)),
			ApprovedTags: nonNilTags(skill.ApprovedTags),
		})
	}
	for _, session := range agent.Sessions {
		summary.Sessions = append(summary.Sessions, ComponentTagRow{
			ID:           session.Name,
			Kind:         "session",
			ProposedTags: nonNilTags(firstTagList(session.ProposedTags, session.Tags)),
			ApprovedTags: nonNilTags(session.ApprovedTags),
		})
	}
	return summary
}

func firstTagList(primary []string, fallback []string) []string {
	if len(primary) > 0 {
		return primary
	}
	return fallback
}

func nonNilTags(tags []string) []string {
	if tags == nil {
		return []string{}
	}
	return tags
}
