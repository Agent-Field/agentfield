package ui

import (
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
)

// IdentityHandlers handles identity and credential UI endpoints
type IdentityHandlers struct {
	storage       storage.StorageProvider
	didWebService *services.DIDWebService
}

// NewIdentityHandlers creates a new identity handlers instance
func NewIdentityHandlers(storage storage.StorageProvider, didWebService *services.DIDWebService) *IdentityHandlers {
	return &IdentityHandlers{
		storage:       storage,
		didWebService: didWebService,
	}
}

// DIDSearchResult represents a search result for DIDs
type DIDSearchResult struct {
	Type           string `json:"type"` // "agent", "reasoner", "skill"
	DID            string `json:"did"`
	Name           string `json:"name"`
	ParentDID      string `json:"parent_did,omitempty"`
	ParentName     string `json:"parent_name,omitempty"`
	DerivationPath string `json:"derivation_path"`
	Status         string `json:"status,omitempty"`
	CreatedAt      string `json:"created_at"`
}

// DIDStatsResponse represents DID statistics
type DIDStatsResponse struct {
	TotalAgents    int `json:"total_agents"`
	TotalReasoners int `json:"total_reasoners"`
	TotalSkills    int `json:"total_skills"`
	TotalDIDs      int `json:"total_dids"`
}

// AgentDIDResponse represents an agent with its DIDs
type AgentDIDResponse struct {
	DID            string             `json:"did"`
	DIDWeb         string             `json:"did_web,omitempty"`
	AgentNodeID    string             `json:"agent_node_id"`
	Status         string             `json:"status"`
	DerivationPath string             `json:"derivation_path"`
	CreatedAt      string             `json:"created_at"`
	ReasonerCount  int                `json:"reasoner_count"`
	SkillCount     int                `json:"skill_count"`
	Reasoners      []ComponentDIDInfo `json:"reasoners,omitempty"`
	Skills         []ComponentDIDInfo `json:"skills,omitempty"`
}

// ComponentDIDInfo represents a reasoner or skill DID
type ComponentDIDInfo struct {
	DID            string `json:"did"`
	Name           string `json:"name"`
	Type           string `json:"type"` // "reasoner" or "skill"
	DerivationPath string `json:"derivation_path"`
	CreatedAt      string `json:"created_at"`
}

// VCSearchResult represents a verifiable credential search result
type VCSearchResult struct {
	VCID         string `json:"vc_id"`
	ExecutionID  string `json:"execution_id"`
	WorkflowID   string `json:"workflow_id"`
	WorkflowName string `json:"workflow_name,omitempty"`
	SessionID    string `json:"session_id"`
	IssuerDID    string `json:"issuer_did"`
	TargetDID    string `json:"target_did"`
	CallerDID    string `json:"caller_did"`
	Status       string `json:"status"`
	CreatedAt    string `json:"created_at"`
	DurationMS   *int   `json:"duration_ms,omitempty"`
	ReasonerName string `json:"reasoner_name,omitempty"`
	AgentName    string `json:"agent_name,omitempty"`
	Verified     bool   `json:"verified"`
}

// GetDIDStats returns statistics about DIDs in the system
// GET /api/ui/v1/identity/dids/stats
func (h *IdentityHandlers) GetDIDStats(c *gin.Context) {
	ctx := c.Request.Context()

	// Get all agent DIDs
	agentDIDs, err := h.storage.ListAgentDIDs(ctx)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to list agent DIDs")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get DID stats"})
		return
	}

	// Get all component DIDs (pass empty string to get all)
	componentDIDs, err := h.storage.ListComponentDIDs(ctx, "")
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to list component DIDs")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get DID stats"})
		return
	}

	// Count reasoners and skills
	reasonerCount := 0
	skillCount := 0
	for i := range componentDIDs {
		comp := componentDIDs[i]
		if comp.ComponentType == "reasoner" {
			reasonerCount++
		} else if comp.ComponentType == "skill" {
			skillCount++
		}
	}

	stats := DIDStatsResponse{
		TotalAgents:    len(agentDIDs),
		TotalReasoners: reasonerCount,
		TotalSkills:    skillCount,
		TotalDIDs:      len(agentDIDs) + len(componentDIDs),
	}

	c.JSON(http.StatusOK, stats)
}

// SearchDIDs searches for DIDs by query string
// GET /api/ui/v1/identity/dids/search?q=greeting&type=all&limit=20&offset=0
func (h *IdentityHandlers) SearchDIDs(c *gin.Context) {
	ctx := c.Request.Context()
	query := strings.ToLower(c.Query("q"))
	didType := c.DefaultQuery("type", "all") // "all", "agent", "reasoner", "skill"
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "20"))
	offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))

	if limit > 100 {
		limit = 100
	}

	var results []DIDSearchResult

	// Search agents if type is "all" or "agent"
	if didType == "all" || didType == "agent" {
		agentDIDs, err := h.storage.ListAgentDIDs(ctx)
		if err == nil {
			for i := range agentDIDs {
				agent := agentDIDs[i]
				if query == "" || strings.Contains(strings.ToLower(agent.AgentNodeID), query) {
					results = append(results, DIDSearchResult{
						Type:           "agent",
						DID:            agent.DID,
						Name:           agent.AgentNodeID,
						DerivationPath: agent.DerivationPath,
						Status:         string(agent.Status),
						CreatedAt:      agent.RegisteredAt.Format("2006-01-02T15:04:05Z"),
					})
				}
			}
		}
	}

	// Search components if type is "all", "reasoner", or "skill"
	if didType == "all" || didType == "reasoner" || didType == "skill" {
		componentDIDs, err := h.storage.ListComponentDIDs(ctx, "")
		if err == nil {
			for i := range componentDIDs {
				comp := componentDIDs[i]
				// Filter by type
				if didType != "all" && comp.ComponentType != didType {
					continue
				}

				// Filter by query
				if query != "" && !strings.Contains(strings.ToLower(comp.ComponentName), query) {
					continue
				}

				results = append(results, DIDSearchResult{
					Type:           comp.ComponentType,
					DID:            comp.ComponentDID,
					Name:           comp.ComponentName,
					ParentDID:      comp.AgentDID,
					DerivationPath: strconv.Itoa(comp.DerivationIndex),
					CreatedAt:      comp.CreatedAt.Format("2006-01-02T15:04:05Z"),
				})
			}
		}
	}

	// Apply pagination
	total := len(results)
	start := offset
	end := offset + limit

	if start > total {
		start = total
	}
	if end > total {
		end = total
	}

	paginatedResults := results[start:end]

	c.JSON(http.StatusOK, gin.H{
		"results":  paginatedResults,
		"total":    total,
		"limit":    limit,
		"offset":   offset,
		"has_more": end < total,
	})
}

// ListAgents returns a paginated list of agent DIDs
// GET /api/ui/v1/identity/agents?limit=10&offset=0
func (h *IdentityHandlers) ListAgents(c *gin.Context) {
	ctx := c.Request.Context()
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "10"))
	offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))

	if limit > 50 {
		limit = 50
	}

	agentDIDs, err := h.storage.ListAgentDIDs(ctx)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to list agent DIDs")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to list agents"})
		return
	}

	// Get component counts for each agent
	componentDIDs, _ := h.storage.ListComponentDIDs(ctx, "")
	componentsByAgent := make(map[string][]*types.ComponentDIDInfo)
	for i := range componentDIDs {
		comp := componentDIDs[i]
		componentsByAgent[comp.AgentDID] = append(componentsByAgent[comp.AgentDID], comp)
	}

	// Build response
	var agents []AgentDIDResponse
	for i := range agentDIDs {
		agent := agentDIDs[i]
		components := componentsByAgent[agent.DID]
		reasonerCount := 0
		skillCount := 0
		for _, comp := range components {
			if comp.ComponentType == "reasoner" {
				reasonerCount++
			} else if comp.ComponentType == "skill" {
				skillCount++
			}
		}

		agents = append(agents, AgentDIDResponse{
			DID:            agent.DID,
			DIDWeb:         h.resolveAgentDIDWeb(c, agent.AgentNodeID),
			AgentNodeID:    agent.AgentNodeID,
			Status:         string(agent.Status),
			DerivationPath: agent.DerivationPath,
			CreatedAt:      agent.RegisteredAt.Format("2006-01-02T15:04:05Z"),
			ReasonerCount:  reasonerCount,
			SkillCount:     skillCount,
		})
	}

	// Apply pagination
	total := len(agents)
	start := offset
	end := offset + limit

	if start > total {
		start = total
	}
	if end > total {
		end = total
	}

	paginatedAgents := agents[start:end]

	c.JSON(http.StatusOK, gin.H{
		"agents":   paginatedAgents,
		"total":    total,
		"limit":    limit,
		"offset":   offset,
		"has_more": end < total,
	})
}

// GetAgentDetails returns detailed information about an agent and its components
// GET /api/ui/v1/identity/agents/:agent_id/details?limit=20&offset=0
func (h *IdentityHandlers) GetAgentDetails(c *gin.Context) {
	ctx := c.Request.Context()
	agentNodeID := c.Param("agent_id")
	limit, _ := strconv.Atoi(c.DefaultQuery("limit", "20"))
	offset, _ := strconv.Atoi(c.DefaultQuery("offset", "0"))

	if limit > 100 {
		limit = 100
	}

	// Find the agent DID
	agentDIDs, err := h.storage.ListAgentDIDs(ctx)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to list agent DIDs")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get agent details"})
		return
	}

	var agentDID *types.AgentDIDInfo
	for i := range agentDIDs {
		if agentDIDs[i].AgentNodeID == agentNodeID {
			agentDID = agentDIDs[i]
			break
		}
	}

	if agentDID == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Agent not found"})
		return
	}

	// Get components for this agent
	componentDIDs, err := h.storage.ListComponentDIDs(ctx, agentDID.DID)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to list component DIDs")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to get agent components"})
		return
	}

	var reasoners []ComponentDIDInfo
	var skills []ComponentDIDInfo

	for i := range componentDIDs {
		comp := componentDIDs[i]

		info := ComponentDIDInfo{
			DID:            comp.ComponentDID,
			Name:           comp.ComponentName,
			Type:           comp.ComponentType,
			DerivationPath: strconv.Itoa(comp.DerivationIndex),
			CreatedAt:      comp.CreatedAt.Format("2006-01-02T15:04:05Z"),
		}

		if comp.ComponentType == "reasoner" {
			reasoners = append(reasoners, info)
		} else if comp.ComponentType == "skill" {
			skills = append(skills, info)
		}
	}

	// Apply pagination to reasoners
	totalReasoners := len(reasoners)
	start := offset
	end := offset + limit

	if start > totalReasoners {
		start = totalReasoners
	}
	if end > totalReasoners {
		end = totalReasoners
	}

	paginatedReasoners := reasoners[start:end]

	response := AgentDIDResponse{
		DID:            agentDID.DID,
		DIDWeb:         h.resolveAgentDIDWeb(c, agentDID.AgentNodeID),
		AgentNodeID:    agentDID.AgentNodeID,
		Status:         string(agentDID.Status),
		DerivationPath: agentDID.DerivationPath,
		CreatedAt:      agentDID.RegisteredAt.Format("2006-01-02T15:04:05Z"),
		ReasonerCount:  len(reasoners),
		SkillCount:     len(skills),
		Reasoners:      paginatedReasoners,
		Skills:         skills,
	}

	c.JSON(http.StatusOK, gin.H{
		"agent":              response,
		"total_reasoners":    totalReasoners,
		"reasoners_limit":    limit,
		"reasoners_offset":   offset,
		"reasoners_has_more": end < totalReasoners,
	})
}

// SearchCredentials searches for verifiable credentials with time-range filtering
// GET /api/ui/v1/identity/credentials/search?start_time=...&end_time=...&workflow_id=...&status=...&limit=50&offset=0
func (h *IdentityHandlers) SearchCredentials(c *gin.Context) {
	ctx := c.Request.Context()

	// Parse filters
	filters := types.VCFilters{
		Limit:  50,
		Offset: 0,
	}

	if limit := c.Query("limit"); limit != "" {
		if l, err := strconv.Atoi(limit); err == nil && l > 0 {
			if l > 100 {
				l = 100
			}
			filters.Limit = l
		}
	}

	if offset := c.Query("offset"); offset != "" {
		if o, err := strconv.Atoi(offset); err == nil && o >= 0 {
			filters.Offset = o
		}
	}

	if workflowID := c.Query("workflow_id"); workflowID != "" {
		filters.WorkflowID = &workflowID
	}

	if sessionID := c.Query("session_id"); sessionID != "" {
		filters.SessionID = &sessionID
	}

	if statusParam := c.Query("status"); statusParam != "" {
		normalized := strings.ToLower(statusParam)
		switch normalized {
		case "all":
			normalized = ""
		case "verified":
			normalized = "completed"
		case "failed":
			normalized = "failed"
		case "pending":
			normalized = "pending"
		case "revoked":
			normalized = "revoked"
		}
		if normalized != "" {
			filters.Status = &normalized
		}
	}

	if issuerDID := c.Query("issuer_did"); issuerDID != "" {
		filters.IssuerDID = &issuerDID
	}

	if executionID := c.Query("execution_id"); executionID != "" {
		filters.ExecutionID = &executionID
	}

	if callerDID := c.Query("caller_did"); callerDID != "" {
		filters.CallerDID = &callerDID
	}

	if targetDID := c.Query("target_did"); targetDID != "" {
		filters.TargetDID = &targetDID
	}

	if agentNodeID := c.Query("agent_node_id"); agentNodeID != "" {
		filters.AgentNodeID = &agentNodeID
	} else if agentNodeID := c.Query("agent_id"); agentNodeID != "" {
		filters.AgentNodeID = &agentNodeID
	}

	if search := strings.TrimSpace(c.Query("query")); search != "" {
		filters.Search = &search
	} else if q := strings.TrimSpace(c.Query("q")); q != "" {
		filters.Search = &q
	}

	// Parse time range filters
	if startTime := c.Query("start_time"); startTime != "" {
		if t, err := time.Parse(time.RFC3339, startTime); err == nil {
			filters.CreatedAfter = &t
		} else {
			logger.Logger.Warn().Str("start_time", startTime).Err(err).Msg("Failed to parse start_time")
		}
	}

	if endTime := c.Query("end_time"); endTime != "" {
		if t, err := time.Parse(time.RFC3339, endTime); err == nil {
			filters.CreatedBefore = &t
		} else {
			logger.Logger.Warn().Str("end_time", endTime).Err(err).Msg("Failed to parse end_time")
		}
	}

	// Query execution VCs
	vcs, err := h.storage.ListExecutionVCs(ctx, filters)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to query execution VCs")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to search credentials"})
		return
	}

	countFilters := filters
	countFilters.Limit = 0
	countFilters.Offset = 0
	totalCount, err := h.storage.CountExecutionVCs(ctx, countFilters)
	if err != nil {
		logger.Logger.Error().Err(err).Msg("Failed to count execution VCs")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to search credentials"})
		return
	}

	// Transform to search results
	var results []VCSearchResult
	for i := range vcs {
		vc := vcs[i]
		var agentName string
		if vc.AgentNodeID != nil {
			agentName = *vc.AgentNodeID
		}

		var workflowName string
		if vc.WorkflowName != nil {
			workflowName = *vc.WorkflowName
		}

		verified := strings.EqualFold(vc.Status, "completed") || strings.EqualFold(vc.Status, "succeeded")

		results = append(results, VCSearchResult{
			VCID:         vc.VCID,
			ExecutionID:  vc.ExecutionID,
			WorkflowID:   vc.WorkflowID,
			WorkflowName: workflowName,
			SessionID:    vc.SessionID,
			IssuerDID:    vc.IssuerDID,
			TargetDID:    vc.TargetDID,
			CallerDID:    vc.CallerDID,
			Status:       vc.Status,
			CreatedAt:    vc.CreatedAt.Format("2006-01-02T15:04:05Z"),
			AgentName:    agentName,
			Verified:     verified,
		})
	}

	c.JSON(http.StatusOK, gin.H{
		"credentials": results,
		"total":       totalCount,
		"limit":       filters.Limit,
		"offset":      filters.Offset,
		"has_more":    filters.Offset+len(results) < totalCount,
	})
}

// resolveAgentDIDWeb returns the did:web identifier for an agent, or empty string if unavailable.
func (h *IdentityHandlers) resolveAgentDIDWeb(c *gin.Context, agentID string) string {
	if h.didWebService == nil {
		return ""
	}
	result, err := h.didWebService.ResolveDIDByAgentID(c.Request.Context(), agentID)
	if err == nil && result != nil && result.DIDDocument != nil {
		return h.didWebService.GenerateDIDWeb(agentID)
	}
	return ""
}

// RegisterRoutes registers all identity UI routes
func (h *IdentityHandlers) RegisterRoutes(router *gin.RouterGroup) {
	identity := router.Group("/identity")
	{
		// DID Explorer endpoints
		identity.GET("/dids/stats", h.GetDIDStats)
		identity.GET("/dids/search", h.SearchDIDs)
		identity.GET("/agents", h.ListAgents)
		identity.GET("/agents/:agent_id/details", h.GetAgentDetails)

		// Credentials endpoints
		identity.GET("/credentials/search", h.SearchCredentials)
	}
}
