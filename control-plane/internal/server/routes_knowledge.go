package server

import (
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"

	"github.com/gin-gonic/gin"
)

// registerKnowledgeRoutes installs the /api/v1/knowledge surface: the native,
// scope-aware RAG store where callers send TEXT and the control plane embeds,
// stores, and scoped-searches it. Skipped when the feature is disabled.
func (s *AgentFieldServer) registerKnowledgeRoutes(agentAPI *gin.RouterGroup) {
	if !s.config.Features.Knowledge.IsEnabled() {
		logger.Logger.Info().Msg("knowledge store disabled; skipping /api/v1/knowledge routes")
		return
	}
	if s.knowledgeService == nil {
		logger.Logger.Warn().Msg("knowledge service not initialized; skipping /api/v1/knowledge routes")
		return
	}

	kg := agentAPI.Group("/knowledge")
	{
		kg.POST("/upsert", handlers.KnowledgeUpsertHandler(s.knowledgeService))
		kg.POST("/search", handlers.KnowledgeSearchHandler(s.knowledgeService))
		kg.DELETE("/source/:id", handlers.KnowledgeDeleteSourceHandler(s.knowledgeService))
	}
}
