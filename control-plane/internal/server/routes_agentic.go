package server

import (
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers/agentic"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"

	"github.com/gin-gonic/gin"
)

// registerAgenticRoutes installs the /api/v1/agentic/* surface — agent-optimized
// endpoints for discovery, query, run inspection, per-agent summaries, batch
// invocation, and aggregate status. These inherit the authenticated agentAPI
// group's middleware stack.
func (s *AgentFieldServer) registerAgenticRoutes(agentAPI *gin.RouterGroup) {
	agenticGroup := agentAPI.Group("/agentic")
	{
		agenticGroup.GET("/discover", agentic.DiscoverHandler(s.apiCatalog))
		agenticGroup.POST("/query", agentic.QueryHandler(s.storage))
		agenticGroup.GET("/run/:run_id", agentic.RunOverviewHandler(s.storage))
		agenticGroup.GET("/agent/:agent_id/summary", agentic.AgentSummaryHandler(s.storage))
		agenticGroup.POST("/batch", agentic.BatchHandler(s.Router))
		agenticGroup.GET("/status", agentic.StatusHandler(s.storage))
	}
	logger.Logger.Info().Msg("🤖 Agentic API routes registered (discover, query, run, agent, batch, status)")
}

// registerKBRoutes installs the public, unauthenticated Knowledge Base tree
// under /api/v1/agentic/kb. Registered directly on the root router so it sits
// outside the authenticated agentAPI group.
func (s *AgentFieldServer) registerKBRoutes() {
	kbGroup := s.Router.Group("/api/v1/agentic/kb")
	{
		kbGroup.GET("/topics", agentic.KBTopicsHandler(s.kb))
		kbGroup.GET("/articles", agentic.KBArticlesHandler(s.kb))
		kbGroup.GET("/articles/:article_id/:sub_id", agentic.KBArticleHandler(s.kb))
		kbGroup.GET("/articles/:article_id", agentic.KBArticleHandler(s.kb))
		kbGroup.GET("/guide", agentic.KBGuideHandler(s.kb))
	}
	logger.Logger.Info().Msg("📚 Knowledge Base routes registered (public, no auth)")
}
