package server

import (
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/handlers/agentic"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"

	"github.com/gin-gonic/gin"
)

// agenticLoadCacheTTL is how long the ambient load snapshot is cached so a
// burst of agentic calls doesn't hammer storage.
const agenticLoadCacheTTL = 2 * time.Second

// registerAgenticRoutes installs the /api/v1/agentic/* surface — agent-optimized
// endpoints for discovery, query, run inspection, per-agent summaries, batch
// invocation, and aggregate status. These inherit the authenticated agentAPI
// group's middleware stack.
func (s *AgentFieldServer) registerAgenticRoutes(agentAPI *gin.RouterGroup) {
	// Ambient machine-load metadata stamped onto every agentic response
	// (meta.load). Cached for a short TTL so bursts of calls don't hammer
	// storage; degrades to omitting meta.load on any error.
	agentic.SetLoadProvider(agentic.NewStorageLoadProvider(s.storage, agenticLoadCacheTTL))

	agenticGroup := agentAPI.Group("/agentic")
	{
		agenticGroup.GET("/discover", agentic.DiscoverHandler(s.apiCatalog))
		agenticGroup.GET("/reasoners", agentic.ReasonersHandler(s.storage))
		agenticGroup.POST("/query", agentic.QueryHandler(s.storage))
		agenticGroup.GET("/run/:run_id", agentic.RunOverviewHandler(s.storage))
		agenticGroup.GET("/agent/:agent_id/summary", agentic.AgentSummaryHandler(s.storage))
		agenticGroup.POST("/batch", agentic.BatchHandler(s.Router))
		agenticGroup.GET("/status", agentic.StatusHandler(s.storage))
	}
	logger.Logger.Info().Msg("🤖 Agentic API routes registered (discover, reasoners, query, run, agent, batch, status)")
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
