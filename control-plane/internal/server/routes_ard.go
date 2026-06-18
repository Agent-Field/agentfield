package server

import (
	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/gin-gonic/gin"
)

func (s *AgentFieldServer) newARDHandler() *handlers.ARDHandler {
	return handlers.NewARDHandler(
		s.storage,
		func() config.ARDConfig {
			s.configMu.RLock()
			defer s.configMu.RUnlock()
			return s.config.AgentField.ARD
		},
		func() bool {
			return s.config.Features.DID.Enabled && s.didService != nil
		},
	)
}

func (s *AgentFieldServer) registerARDPublicRoutes() {
	ardHandler := s.newARDHandler()
	s.Router.GET("/.well-known/ai-catalog.json", ardHandler.GetPublicCatalog)
}

func (s *AgentFieldServer) registerARDRoutes(agentAPI *gin.RouterGroup) {
	ardHandler := s.newARDHandler()
	ardAPI := agentAPI.Group("/ard")
	{
		ardAPI.GET("/artifacts/:entryID", ardHandler.GetArtifact)
		ardAPI.POST("/search", ardHandler.SearchRegistry)
		ardAPI.GET("/agents", ardHandler.ListRegistryAgents)
		ardAPI.POST("/explore", ardHandler.ExploreRegistry)
	}
}
