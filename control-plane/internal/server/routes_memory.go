package server

import (
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"

	"github.com/gin-gonic/gin"
)

// registerMemoryRoutes installs the /api/v1/memory surface: key-value memory,
// vector memory (RESTful + legacy), and memory-event streams.
func (s *AgentFieldServer) registerMemoryRoutes(agentAPI *gin.RouterGroup) {
	memoryGroup := agentAPI.Group("/memory")
	{
		// Apply memory permission middleware if authorization is enabled
		if s.config.Features.DID.Authorization.Enabled && s.accessPolicyService != nil && s.didWebService != nil {
			memPermConfig := middleware.MemoryPermissionConfig{
				Enabled:               true,
				EnforceScopeOwnership: true,
			}
			memoryGroup.Use(middleware.MemoryPermissionMiddleware(
				s.accessPolicyService,
				s.storage,
				s.didWebService,
				memPermConfig,
			))
			logger.Logger.Info().Msg("Memory permission middleware enabled on memory endpoints")
		}

		// Key-value memory endpoints
		memoryGroup.POST("/set", handlers.SetMemoryHandler(s.storage))
		memoryGroup.POST("/get", handlers.GetMemoryHandler(s.storage))
		memoryGroup.POST("/delete", handlers.DeleteMemoryHandler(s.storage))
		memoryGroup.GET("/list", handlers.ListMemoryHandler(s.storage))

		// Vector Memory endpoints (RESTful)
		memoryGroup.POST("/vector", handlers.SetVectorHandler(s.storage))
		memoryGroup.GET("/vector/:key", handlers.GetVectorHandler(s.storage))
		memoryGroup.POST("/vector/search", handlers.SimilaritySearchHandler(s.storage))
		memoryGroup.DELETE("/vector/:key", handlers.DeleteVectorHandler(s.storage))

		// Legacy Vector Memory endpoints (for backward compatibility)
		memoryGroup.POST("/vector/set", handlers.SetVectorHandler(s.storage))
		memoryGroup.POST("/vector/delete", handlers.DeleteVectorHandler(s.storage))
		memoryGroup.DELETE("/vector/namespace", handlers.DeleteNamespaceVectorsHandler(s.storage))

		// Memory events endpoints
		memoryEventsHandler := handlers.NewMemoryEventsHandler(s.storage)
		memoryGroup.GET("/events/ws", memoryEventsHandler.WebSocketHandler)
		memoryGroup.GET("/events/sse", memoryEventsHandler.SSEHandler)
		memoryGroup.GET("/events/history", handlers.GetEventHistoryHandler(s.storage))
	}
}
