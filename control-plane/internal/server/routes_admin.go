package server

import (
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers/admin"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"

	"github.com/gin-gonic/gin"
)

// registerAdminRoutes installs admin-authenticated routes under /api/v1. When
// DID authorization is enabled, tag approval and access policy endpoints are
// wired onto a dedicated group gated by AdminTokenAuth. Config storage routes
// are always registered (they carry their own auth via the handler package).
func (s *AgentFieldServer) registerAdminRoutes(agentAPI *gin.RouterGroup) {
	// Admin routes for tag approval and access policy management (VC-based authorization)
	if s.config.Features.DID.Authorization.Enabled {
		adminGroup := agentAPI.Group("")
		adminGroup.Use(middleware.AdminTokenAuth(s.config.Features.DID.Authorization.AdminToken))

		// Tag approval admin routes
		if s.tagApprovalService != nil {
			tagApprovalHandlers := admin.NewTagApprovalHandlers(s.tagApprovalService, s.storage)
			tagApprovalHandlers.RegisterRoutes(adminGroup)
		}

		// Access policy admin routes
		if s.accessPolicyService != nil {
			accessPolicyHandlers := admin.NewAccessPolicyHandlers(s.accessPolicyService)
			accessPolicyHandlers.RegisterRoutes(adminGroup)
		}

		logger.Logger.Info().Msg("📋 Authorization admin routes registered")
	}

	// Config storage routes (admin-authenticated)
	{
		configHandlers := handlers.NewConfigStorageHandlers(s.storage, s.configReloadFn())
		configHandlers.RegisterRoutes(agentAPI)
		logger.Logger.Info().Msg("Config storage routes registered")
	}
}
