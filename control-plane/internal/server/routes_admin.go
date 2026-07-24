package server

import (
	"net/http/pprof"

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

// registerPprofRoutes installs Go pprof endpoints under /debug/pprof/, gated
// by the admin token from the DID Authorization config. When no admin token
// is configured, they fall back to global API-key authentication. Operators
// should configure an admin token before exposing pprof outside trusted networks.
func (s *AgentFieldServer) registerPprofRoutes() {
	adminToken := s.config.Features.DID.Authorization.AdminToken

	pprofGroup := s.Router.Group("/debug/pprof")
	pprofGroup.Use(middleware.AdminTokenAuth(adminToken))

	pprofGroup.GET("/", gin.WrapF(pprof.Index))
	pprofGroup.GET("/cmdline", gin.WrapF(pprof.Cmdline))
	pprofGroup.GET("/profile", gin.WrapF(pprof.Profile))
	pprofGroup.GET("/symbol", gin.WrapF(pprof.Symbol))
	pprofGroup.POST("/symbol", gin.WrapF(pprof.Symbol))
	pprofGroup.GET("/trace", gin.WrapF(pprof.Trace))

	pprofGroup.Any("/:name", func(c *gin.Context) {
		pprof.Handler(c.Param("name")).ServeHTTP(c.Writer, c.Request)
	})

	if adminToken == "" {
		logger.Logger.Warn().Msg("SECURITY WARNING: pprof debug endpoints have no admin token configured and rely on global API-key authentication. Set AGENTFIELD_AUTHORIZATION_ADMIN_TOKEN before exposing them outside trusted networks.")
	} else {
		logger.Logger.Info().Msg("pprof debug endpoints registered (admin-token gated)")
	}
}
