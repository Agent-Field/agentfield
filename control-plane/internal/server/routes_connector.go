package server

import (
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	connectorpkg "github.com/Agent-Field/agentfield/control-plane/internal/handlers/connector"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"

	"github.com/gin-gonic/gin"
)

// registerConnectorRoutes installs the /api/v1/connector/* surface used by
// external connector integrations. The group is only registered when the
// connector feature is enabled and a token is configured; all routes are
// authenticated with ConnectorTokenAuth and config-management sub-routes are
// additionally capability-gated.
func (s *AgentFieldServer) registerConnectorRoutes(agentAPI *gin.RouterGroup) {
	if !(s.config.Features.Connector.Enabled && s.config.Features.Connector.Token != "") {
		return
	}

	connectorGroup := agentAPI.Group("/connector")
	connectorGroup.Use(middleware.ConnectorTokenAuth(s.config.Features.Connector.Token))

	connectorHandlers := connectorpkg.NewHandlers(
		s.config.Features.Connector,
		s.storage,
		s.statusManager,
		s.accessPolicyService,
		s.tagApprovalService,
		s.didService,
	)
	connectorHandlers.RegisterRoutes(connectorGroup)

	// Config management routes for connector
	configGroup := connectorGroup.Group("")
	configGroup.Use(middleware.ConnectorCapabilityCheck("config_management", s.config.Features.Connector.Capabilities))
	{
		configHandlers := handlers.NewConfigStorageHandlers(s.storage, s.configReloadFn())
		configHandlers.RegisterRoutes(configGroup)
	}

	logger.Logger.Info().Msg("🔌 Connector routes registered")
}
