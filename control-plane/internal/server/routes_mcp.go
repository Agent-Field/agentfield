package server

import (
	"net/http"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"

	"github.com/gin-gonic/gin"
)

// buildVersion is the control plane's release version, surfaced by the embedded
// MCP server's serverInfo. Set once at startup by the server binaries via
// SetBuildVersion; defaults to "dev".
var buildVersion = "dev"

// SetBuildVersion records the control plane's build version for surfaces that
// report it (currently the embedded MCP server's serverInfo). Call once at
// startup, before NewAgentFieldServer.
func SetBuildVersion(v string) {
	if v = strings.TrimSpace(v); v != "" {
		buildVersion = v
	}
}

// registerMCPRoutes installs the embedded Model Context Protocol server at /mcp,
// served on the same port and behind the same global auth/trust domain as the
// REST API. It is enabled by default; set AGENTFIELD_MCP_ENABLED=false to
// disable, in which case the route is not registered and /mcp returns 404.
//
// The endpoint speaks streamable-HTTP JSON-RPC 2.0 over POST. GET returns 405
// (it is not a valid transport verb here) and OPTIONS answers preflight/probe.
func (s *AgentFieldServer) registerMCPRoutes() {
	if !s.config.Features.MCP.IsEnabled() {
		logger.Logger.Info().Msg("🧩 Embedded MCP server disabled (AGENTFIELD_MCP_ENABLED=false)")
		return
	}

	handler := handlers.MCPHandler(
		s.storage,
		s.payloadStore,
		s.webhookDispatcher,
		s.config.AgentField.ExecutionQueue.AgentCallTimeout,
		s.config.Features.DID.Authorization.InternalToken,
		buildVersion,
	)

	s.Router.POST("/mcp", handler)
	s.Router.GET("/mcp", func(c *gin.Context) {
		c.Header("Allow", "POST, OPTIONS")
		c.JSON(http.StatusMethodNotAllowed, gin.H{
			"error": "method not allowed; POST a JSON-RPC 2.0 message to /mcp",
		})
	})
	s.Router.OPTIONS("/mcp", func(c *gin.Context) {
		c.Header("Allow", "POST, OPTIONS")
		c.Status(http.StatusNoContent)
	})

	logger.Logger.Info().Msg("🧩 Embedded MCP server registered at POST /mcp")
}
