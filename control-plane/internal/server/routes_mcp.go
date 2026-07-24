package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"

	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"

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
		s.mcpAuthorizer(),
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

// mcpAuthorizer applies precisely the same permission middleware as REST
// execution. MCP targets live in a JSON-RPC argument rather than the URL, so a
// small synthetic request supplies the target and flat input to the shared
// middleware without trusting any client-supplied identity headers.
func (s *AgentFieldServer) mcpAuthorizer() handlers.MCPAuthorizer {
	if !s.config.Features.DID.Authorization.Enabled || s.accessPolicyService == nil || s.didWebService == nil {
		return nil
	}
	permission := middleware.PermissionCheckMiddleware(
		s.accessPolicyService, s.tagVCVerifier, s.storage, s.didWebService,
		middleware.PermissionConfig{Enabled: true, DefaultDeny: s.config.Features.DID.Authorization.DefaultDeny},
	)
	return func(ctx context.Context, callerDID, target string, input map[string]interface{}) (string, error) {
		body, err := json.Marshal(map[string]interface{}{"input": input})
		if err != nil {
			return "", fmt.Errorf("encode permission input: %w", err)
		}
		request := httptest.NewRequest(http.MethodPost, "/api/v1/execute/"+target, bytes.NewReader(body)).WithContext(ctx)
		recorder := httptest.NewRecorder()
		checkCtx, _ := gin.CreateTestContext(recorder)
		checkCtx.Request = request
		checkCtx.Params = gin.Params{{Key: "target", Value: target}}
		if callerDID != "" {
			checkCtx.Set(string(middleware.VerifiedCallerDIDKey), callerDID)
		}
		permission(checkCtx)
		if recorder.Code >= http.StatusBadRequest || checkCtx.IsAborted() {
			return "", fmt.Errorf("permission check rejected target")
		}
		return middleware.GetTargetDID(checkCtx), nil
	}
}
