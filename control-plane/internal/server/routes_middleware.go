package server

import (
	"context"
	"fmt"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/middleware"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
)

// applyGlobalMiddleware installs CORS, request logging, request timeouts, API
// key auth, and (when enabled) DID auth on the router. It must run before any
// route is registered so that every subsequent route inherits the stack.
func (s *AgentFieldServer) applyGlobalMiddleware() {
	corsConfig := cors.Config{
		AllowOrigins:     s.config.API.CORS.AllowedOrigins,
		AllowMethods:     s.config.API.CORS.AllowedMethods,
		AllowHeaders:     s.config.API.CORS.AllowedHeaders,
		ExposeHeaders:    s.config.API.CORS.ExposedHeaders,
		AllowCredentials: s.config.API.CORS.AllowCredentials,
	}

	// Fallback to defaults if not configured
	if len(corsConfig.AllowOrigins) == 0 {
		corsConfig.AllowOrigins = []string{"http://localhost:3000", "http://localhost:5173"}
	}
	if len(corsConfig.AllowMethods) == 0 {
		corsConfig.AllowMethods = []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}
	}
	if len(corsConfig.AllowHeaders) == 0 {
		corsConfig.AllowHeaders = []string{"Origin", "Content-Type", "Accept", "Authorization", "X-API-Key"}
	}

	s.Router.Use(cors.New(corsConfig))

	s.Router.Use(gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		return fmt.Sprintf("%s - [%s] \"%s %s %s %d %s \"%s\" %s\"\n",
			param.ClientIP,
			param.TimeStamp.Format(time.RFC1123),
			param.Method,
			param.Path,
			param.Request.Proto,
			param.StatusCode,
			param.Latency,
			param.Request.UserAgent(),
			param.ErrorMessage,
		)
	}))

	// Request timeout middleware (1 hour for long-running executions)
	s.Router.Use(func(c *gin.Context) {
		ctx := c.Request.Context()
		timeoutCtx, cancel := context.WithTimeout(ctx, 3600*time.Second)
		defer cancel()

		c.Request = c.Request.WithContext(timeoutCtx)
		c.Next()
	})

	// API key authentication. Header auth is supported for every protected
	// route; query-string auth is restricted to browser streaming endpoints
	// because EventSource and WebSocket clients cannot set custom headers.
	// Note: The approval webhook callback is authenticated via HMAC signature,
	// not the global API key. Always bypass API-key auth on that endpoint.
	skipPaths := append(append([]string{}, s.config.API.Auth.SkipPaths...), "/api/v1/webhooks/approval-response")
	skipPrefixes := []string{}
	if s.config.AgentField.ARD.Enabled && s.config.AgentField.ARD.Publish.Enabled {
		skipPaths = append(skipPaths, "/.well-known/ai-catalog.json")
		skipPrefixes = append(skipPrefixes, "/api/v1/ard/artifacts/")
	}
	if s.config.AgentField.ARD.Enabled && s.config.AgentField.ARD.Registry.Enabled && s.config.AgentField.ARD.Registry.Public {
		skipPrefixes = append(skipPrefixes, "/api/v1/ard/")
	}
	skipPaths = uniqueStrings(skipPaths)
	s.Router.Use(middleware.APIKeyAuth(middleware.AuthConfig{
		APIKey:                  s.config.API.Auth.APIKey,
		SkipPaths:               skipPaths,
		SkipPrefixes:            uniqueStrings(skipPrefixes),
		QueryAPIKeyAllowedPaths: streamingQueryAPIKeyAllowedPaths(),
	}))
	if s.config.API.Auth.APIKey != "" {
		logger.Logger.Info().Msg("🔐 API key authentication enabled")
	}

	// DID authentication middleware (applied globally, but only validates when headers present)
	if s.config.Features.DID.Enabled && s.config.Features.DID.Authorization.DIDAuthEnabled && s.didWebService != nil {
		didAuthConfig := middleware.DIDAuthConfig{
			Enabled:                true,
			TimestampWindowSeconds: s.config.Features.DID.Authorization.TimestampWindowSeconds,
			SkipPaths: []string{
				"/health",
				"/metrics",
				"/api/v1/health",
			},
		}
		s.Router.Use(middleware.DIDAuthMiddleware(s.didWebService, didAuthConfig))
		logger.Logger.Info().Msg("🆔 DID authentication middleware enabled")
	}

	// Warn loudly when the server runs without any authentication: execution-note
	// ownership (and any other identity-dependent guard) cannot be enforced
	// because no trusted caller identity is established for incoming requests.
	if !s.noteOwnershipEnforced() {
		logger.Logger.Warn().Msg("⚠️  No authentication configured (API key and DID auth both disabled): execution-note ownership is NOT enforced. Enable API key or DID auth to protect execution notes from cross-agent reads and writes.")
	}
}

// noteOwnershipEnforced reports whether the server runs with an authentication
// method that establishes a trusted caller identity (API key or DID auth).
// Execution-note ownership can only be enforced when this is true; in a fully
// unauthenticated deployment there is no trustworthy caller identity to compare
// against, so the ownership guard is skipped.
func (s *AgentFieldServer) noteOwnershipEnforced() bool {
	if s.config.API.Auth.APIKey != "" {
		return true
	}
	return s.config.Features.DID.Enabled && s.config.Features.DID.Authorization.DIDAuthEnabled && s.didWebService != nil
}

func streamingQueryAPIKeyAllowedPaths() []string {
	return []string{
		"/api/ui/v1/nodes/events",
		"/api/ui/v1/executions/events",
		"/api/ui/v1/executions/:execution_id/logs/stream",
		"/api/ui/v1/workflows/:workflowId/notes/events",
		"/api/ui/v1/reasoners/events",
		"/api/v1/executions/:execution_id/events",
		"/api/v1/memory/events/ws",
		"/api/v1/memory/events/sse",
		"/api/v1/triggers/:trigger_id/events/stream",
	}
}

func uniqueStrings(in []string) []string {
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}
