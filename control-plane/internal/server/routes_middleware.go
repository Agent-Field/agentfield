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

	// API key authentication (supports headers + api_key query param)
	// Note: The approval webhook callback is authenticated via HMAC signature,
	// not the global API key. Always bypass API-key auth on that endpoint.
	skipPaths := uniqueStrings(append(append([]string{}, s.config.API.Auth.SkipPaths...), "/api/v1/webhooks/approval-response"))
	s.Router.Use(middleware.APIKeyAuth(middleware.AuthConfig{
		APIKey:    s.config.API.Auth.APIKey,
		SkipPaths: skipPaths,
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
