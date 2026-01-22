package middleware

import (
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

// AuthConfig mirrors server configuration for HTTP authentication.
type AuthConfig struct {
	APIKey    string
	SkipPaths []string
}

// APIKeyAuth enforces API key authentication via header, bearer token, or query param.
func APIKeyAuth(config AuthConfig) gin.HandlerFunc {
	skipPathSet := make(map[string]struct{}, len(config.SkipPaths))
	for _, p := range config.SkipPaths {
		skipPathSet[p] = struct{}{}
	}

	return func(c *gin.Context) {
		// No auth configured, allow everything.
		if config.APIKey == "" {
			c.Next()
			return
		}

		// Skip explicit paths
		if _, ok := skipPathSet[c.Request.URL.Path]; ok {
			c.Next()
			return
		}

		// Always allow health and metrics by default
		if strings.HasPrefix(c.Request.URL.Path, "/api/v1/health") || c.Request.URL.Path == "/health" || c.Request.URL.Path == "/metrics" {
			c.Next()
			return
		}

		// Allow UI static files to load (the React app handles auth prompting)
		// Also allow root "/" which redirects to /ui/
		if strings.HasPrefix(c.Request.URL.Path, "/ui") || c.Request.URL.Path == "/" {
			c.Next()
			return
		}

		apiKey := ""

		// Preferred: X-API-Key header
		apiKey = c.GetHeader("X-API-Key")

		// Fallback: Authorization: Bearer <token>
		if apiKey == "" {
			authHeader := c.GetHeader("Authorization")
			if strings.HasPrefix(authHeader, "Bearer ") {
				apiKey = strings.TrimPrefix(authHeader, "Bearer ")
			}
		}

		// SSE/WebSocket friendly: api_key query parameter
		if apiKey == "" {
			apiKey = c.Query("api_key")
		}

		if apiKey != config.APIKey {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error":   "unauthorized",
				"message": "invalid or missing API key",
			})
			return
		}

		c.Next()
	}
}
