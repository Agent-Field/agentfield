package middleware

import (
	"crypto/subtle"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

// AuthConfig mirrors server configuration for HTTP authentication.
type AuthConfig struct {
	APIKey                  string
	SkipPaths               []string
	SkipPrefixes            []string
	QueryAPIKeyAllowedPaths []string
}

// APIKeyAuth enforces API key authentication via header or bearer token.
func APIKeyAuth(config AuthConfig) gin.HandlerFunc {
	skipPathSet := make(map[string]struct{}, len(config.SkipPaths))
	for _, p := range config.SkipPaths {
		skipPathSet[p] = struct{}{}
	}
	queryAPIKeyAllowedPathSet := make(map[string]struct{}, len(config.QueryAPIKeyAllowedPaths))
	for _, p := range config.QueryAPIKeyAllowedPaths {
		queryAPIKeyAllowedPathSet[p] = struct{}{}
	}

	return func(c *gin.Context) {
		// No auth configured, allow everything.
		if config.APIKey == "" {
			// Callers have full access in no-auth mode; say so, or the
			// auth-level filtering in agentic discover / smart 404 treats
			// them as "public" and hides every api_key endpoint from the
			// very callers who can use them.
			c.Set("auth_level", "api_key")
			c.Next()
			return
		}

		// Skip explicit paths
		if _, ok := skipPathSet[c.Request.URL.Path]; ok {
			c.Next()
			return
		}
		for _, prefix := range config.SkipPrefixes {
			if strings.HasPrefix(c.Request.URL.Path, prefix) {
				c.Next()
				return
			}
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

		// Allow public DID document resolution (did:web spec requires public access)
		if strings.HasPrefix(c.Request.URL.Path, "/api/v1/did/document/") || strings.HasPrefix(c.Request.URL.Path, "/api/v1/did/resolve/") {
			c.Next()
			return
		}

		// Allow public Knowledge Base access (no secrets, supports adoption)
		if strings.HasPrefix(c.Request.URL.Path, "/api/v1/agentic/kb/") {
			c.Set("auth_level", "public")
			c.Next()
			return
		}

		// Connector routes use their own ConnectorTokenAuth middleware — skip global API key check.
		// Security: ConnectorTokenAuth enforces X-Connector-Token with constant-time comparison,
		// plus per-route ConnectorCapabilityCheck for fine-grained access control.
		if strings.HasPrefix(c.Request.URL.Path, "/api/v1/connector/") {
			c.Next()
			return
		}

		// Public webhook ingest at /sources/:trigger_id — Stripe / GitHub /
		// Slack / generic providers can't be reconfigured to send the
		// AgentField API key, so we bypass the global key check here.
		// Security: each Source plugin enforces its own signature
		// verification (HMAC-SHA256 with constant-time comparison; Stripe
		// and Slack additionally enforce a timestamp-tolerance window).
		// Disabled triggers and unknown trigger_ids are rejected by the
		// handler before any payload work happens.
		if strings.HasPrefix(c.Request.URL.Path, "/sources/") {
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

		// Browser EventSource and WebSocket clients cannot set custom
		// authentication headers, so query-string API key auth is restricted
		// to an explicit streaming route allowlist.
		if apiKey == "" && queryAPIKeyAllowed(c, queryAPIKeyAllowedPathSet) {
			apiKey = c.Query("api_key")
		}

		if subtle.ConstantTimeCompare([]byte(apiKey), []byte(config.APIKey)) != 1 {
			// Set auth level as public for failed auth (used by smart 404 and agentic handlers)
			c.Set("auth_level", "public")
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error":   "unauthorized",
				"message": "invalid or missing API key. Provide via X-API-Key header or Authorization: Bearer <token>",
				"help": map[string]string{
					"kb":              "GET /api/v1/agentic/kb/topics (public, no auth required)",
					"guide":           "GET /api/v1/agentic/kb/guide?goal=<your goal> (public)",
					"api_discovery":   "GET /api/v1/agentic/discover (requires auth)",
					"agent_discovery": "GET /api/v1/discovery/capabilities (requires auth — lists live agents, reasoners, skills)",
				},
			})
			return
		}

		// Set auth level for downstream handlers (used by agentic API for filtering)
		c.Set("auth_level", "api_key")
		if callerAgentID := strings.TrimSpace(c.GetHeader("X-Caller-Agent-ID")); callerAgentID != "" {
			c.Set(string(CallerAgentIDKey), callerAgentID)
		} else if callerAgentID := strings.TrimSpace(c.GetHeader("X-Agent-Node-ID")); callerAgentID != "" {
			c.Set(string(CallerAgentIDKey), callerAgentID)
		}
		c.Next()
	}
}

func queryAPIKeyAllowed(c *gin.Context, allowedPaths map[string]struct{}) bool {
	if len(allowedPaths) == 0 {
		return false
	}
	if _, ok := allowedPaths[c.Request.URL.Path]; ok {
		return true
	}
	if fullPath := c.FullPath(); fullPath != "" {
		_, ok := allowedPaths[fullPath]
		return ok
	}
	return false
}

// AdminTokenAuth enforces a separate admin token for admin routes.
// If adminToken is empty, the middleware is a no-op (falls back to global API key auth).
// Admin tokens must be sent via the X-Admin-Token header only (not Bearer) to avoid
// collision with the API key Bearer token namespace.
func AdminTokenAuth(adminToken string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if adminToken == "" {
			c.Next()
			return
		}

		token := c.GetHeader("X-Admin-Token")

		if subtle.ConstantTimeCompare([]byte(token), []byte(adminToken)) != 1 {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"error":   "forbidden",
				"message": "admin token required for this operation (use X-Admin-Token header)",
			})
			return
		}

		c.Next()
	}
}
