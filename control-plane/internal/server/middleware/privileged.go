package middleware

import (
	"crypto/subtle"
	"net"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

// PrivilegedAccess guards the control-plane endpoints that can execute
// arbitrary code (package install/update/uninstall) or read and write
// credentials (the encrypted secret store and agent .env files).
//
// These routes would otherwise inherit the global posture, where an empty
// API key disables authentication entirely. Combined with the wildcard bind
// in Start(), that leaves anything on the same network able to make the
// control plane clone and build an arbitrary repository, or overwrite the
// credentials an installed agent runs with.
//
// The rule:
//
//	API key configured -> the key is required, like any other protected route.
//	No API key         -> only loopback callers may use these endpoints, so a
//	                      local CLI or desktop app keeps working untouched
//	                      while remote callers are told to configure a key.
//
// The peer address is read from Request.RemoteAddr rather than c.ClientIP():
// gin trusts every proxy by default, so ClientIP() honours a caller-supplied
// X-Forwarded-For header and a remote host could simply claim to be 127.0.0.1.
//
// The corollary is that a reverse proxy sharing a host with the control plane
// makes every forwarded request look local, and the loopback rule then protects
// nothing. Such a deployment has to configure an API key; the docs say so.
func PrivilegedAccess(config AuthConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		if config.APIKey != "" {
			if privilegedKeyMatches(c, config.APIKey) {
				c.Set("auth_level", "api_key")
				c.Next()
				return
			}
			c.Set("auth_level", "public")
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error":   "unauthorized",
				"message": "invalid or missing API key. This endpoint manages packages and credentials; provide the key via X-API-Key header or Authorization: Bearer <token>",
				"help": map[string]string{
					"cli": "af auth login --server <control-plane-url>",
				},
			})
			return
		}

		if isLoopbackRequest(c.Request) {
			c.Next()
			return
		}

		c.Set("auth_level", "public")
		c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
			"error":   "unauthorized",
			"message": "this endpoint installs packages and manages credentials, so it is restricted to local callers while the control plane runs without authentication. Configure an API key to use it from another host.",
			"help": map[string]string{
				"enable_auth": "set AGENTFIELD_API_KEY on the control plane, or api.auth.api_key in ~/.agentfield/agentfield.yaml, then restart it",
				"cli":         "af auth login --server <control-plane-url>",
			},
		})
	}
}

// privilegedKeyMatches reports whether the request carries the configured API
// key. Unlike the global middleware this deliberately does not accept the key
// as a query parameter: that fallback exists for browser EventSource and
// WebSocket clients, which never call these routes, and query strings leak
// into access logs and browser history.
func privilegedKeyMatches(c *gin.Context, configuredKey string) bool {
	provided := c.GetHeader("X-API-Key")
	if provided == "" {
		if authHeader := c.GetHeader("Authorization"); strings.HasPrefix(authHeader, "Bearer ") {
			provided = strings.TrimPrefix(authHeader, "Bearer ")
		}
	}
	return subtle.ConstantTimeCompare([]byte(provided), []byte(configuredKey)) == 1
}

// isLoopbackRequest reports whether the request's TCP peer is the local host.
// An address that cannot be parsed is treated as remote, so an unexpected
// transport fails closed rather than granting privileged access.
func isLoopbackRequest(r *http.Request) bool {
	if r == nil {
		return false
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		host = r.RemoteAddr
	}
	ip := net.ParseIP(strings.TrimSpace(host))
	if ip == nil {
		return false
	}
	return ip.IsLoopback()
}
