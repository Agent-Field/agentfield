package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Contract A1: APIKeyAuth lets /api/v1/nodes/<id>/link pass without a global
// API key because nodelink.Hub authenticates the WebSocket upgrade itself,
// while refusing paths that only look similar (extra segments, missing node
// segment, different prefix, or different suffix).
func TestAPIKeyAuthNodeLink_A1_AllowsNodeLinkUpgradeAndRejectsLookalikes(t *testing.T) {
	router := gin.New()
	router.Use(APIKeyAuth(AuthConfig{APIKey: "secret-key"}))
	router.GET("/*path", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"passed": true})
	})

	for _, allowedPath := range []string{
		"/api/v1/nodes/worker-1/link",
		"/api/v1/nodes/codeaf-factory/link",
	} {
		req := httptest.NewRequest(http.MethodGet, allowedPath, nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusOK, rec.Code, "node link path %s must pass APIKeyAuth without X-API-Key", allowedPath)
	}

	for _, rejectedPath := range []string{
		"/api/v1/nodes/worker-1/nested/link",
		"/api/v1/nodes/worker-1/link/extra",
		"/api/v1/nodes/worker-1/status",
		"/api/v1/nodes/worker-1/links",
		"/api/v1/nodes/link",
		"/api/v1/agents/worker-1/link",
	} {
		req := httptest.NewRequest(http.MethodGet, rejectedPath, nil)
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		assert.Equal(t, http.StatusUnauthorized, rec.Code, "lookalike path %s must still require an API key", rejectedPath)
	}
}
