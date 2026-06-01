package server

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestAPIKeyAuth_SkipsApprovalWebhookPath(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	cfg := &config.Config{
		API: config.APIConfig{
			Auth: config.AuthConfig{
				APIKey:    "secret-key",
				SkipPaths: nil,
			},
		},
	}

	srv := &AgentFieldServer{
		Router: gin.New(),
		config: cfg,
	}
	srv.applyGlobalMiddleware()

	// Register routes after middleware so they inherit the auth stack.
	srv.Router.POST("/api/v1/webhooks/approval-response", func(c *gin.Context) { c.String(http.StatusOK, "ok") })
	srv.Router.GET("/api/v1/nodes", func(c *gin.Context) { c.String(http.StatusOK, "nodes") })

	// Webhook endpoint should bypass API-key auth.
	recWebhook := httptest.NewRecorder()
	reqWebhook := httptest.NewRequest(http.MethodPost, "/api/v1/webhooks/approval-response", nil)
	srv.Router.ServeHTTP(recWebhook, reqWebhook)
	require.Equal(t, http.StatusOK, recWebhook.Code)

	// Non-skipped endpoints should still require API key.
	recNodes := httptest.NewRecorder()
	reqNodes := httptest.NewRequest(http.MethodGet, "/api/v1/nodes", nil)
	srv.Router.ServeHTTP(recNodes, reqNodes)
	require.Equal(t, http.StatusUnauthorized, recNodes.Code)
}

