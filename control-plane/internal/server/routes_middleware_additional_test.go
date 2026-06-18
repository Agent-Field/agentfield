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

func TestAPIKeyAuth_ARDPublicRoutesRespectFeatureGates(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		ard        config.ARDConfig
		path       string
		wantStatus int
	}{
		{
			name: "catalog requires publish gate",
			ard: config.ARDConfig{
				Enabled: true,
				Publish: config.ARDPublishConfig{
					Enabled: true,
				},
			},
			path:       "/.well-known/ai-catalog.json",
			wantStatus: http.StatusOK,
		},
		{
			name: "catalog stays private when publish disabled",
			ard: config.ARDConfig{
				Enabled: true,
				Publish: config.ARDPublishConfig{
					Enabled: false,
				},
			},
			path:       "/.well-known/ai-catalog.json",
			wantStatus: http.StatusUnauthorized,
		},
		{
			name: "registry requires public registry gate",
			ard: config.ARDConfig{
				Enabled: true,
				Registry: config.ARDRegistryConfig{
					Enabled: true,
					Public:  true,
				},
			},
			path:       "/api/v1/ard/search",
			wantStatus: http.StatusOK,
		},
		{
			name: "registry stays private when public disabled",
			ard: config.ARDConfig{
				Enabled: true,
				Registry: config.ARDRegistryConfig{
					Enabled: true,
					Public:  false,
				},
			},
			path:       "/api/v1/ard/search",
			wantStatus: http.StatusUnauthorized,
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			gin.SetMode(gin.TestMode)
			cfg := &config.Config{
				AgentField: config.AgentFieldConfig{ARD: tc.ard},
				API: config.APIConfig{
					Auth: config.AuthConfig{APIKey: "secret-key"},
				},
			}
			srv := &AgentFieldServer{
				Router: gin.New(),
				config: cfg,
			}
			srv.applyGlobalMiddleware()
			srv.Router.GET("/.well-known/ai-catalog.json", func(c *gin.Context) { c.String(http.StatusOK, "catalog") })
			srv.Router.POST("/api/v1/ard/search", func(c *gin.Context) { c.String(http.StatusOK, "search") })

			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodGet, tc.path, nil)
			if tc.path == "/api/v1/ard/search" {
				req = httptest.NewRequest(http.MethodPost, tc.path, nil)
			}
			srv.Router.ServeHTTP(rec, req)
			require.Equal(t, tc.wantStatus, rec.Code)
		})
	}
}
