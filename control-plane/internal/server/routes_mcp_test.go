package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/server/apicatalog"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// newMCPTestServer wires a real router through setupRoutes with the MCP toggle
// set as requested, so we exercise the actual route registration rather than a
// hand-rolled handler.
func newMCPTestServer(t *testing.T, mcpEnabled *bool) *AgentFieldServer {
	t.Helper()
	gin.SetMode(gin.TestMode)
	srv := &AgentFieldServer{
		Router:            gin.New(),
		storage:           newStubStorage(),
		payloadStore:      &stubPayloadStore{},
		webhookDispatcher: &stubWebhookDispatcher{},
		apiCatalog:        apicatalog.New(),
		config: &config.Config{
			UI:  config.UIConfig{Enabled: false},
			API: config.APIConfig{},
			Features: config.FeatureConfig{
				MCP: config.MCPConfig{Enabled: mcpEnabled},
			},
		},
	}
	srv.setupRoutes()
	return srv
}

func TestMCPRoutes_EnabledByDefault(t *testing.T) {
	srv := newMCPTestServer(t, nil) // nil => default on

	t.Run("POST /mcp handles initialize", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/mcp",
			strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18"}}`))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		srv.Router.ServeHTTP(w, req)

		require.Equal(t, http.StatusOK, w.Code)
		require.Contains(t, w.Body.String(), `"serverInfo"`)
		require.Contains(t, w.Body.String(), `"agentfield"`)
	})

	t.Run("GET /mcp returns 405", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/mcp", nil)
		w := httptest.NewRecorder()
		srv.Router.ServeHTTP(w, req)
		require.Equal(t, http.StatusMethodNotAllowed, w.Code)
	})
}

func TestMCPRoutes_DisabledReturns404(t *testing.T) {
	disabled := false
	srv := newMCPTestServer(t, &disabled)

	req := httptest.NewRequest(http.MethodPost, "/mcp",
		strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"initialize"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Router.ServeHTTP(w, req)

	require.Equal(t, http.StatusNotFound, w.Code)
}

func TestSetBuildVersion(t *testing.T) {
	original := buildVersion
	defer func() { buildVersion = original }()

	buildVersion = "dev"
	SetBuildVersion("  ") // blank must not overwrite
	require.Equal(t, "dev", buildVersion)

	SetBuildVersion("v9.9.9")
	require.Equal(t, "v9.9.9", buildVersion)
}
