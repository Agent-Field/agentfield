package server

import (
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestPackageRoutesRegistered(t *testing.T) {
	gin.SetMode(gin.TestMode)
	srv := &AgentFieldServer{
		Router:            gin.New(),
		storage:           newStubStorage(),
		payloadStore:      &stubPayloadStore{},
		webhookDispatcher: &stubWebhookDispatcher{},
		config: &config.Config{
			UI: config.UIConfig{Enabled: true},
		},
	}

	srv.registerUIAPI()

	routes := make(map[string]struct{})
	for _, route := range srv.Router.Routes() {
		routes[route.Method+" "+route.Path] = struct{}{}
	}
	for _, want := range []string{
		"GET /api/ui/v1/agents/packages",
		"POST /api/ui/v1/agents/packages/install",
		"GET /api/ui/v1/agents/packages/install/jobs",
		"GET /api/ui/v1/agents/packages/install/jobs/:jobId",
		"POST /api/ui/v1/agents/packages/:packageId/uninstall",
		"POST /api/ui/v1/agents/packages/:packageId/update",
	} {
		_, ok := routes[want]
		require.Truef(t, ok, "route %q is not registered; routes=%v", want, srv.Router.Routes())
	}
}
