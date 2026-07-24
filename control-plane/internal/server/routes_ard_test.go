package server

import (
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/gin-gonic/gin"
)

func TestRegisterARDRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	srv := &AgentFieldServer{
		Router: gin.New(),
		config: &config.Config{
			AgentField: config.AgentFieldConfig{
				ARD: config.ARDConfig{Enabled: true, Publish: config.ARDPublishConfig{Enabled: true}},
			},
			Features: config.FeatureConfig{
				DID: config.DIDConfig{Enabled: true},
			},
		},
	}

	srv.registerARDPublicRoutes()
	srv.registerARDRoutes(srv.Router.Group("/api/v1"))
	handler := srv.newARDHandler()
	if !handler.ReadConfig().Enabled {
		t.Fatal("new ARD handler did not read server config")
	}
	if handler.DIDAvailable() {
		t.Fatal("DID should require an initialized DID service")
	}

	seen := map[string]string{}
	for _, route := range srv.Router.Routes() {
		seen[route.Method+" "+route.Path] = route.Handler
	}
	for _, route := range []string{
		"GET /.well-known/ai-catalog.json",
		"GET /api/v1/ard/artifacts/:entryID",
		"POST /api/v1/ard/search",
		"GET /api/v1/ard/agents",
		"POST /api/v1/ard/explore",
	} {
		if seen[route] == "" {
			t.Fatalf("missing ARD route %s in %#v", route, seen)
		}
	}
}
