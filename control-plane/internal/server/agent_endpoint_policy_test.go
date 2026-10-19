package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/agenturl"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

// C9: opening an existing database and recovering the health registry tolerate
// invalid historical endpoints; listing works while polling refuses the URL.
// Constructor/recovery are exercised without starting listeners on wildcard IPs.
func TestAgentEndpointLegacyStartup(t *testing.T) {
	cfg := baseConfigForDBTests()
	cfg.UI.Enabled = false
	cfg.Features.DID.Enabled = false
	cfg.Features.Connector.Enabled = false
	cfg.API.Auth.APIKey = ""
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("AGENTFIELD_CONFIG_SOURCE", "")
	cfg.Storage.Local.DatabasePath = filepath.Join(home, "legacy.db")
	cfg.Storage.Local.KVStorePath = filepath.Join(home, "legacy.bolt")
	ctx := context.Background()
	seed := storage.NewLocalStorage(cfg.Storage.Local)
	require.NoError(t, seed.Initialize(ctx, storage.StorageConfig{Mode: "local", Local: cfg.Storage.Local}))
	require.NoError(t, seed.RegisterAgent(ctx, &types.AgentNode{ID: "legacy", BaseURL: "http://169.254.169.254/prefix?bad", HealthStatus: types.HealthStatusActive, LifecycleStatus: types.AgentStatusReady, RegisteredAt: time.Now(), LastHeartbeat: time.Now()}))
	require.NoError(t, seed.Close(ctx))
	srv, err := NewAgentFieldServer(&cfg)
	require.NoError(t, err)
	defer func() { require.NoError(t, srv.Stop()); agenturl.Configure(false, nil) }()
	require.NoError(t, srv.healthMonitor.RecoverFromDatabase(ctx))
	router := gin.New()
	router.GET("/nodes", handlers.ListNodesHandler(srv.storage))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/nodes?show_all=true", nil))
	require.Equal(t, 200, rec.Code)
	require.Contains(t, rec.Body.String(), "legacy")
	_, err = srv.agentClient.GetAgentStatus(ctx, "legacy")
	require.ErrorContains(t, err, "query")
}

// Review finding 5: constructor wiring must apply both strict mode and its
// configured exceptions, even if the previous package-level policy was default.
func TestReviewStartupAppliesAgentEndpointPolicy(t *testing.T) {
	agenturl.Configure(false, nil)
	defer agenturl.Configure(false, nil)
	cfg := baseConfigForDBTests()
	cfg.UI.Enabled = false
	cfg.Features.DID.Enabled = false
	cfg.Features.Connector.Enabled = false
	cfg.API.Auth.APIKey = ""
	cfg.AgentField.Registration.AgentEndpointStrict = true
	cfg.AgentField.Registration.AgentEndpointAllowedHosts = []string{"127.0.0.2/32"}
	home := t.TempDir()
	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("AGENTFIELD_CONFIG_SOURCE", "")
	cfg.Storage.Local.DatabasePath = filepath.Join(home, "policy.db")
	cfg.Storage.Local.KVStorePath = filepath.Join(home, "policy.bolt")
	srv, err := NewAgentFieldServer(&cfg)
	require.NoError(t, err)
	defer func() { require.NoError(t, srv.Stop()) }()
	_, err = agenturl.Validate(context.Background(), "http://127.0.0.1:18190")
	require.ErrorContains(t, err, "strict mode")
	_, err = agenturl.Validate(context.Background(), "http://127.0.0.2:18190")
	require.NoError(t, err)
}
