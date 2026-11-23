package server

import (
	"context"
	"net"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestServerHostRestrictsBothListeners(t *testing.T) {
	t.Setenv("AGENTFIELD_HOME", t.TempDir())
	t.Setenv("AGENTFIELD_CONFIG_SOURCE", "")
	t.Setenv("AGENTFIELD_TELEMETRY_ENABLED", "false")
	for _, host := range []string{"127.0.0.1", "::1", ""} {
		t.Run(host, func(t *testing.T) {
			cfg := baseConfigForDBTests()
			require.NoError(t, yaml.Unmarshal([]byte("agentfield:\n  host: '"+host+"'\n"), &cfg))
			cfg.AgentField.Port = 0
			cfg.AgentField.ShutdownTimeout = time.Second
			cfg.UI.Enabled, cfg.Features.DID.Enabled, cfg.Features.Connector.Enabled = false, false, false
			disabled := false
			cfg.Telemetry.Enabled = &disabled
			srv := newRouteTestServer(&cfg)
			srv.agentfieldHome = t.TempDir()
			srv.streamCtx = context.Background()
			srv.maintenanceReady = make(chan struct{})
			srv.statusManager = services.NewStatusManager(srv.storage, services.StatusManagerConfig{
				ReconcileInterval: time.Hour, StatusCacheTTL: time.Minute, MaxTransitionTime: time.Minute,
			}, nil, nil)
			srv.healthMonitor = services.NewHealthMonitor(srv.storage, services.HealthMonitorConfig{
				CheckInterval: time.Hour, CheckTimeout: time.Second,
			}, nil, nil, srv.statusManager, nil)
			srv.cleanupService = handlers.NewExecutionCleanupService(srv.storage, cfg.AgentField.ExecutionCleanup)
			srv.adminGRPCPort = 0
			done := make(chan error, 1)
			go func() { done <- srv.Start() }()
			t.Cleanup(func() {
				require.NoError(t, srv.Stop())
				select {
				case err := <-done:
					require.NoError(t, err)
				case <-time.After(5 * time.Second):
					t.Error("server did not stop")
				}
			})
			select {
			case <-srv.maintenanceReady:
			case err := <-done:
				t.Fatalf("server stopped before binding: %v", err)
			case <-time.After(5 * time.Second):
				t.Fatal("server did not bind")
			}
			assert.Equal(t, net.JoinHostPort(host, "0"), srv.getHTTPServer().Addr)
			actual := srv.adminListener.Addr().(*net.TCPAddr).IP
			if host == "" {
				assert.True(t, actual.IsUnspecified())
			} else {
				assert.True(t, net.ParseIP(host).Equal(actual), "admin gRPC bound %s, want %s", actual, host)
			}
		})
	}
}
