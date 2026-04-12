package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type forwarderStub struct {
	stopErr error
}

func (f *forwarderStub) Start(context.Context) error                    { return nil }
func (f *forwarderStub) Stop(context.Context) error                     { return f.stopErr }
func (f *forwarderStub) ReloadConfig(context.Context) error             { return nil }
func (f *forwarderStub) GetStatus() types.ObservabilityForwarderStatus  { return types.ObservabilityForwarderStatus{} }
func (f *forwarderStub) Redrive(context.Context) types.ObservabilityRedriveResponse {
	return types.ObservabilityRedriveResponse{}
}

func TestNewAgentFieldServerCoversFallbacksAndOptionalServices(t *testing.T) {
	cfg := baseConfigForDBTests()
	cfg.UI.Enabled = false
	cfg.API.Auth.APIKey = ""
	cfg.Features.Connector.Enabled = false
	cfg.Features.DID.Enabled = true
	cfg.Features.DID.KeyAlgorithm = "Ed25519"
	cfg.Features.DID.Authorization.Enabled = true
	cfg.Features.DID.Authorization.Domain = ""
	cfg.Features.DID.Authorization.TagApprovalRules.DefaultMode = "auto"
	cfg.Features.DID.Authorization.AccessPolicies = []config.AccessPolicyConfig{
		{
			Name:           "allow-observe",
			CallerTags:     []string{"ops"},
			TargetTags:     []string{"prod"},
			AllowFunctions: []string{"observe"},
			Action:         "allow",
			Priority:       10,
		},
	}
	cfg.Features.DID.Keystore.Path = filepath.Join(t.TempDir(), "keys")
	cfg.Features.DID.Keystore.EncryptionPassphrase = "passphrase"
	cfg.Features.Tracing.Enabled = true
	cfg.Features.Tracing.Insecure = true
	cfg.Features.Tracing.ServiceName = "agentfield-test"

	storageDir := t.TempDir()
	cfg.Storage.Local.DatabasePath = filepath.Join(storageDir, "server.db")
	cfg.Storage.Local.KVStorePath = filepath.Join(storageDir, "server.bolt")

	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("AGENTFIELD_HOME", "")
	t.Setenv("AGENTFIELD_CONFIG_SOURCE", "db")

	srv, err := NewAgentFieldServer(&cfg)
	require.NoError(t, err)
	require.Equal(t, filepath.Join(home, ".agentfield"), srv.agentfieldHome)
	require.NotNil(t, srv.didService)
	require.NotNil(t, srv.didWebService)
	require.NotNil(t, srv.accessPolicyService)
	require.NotNil(t, srv.tagApprovalService)
	require.NotNil(t, srv.tagVCVerifier)
	require.NotNil(t, srv.executionTracer)
	require.NotNil(t, srv.tracerShutdown)

	stopServerBackgrounds(t, srv)
}

func TestNewAgentFieldServerContinuesAfterInvalidDBOverlay(t *testing.T) {
	cfg := baseConfigForDBTests()
	cfg.UI.Enabled = false
	cfg.Features.DID.Enabled = false
	cfg.Features.Connector.Enabled = false

	home := t.TempDir()
	cfg.Storage.Local.DatabasePath = filepath.Join(home, "server.db")
	cfg.Storage.Local.KVStorePath = filepath.Join(home, "server.bolt")

	ctx := context.Background()
	provider := storage.NewLocalStorage(storage.LocalStorageConfig{})
	require.NoError(t, provider.Initialize(ctx, cfg.Storage))
	require.NoError(t, provider.SetConfig(ctx, dbConfigKey, "agentfield: [", "tester"))
	require.NoError(t, provider.Close(ctx))

	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("AGENTFIELD_CONFIG_SOURCE", "db")

	srv, err := NewAgentFieldServer(&cfg)
	require.NoError(t, err)
	require.Equal(t, 8080, srv.config.AgentField.Port)

	stopServerBackgrounds(t, srv)
}

func TestStartAndStopCoverAdditionalBranches(t *testing.T) {
	cfg := baseConfigForDBTests()
	cfg.UI.Enabled = false
	cfg.Features.DID.Enabled = false
	cfg.Features.Connector.Enabled = false
	cfg.API.Auth.APIKey = ""
	cfg.Features.Tracing.Enabled = true
	cfg.Features.Tracing.Insecure = true
	cfg.AgentField.LLMHealth.Enabled = true
	cfg.AgentField.LLMHealth.CheckInterval = time.Hour
	cfg.AgentField.LLMHealth.Endpoints = []config.LLMEndpoint{{
		Name: "stub",
		URL:  "http://127.0.0.1:1/health",
	}}

	home := t.TempDir()
	cfg.Storage.Local.DatabasePath = filepath.Join(home, "server.db")
	cfg.Storage.Local.KVStorePath = filepath.Join(home, "server.bolt")

	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("AGENTFIELD_CONFIG_SOURCE", "")

	srv, err := NewAgentFieldServer(&cfg)
	require.NoError(t, err)

	srv.agentfieldHome = filepath.Join(home, "missing-home")
	srv.config.AgentField.Port = -1

	err = srv.Start()
	require.Error(t, err)
	require.NotNil(t, srv.adminGRPCServer)
	require.NotNil(t, srv.adminListener)

	cancelCalled := false
	srv.registryWatcherCancel = func() { cancelCalled = true }
	srv.observabilityForwarder = &forwarderStub{stopErr: errors.New("stop failed")}
	srv.tracerShutdown = func(context.Context) error { return errors.New("shutdown failed") }

	if srv.llmHealthMonitor != nil {
		srv.llmHealthMonitor.Stop()
	}

	require.NoError(t, srv.Stop())
	require.True(t, cancelCalled)
}

func TestStartCoversRecoveryErrorBranches(t *testing.T) {
	cfg := baseConfigForDBTests()
	cfg.UI.Enabled = false
	cfg.Features.DID.Enabled = false
	cfg.Features.Connector.Enabled = false
	cfg.API.Auth.APIKey = ""

	errStorage := &listAgentsStorage{stubStorage: newStubStorage(), err: errors.New("list failed")}
	statusManager := services.NewStatusManager(errStorage, services.StatusManagerConfig{
		ReconcileInterval: 30 * time.Second,
		StatusCacheTTL:    time.Minute,
		MaxTransitionTime: time.Minute,
	}, nil, nil)
	uiService := services.NewUIService(errStorage, nil, nil, statusManager)
	healthMonitor := services.NewHealthMonitor(errStorage, services.HealthMonitorConfig{
		CheckInterval: time.Hour,
		CheckTimeout:  time.Second,
	}, uiService, nil, statusManager, nil)
	presenceManager := services.NewPresenceManager(statusManager, services.PresenceManagerConfig{
		HeartbeatTTL:  time.Minute,
		SweepInterval: time.Hour,
		HardEvictTTL:  time.Hour,
	})

	srv := &AgentFieldServer{
		storage:           errStorage,
		Router:            gin.New(),
		uiService:         uiService,
		healthMonitor:     healthMonitor,
		presenceManager:   presenceManager,
		statusManager:     statusManager,
		config:            &cfg,
		cleanupService:    handlers.NewExecutionCleanupService(errStorage, cfg.AgentField.ExecutionCleanup),
		payloadStore:      &stubPayloadStore{},
		webhookDispatcher: &stubWebhookDispatcher{},
		adminGRPCPort:     0,
		apiCatalog:        initAPICatalog(),
		kb:                initKnowledgeBase(),
	}
	srv.config.AgentField.Port = -1

	err := srv.Start()
	require.Error(t, err)
	require.NoError(t, srv.Stop())
}

func TestSetupRoutesFilesystemFallbackDistPath(t *testing.T) {
	gin.SetMode(gin.TestMode)

	root := t.TempDir()
	t.Chdir(root)
	distDir := filepath.Join(root, "web", "client", "dist")
	require.NoError(t, os.MkdirAll(distDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(distDir, "index.html"), []byte("fallback index"), 0o644))
	require.NoError(t, os.WriteFile(filepath.Join(distDir, "app.js"), []byte("console.log('ok')"), 0o644))

	cfg := baseConfigForDBTests()
	cfg.UI.Enabled = true
	cfg.UI.Mode = "filesystem"
	cfg.UI.DistPath = ""
	cfg.API.Auth.APIKey = ""
	cfg.API.CORS = config.CORSConfig{}
	cfg.Features.DID.Enabled = false
	cfg.Features.Connector.Enabled = false

	srv := newRouteTestServer(&cfg)
	srv.setupRoutes()

	t.Run("serves static file from discovered dist path", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/ui/app.js", nil)

		srv.Router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Body.String(), "console.log('ok')")
	})

	t.Run("serves spa fallback from discovered dist path", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/ui/dashboard/executions", nil)

		srv.Router.ServeHTTP(rec, req)

		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Body.String(), "fallback index")
	})

	t.Run("exercises inline ui handlers", func(t *testing.T) {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, "/api/ui/v1/agents/agent-1/details", nil)
		srv.Router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusOK, rec.Code)

		rec = httptest.NewRecorder()
		req = httptest.NewRequest(http.MethodGet, "/api/ui/v1/settings/node-log-proxy", nil)
		srv.Router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusOK, rec.Code)

		rec = httptest.NewRecorder()
		req = httptest.NewRequest(http.MethodPut, "/api/ui/v1/settings/node-log-proxy", bytes.NewBufferString(`{"connect_timeout":"3s","stream_idle_timeout":"4s","max_stream_duration":"5s","max_tail_lines":12}`))
		req.Header.Set("Content-Type", "application/json")
		srv.Router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusOK, rec.Code)

		rec = httptest.NewRecorder()
		req = httptest.NewRequest(http.MethodGet, "/api/ui/v1/llm/health", nil)
		srv.Router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusOK, rec.Code)

		rec = httptest.NewRecorder()
		req = httptest.NewRequest(http.MethodGet, "/api/ui/v1/queue/status", nil)
		srv.Router.ServeHTTP(rec, req)
		require.Equal(t, http.StatusOK, rec.Code)
	})
}

func TestSyncPackagesFromRegistryErrorPaths(t *testing.T) {
	t.Run("returns yaml parse error for invalid registry", func(t *testing.T) {
		home := t.TempDir()
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte("installed: ["), 0o644))

		err := SyncPackagesFromRegistry(home, newStubPackageStorage())
		require.Error(t, err)
	})

	t.Run("skips packages with missing or invalid package yaml", func(t *testing.T) {
		home := t.TempDir()
		missingDir := filepath.Join(home, "missing-pkg")
		badDir := filepath.Join(home, "bad-pkg")
		require.NoError(t, os.MkdirAll(missingDir, 0o755))
		require.NoError(t, os.MkdirAll(badDir, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(badDir, "agentfield-package.yaml"), []byte("name: ["), 0o644))

		registry := []byte(
			"installed:\n" +
				"  missing:\n" +
				"    name: Missing Package\n" +
				"    version: 1.0.0\n" +
				"    path: " + missingDir + "\n" +
				"  bad:\n" +
				"    name: Bad Package\n" +
				"    version: 1.0.0\n" +
				"    path: " + badDir + "\n",
		)
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), registry, 0o644))

		store := newStubPackageStorage()
		require.NoError(t, SyncPackagesFromRegistry(home, store))
		require.Empty(t, store.packages)
	})
}

func TestStartPackageRegistryWatcherIgnoresUnrelatedChangesAndHandlesRename(t *testing.T) {
	home := t.TempDir()
	pkgDir := filepath.Join(home, "pkg-a")
	require.NoError(t, os.MkdirAll(pkgDir, 0o755))
	require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte("name: Package A\nschema:\n  type: object\n"), 0o644))

	store := newStubPackageStorage()
	cancel, err := StartPackageRegistryWatcher(context.Background(), home, store)
	require.NoError(t, err)
	t.Cleanup(cancel)

	require.NoError(t, os.WriteFile(filepath.Join(home, "notes.txt"), []byte("ignore me"), 0o644))
	time.Sleep(400 * time.Millisecond)
	require.Empty(t, store.packages)

	registry := []byte(
		"installed:\n" +
			"  pkg-a:\n" +
			"    name: Package A\n" +
			"    version: 1.2.3\n" +
			"    description: watched package\n" +
			"    path: " + pkgDir + "\n",
	)
	tmpPath := filepath.Join(home, "installed.yaml.tmp")
	require.NoError(t, os.WriteFile(tmpPath, registry, 0o644))
	require.NoError(t, os.Rename(tmpPath, filepath.Join(home, "installed.yaml")))

	require.Eventually(t, func() bool {
		_, ok := store.packages["pkg-a"]
		return ok
	}, 5*time.Second, 100*time.Millisecond)
}

func TestStartPackageRegistryWatcherHandlesSyncErrors(t *testing.T) {
	home := t.TempDir()
	store := newStubPackageStorage()
	cancel, err := StartPackageRegistryWatcher(context.Background(), home, store)
	require.NoError(t, err)
	t.Cleanup(cancel)

	tmpPath := filepath.Join(home, "installed.yaml.tmp")
	require.NoError(t, os.WriteFile(tmpPath, []byte("installed: ["), 0o644))
	require.NoError(t, os.Rename(tmpPath, filepath.Join(home, "installed.yaml")))

	time.Sleep(400 * time.Millisecond)
	require.Empty(t, store.packages)
}

func TestHandleDIDWebServerDocumentNotFoundWhenRegistryMissing(t *testing.T) {
	store := &didDocStorage{stubStorage: newStubStorage(), docs: make(map[string]*types.DIDDocumentRecord)}
	registry := services.NewDIDRegistryWithStorage(store)
	didService := services.NewDIDService(&config.DIDConfig{Enabled: true}, nil, registry)
	require.NoError(t, didService.Initialize("server-id"))
	require.NoError(t, registry.DeleteRegistry("server-id"))

	srv := &AgentFieldServer{didService: didService}
	rec := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(rec)
	ctx.Request = httptest.NewRequest(http.MethodGet, "/.well-known/did.json", nil)

	srv.handleDIDWebServerDocument(ctx)

	require.Equal(t, http.StatusNotFound, rec.Code)
}

func TestSetupRoutesAgentFieldDIDRouteHandlesUninitializedService(t *testing.T) {
	cfg := baseConfigForDBTests()
	cfg.UI.Enabled = false
	cfg.API.Auth.APIKey = ""
	cfg.Features.DID.Enabled = true
	cfg.Features.Connector.Enabled = false

	didService := services.NewDIDService(&config.DIDConfig{Enabled: true}, nil, services.NewDIDRegistryWithStorage(newStubStorage()))
	vcService := services.NewVCService(&cfg.Features.DID, didService, newStubStorage())

	srv := newRouteTestServer(&cfg)
	srv.didService = didService
	srv.vcService = vcService
	srv.didWebService = services.NewDIDWebService("example.com", didService, newStubStorage())
	srv.setupRoutes()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/did/agentfield-server", nil)
	srv.Router.ServeHTTP(rec, req)

	require.Equal(t, http.StatusInternalServerError, rec.Code)
}

func TestGenerateAgentFieldServerIDFallsBackWhenGetwdFails(t *testing.T) {
	originalWD, err := os.Getwd()
	require.NoError(t, err)

	brokenWD := t.TempDir()
	require.NoError(t, os.Chdir(brokenWD))
	require.NoError(t, os.Remove(brokenWD))
	t.Cleanup(func() {
		_ = os.Chdir(originalWD)
	})

	got := generateAgentFieldServerID("relative/path")
	sum := sha256.Sum256([]byte("relative/path"))
	want := hex.EncodeToString(sum[:])[:16]

	require.Equal(t, want, got)
}

func TestStartAdminGRPCServerReturnsListenError(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	defer listener.Close()

	cfg := baseConfigForDBTests()
	cfg.API.Auth.APIKey = "secret"
	srv := &AgentFieldServer{
		config: &cfg,
		adminGRPCPort: listener.Addr().(*net.TCPAddr).Port,
	}

	err = srv.startAdminGRPCServer()
	require.Error(t, err)
	require.Nil(t, srv.adminGRPCServer)
	require.Nil(t, srv.adminListener)
}
