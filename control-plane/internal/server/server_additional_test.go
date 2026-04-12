package server

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/adminpb"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func TestInitCatalogAndKnowledgeBase(t *testing.T) {
	t.Parallel()

	catalog := initAPICatalog()
	require.NotNil(t, catalog)
	require.NotEmpty(t, catalog.All())

	kb := initKnowledgeBase()
	require.NotNil(t, kb)
	require.NotEmpty(t, kb.Topics())
	require.NotNil(t, kb.Get("observability/metrics"))
}

func TestConfigReloadFn(t *testing.T) {
	t.Run("disabled when config source is not db", func(t *testing.T) {
		t.Setenv("AGENTFIELD_CONFIG_SOURCE", "")

		srv := &AgentFieldServer{}
		require.Nil(t, srv.configReloadFn())
	})

	t.Run("reloads config from database", func(t *testing.T) {
		t.Setenv("AGENTFIELD_CONFIG_SOURCE", "db")

		cfg := baseConfigForDBTests()
		srv := &AgentFieldServer{
			config: &cfg,
			storage: &configStoreStub{
				entry: &storage.ConfigEntry{
					Key:       dbConfigKey,
					Value:     "agentfield:\n  port: 9191\n",
					Version:   3,
					UpdatedAt: time.Date(2026, 4, 8, 12, 0, 0, 0, time.UTC),
				},
			},
		}

		reload := srv.configReloadFn()
		require.NotNil(t, reload)
		require.NoError(t, reload())
		require.Equal(t, 9191, srv.config.AgentField.Port)
	})
}

func TestNewAgentFieldServer(t *testing.T) {
	t.Run("uses env home and explicit admin port", func(t *testing.T) {
		cfg := baseConfigForDBTests()
		cfg.UI.Enabled = false
		cfg.Features.DID.Enabled = false
		cfg.Features.Connector.Enabled = false
		cfg.AgentField.Port = 18080

		home := t.TempDir()
		cfg.Storage.Local.DatabasePath = filepath.Join(home, "server.db")
		cfg.Storage.Local.KVStorePath = filepath.Join(home, "server.bolt")

		t.Setenv("AGENTFIELD_HOME", home)
		t.Setenv("AGENTFIELD_ADMIN_GRPC_PORT", "19555")
		t.Setenv("AGENTFIELD_CONFIG_SOURCE", "")

		srv, err := NewAgentFieldServer(&cfg)
		require.NoError(t, err)
		require.Equal(t, home, srv.agentfieldHome)
		require.Equal(t, 19555, srv.adminGRPCPort)
		require.NotNil(t, srv.apiCatalog)
		require.NotNil(t, srv.kb)

		stopServerBackgrounds(t, srv)
	})

	t.Run("falls back to default admin port when env is invalid", func(t *testing.T) {
		cfg := baseConfigForDBTests()
		cfg.UI.Enabled = false
		cfg.Features.DID.Enabled = false
		cfg.Features.Connector.Enabled = false
		cfg.AgentField.Port = 18081

		home := t.TempDir()
		cfg.Storage.Local.DatabasePath = filepath.Join(home, "server.db")
		cfg.Storage.Local.KVStorePath = filepath.Join(home, "server.bolt")

		t.Setenv("AGENTFIELD_HOME", home)
		t.Setenv("AGENTFIELD_ADMIN_GRPC_PORT", "not-a-port")
		t.Setenv("AGENTFIELD_CONFIG_SOURCE", "db")

		srv, err := NewAgentFieldServer(&cfg)
		require.NoError(t, err)
		require.Equal(t, cfg.AgentField.Port+100, srv.adminGRPCPort)

		stopServerBackgrounds(t, srv)
	})

	t.Run("initializes did services when enabled", func(t *testing.T) {
		cfg := baseConfigForDBTests()
		cfg.UI.Enabled = false
		cfg.API.Auth.APIKey = ""
		cfg.Features.DID.Enabled = true
		cfg.Features.DID.KeyAlgorithm = "Ed25519"
		cfg.Features.DID.Authorization.Enabled = true
		cfg.Features.DID.Authorization.DIDAuthEnabled = true
		cfg.Features.DID.Authorization.Domain = "example.com"
		cfg.Features.DID.Authorization.InternalToken = "internal-token"
		cfg.Features.DID.Authorization.TagApprovalRules.DefaultMode = "manual"
		cfg.Features.Connector.Enabled = false

		home := t.TempDir()
		cfg.Storage.Local.DatabasePath = filepath.Join(home, "server.db")
		cfg.Storage.Local.KVStorePath = filepath.Join(home, "server.bolt")
		cfg.Features.DID.Keystore.Path = filepath.Join(home, "keys")

		t.Setenv("AGENTFIELD_HOME", home)
		t.Setenv("AGENTFIELD_CONFIG_SOURCE", "")

		srv, err := NewAgentFieldServer(&cfg)
		require.NoError(t, err)
		require.NotNil(t, srv.didService)
		require.NotNil(t, srv.vcService)
		require.NotNil(t, srv.didWebService)
		require.NotNil(t, srv.accessPolicyService)
		require.NotNil(t, srv.tagApprovalService)
		require.NotNil(t, srv.tagVCVerifier)

		stopServerBackgrounds(t, srv)
	})
}

func TestStartAdminGRPCServer(t *testing.T) {
	t.Parallel()

	cfg := baseConfigForDBTests()
	cfg.API.Auth.APIKey = "secret"

	srv := &AgentFieldServer{
		config:        &cfg,
		adminGRPCPort: 0,
	}

	require.NoError(t, srv.startAdminGRPCServer())
	require.NotNil(t, srv.adminGRPCServer)
	require.NotNil(t, srv.adminListener)
	require.NoError(t, srv.startAdminGRPCServer())

	srv.adminGRPCServer.GracefulStop()
	require.NoError(t, srv.adminListener.Close())
}

func TestStartAndStop(t *testing.T) {
	cfg := baseConfigForDBTests()
	cfg.UI.Enabled = false
	cfg.Features.DID.Enabled = false
	cfg.Features.Connector.Enabled = false
	cfg.AgentField.Port = 18082

	home := t.TempDir()
	cfg.Storage.Local.DatabasePath = filepath.Join(home, "server.db")
	cfg.Storage.Local.KVStorePath = filepath.Join(home, "server.bolt")

	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("AGENTFIELD_CONFIG_SOURCE", "")

	srv, err := NewAgentFieldServer(&cfg)
	require.NoError(t, err)

	srv.config.AgentField.Port = -1
	srv.adminGRPCPort = 0

	err = srv.Start()
	require.Error(t, err)
	require.NoError(t, srv.Stop())
}

func TestListReasonersErrorAndSkipsNilNodes(t *testing.T) {
	t.Parallel()

	t.Run("returns internal error on storage failure", func(t *testing.T) {
		srv := &AgentFieldServer{
			storage: &listAgentsStorage{stubStorage: newStubStorage(), err: fmt.Errorf("boom")},
		}

		resp, err := srv.ListReasoners(context.Background(), &adminpb.ListReasonersRequest{})
		require.Nil(t, resp)
		require.Error(t, err)
		require.Equal(t, codes.Internal, status.Code(err))
	})

	t.Run("skips nil nodes and empty reasoners", func(t *testing.T) {
		now := time.Date(2026, 4, 8, 14, 0, 0, 0, time.UTC)
		srv := &AgentFieldServer{
			storage: &listAgentsStorage{
				stubStorage: newStubStorage(),
				nodes: []*types.AgentNode{
					nil,
					{ID: "empty", LastHeartbeat: now},
					{
						ID:            "node-2",
						HealthStatus:  types.HealthStatusActive,
						Version:       "2.0.0",
						LastHeartbeat: now,
						Reasoners:     []types.ReasonerDefinition{{ID: "alpha"}},
					},
				},
			},
		}

		resp, err := srv.ListReasoners(context.Background(), &adminpb.ListReasonersRequest{})
		require.NoError(t, err)
		require.Len(t, resp.Reasoners, 1)
		require.Equal(t, "node-2.alpha", resp.Reasoners[0].ReasonerId)
	})
}

func TestSetupRoutesFilesystemUIAndNoRouteFallback(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	distDir := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(distDir, "index.html"), []byte("ui index"), 0o644))

	cfg := baseConfigForDBTests()
	cfg.UI.Enabled = true
	cfg.UI.Mode = "filesystem"
	cfg.UI.DistPath = distDir
	cfg.API.Auth.APIKey = ""
	cfg.API.Auth.SkipPaths = nil
	cfg.API.CORS = config.CORSConfig{}
	cfg.Features.DID.Enabled = false
	cfg.Features.Connector.Enabled = false

	srv := newRouteTestServer(&cfg)
	srv.setupRoutes()

	testCases := []struct {
		name       string
		path       string
		wantCode   int
		wantBody   string
		wantHeader string
	}{
		{name: "root redirect", path: "/", wantCode: http.StatusMovedPermanently, wantHeader: "/ui/"},
		{name: "knowledge base topics", path: "/api/v1/agentic/kb/topics", wantCode: http.StatusOK},
		{name: "metrics", path: "/metrics", wantCode: http.StatusOK},
		{name: "spa fallback", path: "/ui/dashboard/runs", wantCode: http.StatusOK, wantBody: "ui index"},
		{name: "static asset 404", path: "/missing.js", wantCode: http.StatusNotFound, wantBody: `"error":"endpoint_not_found"`},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodGet, tc.path, nil)

			srv.Router.ServeHTTP(rec, req)

			require.Equal(t, tc.wantCode, rec.Code)
			if tc.wantBody != "" {
				require.Contains(t, rec.Body.String(), tc.wantBody)
			}
			if tc.wantHeader != "" {
				require.Equal(t, tc.wantHeader, rec.Header().Get("Location"))
			}
		})
	}
}

func TestSetupRoutesRegistersAuthorizationAndConnectorBranches(t *testing.T) {
	t.Parallel()

	gin.SetMode(gin.TestMode)

	cfg := baseConfigForDBTests()
	cfg.UI.Enabled = false
	cfg.API.Auth.APIKey = ""
	cfg.Features.DID.Enabled = true
	cfg.Features.DID.Authorization.Enabled = true
	cfg.Features.DID.Authorization.DIDAuthEnabled = true
	cfg.Features.DID.Authorization.AdminToken = "admin-token"
	cfg.Features.Connector.Enabled = true
	cfg.Features.Connector.Token = "connector-token"
	cfg.Features.Connector.Capabilities = map[string]config.ConnectorCapability{
		"config_management": {Enabled: true},
	}

	srv := newRouteTestServer(&cfg)
	srv.didWebService = services.NewDIDWebService("example.com", nil, srv.storage)
	srv.accessPolicyService = services.NewAccessPolicyService(srv.storage)
	srv.tagApprovalService = services.NewTagApprovalService(config.TagApprovalRulesConfig{DefaultMode: "manual"}, srv.storage)

	require.NotPanics(t, srv.setupRoutes)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/api/v1/registered-dids", nil)
	srv.Router.ServeHTTP(rec, req)
	require.NotEqual(t, http.StatusNotFound, rec.Code)
}

func TestSetupRoutesWithDIDServices(t *testing.T) {
	cfg := baseConfigForDBTests()
	cfg.UI.Enabled = false
	cfg.API.Auth.APIKey = ""
	cfg.Features.DID.Enabled = true
	cfg.Features.DID.KeyAlgorithm = "Ed25519"
	cfg.Features.DID.Authorization.Enabled = true
	cfg.Features.DID.Authorization.DIDAuthEnabled = true
	cfg.Features.DID.Authorization.Domain = "example.com"
	cfg.Features.DID.Authorization.InternalToken = "internal-token"
	cfg.Features.DID.Authorization.TagApprovalRules.DefaultMode = "manual"
	cfg.Features.Connector.Enabled = false

	home := t.TempDir()
	cfg.Storage.Local.DatabasePath = filepath.Join(home, "server.db")
	cfg.Storage.Local.KVStorePath = filepath.Join(home, "server.bolt")
	cfg.Features.DID.Keystore.Path = filepath.Join(home, "keys")

	t.Setenv("AGENTFIELD_HOME", home)
	t.Setenv("AGENTFIELD_CONFIG_SOURCE", "")

	srv, err := NewAgentFieldServer(&cfg)
	require.NoError(t, err)
	defer stopServerBackgrounds(t, srv)

	gin.SetMode(gin.TestMode)
	srv.Router = gin.New()
	srv.setupRoutes()

	testCases := []struct {
		path     string
		wantCode int
		wantBody string
	}{
		{path: "/.well-known/did.json", wantCode: http.StatusOK, wantBody: `"verificationMethod"`},
		{path: "/api/v1/did/agentfield-server", wantCode: http.StatusOK, wantBody: `"agentfield_server_did"`},
		{path: "/api/v1/policies", wantCode: http.StatusOK, wantBody: `"policies"`},
		{path: "/api/v1/revocations", wantCode: http.StatusOK, wantBody: `"revoked_dids"`},
		{path: "/api/v1/did/issuer-public-key", wantCode: http.StatusOK, wantBody: `"public_key_jwk"`},
	}

	for _, tc := range testCases {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodGet, tc.path, nil)
		srv.Router.ServeHTTP(rec, req)

		require.Equal(t, tc.wantCode, rec.Code)
		require.Contains(t, rec.Body.String(), tc.wantBody)
	}
}

func TestDIDHandlersAndServeDocument(t *testing.T) {
	t.Parallel()

	t.Run("server document returns 500 before DID service initialization", func(t *testing.T) {
		didService := services.NewDIDService(&config.DIDConfig{Enabled: true}, nil, services.NewDIDRegistryWithStorage(newStubStorage()))
		srv := &AgentFieldServer{didService: didService}

		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/.well-known/did.json", nil)

		srv.handleDIDWebServerDocument(ctx)

		require.Equal(t, http.StatusInternalServerError, rec.Code)
	})

	t.Run("serves server did:key document", func(t *testing.T) {
		didService, _, _ := newTestDIDServices(t)
		serverID, err := didService.GetAgentFieldServerID()
		require.NoError(t, err)
		registry, err := didService.GetRegistry(serverID)
		require.NoError(t, err)

		srv := &AgentFieldServer{didService: didService}
		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/.well-known/did.json", nil)

		srv.handleDIDWebServerDocument(ctx)

		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Body.String(), registry.RootDID)
		require.Contains(t, rec.Body.String(), `"authentication"`)
	})

	t.Run("agent did handler validates required param", func(t *testing.T) {
		srv := &AgentFieldServer{}
		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/agents//did.json", nil)

		srv.handleDIDWebAgentDocument(ctx)

		require.Equal(t, http.StatusBadRequest, rec.Code)
	})

	t.Run("serves agent did:web document and revoked documents", func(t *testing.T) {
		didService, didWebService, didStore := newTestDIDServices(t)
		srv := &AgentFieldServer{
			didService:    didService,
			didWebService: didWebService,
		}

		_, err := didWebService.CreateDIDDocument(context.Background(), "agent-1", []byte(`{"kty":"OKP","crv":"Ed25519","x":"abc"}`))
		require.NoError(t, err)

		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Params = gin.Params{{Key: "agentID", Value: "agent-1"}}
		ctx.Request = httptest.NewRequest(http.MethodGet, "/agents/agent-1/did.json", nil)

		srv.handleDIDWebAgentDocument(ctx)

		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Body.String(), "did:web:example.com:agents:agent-1")

		did := didWebService.GenerateDIDWeb("agent-1")
		require.NoError(t, didStore.RevokeDIDDocument(context.Background(), did))

		rec = httptest.NewRecorder()
		ctx, _ = gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/agents/agent-1/did.json", nil)

		srv.serveDIDDocument(ctx, did)

		require.Equal(t, http.StatusGone, rec.Code)
	})

	t.Run("returns not found for unknown DID", func(t *testing.T) {
		didService, _, _ := newTestDIDServices(t)
		srv := &AgentFieldServer{didService: didService}

		rec := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(rec)
		ctx.Request = httptest.NewRequest(http.MethodGet, "/unknown", nil)

		srv.serveDIDDocument(ctx, "did:key:missing")

		require.Equal(t, http.StatusNotFound, rec.Code)
	})
}

func TestStartPackageRegistryWatcher(t *testing.T) {
	t.Parallel()

	t.Run("returns error for missing directory", func(t *testing.T) {
		cancel, err := StartPackageRegistryWatcher(context.Background(), filepath.Join(t.TempDir(), "missing"), newStubPackageStorage())
		require.Nil(t, cancel)
		require.Error(t, err)
	})

	t.Run("syncs packages after registry write", func(t *testing.T) {
		home := t.TempDir()
		pkgDir := filepath.Join(home, "pkg-a")
		require.NoError(t, os.MkdirAll(pkgDir, 0o755))
		require.NoError(t, os.WriteFile(filepath.Join(pkgDir, "agentfield-package.yaml"), []byte("name: Package A\nschema:\n  type: object\n"), 0o644))

		store := newStubPackageStorage()
		cancel, err := StartPackageRegistryWatcher(context.Background(), home, store)
		require.NoError(t, err)
		t.Cleanup(cancel)

		registry := strings.Join([]string{
			"installed:",
			"  pkg-a:",
			"    name: Package A",
			"    version: 1.2.3",
			"    description: watched package",
			"    path: " + pkgDir,
			"",
		}, "\n")
		require.NoError(t, os.WriteFile(filepath.Join(home, "installed.yaml"), []byte(registry), 0o644))

		require.Eventually(t, func() bool {
			_, ok := store.packages["pkg-a"]
			return ok
		}, 5*time.Second, 100*time.Millisecond)
	})
}

type listAgentsStorage struct {
	*stubStorage
	nodes []*types.AgentNode
	err   error
}

func (s *listAgentsStorage) ListAgents(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
	return s.nodes, s.err
}

func newRouteTestServer(cfg *config.Config) *AgentFieldServer {
	return &AgentFieldServer{
		storage:           newStubStorage(),
		Router:            gin.New(),
		config:            cfg,
		apiCatalog:        initAPICatalog(),
		kb:                initKnowledgeBase(),
		payloadStore:      &stubPayloadStore{},
		adminGRPCPort:     cfg.AgentField.Port + 100,
		webhookDispatcher: &stubWebhookDispatcher{},
	}
}

func stopServerBackgrounds(t *testing.T, srv *AgentFieldServer) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if srv.webhookDispatcher != nil {
		require.NoError(t, srv.webhookDispatcher.Stop(ctx))
	}
	if srv.observabilityForwarder != nil {
		require.NoError(t, srv.observabilityForwarder.Stop(ctx))
	}
	if srv.storage != nil {
		require.NoError(t, srv.storage.Close(ctx))
	}
}

type didDocStorage struct {
	*stubStorage
	docs map[string]*types.DIDDocumentRecord
}

func (s *didDocStorage) StoreDIDDocument(_ context.Context, record *types.DIDDocumentRecord) error {
	if s.docs == nil {
		s.docs = make(map[string]*types.DIDDocumentRecord)
	}
	copyRecord := *record
	s.docs[record.DID] = &copyRecord
	return nil
}

func (s *didDocStorage) GetDIDDocument(_ context.Context, did string) (*types.DIDDocumentRecord, error) {
	record, ok := s.docs[did]
	if !ok {
		return nil, fmt.Errorf("document not found")
	}
	return record, nil
}

func (s *didDocStorage) GetDIDDocumentByAgentID(_ context.Context, agentID string) (*types.DIDDocumentRecord, error) {
	for _, record := range s.docs {
		if record.AgentID == agentID {
			return record, nil
		}
	}
	return nil, fmt.Errorf("document not found")
}

func (s *didDocStorage) RevokeDIDDocument(_ context.Context, did string) error {
	record, ok := s.docs[did]
	if !ok {
		return fmt.Errorf("document not found")
	}
	now := time.Now().UTC()
	record.RevokedAt = &now
	record.UpdatedAt = now
	return nil
}

func (s *didDocStorage) ListDIDDocuments(context.Context) ([]*types.DIDDocumentRecord, error) {
	records := make([]*types.DIDDocumentRecord, 0, len(s.docs))
	for _, record := range s.docs {
		records = append(records, record)
	}
	return records, nil
}

func newTestDIDServices(t *testing.T) (*services.DIDService, *services.DIDWebService, *didDocStorage) {
	t.Helper()

	store := &didDocStorage{stubStorage: newStubStorage(), docs: make(map[string]*types.DIDDocumentRecord)}
	registry := services.NewDIDRegistryWithStorage(store)
	didService := services.NewDIDService(&config.DIDConfig{Enabled: true}, nil, registry)
	require.NoError(t, didService.Initialize("server-id"))

	return didService, services.NewDIDWebService("example.com", didService, store), store
}
