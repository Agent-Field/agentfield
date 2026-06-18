package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/ard"
	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

type ardHandlerStore struct {
	storage.StorageProvider
	agents  []*types.AgentNode
	config  map[string]string
	getErr  error
	setErr  error
	listErr error
}

func (s *ardHandlerStore) ListAgents(context.Context, types.AgentFilters) ([]*types.AgentNode, error) {
	if s.listErr != nil {
		return nil, s.listErr
	}
	return s.agents, nil
}

func (s *ardHandlerStore) GetConfig(_ context.Context, key string) (*storage.ConfigEntry, error) {
	if s.getErr != nil {
		return nil, s.getErr
	}
	value, ok := s.config[key]
	if !ok {
		return nil, nil
	}
	return &storage.ConfigEntry{Key: key, Value: value}, nil
}

func (s *ardHandlerStore) SetConfig(_ context.Context, key string, value string, _ string) error {
	if s.setErr != nil {
		return s.setErr
	}
	if s.config == nil {
		s.config = map[string]string{}
	}
	s.config[key] = value
	return nil
}

func TestARDRegistryHandlersRequireRuntimePublicGate(t *testing.T) {
	gin.SetMode(gin.TestMode)

	registryPublic := false
	store := &ardHandlerStore{
		agents: []*types.AgentNode{{
			ID:           "node-1",
			Version:      "1.0.0",
			HealthStatus: types.HealthStatusActive,
			Reasoners: []types.ReasonerDefinition{{
				ID: "review_contract",
			}},
		}},
		config: map[string]string{},
	}
	state := ard.State{
		Settings: ard.RuntimeSettings{RegistryPublic: &registryPublic},
		Publications: map[string]ard.Publication{
			ard.TargetKey("reasoner", "node-1", "review_contract"): {
				TargetKind:            "reasoner",
				NodeID:                "node-1",
				TargetID:              "review_contract",
				Published:             true,
				DisplayName:           "Review Contract",
				Description:           "Review contract language.",
				RepresentativeQueries: []string{"review this MSA", "find risky clauses"},
				ArtifactType:          "application/openapi+json",
			},
		},
	}
	rawState, err := json.Marshal(state)
	if err != nil {
		t.Fatalf("marshal state: %v", err)
	}
	store.config[ard.StateConfigKey] = string(rawState)

	handler := NewARDHandler(store, func() config.ARDConfig {
		return config.ARDConfig{
			Enabled:         true,
			PublicBaseURL:   "https://cp.example.com",
			PublisherDomain: "example.com",
			Publish: config.ARDPublishConfig{
				Enabled:               true,
				IncludeHealthStatuses: []string{"active"},
			},
			Registry: config.ARDRegistryConfig{
				Enabled: true,
				Public:  true,
			},
		}
	}, func() bool { return false })
	router := gin.New()
	router.POST("/api/v1/ard/search", handler.SearchRegistry)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/api/v1/ard/search", bytes.NewBufferString(`{"query":{"text":"review"},"pageSize":10}`))
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected runtime-private registry to return 404, got %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestARDExploreRejectsMalformedPayload(t *testing.T) {
	gin.SetMode(gin.TestMode)

	store := &ardHandlerStore{
		agents: []*types.AgentNode{{
			ID:            "node-1",
			Version:       "1.0.0",
			HealthStatus:  types.HealthStatusActive,
			LastHeartbeat: time.Now().UTC(),
		}},
		config: map[string]string{},
	}
	handler := NewARDHandler(store, func() config.ARDConfig {
		return config.ARDConfig{
			Enabled:         true,
			PublicBaseURL:   "https://cp.example.com",
			PublisherDomain: "example.com",
			Publish: config.ARDPublishConfig{
				Enabled:               true,
				IncludeHealthStatuses: []string{"active"},
			},
			Registry: config.ARDRegistryConfig{
				Enabled: true,
				Public:  true,
			},
		}
	}, func() bool { return false })
	router := gin.New()
	router.POST("/api/v1/ard/explore", handler.ExploreRegistry)

	for _, body := range []string{`{`, `{}`} {
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(http.MethodPost, "/api/v1/ard/explore", bytes.NewBufferString(body))
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusBadRequest {
			t.Fatalf("expected %q to return 400, got %d body=%s", body, rec.Code, rec.Body.String())
		}
	}
}

func TestARDUIHandlersPersistRuntimeState(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &ardHandlerStore{
		agents: []*types.AgentNode{{
			ID:           "node-1",
			Version:      "1.0.0",
			HealthStatus: types.HealthStatusActive,
			Reasoners: []types.ReasonerDefinition{{
				ID:          "review_contract",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}}}`),
			}},
			Skills: []types.SkillDefinition{{
				ID: "summarize",
			}},
		}},
		config: map[string]string{},
	}
	handler := NewARDHandler(store, func() config.ARDConfig {
		return config.ARDConfig{
			Enabled:         true,
			PublicBaseURL:   "https://cp.example.com",
			PublisherDomain: "example.com",
			Host:            config.ARDHostConfig{Identifier: "did:web:example.com"},
			Publish: config.ARDPublishConfig{
				Enabled:               true,
				IncludeHealthStatuses: []string{"active"},
			},
			Registry: config.ARDRegistryConfig{Enabled: true, Public: true},
			External: config.ARDExternalConfig{
				SearchEnabled:      true,
				InvocationEnabled:  true,
				AllowedRegistries:  []string{"https://registry.example.com/api/v1/ard"},
				DefaultSearchLimit: 7,
			},
		}
	}, func() bool { return true })
	router := gin.New()
	router.GET("/api/ui/v1/ard", handler.GetDashboard)
	router.PUT("/api/ui/v1/ard/settings", handler.UpdateSettings)
	router.PUT("/api/ui/v1/ard/publications", handler.SavePublication)
	router.POST("/api/ui/v1/ard/imports", handler.ImportExternal)
	router.PUT("/api/ui/v1/ard/imports/:entryID/binding", handler.SaveExternalBinding)
	router.PUT("/api/ui/v1/ard/registries", handler.SaveRegistries)

	rec := serveARD(router, http.MethodGet, "/api/ui/v1/ard", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("dashboard failed: %d %s", rec.Code, rec.Body.String())
	}
	var dashboard ard.Dashboard
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &dashboard))
	if len(dashboard.Publications) != 2 || dashboard.Summary.CatalogURL != "https://cp.example.com/.well-known/ai-catalog.json" {
		t.Fatalf("unexpected dashboard: %#v", dashboard)
	}

	rec = serveARD(router, http.MethodPut, "/api/ui/v1/ard/settings", map[string]any{
		"enabled":         true,
		"registry_public": false,
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("settings failed: %d %s", rec.Code, rec.Body.String())
	}
	state, err := ard.LoadState(context.Background(), store)
	require.NoError(t, err)
	if state.Settings.Enabled == nil || !*state.Settings.Enabled {
		t.Fatalf("settings were not saved: %#v", state.Settings)
	}

	publication := ard.Publication{
		TargetKind:            "reasoner",
		NodeID:                "node-1",
		TargetID:              "review_contract",
		Published:             true,
		DisplayName:           "Review Contract",
		Description:           "Review contract language.",
		RepresentativeQueries: []string{"review this MSA", "find risky indemnity clauses"},
		ArtifactType:          "application/openapi+json",
	}
	rec = serveARD(router, http.MethodPut, "/api/ui/v1/ard/publications", publication)
	if rec.Code != http.StatusOK {
		t.Fatalf("publication failed: %d %s", rec.Code, rec.Body.String())
	}
	state, err = ard.LoadState(context.Background(), store)
	require.NoError(t, err)
	saved := state.Publications[ard.TargetKey("reasoner", "node-1", "review_contract")]
	if saved.ValidationStatus != "valid" || saved.LastValidatedAt.IsZero() {
		t.Fatalf("publication validation not saved: %#v", saved)
	}

	importPayload := map[string]any{
		"source_registry": "https://registry.example.com/api/v1/ard",
		"entry": map[string]any{
			"identifier":            "urn:ai:vendor.example:agent:review",
			"displayName":           "Vendor Review",
			"description":           "External reviewer.",
			"type":                  "application/a2a-agent-card+json",
			"url":                   "https://vendor.example/agent-card.json",
			"representativeQueries": []string{"review a contract"},
			"trustManifest": map[string]any{
				"identity":     "did:web:vendor.example",
				"identityType": "did",
			},
		},
	}
	rec = serveARD(router, http.MethodPost, "/api/ui/v1/ard/imports", importPayload)
	if rec.Code != http.StatusOK {
		t.Fatalf("import failed: %d %s", rec.Code, rec.Body.String())
	}
	externalID := ard.ImportID("urn:ai:vendor.example:agent:review")
	rec = serveARD(router, http.MethodPut, "/api/ui/v1/ard/imports/"+externalID+"/binding", map[string]any{
		"callable":     true,
		"local_target": "external.vendor.review",
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("binding failed: %d %s", rec.Code, rec.Body.String())
	}
	state, err = ard.LoadState(context.Background(), store)
	require.NoError(t, err)
	if len(state.Imports) != 1 || state.Imports[0].Publisher != "vendor.example" || state.Imports[0].TrustSummary != "did:web:vendor.example" {
		t.Fatalf("import not persisted with publisher/trust: %#v", state.Imports)
	}
	if binding := state.Bindings[externalID]; binding.Adapter != "openapi" || binding.TimeoutMS != 30000 || !binding.Callable {
		t.Fatalf("binding defaults not applied: %#v", binding)
	}
	rec = serveARD(router, http.MethodPut, "/api/ui/v1/ard/imports/"+externalID+"/binding", map[string]any{
		"callable":     true,
		"local_target": "vendor.review",
	})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("callable binding without external target should fail: %d %s", rec.Code, rec.Body.String())
	}
	secondImportPayload := map[string]any{
		"source_registry": "https://registry.example.com/api/v1/ard",
		"entry": map[string]any{
			"identifier":  "urn:ai:vendor.example:agent:review-secondary",
			"displayName": "Vendor Review Secondary",
			"type":        "application/a2a-agent-card+json",
			"url":         "https://vendor.example/agent-card-secondary.json",
		},
	}
	rec = serveARD(router, http.MethodPost, "/api/ui/v1/ard/imports", secondImportPayload)
	if rec.Code != http.StatusOK {
		t.Fatalf("second import failed: %d %s", rec.Code, rec.Body.String())
	}
	secondExternalID := ard.ImportID("urn:ai:vendor.example:agent:review-secondary")
	rec = serveARD(router, http.MethodPut, "/api/ui/v1/ard/imports/"+secondExternalID+"/binding", map[string]any{
		"callable":     true,
		"local_target": "external.vendor.review",
	})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("duplicate callable target should fail: %d %s", rec.Code, rec.Body.String())
	}

	rec = serveARD(router, http.MethodPut, "/api/ui/v1/ard/registries", map[string]any{
		"registries": []map[string]any{{"url": "https://registry.example.com/api/v1/ard", "name": "Registry"}},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("registries failed: %d %s", rec.Code, rec.Body.String())
	}
	state, err = ard.LoadState(context.Background(), store)
	require.NoError(t, err)
	if len(state.Registries) != 1 || state.Registries[0].Name != "Registry" {
		t.Fatalf("registries not saved: %#v", state.Registries)
	}
}

func TestARDPublicAndRegistryHandlers(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &ardHandlerStore{
		agents: []*types.AgentNode{{
			ID:           "node-1",
			Version:      "1.0.0",
			HealthStatus: types.HealthStatusActive,
			Reasoners: []types.ReasonerDefinition{{
				ID:          "review_contract",
				InputSchema: json.RawMessage(`{"type":"object","properties":{"text":{"type":"string"}}}`),
			}},
			Skills: []types.SkillDefinition{{ID: "summarize"}},
		}},
		config: map[string]string{},
	}
	publicationState := ard.State{
		Settings: ard.RuntimeSettings{},
		Publications: map[string]ard.Publication{
			ard.TargetKey("reasoner", "node-1", "review_contract"): {
				TargetKind:            "reasoner",
				NodeID:                "node-1",
				TargetID:              "review_contract",
				Published:             true,
				DisplayName:           "Review Contract",
				Description:           "Review contract language.",
				Tags:                  []string{"legal"},
				Capabilities:          []string{"ContractReview"},
				RepresentativeQueries: []string{"review this MSA", "find risky indemnity clauses"},
				ArtifactType:          "application/openapi+json",
			},
			ard.TargetKey("skill", "node-1", "summarize"): {
				TargetKind:            "skill",
				NodeID:                "node-1",
				TargetID:              "summarize",
				Published:             true,
				DisplayName:           "Summarize",
				Description:           "Summarize text.",
				RepresentativeQueries: []string{"summarize this note", "make this shorter"},
				ArtifactType:          "application/openapi+json",
			},
		},
	}
	require.NoError(t, ard.SaveState(context.Background(), store, publicationState, "test"))
	handler := NewARDHandler(store, func() config.ARDConfig {
		return config.ARDConfig{
			Enabled:         true,
			PublicBaseURL:   "https://cp.example.com",
			PublisherDomain: "example.com",
			Publish: config.ARDPublishConfig{
				Enabled:               true,
				IncludeHealthStatuses: []string{"active"},
			},
			Registry: config.ARDRegistryConfig{Enabled: true, Public: true},
		}
	}, func() bool { return false })
	router := gin.New()
	router.GET("/.well-known/ai-catalog.json", handler.GetPublicCatalog)
	router.GET("/api/v1/ard/artifacts/:entryID", handler.GetArtifact)
	router.POST("/api/v1/ard/search", handler.SearchRegistry)
	router.GET("/api/v1/ard/agents", handler.ListRegistryAgents)
	router.POST("/api/v1/ard/explore", handler.ExploreRegistry)

	rec := serveARD(router, http.MethodGet, "/.well-known/ai-catalog.json", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("catalog failed: %d %s", rec.Code, rec.Body.String())
	}
	if rec.Header().Get("Access-Control-Allow-Origin") != "*" || rec.Header().Get("Cache-Control") == "" {
		t.Fatalf("public cache/cors headers missing: %#v", rec.Header())
	}
	var catalog ard.CatalogManifest
	require.NoError(t, json.Unmarshal(rec.Body.Bytes(), &catalog))
	if len(catalog.Entries) != 2 {
		t.Fatalf("expected two public entries, got %#v", catalog.Entries)
	}

	rec = serveARD(router, http.MethodGet, "/api/v1/ard/artifacts/node-1.review_contract", nil)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), "/api/v1/execute/node-1.review_contract") {
		t.Fatalf("artifact failed: %d %s", rec.Code, rec.Body.String())
	}
	rec = serveARD(router, http.MethodPost, "/api/v1/ard/search", map[string]any{
		"query": map[string]any{"text": "review", "filter": map[string]any{"tags": "legal"}},
	})
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"score"`) {
		t.Fatalf("search failed: %d %s", rec.Code, rec.Body.String())
	}
	rec = serveARD(router, http.MethodGet, "/api/v1/ard/agents?pageSize=1", nil)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"pageToken":"1"`) {
		t.Fatalf("agents pagination failed: %d %s", rec.Code, rec.Body.String())
	}
	rec = serveARD(router, http.MethodGet, "/api/v1/ard/agents?pageSize=x", nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("bad page size should fail: %d %s", rec.Code, rec.Body.String())
	}
	rec = serveARD(router, http.MethodPost, "/api/v1/ard/explore", map[string]any{
		"query": map[string]any{"text": "review"},
		"resultType": map[string]any{
			"facets": []map[string]any{{"field": "type"}, {"field": "tags"}},
		},
	})
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"resultType":"facets"`) {
		t.Fatalf("explore failed: %d %s", rec.Code, rec.Body.String())
	}
}

func TestARDErrorBranches(t *testing.T) {
	gin.SetMode(gin.TestMode)
	store := &ardHandlerStore{config: map[string]string{}}
	handler := NewARDHandler(store, func() config.ARDConfig {
		return config.ARDConfig{
			Enabled:         true,
			PublicBaseURL:   "https://cp.example.com",
			PublisherDomain: "example.com",
			Publish:         config.ARDPublishConfig{Enabled: true},
			External:        config.ARDExternalConfig{InvocationEnabled: false},
		}
	}, nil)
	router := gin.New()
	router.PUT("/api/ui/v1/ard/publications", handler.SavePublication)
	router.POST("/api/ui/v1/ard/imports", handler.ImportExternal)
	router.PUT("/api/ui/v1/ard/imports/:entryID/binding", handler.SaveExternalBinding)
	router.POST("/api/ui/v1/ard/external/search", handler.SearchExternal)
	router.GET("/api/v1/ard/artifacts/:entryID", handler.GetArtifact)

	rec := serveARD(router, http.MethodPut, "/api/ui/v1/ard/publications", map[string]any{"target_kind": "other"})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("bad publication should fail: %d %s", rec.Code, rec.Body.String())
	}
	rec = serveARD(router, http.MethodPost, "/api/ui/v1/ard/imports", map[string]any{"entry": map[string]any{"displayName": "Missing ID"}})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("bad import should fail: %d %s", rec.Code, rec.Body.String())
	}
	rec = serveARD(router, http.MethodPut, "/api/ui/v1/ard/imports/ext_missing/binding", map[string]any{
		"external_entry_id": "other",
	})
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("mismatched binding should fail: %d %s", rec.Code, rec.Body.String())
	}
	rec = serveARD(router, http.MethodPut, "/api/ui/v1/ard/imports/ext_missing/binding", map[string]any{
		"callable": true,
	})
	if rec.Code != http.StatusForbidden {
		t.Fatalf("disabled invocation should fail: %d %s", rec.Code, rec.Body.String())
	}
	rec = serveARD(router, http.MethodPost, "/api/ui/v1/ard/external/search", map[string]any{"query": map[string]any{"text": "review"}})
	if rec.Code != http.StatusForbidden {
		t.Fatalf("disabled external search should fail: %d %s", rec.Code, rec.Body.String())
	}
	rec = serveARD(router, http.MethodGet, "/api/v1/ard/artifacts/missing", nil)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("missing artifact should 404: %d %s", rec.Code, rec.Body.String())
	}
}

func TestARDHandlerRequestAndStorageErrorBranches(t *testing.T) {
	gin.SetMode(gin.TestMode)
	cfg := func() config.ARDConfig {
		return config.ARDConfig{
			Enabled:         true,
			PublicBaseURL:   "https://cp.example.com",
			PublisherDomain: "example.com",
			Publish: config.ARDPublishConfig{
				Enabled:               true,
				IncludeHealthStatuses: []string{"active"},
			},
			Registry: config.ARDRegistryConfig{Enabled: true, Public: true},
			External: config.ARDExternalConfig{
				SearchEnabled:      true,
				InvocationEnabled:  true,
				AllowedRegistries:  []string{"https://registry.example.com/api/v1/ard"},
				DefaultSearchLimit: 3,
			},
		}
	}
	makeStore := func() *ardHandlerStore {
		store := &ardHandlerStore{
			agents: []*types.AgentNode{{
				ID:           "node-1",
				Version:      "1.0.0",
				HealthStatus: types.HealthStatusActive,
				Reasoners: []types.ReasonerDefinition{{
					ID:          "review_contract",
					InputSchema: json.RawMessage(`{"type":"object"}`),
				}},
			}},
			config: map[string]string{},
		}
		require.NoError(t, ard.SaveState(context.Background(), store, ard.State{
			Publications: map[string]ard.Publication{
				ard.TargetKey("reasoner", "node-1", "review_contract"): {
					TargetKind:            "reasoner",
					NodeID:                "node-1",
					TargetID:              "review_contract",
					Published:             true,
					DisplayName:           "Review Contract",
					Description:           "Review contract language.",
					RepresentativeQueries: []string{"review this MSA", "find risky indemnity clauses"},
					ArtifactType:          "application/openapi+json",
				},
			},
			Imports: []ard.ExternalEntry{{ID: "ext_1", Identifier: "urn:ai:vendor.example:agent:review"}},
		}, "test"))
		return store
	}
	serve := func(store *ardHandlerStore, configure func(*ardHandlerStore), method, path, body string) *httptest.ResponseRecorder {
		if configure != nil {
			configure(store)
		}
		handler := NewARDHandler(store, cfg, func() bool { return false })
		router := gin.New()
		router.GET("/api/ui/v1/ard", handler.GetDashboard)
		router.PUT("/api/ui/v1/ard/settings", handler.UpdateSettings)
		router.PUT("/api/ui/v1/ard/publications", handler.SavePublication)
		router.POST("/api/ui/v1/ard/external/search", handler.SearchExternal)
		router.POST("/api/ui/v1/ard/imports", handler.ImportExternal)
		router.PUT("/api/ui/v1/ard/imports/:entryID/binding", handler.SaveExternalBinding)
		router.PUT("/api/ui/v1/ard/registries", handler.SaveRegistries)
		router.GET("/.well-known/ai-catalog.json", handler.GetPublicCatalog)
		router.GET("/api/v1/ard/artifacts/:entryID", handler.GetArtifact)
		router.POST("/api/v1/ard/search", handler.SearchRegistry)
		rec := httptest.NewRecorder()
		req := httptest.NewRequest(method, path, strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		router.ServeHTTP(rec, req)
		return rec
	}

	for name, tc := range map[string]struct {
		method string
		path   string
		body   string
		mutate func(*ardHandlerStore)
		want   int
	}{
		"dashboard get error": {
			method: http.MethodGet, path: "/api/ui/v1/ard", mutate: func(s *ardHandlerStore) { s.getErr = errors.New("get failed") }, want: http.StatusInternalServerError,
		},
		"settings bad json": {
			method: http.MethodPut, path: "/api/ui/v1/ard/settings", body: "{", want: http.StatusBadRequest,
		},
		"settings get error": {
			method: http.MethodPut, path: "/api/ui/v1/ard/settings", body: `{}`, mutate: func(s *ardHandlerStore) { s.getErr = errors.New("get failed") }, want: http.StatusInternalServerError,
		},
		"settings set error": {
			method: http.MethodPut, path: "/api/ui/v1/ard/settings", body: `{}`, mutate: func(s *ardHandlerStore) { s.setErr = errors.New("set failed") }, want: http.StatusInternalServerError,
		},
		"publication bad json": {
			method: http.MethodPut, path: "/api/ui/v1/ard/publications", body: "{", want: http.StatusBadRequest,
		},
		"publication invalid metadata still saves": {
			method: http.MethodPut, path: "/api/ui/v1/ard/publications", body: `{"target_kind":"reasoner","node_id":"node-1","target_id":"review_contract","published":true}`, want: http.StatusOK,
		},
		"publication set error": {
			method: http.MethodPut, path: "/api/ui/v1/ard/publications", body: `{"target_kind":"reasoner","node_id":"node-1","target_id":"review_contract"}`, mutate: func(s *ardHandlerStore) { s.setErr = errors.New("set failed") }, want: http.StatusInternalServerError,
		},
		"external search get error": {
			method: http.MethodPost, path: "/api/ui/v1/ard/external/search", body: `{"query":{"text":"review"}}`, mutate: func(s *ardHandlerStore) { s.getErr = errors.New("get failed") }, want: http.StatusInternalServerError,
		},
		"external search bad json": {
			method: http.MethodPost, path: "/api/ui/v1/ard/external/search", body: "{", want: http.StatusBadRequest,
		},
		"external search default page size": {
			method: http.MethodPost, path: "/api/ui/v1/ard/external/search", body: `{"query":{"text":"review"},"registries":["https://blocked.example.com/api/v1/ard"]}`, want: http.StatusOK,
		},
		"import bad json": {
			method: http.MethodPost, path: "/api/ui/v1/ard/imports", body: "{", want: http.StatusBadRequest,
		},
		"import get error": {
			method: http.MethodPost, path: "/api/ui/v1/ard/imports", body: `{"entry":{"identifier":"urn:ai:vendor.example:agent:new"}}`, mutate: func(s *ardHandlerStore) { s.getErr = errors.New("get failed") }, want: http.StatusInternalServerError,
		},
		"import set error": {
			method: http.MethodPost, path: "/api/ui/v1/ard/imports", body: `{"entry":{"identifier":"urn:ai:vendor.example:agent:new"}}`, mutate: func(s *ardHandlerStore) { s.setErr = errors.New("set failed") }, want: http.StatusInternalServerError,
		},
		"binding bad json": {
			method: http.MethodPut, path: "/api/ui/v1/ard/imports/ext_1/binding", body: "{", want: http.StatusBadRequest,
		},
		"binding get error": {
			method: http.MethodPut, path: "/api/ui/v1/ard/imports/ext_1/binding", body: `{}`, mutate: func(s *ardHandlerStore) { s.getErr = errors.New("get failed") }, want: http.StatusInternalServerError,
		},
		"binding missing import": {
			method: http.MethodPut, path: "/api/ui/v1/ard/imports/ext_missing/binding", body: `{}`, want: http.StatusNotFound,
		},
		"binding set error": {
			method: http.MethodPut, path: "/api/ui/v1/ard/imports/ext_1/binding", body: `{}`, mutate: func(s *ardHandlerStore) { s.setErr = errors.New("set failed") }, want: http.StatusInternalServerError,
		},
		"registries bad json": {
			method: http.MethodPut, path: "/api/ui/v1/ard/registries", body: "{", want: http.StatusBadRequest,
		},
		"registries get error": {
			method: http.MethodPut, path: "/api/ui/v1/ard/registries", body: `{"registries":[]}`, mutate: func(s *ardHandlerStore) { s.getErr = errors.New("get failed") }, want: http.StatusInternalServerError,
		},
		"registries set error": {
			method: http.MethodPut, path: "/api/ui/v1/ard/registries", body: `{"registries":[]}`, mutate: func(s *ardHandlerStore) { s.setErr = errors.New("set failed") }, want: http.StatusInternalServerError,
		},
		"artifact get error": {
			method: http.MethodGet, path: "/api/v1/ard/artifacts/node-1.review_contract", mutate: func(s *ardHandlerStore) { s.getErr = errors.New("get failed") }, want: http.StatusInternalServerError,
		},
		"artifact build error": {
			method: http.MethodGet, path: "/api/v1/ard/artifacts/node-1.review_contract", mutate: func(s *ardHandlerStore) { s.listErr = errors.New("list failed") }, want: http.StatusInternalServerError,
		},
		"registry search bad json": {
			method: http.MethodPost, path: "/api/v1/ard/search", body: "{", want: http.StatusBadRequest,
		},
		"registry search missing query": {
			method: http.MethodPost, path: "/api/v1/ard/search", body: `{"query":{"text":" "}}`, want: http.StatusBadRequest,
		},
	} {
		t.Run(name, func(t *testing.T) {
			rec := serve(makeStore(), tc.mutate, tc.method, tc.path, tc.body)
			if rec.Code != tc.want {
				t.Fatalf("expected %d, got %d body=%s", tc.want, rec.Code, rec.Body.String())
			}
		})
	}

	t.Run("public catalog disabled", func(t *testing.T) {
		handler := NewARDHandler(makeStore(), func() config.ARDConfig { return config.ARDConfig{} }, nil)
		router := gin.New()
		router.GET("/.well-known/ai-catalog.json", handler.GetPublicCatalog)
		rec := serveARD(router, http.MethodGet, "/.well-known/ai-catalog.json", nil)
		if rec.Code != http.StatusNotFound {
			t.Fatalf("expected unpublished catalog to 404, got %d %s", rec.Code, rec.Body.String())
		}
	})
}

func serveARD(router *gin.Engine, method string, path string, payload any) *httptest.ResponseRecorder {
	var body *bytes.Buffer
	if payload == nil {
		body = bytes.NewBuffer(nil)
	} else {
		raw, _ := json.Marshal(payload)
		body = bytes.NewBuffer(raw)
	}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(method, path, body)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Caller-Agent-ID", "tester")
	router.ServeHTTP(rec, req)
	return rec
}
