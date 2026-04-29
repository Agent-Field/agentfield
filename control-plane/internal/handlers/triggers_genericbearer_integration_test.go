package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	// Import sources to register them
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/genericbearer"
)

// setupGenericBearerTestEnv creates storage and dispatcher for generic_bearer webhook tests.
func setupGenericBearerTestEnv(t *testing.T) (storage.StorageProvider, *services.TriggerDispatcher, context.Context) {
	t.Helper()

	ctx := context.Background()
	tempDir := t.TempDir()

	cfg := storage.StorageConfig{
		Mode: "local",
		Local: storage.LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "agentfield.db"),
			KVStorePath:  filepath.Join(tempDir, "agentfield.bolt"),
		},
	}

	provider := storage.NewLocalStorage(storage.LocalStorageConfig{})
	require.NoError(t, provider.Initialize(ctx, cfg))

	t.Cleanup(func() {
		_ = provider.Close(ctx)
	})

	// VCService with DID disabled for testing
	disabledCfg := &config.DIDConfig{Enabled: false}
	vcService := services.NewVCService(disabledCfg, nil, provider)
	dispatcher := services.NewTriggerDispatcher(provider, vcService)

	return provider, dispatcher, ctx
}

// TestGenericBearerIngest_DefaultBearerScheme tests the full ingest flow with default Authorization Bearer header.
func TestGenericBearerIngest_DefaultBearerScheme(t *testing.T) {
	provider, dispatcher, ctx := setupGenericBearerTestEnv(t)
	secret := "tok_test_default_123"

	// Set up fake target server
	var (
		mu           sync.Mutex
		targetCalled bool
		gotSourceName string
	)
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		targetCalled = true
		gotSourceName = r.Header.Get("X-Source-Name")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer target.Close()

	// Register target node
	node := &types.AgentNode{
		ID:              "bearer-target-default",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_event"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	// Create trigger with default config (Bearer scheme)
	trig := &types.Trigger{
		ID:             "bearer_trigger_default",
		SourceName:     "generic_bearer",
		TargetNodeID:   "bearer-target-default",
		TargetReasoner: "handle_event",
		SecretEnvVar:   "BEARER_TOKEN",
		// generic_bearer doesn't extract event types from the body, so leave
		// EventTypes empty (match-all). Filtering by event type with
		// generic_bearer requires an event_type_header config.
		ManagedBy: types.ManagedByUI,
		Enabled:   true,
		Config:    json.RawMessage(`{}`),
		CreatedAt: time.Now().UTC(),
		UpdatedAt: time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	// Set up Gin router with ingest handler
	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, dispatcher, nil)
	r.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	// Prepare request
	body := []byte(`{"data":"test"}`)
	t.Setenv("BEARER_TOKEN", secret)

	req := httptest.NewRequest("POST", "/sources/bearer_trigger_default", strings.NewReader(string(body)))
	req.Header.Set("Authorization", "Bearer "+secret)
	req.Header.Set("Content-Type", "application/json")

	// Execute
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	// Assertions
	assert.Equal(t, http.StatusOK, w.Code)
	var resp map[string]interface{}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	assert.Equal(t, "ok", resp["status"])
	assert.Equal(t, float64(1), resp["received"])

	// Verify inbound event persisted
	events, err := provider.ListInboundEvents(ctx, "bearer_trigger_default", 10)
	require.NoError(t, err)
	require.Equal(t, 1, len(events))

	// Give dispatch time to complete
	time.Sleep(100 * time.Millisecond)

	mu.Lock()
	assert.True(t, targetCalled, "target reasoner should have been invoked")
	assert.Equal(t, "generic_bearer", gotSourceName)
	mu.Unlock()
}

// TestGenericBearerIngest_CustomHeaderEmptyScheme tests custom header with no scheme prefix (raw token).
func TestGenericBearerIngest_CustomHeaderEmptyScheme(t *testing.T) {
	provider, dispatcher, ctx := setupGenericBearerTestEnv(t)
	secret := "raw_token_xyz"

	var (
		mu           sync.Mutex
		targetCalled bool
	)
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		targetCalled = true
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	node := &types.AgentNode{
		ID:              "bearer-target-custom",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_event"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	// Create trigger with custom header and empty scheme
	cfg := json.RawMessage(`{
		"header": "X-API-Key",
		"scheme": ""
	}`)
	trig := &types.Trigger{
		ID:             "bearer_trigger_custom",
		SourceName:     "generic_bearer",
		TargetNodeID:   "bearer-target-custom",
		TargetReasoner: "handle_event",
		SecretEnvVar:   "API_KEY",
		ManagedBy: types.ManagedByUI,
		Enabled:   true,
		Config:    cfg,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, dispatcher, nil)
	r.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	body := []byte(`{"msg":"hello"}`)
	t.Setenv("API_KEY", secret)

	req := httptest.NewRequest("POST", "/sources/bearer_trigger_custom", strings.NewReader(string(body)))
	req.Header.Set("X-API-Key", secret) // Raw token, no "Bearer " prefix

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var resp map[string]interface{}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	assert.Equal(t, float64(1), resp["received"])

	events, err := provider.ListInboundEvents(ctx, "bearer_trigger_custom", 10)
	require.NoError(t, err)
	require.Equal(t, 1, len(events))

	time.Sleep(100 * time.Millisecond)
	mu.Lock()
	assert.True(t, targetCalled)
	mu.Unlock()
}

// TestGenericBearerIngest_WrongToken tests rejection of a request with wrong token.
func TestGenericBearerIngest_WrongToken(t *testing.T) {
	provider, _, ctx := setupGenericBearerTestEnv(t)
	secret := "correct_token"

	node := &types.AgentNode{
		ID:              "bearer-target-wrong",
		BaseURL:         "http://localhost:9999",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_event"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "bearer_trigger_wrong",
		SourceName:     "generic_bearer",
		TargetNodeID:   "bearer-target-wrong",
		TargetReasoner: "handle_event",
		SecretEnvVar:   "BEARER_TOKEN",
		EventTypes:     []string{"webhook"},
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		Config:         json.RawMessage(`{}`),
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, nil, nil)
	r.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	t.Setenv("BEARER_TOKEN", secret)

	body := []byte(`{"data":"test"}`)
	req := httptest.NewRequest("POST", "/sources/bearer_trigger_wrong", strings.NewReader(string(body)))
	req.Header.Set("Authorization", "Bearer wrong_token")

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "token")

	// Verify no event persisted
	events, err := provider.ListInboundEvents(ctx, "bearer_trigger_wrong", 10)
	require.NoError(t, err)
	assert.Equal(t, 0, len(events))
}

// TestGenericBearerIngest_MissingHeader tests rejection of a request without auth header.
func TestGenericBearerIngest_MissingHeader(t *testing.T) {
	provider, _, ctx := setupGenericBearerTestEnv(t)

	node := &types.AgentNode{
		ID:              "bearer-target-noauth",
		BaseURL:         "http://localhost:9999",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_event"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "bearer_trigger_noauth",
		SourceName:     "generic_bearer",
		TargetNodeID:   "bearer-target-noauth",
		TargetReasoner: "handle_event",
		SecretEnvVar:   "BEARER_TOKEN",
		EventTypes:     []string{"webhook"},
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		Config:         json.RawMessage(`{}`),
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, nil, nil)
	r.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	t.Setenv("BEARER_TOKEN", "secret123")

	body := []byte(`{"data":"test"}`)
	req := httptest.NewRequest("POST", "/sources/bearer_trigger_noauth", strings.NewReader(string(body)))
	// Intentionally omit Authorization header

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "missing")

	events, err := provider.ListInboundEvents(ctx, "bearer_trigger_noauth", 10)
	require.NoError(t, err)
	assert.Equal(t, 0, len(events))
}
