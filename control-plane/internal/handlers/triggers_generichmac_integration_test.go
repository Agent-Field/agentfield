package handlers

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
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
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/generichmac"
)

// hmacSign creates a HMAC-SHA256 signature over the body.
func hmacSign(body []byte, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write(body)
	return hex.EncodeToString(mac.Sum(nil))
}

// setupGenericHMACTestEnv creates storage and dispatcher for generic_hmac webhook tests.
func setupGenericHMACTestEnv(t *testing.T) (storage.StorageProvider, *services.TriggerDispatcher, context.Context) {
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

// TestGenericHMACIngest_DefaultHeader tests the full ingest flow with default X-Signature header.
func TestGenericHMACIngest_DefaultHeader(t *testing.T) {
	provider, dispatcher, ctx := setupGenericHMACTestEnv(t)
	secret := "hmac_test_secret_default"

	// Set up fake target server
	var (
		mu            sync.Mutex
		targetCalled  bool
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
		ID:              "hmac-target-default",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_webhook"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	// Default config has no event_type_header — leave EventTypes empty so the
	// match-all path applies. (Filtering by event type with generic_hmac
	// requires configuring event_type_header; see CustomHeaderAndPrefix.)
	trig := &types.Trigger{
		ID:             "hmac_trigger_default",
		SourceName:     "generic_hmac",
		TargetNodeID:   "hmac-target-default",
		TargetReasoner: "handle_webhook",
		SecretEnvVar:   "HMAC_TEST_SECRET",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		Config:         json.RawMessage(`{}`),
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	// Set up Gin router with ingest handler
	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, dispatcher, nil)
	r.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	// Prepare request
	body := []byte(`{"order_id":"123","amount":99.99}`)
	sig := hmacSign(body, secret)
	t.Setenv("HMAC_TEST_SECRET", secret)

	req := httptest.NewRequest("POST", "/sources/hmac_trigger_default", strings.NewReader(string(body)))
	req.Header.Set("X-Signature", sig)
	req.Header.Set("X-Event-Type", "order.created")
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

	// Verify inbound event persisted. EventType stays empty here because
	// the default config doesn't extract it from any header — that's tested
	// in CustomHeaderAndPrefix below.
	events, err := provider.ListInboundEvents(ctx, "hmac_trigger_default", 10)
	require.NoError(t, err)
	require.Equal(t, 1, len(events))

	// Give dispatch time to complete
	time.Sleep(100 * time.Millisecond)

	mu.Lock()
	assert.True(t, targetCalled, "target reasoner should have been invoked")
	assert.Equal(t, "generic_hmac", gotSourceName)
	mu.Unlock()
}

// TestGenericHMACIngest_CustomHeaderAndPrefix tests custom signature header and prefix.
func TestGenericHMACIngest_CustomHeaderAndPrefix(t *testing.T) {
	provider, dispatcher, ctx := setupGenericHMACTestEnv(t)
	secret := "hmac_test_secret_custom"

	var (
		mu           sync.Mutex
		targetCalled bool
		gotEventType string
	)
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		targetCalled = true
		gotEventType = r.Header.Get("X-Event-Type")
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	node := &types.AgentNode{
		ID:              "hmac-target-custom",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_webhook"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	// Create trigger with custom config
	cfg := json.RawMessage(`{
		"signature_header": "X-Custom-Sig",
		"signature_prefix": "v1=",
		"event_type_header": "X-Event-Type",
		"idempotency_header": "X-Idempotency"
	}`)
	trig := &types.Trigger{
		ID:             "hmac_trigger_custom",
		SourceName:     "generic_hmac",
		TargetNodeID:   "hmac-target-custom",
		TargetReasoner: "handle_webhook",
		SecretEnvVar:   "HMAC_TEST_SECRET",
		EventTypes:     []string{"payment.processed"},
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		Config:         cfg,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, dispatcher, nil)
	r.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	body := []byte(`{"tx_id":"tx_001","amount":500}`)
	sig := hmacSign(body, secret)
	t.Setenv("HMAC_TEST_SECRET", secret)

	req := httptest.NewRequest("POST", "/sources/hmac_trigger_custom", strings.NewReader(string(body)))
	req.Header.Set("X-Custom-Sig", "v1="+sig)
	req.Header.Set("X-Event-Type", "payment.processed")
	req.Header.Set("X-Idempotency", "idempotent_key_123")

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var resp map[string]interface{}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	assert.Equal(t, float64(1), resp["received"])

	events, err := provider.ListInboundEvents(ctx, "hmac_trigger_custom", 10)
	require.NoError(t, err)
	require.Equal(t, 1, len(events))
	assert.Equal(t, "payment.processed", events[0].EventType)
	assert.Equal(t, "idempotent_key_123", events[0].IdempotencyKey)

	// Idempotency is asserted on the persisted event row above. The
	// dispatcher does not propagate it as an outbound header today.
	time.Sleep(100 * time.Millisecond)
	mu.Lock()
	assert.True(t, targetCalled)
	assert.Equal(t, "payment.processed", gotEventType)
	mu.Unlock()
}

// TestGenericHMACIngest_TamperedBody tests rejection of a request with tampered body.
func TestGenericHMACIngest_TamperedBody(t *testing.T) {
	provider, _, ctx := setupGenericHMACTestEnv(t)
	secret := "hmac_test_secret_tampered"

	node := &types.AgentNode{
		ID:              "hmac-target-tamper",
		BaseURL:         "http://localhost:9999",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_webhook"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "hmac_trigger_tamper",
		SourceName:     "generic_hmac",
		TargetNodeID:   "hmac-target-tamper",
		TargetReasoner: "handle_webhook",
		SecretEnvVar:   "HMAC_TEST_SECRET",
		EventTypes:     []string{"action"},
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

	// Sign correct body
	body := []byte(`{"action":"create"}`)
	sig := hmacSign(body, secret)
	t.Setenv("HMAC_TEST_SECRET", secret)

	// But send tampered body
	tampered := []byte(`{"action":"delete"}`)
	req := httptest.NewRequest("POST", "/sources/hmac_trigger_tamper", strings.NewReader(string(tampered)))
	req.Header.Set("X-Signature", sig)
	req.Header.Set("X-Event-Type", "action")

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "signature")

	// Verify no event persisted
	events, err := provider.ListInboundEvents(ctx, "hmac_trigger_tamper", 10)
	require.NoError(t, err)
	assert.Equal(t, 0, len(events))
}

// TestGenericHMACIngest_MissingSignature tests rejection of a request without signature.
func TestGenericHMACIngest_MissingSignature(t *testing.T) {
	provider, _, ctx := setupGenericHMACTestEnv(t)

	node := &types.AgentNode{
		ID:              "hmac-target-nosig",
		BaseURL:         "http://localhost:9999",
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_webhook"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "hmac_trigger_nosig",
		SourceName:     "generic_hmac",
		TargetNodeID:   "hmac-target-nosig",
		TargetReasoner: "handle_webhook",
		SecretEnvVar:   "HMAC_TEST_SECRET",
		EventTypes:     []string{"action"},
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

	body := []byte(`{"action":"create"}`)
	t.Setenv("HMAC_TEST_SECRET", "secret123")

	req := httptest.NewRequest("POST", "/sources/hmac_trigger_nosig", strings.NewReader(string(body)))
	// Intentionally omit X-Signature header

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusUnauthorized, w.Code)
	assert.Contains(t, w.Body.String(), "missing")

	events, err := provider.ListInboundEvents(ctx, "hmac_trigger_nosig", 10)
	require.NoError(t, err)
	assert.Equal(t, 0, len(events))
}
