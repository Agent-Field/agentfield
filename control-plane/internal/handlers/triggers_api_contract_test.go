package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
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
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/all"
)

// setupAPIContractTestEnv creates storage and dispatcher for Phase 4 API tests.
func setupAPIContractTestEnv(t *testing.T) (storage.StorageProvider, *services.TriggerDispatcher, context.Context) {
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

// TestGetSource_HappyPath tests GET /api/v1/sources/:name returns complete metadata.
func TestGetSource_HappyPath(t *testing.T) {
	provider, _, _ := setupAPIContractTestEnv(t)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, nil, nil)
	r.GET("/api/v1/sources/:name", handlers.GetSource())

	req := httptest.NewRequest("GET", "/api/v1/sources/generic_hmac", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var resp map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)
	assert.Equal(t, "generic_hmac", resp["name"])
	assert.Equal(t, "http", resp["kind"])
	assert.Equal(t, true, resp["secret_required"])
	assert.NotNil(t, resp["config_schema"])
}

// TestGetSource_Unknown tests GET /api/v1/sources/:name with nonexistent source.
func TestGetSource_Unknown(t *testing.T) {
	provider, _, _ := setupAPIContractTestEnv(t)

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, nil, nil)
	r.GET("/api/v1/sources/:name", handlers.GetSource())

	req := httptest.NewRequest("GET", "/api/v1/sources/nonexistent_source_xyz", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusNotFound, w.Code)
	var resp map[string]interface{}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	assert.Equal(t, "source not found", resp["error"])
}

// TestGetTriggerEvent_HappyPath tests GET /api/v1/triggers/:trigger_id/events/:event_id.
func TestGetTriggerEvent_HappyPath(t *testing.T) {
	provider, _, ctx := setupAPIContractTestEnv(t)

	// Create a test trigger
	trig := &types.Trigger{
		ID:             "test-trigger-1",
		SourceName:     "generic_hmac",
		Config:         json.RawMessage("{}"),
		TargetNodeID:   "test-node",
		TargetReasoner: "test-reasoner",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	// Create an inbound event
	testPayload := json.RawMessage(`{"test":"data"}`)
	event := &types.InboundEvent{
		ID:                "event-123",
		TriggerID:         trig.ID,
		SourceName:        "generic_hmac",
		EventType:         "test.event",
		RawPayload:        testPayload,
		NormalizedPayload: testPayload,
		Status:            types.InboundEventStatusReceived,
		ReceivedAt:        time.Now().UTC(),
	}
	require.NoError(t, provider.InsertInboundEvent(ctx, event))

	// Setup router and handler
	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, nil, nil)
	r.GET("/api/v1/triggers/:trigger_id/events/:event_id", handlers.GetTriggerEvent())

	req := httptest.NewRequest("GET", "/api/v1/triggers/test-trigger-1/events/event-123", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var resp types.InboundEvent
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)
	assert.Equal(t, event.ID, resp.ID)
	assert.Equal(t, event.TriggerID, resp.TriggerID)
	assert.Equal(t, event.EventType, resp.EventType)
	assert.Equal(t, testPayload, resp.RawPayload)
}

// TestGetTriggerEvent_WrongTrigger tests enumeration prevention.
func TestGetTriggerEvent_WrongTrigger(t *testing.T) {
	provider, _, ctx := setupAPIContractTestEnv(t)

	// Create two triggers
	trig1 := &types.Trigger{
		ID:             "trigger-1",
		SourceName:     "generic_hmac",
		Config:         json.RawMessage("{}"),
		TargetNodeID:   "node1",
		TargetReasoner: "reasoner1",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	trig2 := &types.Trigger{
		ID:             "trigger-2",
		SourceName:     "generic_hmac",
		Config:         json.RawMessage("{}"),
		TargetNodeID:   "node2",
		TargetReasoner: "reasoner2",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig1))
	require.NoError(t, provider.CreateTrigger(ctx, trig2))

	// Create event for trigger1
	event := &types.InboundEvent{
		ID:                "event-xyz",
		TriggerID:         trig1.ID,
		SourceName:        "generic_hmac",
		EventType:         "test",
		RawPayload:        json.RawMessage("{}"),
		NormalizedPayload: json.RawMessage("{}"),
		Status:            types.InboundEventStatusReceived,
		ReceivedAt:        time.Now().UTC(),
	}
	require.NoError(t, provider.InsertInboundEvent(ctx, event))

	// Try to access the event via trigger2's path (should fail)
	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, nil, nil)
	r.GET("/api/v1/triggers/:trigger_id/events/:event_id", handlers.GetTriggerEvent())

	req := httptest.NewRequest("GET", "/api/v1/triggers/trigger-2/events/event-xyz", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusNotFound, w.Code)
	var resp map[string]interface{}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	assert.Equal(t, "event not found", resp["error"])
}

// TestGetSecretStatus_Set tests secret status when env var is set.
func TestGetSecretStatus_Set(t *testing.T) {
	provider, _, ctx := setupAPIContractTestEnv(t)

	trig := &types.Trigger{
		ID:             "secret-test-1",
		SourceName:     "generic_hmac",
		SecretEnvVar:   "TEST_SECRET_VAR",
		Config:         json.RawMessage("{}"),
		TargetNodeID:   "node",
		TargetReasoner: "reasoner",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, nil, nil)
	r.GET("/api/v1/triggers/:trigger_id/secret-status", handlers.GetSecretStatus())

	// Set the env var
	t.Setenv("TEST_SECRET_VAR", "secret_value_123")

	req := httptest.NewRequest("GET", "/api/v1/triggers/secret-test-1/secret-status", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var resp map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)
	assert.Equal(t, "TEST_SECRET_VAR", resp["env_var"])
	assert.Equal(t, true, resp["set"])
}

// TestGetSecretStatus_Unset tests secret status when env var is NOT set.
func TestGetSecretStatus_Unset(t *testing.T) {
	provider, _, ctx := setupAPIContractTestEnv(t)

	trig := &types.Trigger{
		ID:             "secret-test-2",
		SourceName:     "generic_hmac",
		SecretEnvVar:   "UNSET_SECRET_VAR_XYZ",
		Config:         json.RawMessage("{}"),
		TargetNodeID:   "node",
		TargetReasoner: "reasoner",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, nil, nil)
	r.GET("/api/v1/triggers/:trigger_id/secret-status", handlers.GetSecretStatus())

	// Don't set the env var — verify it's not in environment
	req := httptest.NewRequest("GET", "/api/v1/triggers/secret-test-2/secret-status", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)
	var resp map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)
	assert.Equal(t, "UNSET_SECRET_VAR_XYZ", resp["env_var"])
	assert.Equal(t, false, resp["set"])
}

// TestTestTrigger_GenericHMAC tests synthetic test event for generic_hmac source.
func TestTestTrigger_GenericHMAC(t *testing.T) {
	provider, dispatcher, ctx := setupAPIContractTestEnv(t)

	// Set up a test target node
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer target.Close()

	node := &types.AgentNode{
		ID:              "test-node",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "test-reasoner"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	// Create a generic_hmac trigger
	trig := &types.Trigger{
		ID:             "hmac-test-trigger",
		SourceName:     "generic_hmac",
		SecretEnvVar:   "TEST_HMAC_SECRET",
		Config:         json.RawMessage(`{"signature_header":"X-Signature"}`),
		TargetNodeID:   "test-node",
		TargetReasoner: "test-reasoner",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	// Set the secret env var
	t.Setenv("TEST_HMAC_SECRET", "test_secret_123")

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, dispatcher, nil)
	r.POST("/api/v1/triggers/:trigger_id/test", handlers.TestTrigger())

	// Send test request
	body := `{"payload":{"test":"data"},"event_type":"test.hmac"}`
	req := httptest.NewRequest("POST", "/api/v1/triggers/hmac-test-trigger/test",
		strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusAccepted, w.Code)
	var resp map[string]interface{}
	err := json.Unmarshal(w.Body.Bytes(), &resp)
	require.NoError(t, err)
	assert.NotEmpty(t, resp["event_id"])
	assert.Equal(t, "accepted", resp["status"])

	// Verify event was persisted
	eventID := resp["event_id"].(string)
	ev, err := provider.GetInboundEvent(ctx, eventID)
	require.NoError(t, err)
	assert.Equal(t, "test.hmac", ev.EventType)
	assert.Equal(t, "hmac-test-trigger", ev.TriggerID)
}

// TestTestTrigger_UnsupportedSource tests synthetic test for unsupported source (stripe).
func TestTestTrigger_UnsupportedSource(t *testing.T) {
	provider, _, ctx := setupAPIContractTestEnv(t)

	trig := &types.Trigger{
		ID:             "stripe-test-trigger",
		SourceName:     "stripe",
		SecretEnvVar:   "STRIPE_SECRET",
		Config:         json.RawMessage(`{}`),
		TargetNodeID:   "node",
		TargetReasoner: "reasoner",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, nil, nil)
	r.POST("/api/v1/triggers/:trigger_id/test", handlers.TestTrigger())

	body := `{"payload":{},"event_type":"charge.succeeded"}`
	req := httptest.NewRequest("POST", "/api/v1/triggers/stripe-test-trigger/test",
		strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusNotImplemented, w.Code)
	var resp map[string]interface{}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	assert.Contains(t, resp["error"].(string), "synthetic test events not yet supported for source")
}

// TestTestTrigger_MissingSecret tests synthetic test when required secret is not set.
func TestTestTrigger_MissingSecret(t *testing.T) {
	provider, _, ctx := setupAPIContractTestEnv(t)

	trig := &types.Trigger{
		ID:             "missing-secret-trigger",
		SourceName:     "generic_hmac",
		SecretEnvVar:   "UNSET_SECRET_XYZ",
		Config:         json.RawMessage(`{}`),
		TargetNodeID:   "node",
		TargetReasoner: "reasoner",
		ManagedBy:      types.ManagedByUI,
		Enabled:        true,
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	gin.SetMode(gin.TestMode)
	r := gin.New()
	handlers := NewTriggerHandlers(provider, nil, nil)
	r.POST("/api/v1/triggers/:trigger_id/test", handlers.TestTrigger())

	req := httptest.NewRequest("POST", "/api/v1/triggers/missing-secret-trigger/test",
		strings.NewReader(`{}`))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)

	assert.Equal(t, http.StatusBadRequest, w.Code)
	var resp map[string]interface{}
	_ = json.Unmarshal(w.Body.Bytes(), &resp)
	assert.Contains(t, resp["error"].(string), "not set")
}
