package handlers

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/config"
	"github.com/Agent-Field/agentfield/control-plane/internal/services"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	_ "github.com/Agent-Field/agentfield/control-plane/internal/sources/stripe"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// setupStripeTestEnv initializes storage, dispatcher, and context for Stripe webhook tests.
func setupStripeTestEnv(t *testing.T) (storage.StorageProvider, *services.TriggerDispatcher, context.Context) {
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

	// VCService with DID disabled for testing (focus on ingest, not VC chain).
	disabledCfg := &config.DIDConfig{Enabled: false}
	vcService := services.NewVCService(disabledCfg, nil, provider)
	dispatcher := services.NewTriggerDispatcher(provider, vcService)

	return provider, dispatcher, ctx
}

// computeStripeSignature generates a valid Stripe-Signature header value
// for the given timestamp and body using the secret.
func computeStripeSignature(ts int64, body []byte, secret string) string {
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(strconv.FormatInt(ts, 10)))
	mac.Write([]byte("."))
	mac.Write(body)
	sig := hex.EncodeToString(mac.Sum(nil))
	return fmt.Sprintf("t=%d,v1=%s", ts, sig)
}

// TestStripeIngest_HappyPath exercises the full ingest flow with valid Stripe signature,
// persistence to storage, and async dispatch to the target reasoner.
func TestStripeIngest_HappyPath(t *testing.T) {
	provider, dispatcher, ctx := setupStripeTestEnv(t)
	secret := "whsec_test_happy"
	body := []byte(`{"id":"evt_test_001","type":"payment_intent.succeeded","data":{"object":{"id":"pi_1"}}}`)
	ts := time.Now().Unix()
	sig := computeStripeSignature(ts, body, secret)

	// Setup fake target server that captures dispatch.
	var (
		mu              sync.Mutex
		dispatchedCount int
		gotSourceName   string
		gotEventType    string
	)
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		dispatchedCount++
		gotSourceName = r.Header.Get("X-Source-Name")
		gotEventType = r.Header.Get("X-Event-Type")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer target.Close()

	// Register target node.
	node := &types.AgentNode{
		ID:              "node-stripe-test",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_payment"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	// Create trigger with Stripe source.
	trig := &types.Trigger{
		ID:             "trg_stripe_001",
		SourceName:     "stripe",
		TargetNodeID:   "node-stripe-test",
		TargetReasoner: "handle_payment",
		ManagedBy:      types.ManagedByCode,
		Enabled:        true,
		SecretEnvVar:   "STRIPE_TEST_SECRET",
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	// Set environment variable for handler to read.
	t.Setenv("STRIPE_TEST_SECRET", secret)

	// Setup handler.
	sourceManager := services.NewSourceManager(provider, dispatcher)
	handlers := NewTriggerHandlers(provider, dispatcher, sourceManager)

	// Create router and register handler.
	router := gin.New()
	router.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	// POST to ingest endpoint.
	req, _ := http.NewRequest("POST", "/sources/trg_stripe_001", bytes.NewReader(body))
	req.Header.Set("Stripe-Signature", sig)
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	// Verify response.
	assert.Equal(t, http.StatusOK, w.Code)
	var respBody struct {
		Status   string `json:"status"`
		Received int    `json:"received"`
	}
	require.NoError(t, json.Unmarshal(w.Body.Bytes(), &respBody))
	assert.Equal(t, "ok", respBody.Status)
	assert.Equal(t, 1, respBody.Received)

	// Poll storage for persisted event (with deadline to avoid busy-wait).
	var storedEvent *types.InboundEvent
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		events, err := provider.ListInboundEvents(ctx, trig.ID, 100)
		if err == nil && len(events) > 0 {
			storedEvent = events[0]
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	require.NotNil(t, storedEvent, "inbound event must be persisted")
	assert.Equal(t, "evt_test_001", storedEvent.IdempotencyKey)
	assert.Equal(t, "payment_intent.succeeded", storedEvent.EventType)
	assert.Equal(t, json.RawMessage(body), storedEvent.RawPayload)

	// Poll dispatcher (async, so give it time).
	deadline = time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		mu.Lock()
		if dispatchedCount > 0 {
			mu.Unlock()
			break
		}
		mu.Unlock()
		time.Sleep(10 * time.Millisecond)
	}

	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, 1, dispatchedCount, "target reasoner must be invoked exactly once")
	assert.Equal(t, "stripe", gotSourceName)
	assert.Equal(t, "payment_intent.succeeded", gotEventType)
}

// TestStripeIngest_IdempotencyDedup verifies that posting the same event twice
// results in only one storage row and one dispatch.
func TestStripeIngest_IdempotencyDedup(t *testing.T) {
	provider, dispatcher, ctx := setupStripeTestEnv(t)
	secret := "whsec_test_idempotent"
	body := []byte(`{"id":"evt_idem_001","type":"charge.completed","data":{"object":{}}}`)
	ts := time.Now().Unix()
	sig := computeStripeSignature(ts, body, secret)

	// Setup fake target server.
	var (
		mu              sync.Mutex
		dispatchedCount int
	)
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		dispatchedCount++
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	// Register target node.
	node := &types.AgentNode{
		ID:              "node-idem-test",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "process_charge"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	// Create trigger.
	trig := &types.Trigger{
		ID:             "trg_idem_001",
		SourceName:     "stripe",
		TargetNodeID:   "node-idem-test",
		TargetReasoner: "process_charge",
		ManagedBy:      types.ManagedByCode,
		Enabled:        true,
		SecretEnvVar:   "STRIPE_IDEM_SECRET",
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	t.Setenv("STRIPE_IDEM_SECRET", secret)

	// Setup handler.
	sourceManager := services.NewSourceManager(provider, dispatcher)
	handlers := NewTriggerHandlers(provider, dispatcher, sourceManager)

	router := gin.New()
	router.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	// POST first request.
	req1, _ := http.NewRequest("POST", "/sources/trg_idem_001", bytes.NewReader(body))
	req1.Header.Set("Stripe-Signature", sig)
	req1.Header.Set("Content-Type", "application/json")

	w1 := httptest.NewRecorder()
	router.ServeHTTP(w1, req1)

	assert.Equal(t, http.StatusOK, w1.Code)
	var resp1 struct {
		Received int `json:"received"`
	}
	require.NoError(t, json.Unmarshal(w1.Body.Bytes(), &resp1))
	assert.Equal(t, 1, resp1.Received)

	// Wait for dispatch to complete.
	time.Sleep(100 * time.Millisecond)

	// POST second request (same body, timestamp).
	req2, _ := http.NewRequest("POST", "/sources/trg_idem_001", bytes.NewReader(body))
	req2.Header.Set("Stripe-Signature", sig)
	req2.Header.Set("Content-Type", "application/json")

	w2 := httptest.NewRecorder()
	router.ServeHTTP(w2, req2)

	assert.Equal(t, http.StatusOK, w2.Code)
	var resp2 struct {
		Received int `json:"received"`
	}
	require.NoError(t, json.Unmarshal(w2.Body.Bytes(), &resp2))
	assert.Equal(t, 0, resp2.Received, "duplicate event should not be persisted")

	// Verify exactly one event persisted and exactly one dispatch happened.
	events, err := provider.ListInboundEvents(ctx, trig.ID, 100)
	require.NoError(t, err)
	assert.Equal(t, 1, len(events), "only one inbound event should exist")

	time.Sleep(100 * time.Millisecond) // give async dispatch time if any
	mu.Lock()
	defer mu.Unlock()
	assert.Equal(t, 1, dispatchedCount, "dispatcher should have been called exactly once")
}

// TestStripeIngest_BadSignature verifies that an invalid signature is rejected
// with 401 and no event is persisted.
func TestStripeIngest_BadSignature(t *testing.T) {
	provider, dispatcher, ctx := setupStripeTestEnv(t)
	secret := "whsec_test_bad"
	body := []byte(`{"id":"evt_bad_sig","type":"charge.failed","data":{"object":{}}}`)
	ts := time.Now().Unix()

	// Compute signature with wrong secret.
	wrongSig := computeStripeSignature(ts, body, "wrong_secret")

	// Setup trivial target (should not be called).
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("target should not be invoked for bad signature")
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	node := &types.AgentNode{
		ID:              "node-bad-sig",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "reject_charge"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "trg_bad_sig",
		SourceName:     "stripe",
		TargetNodeID:   "node-bad-sig",
		TargetReasoner: "reject_charge",
		ManagedBy:      types.ManagedByCode,
		Enabled:        true,
		SecretEnvVar:   "STRIPE_BAD_SECRET",
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	t.Setenv("STRIPE_BAD_SECRET", secret)

	sourceManager := services.NewSourceManager(provider, dispatcher)
	handlers := NewTriggerHandlers(provider, dispatcher, sourceManager)

	router := gin.New()
	router.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	// POST with wrong signature.
	req, _ := http.NewRequest("POST", "/sources/trg_bad_sig", bytes.NewReader(body))
	req.Header.Set("Stripe-Signature", wrongSig)
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	// Expect 401 Unauthorized.
	assert.Equal(t, http.StatusUnauthorized, w.Code)

	// Verify no event was persisted.
	events, err := provider.ListInboundEvents(ctx, trig.ID, 100)
	require.NoError(t, err)
	assert.Equal(t, 0, len(events), "no event should be persisted for bad signature")
}

// TestStripeIngest_ExpiredTimestamp verifies that a signature with a timestamp
// outside the tolerance window is rejected with 401.
func TestStripeIngest_ExpiredTimestamp(t *testing.T) {
	provider, dispatcher, ctx := setupStripeTestEnv(t)
	secret := "whsec_test_expired"
	body := []byte(`{"id":"evt_expired","type":"invoice.payment_failed","data":{"object":{}}}`)

	// Timestamp 10 minutes in the past (beyond default 5-minute tolerance).
	ts := time.Now().Add(-10 * time.Minute).Unix()
	sig := computeStripeSignature(ts, body, secret)

	// Setup trivial target.
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("target should not be invoked for expired timestamp")
		w.WriteHeader(http.StatusOK)
	}))
	defer target.Close()

	node := &types.AgentNode{
		ID:              "node-expired",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "handle_invoice"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "trg_expired",
		SourceName:     "stripe",
		TargetNodeID:   "node-expired",
		TargetReasoner: "handle_invoice",
		ManagedBy:      types.ManagedByCode,
		Enabled:        true,
		SecretEnvVar:   "STRIPE_EXPIRED_SECRET",
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	t.Setenv("STRIPE_EXPIRED_SECRET", secret)

	sourceManager := services.NewSourceManager(provider, dispatcher)
	handlers := NewTriggerHandlers(provider, dispatcher, sourceManager)

	router := gin.New()
	router.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	// POST with expired timestamp.
	req, _ := http.NewRequest("POST", "/sources/trg_expired", bytes.NewReader(body))
	req.Header.Set("Stripe-Signature", sig)
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	// Expect 401 Unauthorized.
	assert.Equal(t, http.StatusUnauthorized, w.Code)

	// Verify no event was persisted.
	events, err := provider.ListInboundEvents(ctx, trig.ID, 100)
	require.NoError(t, err)
	assert.Equal(t, 0, len(events), "no event should be persisted for expired timestamp")
}

// TestStripeIngest_DispatchedEventStatusUpdate verifies that the inbound event's
// status is correctly updated after dispatch.
func TestStripeIngest_DispatchedEventStatusUpdate(t *testing.T) {
	provider, dispatcher, ctx := setupStripeTestEnv(t)
	secret := "whsec_test_status"
	body := []byte(`{"id":"evt_status_001","type":"payment_intent.amount_capturable_updated","data":{"object":{}}}`)
	ts := time.Now().Unix()
	sig := computeStripeSignature(ts, body, secret)

	// Setup fake target that responds successfully.
	target := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer target.Close()

	node := &types.AgentNode{
		ID:              "node-status",
		BaseURL:         target.URL,
		HealthStatus:    types.HealthStatusActive,
		LifecycleStatus: types.AgentStatusReady,
		Reasoners:       []types.ReasonerDefinition{{ID: "update_payment"}},
	}
	require.NoError(t, provider.RegisterAgent(ctx, node))

	trig := &types.Trigger{
		ID:             "trg_status",
		SourceName:     "stripe",
		TargetNodeID:   "node-status",
		TargetReasoner: "update_payment",
		ManagedBy:      types.ManagedByCode,
		Enabled:        true,
		SecretEnvVar:   "STRIPE_STATUS_SECRET",
		CreatedAt:      time.Now().UTC(),
		UpdatedAt:      time.Now().UTC(),
	}
	require.NoError(t, provider.CreateTrigger(ctx, trig))

	t.Setenv("STRIPE_STATUS_SECRET", secret)

	sourceManager := services.NewSourceManager(provider, dispatcher)
	handlers := NewTriggerHandlers(provider, dispatcher, sourceManager)

	router := gin.New()
	router.POST("/sources/:trigger_id", handlers.IngestSourceHandler())

	req, _ := http.NewRequest("POST", "/sources/trg_status", bytes.NewReader(body))
	req.Header.Set("Stripe-Signature", sig)
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	// Wait for async dispatch.
	deadline := time.Now().Add(2 * time.Second)
	var storedEvent *types.InboundEvent
	for time.Now().Before(deadline) {
		events, err := provider.ListInboundEvents(ctx, trig.ID, 100)
		if err == nil && len(events) > 0 {
			storedEvent = events[0]
			// Check if status changed from Received to Dispatched.
			if storedEvent.Status == types.InboundEventStatusDispatched {
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
	}

	require.NotNil(t, storedEvent)
	assert.Equal(t, types.InboundEventStatusDispatched, storedEvent.Status)
}
