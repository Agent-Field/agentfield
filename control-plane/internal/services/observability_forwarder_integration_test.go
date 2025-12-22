//go:build integration
// +build integration

package services

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// Integration tests for the observability forwarder with real storage.
// Run with: go test -tags=integration ./internal/services/...

// setupIntegrationTest creates a real storage instance for integration testing.
func setupIntegrationTest(t *testing.T) (*storage.LocalStorage, context.Context) {
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

	ls := storage.NewLocalStorage(storage.LocalStorageConfig{})
	if err := ls.Initialize(ctx, cfg); err != nil {
		if strings.Contains(err.Error(), "no such module: fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping integration test")
		}
		t.Fatalf("initialize storage: %v", err)
	}
	t.Cleanup(func() {
		_ = ls.Close(ctx)
	})

	return ls, ctx
}

// TestIntegration_EndToEndWebhookDelivery tests the complete flow from event publishing
// through storage, forwarder, and HTTP delivery.
func TestIntegration_EndToEndWebhookDelivery(t *testing.T) {
	store, ctx := setupIntegrationTest(t)

	// Track received events
	var (
		mu             sync.Mutex
		receivedEvents []types.ObservabilityEventBatch
	)

	// Start mock webhook server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Logf("error reading body: %v", err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}

		var batch types.ObservabilityEventBatch
		if err := json.Unmarshal(body, &batch); err != nil {
			t.Logf("error unmarshaling: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}

		mu.Lock()
		receivedEvents = append(receivedEvents, batch)
		mu.Unlock()

		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Configure webhook in storage
	secret := "integration-test-secret"
	config := &types.ObservabilityWebhookConfig{
		ID:     "global",
		URL:    server.URL,
		Secret: &secret,
		Headers: map[string]string{
			"X-Integration-Test": "true",
		},
		Enabled: true,
	}
	err := store.SetObservabilityWebhook(ctx, config)
	require.NoError(t, err)

	// Create and start forwarder
	cfg := ObservabilityForwarderConfig{
		BatchSize:    2,
		BatchTimeout: 100 * time.Millisecond,
		WorkerCount:  1,
		MaxAttempts:  2,
	}
	forwarder := NewObservabilityForwarder(store, cfg)

	err = forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Publish various event types
	events.PublishExecutionCreated("exec-int-1", "wf-1", "agent-1", nil)
	events.PublishExecutionStarted("exec-int-1", "wf-1", "agent-1", nil)
	events.PublishNodeOnline("node-int-1", map[string]interface{}{"ip": "10.0.0.1"})
	events.PublishExecutionCompleted("exec-int-1", "wf-1", "agent-1", map[string]interface{}{"result": "success"})

	// Wait for batches to be sent
	time.Sleep(500 * time.Millisecond)

	// Verify events were received
	mu.Lock()
	defer mu.Unlock()

	require.Greater(t, len(receivedEvents), 0, "should have received at least one batch")

	// Count total events
	totalEvents := 0
	for _, batch := range receivedEvents {
		totalEvents += batch.EventCount
	}
	require.GreaterOrEqual(t, totalEvents, 4, "should have received at least 4 events")

	// Verify forwarder status
	status := forwarder.GetStatus()
	require.True(t, status.Enabled)
	require.GreaterOrEqual(t, status.EventsForwarded, int64(4))
	require.Equal(t, int64(0), status.EventsDropped)
}

// TestIntegration_DeadLetterQueueFlow tests the complete DLQ flow including
// failed delivery, storage in DLQ, and successful redrive.
func TestIntegration_DeadLetterQueueFlow(t *testing.T) {
	store, ctx := setupIntegrationTest(t)

	failCount := int32(0)
	shouldFail := atomic.Bool{}
	shouldFail.Store(true)

	// Start mock webhook server that can be toggled to fail/succeed
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&failCount, 1)
		if shouldFail.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Configure webhook
	config := &types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Enabled: true,
	}
	err := store.SetObservabilityWebhook(ctx, config)
	require.NoError(t, err)

	// Create forwarder with quick retries
	cfg := ObservabilityForwarderConfig{
		BatchSize:       1,
		BatchTimeout:    50 * time.Millisecond,
		WorkerCount:     1,
		MaxAttempts:     2,
		RetryBackoff:    10 * time.Millisecond,
		MaxRetryBackoff: 50 * time.Millisecond,
	}
	forwarder := NewObservabilityForwarder(store, cfg)

	err = forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Publish events that will fail
	events.PublishExecutionFailed("exec-dlq-1", "wf-1", "agent-1", map[string]interface{}{"error": "test"})

	// Wait for retries and DLQ insertion
	time.Sleep(500 * time.Millisecond)

	// Verify events are in DLQ
	dlqCount, err := store.GetDeadLetterQueueCount(ctx)
	require.NoError(t, err)
	require.Greater(t, dlqCount, int64(0), "events should be in DLQ")

	// Verify forwarder shows dropped events
	status := forwarder.GetStatus()
	require.Greater(t, status.EventsDropped, int64(0))
	require.Greater(t, status.DeadLetterCount, int64(0))

	// Now make the server succeed
	shouldFail.Store(false)

	// Redrive the events
	response := forwarder.Redrive(ctx)
	require.True(t, response.Success, "redrive should succeed: %s", response.Message)
	require.Greater(t, response.Processed, 0)

	// Verify DLQ is empty
	dlqCount, err = store.GetDeadLetterQueueCount(ctx)
	require.NoError(t, err)
	require.Equal(t, int64(0), dlqCount, "DLQ should be empty after successful redrive")
}

// TestIntegration_ConfigReload tests that the forwarder correctly reloads
// configuration when webhook settings change.
func TestIntegration_ConfigReload(t *testing.T) {
	store, ctx := setupIntegrationTest(t)

	var receivedCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&receivedCount, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Start with webhook disabled (no config)
	cfg := ObservabilityForwarderConfig{
		BatchSize:    1,
		BatchTimeout: 50 * time.Millisecond,
		WorkerCount:  1,
	}
	forwarder := NewObservabilityForwarder(store, cfg)

	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Publish event - should NOT be forwarded
	events.PublishExecutionCreated("exec-reload-1", "wf-1", "agent-1", nil)
	time.Sleep(200 * time.Millisecond)

	require.Equal(t, int32(0), atomic.LoadInt32(&receivedCount), "no events should be forwarded when disabled")

	// Enable webhook
	config := &types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Enabled: true,
	}
	err = store.SetObservabilityWebhook(ctx, config)
	require.NoError(t, err)

	// Reload config
	err = forwarder.ReloadConfig(ctx)
	require.NoError(t, err)

	// Verify status
	status := forwarder.GetStatus()
	require.True(t, status.Enabled)
	require.Equal(t, server.URL, status.WebhookURL)

	// Publish event - should be forwarded now
	events.PublishExecutionCreated("exec-reload-2", "wf-1", "agent-1", nil)
	time.Sleep(200 * time.Millisecond)

	require.Greater(t, atomic.LoadInt32(&receivedCount), int32(0), "events should be forwarded after enabling")
}

// TestIntegration_HeartbeatFiltering verifies that heartbeat events are filtered
// and not forwarded to the webhook.
func TestIntegration_HeartbeatFiltering(t *testing.T) {
	store, ctx := setupIntegrationTest(t)

	var (
		mu             sync.Mutex
		receivedEvents []types.ObservabilityEvent
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var batch types.ObservabilityEventBatch
		json.Unmarshal(body, &batch)

		mu.Lock()
		receivedEvents = append(receivedEvents, batch.Events...)
		mu.Unlock()

		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	config := &types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Enabled: true,
	}
	err := store.SetObservabilityWebhook(ctx, config)
	require.NoError(t, err)

	cfg := ObservabilityForwarderConfig{
		BatchSize:    10,
		BatchTimeout: 200 * time.Millisecond,
		WorkerCount:  1,
	}
	forwarder := NewObservabilityForwarder(store, cfg)

	err = forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Publish mix of real events and heartbeats
	events.PublishNodeOnline("node-filter-1", nil)
	events.PublishNodeHeartbeat() // Should be filtered
	events.PublishReasonerOnline("reasoner-1", "node-1", nil)
	events.PublishHeartbeat() // Should be filtered
	events.PublishNodeOffline("node-filter-1", nil)
	events.PublishNodeHeartbeat() // Should be filtered

	// Wait for batch
	time.Sleep(500 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()

	// Verify no heartbeat events
	for _, event := range receivedEvents {
		require.NotEqual(t, "heartbeat", event.EventType, "reasoner heartbeat should be filtered")
		require.NotEqual(t, "node_heartbeat", event.EventType, "node heartbeat should be filtered")
	}

	// Should have received exactly 3 real events
	require.Equal(t, 3, len(receivedEvents), "should have 3 non-heartbeat events")
}

// TestIntegration_SignatureVerification tests that HMAC signatures are correctly
// generated and can be verified.
func TestIntegration_SignatureVerification(t *testing.T) {
	store, ctx := setupIntegrationTest(t)

	secret := "test-verification-secret"
	var (
		mu             sync.Mutex
		receivedSig    string
		receivedBody   []byte
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		receivedSig = r.Header.Get("X-AgentField-Signature")
		receivedBody, _ = io.ReadAll(r.Body)
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	config := &types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Secret:  &secret,
		Enabled: true,
	}
	err := store.SetObservabilityWebhook(ctx, config)
	require.NoError(t, err)

	cfg := ObservabilityForwarderConfig{
		BatchSize:    1,
		BatchTimeout: 50 * time.Millisecond,
		WorkerCount:  1,
	}
	forwarder := NewObservabilityForwarder(store, cfg)

	err = forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	events.PublishExecutionCreated("exec-sig-verify", "wf-1", "agent-1", nil)
	time.Sleep(300 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()

	require.NotEmpty(t, receivedSig)
	require.Contains(t, receivedSig, "sha256=")
	require.NotEmpty(t, receivedBody)

	// Verify signature
	expectedSig := generateObservabilitySignature(secret, receivedBody)
	require.Equal(t, expectedSig, receivedSig, "signature should match")
}

// TestIntegration_BatchAggregation tests that events are correctly aggregated
// into batches.
func TestIntegration_BatchAggregation(t *testing.T) {
	store, ctx := setupIntegrationTest(t)

	var (
		mu           sync.Mutex
		batchSizes   []int
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var batch types.ObservabilityEventBatch
		json.Unmarshal(body, &batch)

		mu.Lock()
		batchSizes = append(batchSizes, batch.EventCount)
		mu.Unlock()

		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	config := &types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Enabled: true,
	}
	err := store.SetObservabilityWebhook(ctx, config)
	require.NoError(t, err)

	cfg := ObservabilityForwarderConfig{
		BatchSize:    5,                     // Wait for 5 events
		BatchTimeout: 10 * time.Second,      // Long timeout
		WorkerCount:  1,
	}
	forwarder := NewObservabilityForwarder(store, cfg)

	err = forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Send exactly 5 events to trigger batch
	for i := 0; i < 5; i++ {
		events.PublishExecutionCreated("exec-batch-"+string(rune('a'+i)), "wf-1", "agent-1", nil)
	}

	time.Sleep(300 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()

	require.GreaterOrEqual(t, len(batchSizes), 1, "should have at least one batch")
	require.Equal(t, 5, batchSizes[0], "first batch should have 5 events")
}

// TestIntegration_StoragePersistence tests that webhook configuration persists
// across forwarder restarts.
func TestIntegration_StoragePersistence(t *testing.T) {
	store, ctx := setupIntegrationTest(t)

	var receivedCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&receivedCount, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	// Configure webhook
	secret := "persistent-secret"
	config := &types.ObservabilityWebhookConfig{
		ID:     "global",
		URL:    server.URL,
		Secret: &secret,
		Headers: map[string]string{
			"X-Persistent": "true",
		},
		Enabled: true,
	}
	err := store.SetObservabilityWebhook(ctx, config)
	require.NoError(t, err)

	cfg := ObservabilityForwarderConfig{
		BatchSize:    1,
		BatchTimeout: 50 * time.Millisecond,
		WorkerCount:  1,
	}

	// Start first forwarder instance
	forwarder1 := NewObservabilityForwarder(store, cfg)
	err = forwarder1.Start(ctx)
	require.NoError(t, err)

	events.PublishExecutionCreated("exec-persist-1", "wf-1", "agent-1", nil)
	time.Sleep(200 * time.Millisecond)

	// Stop first forwarder
	err = forwarder1.Stop(ctx)
	require.NoError(t, err)

	// Start second forwarder instance (simulates restart)
	forwarder2 := NewObservabilityForwarder(store, cfg)
	err = forwarder2.Start(ctx)
	require.NoError(t, err)
	defer forwarder2.Stop(ctx)

	// Verify config was loaded from storage
	status := forwarder2.GetStatus()
	require.True(t, status.Enabled)
	require.Equal(t, server.URL, status.WebhookURL)

	// Verify new forwarder can forward events
	beforeCount := atomic.LoadInt32(&receivedCount)
	events.PublishExecutionCreated("exec-persist-2", "wf-1", "agent-1", nil)
	time.Sleep(200 * time.Millisecond)

	require.Greater(t, atomic.LoadInt32(&receivedCount), beforeCount, "new forwarder should forward events")
}
