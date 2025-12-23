package services

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// mockObservabilityStore is a test implementation of ObservabilityWebhookStore.
type mockObservabilityStore struct {
	mu            sync.Mutex
	webhookConfig *types.ObservabilityWebhookConfig
	dlqEntries    []types.ObservabilityDeadLetterEntry
	dlqNextID     int64
}

func newMockObservabilityStore() *mockObservabilityStore {
	return &mockObservabilityStore{
		dlqEntries: make([]types.ObservabilityDeadLetterEntry, 0),
		dlqNextID:  1,
	}
}

func (m *mockObservabilityStore) GetObservabilityWebhook(ctx context.Context) (*types.ObservabilityWebhookConfig, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.webhookConfig, nil
}

func (m *mockObservabilityStore) SetWebhookConfig(config *types.ObservabilityWebhookConfig) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.webhookConfig = config
}

func (m *mockObservabilityStore) AddToDeadLetterQueue(ctx context.Context, event *types.ObservabilityEvent, errorMessage string, retryCount int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	payload, _ := json.Marshal(event.Data)
	entry := types.ObservabilityDeadLetterEntry{
		ID:             m.dlqNextID,
		EventType:      event.EventType,
		EventSource:    event.EventSource,
		EventTimestamp: time.Now().UTC(),
		Payload:        string(payload),
		ErrorMessage:   errorMessage,
		RetryCount:     retryCount,
		CreatedAt:      time.Now().UTC(),
	}
	m.dlqNextID++
	m.dlqEntries = append(m.dlqEntries, entry)
	return nil
}

func (m *mockObservabilityStore) GetDeadLetterQueueCount(ctx context.Context) (int64, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return int64(len(m.dlqEntries)), nil
}

func (m *mockObservabilityStore) GetDeadLetterQueue(ctx context.Context, limit, offset int) ([]types.ObservabilityDeadLetterEntry, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if offset >= len(m.dlqEntries) {
		return []types.ObservabilityDeadLetterEntry{}, nil
	}

	end := offset + limit
	if end > len(m.dlqEntries) {
		end = len(m.dlqEntries)
	}

	return m.dlqEntries[offset:end], nil
}

func (m *mockObservabilityStore) DeleteFromDeadLetterQueue(ctx context.Context, ids []int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	idSet := make(map[int64]bool)
	for _, id := range ids {
		idSet[id] = true
	}

	newEntries := make([]types.ObservabilityDeadLetterEntry, 0)
	for _, entry := range m.dlqEntries {
		if !idSet[entry.ID] {
			newEntries = append(newEntries, entry)
		}
	}
	m.dlqEntries = newEntries
	return nil
}

func (m *mockObservabilityStore) ClearDeadLetterQueue(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.dlqEntries = make([]types.ObservabilityDeadLetterEntry, 0)
	return nil
}

// Test config normalization
func TestNormalizeObservabilityConfig(t *testing.T) {
	t.Run("uses defaults when values are zero", func(t *testing.T) {
		cfg := ObservabilityForwarderConfig{}
		normalized := normalizeObservabilityConfig(cfg)

		require.Equal(t, 10, normalized.BatchSize)
		require.Equal(t, time.Second, normalized.BatchTimeout)
		require.Equal(t, 10*time.Second, normalized.HTTPTimeout)
		require.Equal(t, 3, normalized.MaxAttempts)
		require.Equal(t, time.Second, normalized.RetryBackoff)
		require.Equal(t, 30*time.Second, normalized.MaxRetryBackoff)
		require.Equal(t, 2, normalized.WorkerCount)
		require.Equal(t, 1000, normalized.QueueSize)
		require.Equal(t, 16*1024, normalized.ResponseBodyLimit)
	})

	t.Run("preserves custom values", func(t *testing.T) {
		cfg := ObservabilityForwarderConfig{
			BatchSize:         50,
			BatchTimeout:      5 * time.Second,
			HTTPTimeout:       30 * time.Second,
			MaxAttempts:       5,
			RetryBackoff:      2 * time.Second,
			MaxRetryBackoff:   60 * time.Second,
			WorkerCount:       4,
			QueueSize:         2000,
			ResponseBodyLimit: 32 * 1024,
		}
		normalized := normalizeObservabilityConfig(cfg)

		require.Equal(t, 50, normalized.BatchSize)
		require.Equal(t, 5*time.Second, normalized.BatchTimeout)
		require.Equal(t, 30*time.Second, normalized.HTTPTimeout)
		require.Equal(t, 5, normalized.MaxAttempts)
		require.Equal(t, 2*time.Second, normalized.RetryBackoff)
		require.Equal(t, 60*time.Second, normalized.MaxRetryBackoff)
		require.Equal(t, 4, normalized.WorkerCount)
		require.Equal(t, 2000, normalized.QueueSize)
		require.Equal(t, 32*1024, normalized.ResponseBodyLimit)
	})
}

// Test forwarder creation
func TestNewObservabilityForwarder(t *testing.T) {
	store := newMockObservabilityStore()
	cfg := ObservabilityForwarderConfig{
		BatchSize: 5,
	}

	forwarder := NewObservabilityForwarder(store, cfg)
	require.NotNil(t, forwarder)
}

// Test forwarder start/stop lifecycle
func TestObservabilityForwarder_StartStop(t *testing.T) {
	store := newMockObservabilityStore()
	cfg := ObservabilityForwarderConfig{
		BatchSize:    5,
		BatchTimeout: 100 * time.Millisecond,
		WorkerCount:  1,
	}

	forwarder := NewObservabilityForwarder(store, cfg)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)

	// Give it a moment to start
	time.Sleep(50 * time.Millisecond)

	// Stop the forwarder
	stopCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	err = forwarder.Stop(stopCtx)
	require.NoError(t, err)
}

// Test forwarder requires store
func TestObservabilityForwarder_RequiresStore(t *testing.T) {
	cfg := ObservabilityForwarderConfig{}
	forwarder := NewObservabilityForwarder(nil, cfg)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.Error(t, err)
	require.Contains(t, err.Error(), "requires a store")
}

// Test config reload
func TestObservabilityForwarder_ReloadConfig(t *testing.T) {
	store := newMockObservabilityStore()
	cfg := ObservabilityForwarderConfig{
		BatchSize:    5,
		BatchTimeout: 100 * time.Millisecond,
		WorkerCount:  1,
	}

	forwarder := NewObservabilityForwarder(store, cfg)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Initially no config
	status := forwarder.GetStatus()
	require.False(t, status.Enabled)

	// Set webhook config
	secret := "test-secret"
	store.SetWebhookConfig(&types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     "https://example.com/webhook",
		Secret:  &secret,
		Enabled: true,
	})

	// Reload config
	err = forwarder.ReloadConfig(ctx)
	require.NoError(t, err)

	// Now should be enabled
	status = forwarder.GetStatus()
	require.True(t, status.Enabled)
	require.Equal(t, "https://example.com/webhook", status.WebhookURL)
}

// Test status reporting
func TestObservabilityForwarder_GetStatus(t *testing.T) {
	store := newMockObservabilityStore()
	cfg := ObservabilityForwarderConfig{
		BatchSize:    5,
		BatchTimeout: 100 * time.Millisecond,
		WorkerCount:  1,
	}

	forwarder := NewObservabilityForwarder(store, cfg)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Enable webhook
	store.SetWebhookConfig(&types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     "https://example.com/webhook",
		Enabled: true,
	})
	forwarder.ReloadConfig(ctx)

	status := forwarder.GetStatus()
	require.True(t, status.Enabled)
	require.Equal(t, int64(0), status.EventsForwarded)
	require.Equal(t, int64(0), status.EventsDropped)
	require.Equal(t, int64(0), status.DeadLetterCount)
}

// Test event transformation - execution events
func TestObservabilityForwarder_TransformExecutionEvent(t *testing.T) {
	store := newMockObservabilityStore()
	forwarder := NewObservabilityForwarder(store, ObservabilityForwarderConfig{}).(*observabilityForwarder)

	execEvent := events.ExecutionEvent{
		Type:        events.ExecutionCompleted,
		ExecutionID: "exec-123",
		WorkflowID:  "wf-456",
		AgentNodeID: "agent-789",
		Status:      "succeeded",
		Timestamp:   time.Now(),
		Data:        map[string]interface{}{"key": "value"},
	}

	obsEvent := forwarder.transformExecutionEvent(execEvent)

	require.Equal(t, "execution_completed", obsEvent.EventType)
	require.Equal(t, "execution", obsEvent.EventSource)
	require.NotEmpty(t, obsEvent.Timestamp)

	data, ok := obsEvent.Data.(map[string]interface{})
	require.True(t, ok)
	require.Equal(t, "exec-123", data["execution_id"])
	require.Equal(t, "wf-456", data["workflow_id"])
	require.Equal(t, "agent-789", data["agent_node_id"])
	require.Equal(t, "succeeded", data["status"])
	require.Equal(t, execEvent.Data, data["payload"])
}

// Test event transformation - node events
func TestObservabilityForwarder_TransformNodeEvent(t *testing.T) {
	store := newMockObservabilityStore()
	forwarder := NewObservabilityForwarder(store, ObservabilityForwarderConfig{}).(*observabilityForwarder)

	nodeEvent := events.NodeEvent{
		Type:      events.NodeOnline,
		NodeID:    "node-123",
		Status:    "online",
		Timestamp: time.Now(),
		Source:    "registration",
		Reason:    "agent connected",
		OldStatus: "offline",
		NewStatus: "online",
		Data:      map[string]interface{}{"ip": "192.168.1.1"},
	}

	obsEvent := forwarder.transformNodeEvent(nodeEvent)

	require.Equal(t, "node_online", obsEvent.EventType)
	require.Equal(t, "node", obsEvent.EventSource)
	require.NotEmpty(t, obsEvent.Timestamp)

	data, ok := obsEvent.Data.(map[string]interface{})
	require.True(t, ok)
	require.Equal(t, "node-123", data["node_id"])
	require.Equal(t, "online", data["status"])
	require.Equal(t, "registration", data["source"])
	require.Equal(t, "agent connected", data["reason"])
	require.Equal(t, "offline", data["old_status"])
	require.Equal(t, "online", data["new_status"])
	require.Equal(t, nodeEvent.Data, data["payload"])
}

// Test event transformation - reasoner events
func TestObservabilityForwarder_TransformReasonerEvent(t *testing.T) {
	store := newMockObservabilityStore()
	forwarder := NewObservabilityForwarder(store, ObservabilityForwarderConfig{}).(*observabilityForwarder)

	reasonerEvent := events.ReasonerEvent{
		Type:       events.ReasonerOnline,
		ReasonerID: "reasoner-123",
		NodeID:     "node-456",
		Status:     "online",
		Timestamp:  time.Now(),
		Data:       map[string]interface{}{"version": "1.0"},
	}

	obsEvent := forwarder.transformReasonerEvent(reasonerEvent)

	require.Equal(t, "reasoner_online", obsEvent.EventType)
	require.Equal(t, "reasoner", obsEvent.EventSource)
	require.NotEmpty(t, obsEvent.Timestamp)

	data, ok := obsEvent.Data.(map[string]interface{})
	require.True(t, ok)
	require.Equal(t, "reasoner-123", data["reasoner_id"])
	require.Equal(t, "node-456", data["node_id"])
	require.Equal(t, "online", data["status"])
	require.Equal(t, reasonerEvent.Data, data["payload"])
}

// Test backoff computation
func TestObservabilityForwarder_ComputeBackoff(t *testing.T) {
	store := newMockObservabilityStore()
	cfg := ObservabilityForwarderConfig{
		RetryBackoff:    time.Second,
		MaxRetryBackoff: 30 * time.Second,
	}
	forwarder := NewObservabilityForwarder(store, cfg).(*observabilityForwarder)

	tests := []struct {
		attempt  int
		expected time.Duration
	}{
		{0, time.Second},       // Treated as 1
		{1, time.Second},       // 1 * 2^0 = 1s
		{2, 2 * time.Second},   // 1 * 2^1 = 2s
		{3, 4 * time.Second},   // 1 * 2^2 = 4s
		{4, 8 * time.Second},   // 1 * 2^3 = 8s
		{5, 16 * time.Second},  // 1 * 2^4 = 16s
		{6, 30 * time.Second},  // Would be 32s, but capped at 30s
		{10, 30 * time.Second}, // Still capped
	}

	for _, tt := range tests {
		backoff := forwarder.computeBackoff(tt.attempt)
		require.Equal(t, tt.expected, backoff, "attempt %d should produce backoff %v", tt.attempt, tt.expected)
	}
}

// Test HMAC signature generation
func TestGenerateObservabilitySignature(t *testing.T) {
	secret := "my-secret-key"
	body := []byte(`{"event":"test"}`)

	sig := generateObservabilitySignature(secret, body)

	require.True(t, len(sig) > 0)
	require.True(t, len(sig) > len("sha256="))
	require.Contains(t, sig, "sha256=")

	// Same input should produce same signature
	sig2 := generateObservabilitySignature(secret, body)
	require.Equal(t, sig, sig2)

	// Different secret should produce different signature
	sig3 := generateObservabilitySignature("different-secret", body)
	require.NotEqual(t, sig, sig3)

	// Different body should produce different signature
	sig4 := generateObservabilitySignature(secret, []byte(`{"event":"other"}`))
	require.NotEqual(t, sig, sig4)
}

// Test webhook delivery with mock HTTP server
// Note: This test uses the internal forwarder directly to avoid race conditions with global event buses
func TestObservabilityForwarder_WebhookDelivery(t *testing.T) {
	var (
		mu            sync.Mutex
		receivedBatch *types.ObservabilityEventBatch
		callCount     int32
	)

	// Create mock webhook endpoint
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&callCount, 1)

		// Verify headers
		require.Equal(t, "application/json", r.Header.Get("Content-Type"))
		require.Equal(t, "AgentField-Observability/1.0", r.Header.Get("User-Agent"))

		// Read and parse body
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)

		mu.Lock()
		receivedBatch = &types.ObservabilityEventBatch{}
		err = json.Unmarshal(body, receivedBatch)
		require.NoError(t, err)
		mu.Unlock()

		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	store := newMockObservabilityStore()
	store.SetWebhookConfig(&types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Enabled: true,
	})

	cfg := ObservabilityForwarderConfig{
		BatchSize:    2,
		BatchTimeout: 100 * time.Millisecond,
		WorkerCount:  1,
		HTTPTimeout:  5 * time.Second,
	}

	forwarder := NewObservabilityForwarder(store, cfg).(*observabilityForwarder)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Wait for forwarder to be fully started
	time.Sleep(100 * time.Millisecond)

	// Directly enqueue events to avoid global event bus timing issues
	forwarder.enqueueEvent(types.ObservabilityEvent{
		EventType:   "execution_completed",
		EventSource: "execution",
		Timestamp:   time.Now().Format(time.RFC3339),
		Data:        map[string]interface{}{"execution_id": "exec-1"},
	})
	forwarder.enqueueEvent(types.ObservabilityEvent{
		EventType:   "execution_completed",
		EventSource: "execution",
		Timestamp:   time.Now().Format(time.RFC3339),
		Data:        map[string]interface{}{"execution_id": "exec-2"},
	})

	// Wait for batch to be sent (batch size is 2, so should trigger immediately)
	time.Sleep(500 * time.Millisecond)

	// Verify delivery
	require.GreaterOrEqual(t, atomic.LoadInt32(&callCount), int32(1))

	mu.Lock()
	require.NotNil(t, receivedBatch)
	require.Greater(t, receivedBatch.EventCount, 0)
	mu.Unlock()

	// Check metrics
	status := forwarder.GetStatus()
	require.Greater(t, status.EventsForwarded, int64(0))
}

// Test webhook delivery with HMAC signature
func TestObservabilityForwarder_WebhookWithSignature(t *testing.T) {
	var (
		mu                sync.Mutex
		receivedSignature string
	)
	secret := "test-secret-123"

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		receivedSignature = r.Header.Get("X-AgentField-Signature")
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	store := newMockObservabilityStore()
	store.SetWebhookConfig(&types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Secret:  &secret,
		Enabled: true,
	})

	cfg := ObservabilityForwarderConfig{
		BatchSize:    1,
		BatchTimeout: 50 * time.Millisecond,
		WorkerCount:  1,
	}

	forwarder := NewObservabilityForwarder(store, cfg).(*observabilityForwarder)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Wait for forwarder to be fully started
	time.Sleep(100 * time.Millisecond)

	// Directly enqueue event
	forwarder.enqueueEvent(types.ObservabilityEvent{
		EventType:   "execution_started",
		EventSource: "execution",
		Timestamp:   time.Now().Format(time.RFC3339),
		Data:        map[string]interface{}{"execution_id": "exec-sig-test"},
	})

	// Wait for delivery
	time.Sleep(300 * time.Millisecond)

	mu.Lock()
	sig := receivedSignature
	mu.Unlock()

	require.NotEmpty(t, sig)
	require.Contains(t, sig, "sha256=")
}

// Test webhook delivery with custom headers
func TestObservabilityForwarder_WebhookWithCustomHeaders(t *testing.T) {
	var (
		mu                 sync.Mutex
		customHeader       string
		authorizationHeader string
	)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		customHeader = r.Header.Get("X-Custom-Header")
		authorizationHeader = r.Header.Get("Authorization")
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	store := newMockObservabilityStore()
	store.SetWebhookConfig(&types.ObservabilityWebhookConfig{
		ID:  "global",
		URL: server.URL,
		Headers: map[string]string{
			"X-Custom-Header": "custom-value",
			"Authorization":   "Bearer token123",
		},
		Enabled: true,
	})

	cfg := ObservabilityForwarderConfig{
		BatchSize:    1,
		BatchTimeout: 50 * time.Millisecond,
		WorkerCount:  1,
	}

	forwarder := NewObservabilityForwarder(store, cfg).(*observabilityForwarder)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Wait for forwarder to be fully started
	time.Sleep(100 * time.Millisecond)

	// Directly enqueue event
	forwarder.enqueueEvent(types.ObservabilityEvent{
		EventType:   "execution_failed",
		EventSource: "execution",
		Timestamp:   time.Now().Format(time.RFC3339),
		Data:        map[string]interface{}{"execution_id": "exec-header-test"},
	})

	// Wait for delivery
	time.Sleep(300 * time.Millisecond)

	mu.Lock()
	ch := customHeader
	ah := authorizationHeader
	mu.Unlock()

	require.Equal(t, "custom-value", ch)
	require.Equal(t, "Bearer token123", ah)
}

// Test DLQ on delivery failure
func TestObservabilityForwarder_DeadLetterQueueOnFailure(t *testing.T) {
	failureCount := int32(0)

	// Server that always fails
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&failureCount, 1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	store := newMockObservabilityStore()
	store.SetWebhookConfig(&types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Enabled: true,
	})

	cfg := ObservabilityForwarderConfig{
		BatchSize:       1,
		BatchTimeout:    50 * time.Millisecond,
		WorkerCount:     1,
		MaxAttempts:     2, // Only 2 retries to speed up test
		RetryBackoff:    10 * time.Millisecond,
		MaxRetryBackoff: 50 * time.Millisecond,
	}

	forwarder := NewObservabilityForwarder(store, cfg).(*observabilityForwarder)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Wait for forwarder to be fully started
	time.Sleep(100 * time.Millisecond)

	// Directly enqueue event
	forwarder.enqueueEvent(types.ObservabilityEvent{
		EventType:   "execution_created",
		EventSource: "execution",
		Timestamp:   time.Now().Format(time.RFC3339),
		Data:        map[string]interface{}{"execution_id": "exec-dlq-1"},
	})

	// Wait for retries and DLQ
	time.Sleep(500 * time.Millisecond)

	// Verify failures occurred
	require.GreaterOrEqual(t, atomic.LoadInt32(&failureCount), int32(2), "should have retried at least twice")

	// Verify DLQ
	count, err := store.GetDeadLetterQueueCount(ctx)
	require.NoError(t, err)
	require.Greater(t, count, int64(0), "events should be in DLQ after failures")

	// Verify metrics
	status := forwarder.GetStatus()
	require.Greater(t, status.EventsDropped, int64(0))
	require.NotNil(t, status.LastError)
}

// Test redrive functionality
func TestObservabilityForwarder_Redrive(t *testing.T) {
	successCount := int32(0)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&successCount, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	store := newMockObservabilityStore()
	store.SetWebhookConfig(&types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Enabled: true,
	})

	// Pre-populate DLQ with entries
	for i := 0; i < 3; i++ {
		event := &types.ObservabilityEvent{
			EventType:   "test_event",
			EventSource: "test",
			Timestamp:   time.Now().Format(time.RFC3339),
			Data:        map[string]interface{}{"id": i},
		}
		store.AddToDeadLetterQueue(context.Background(), event, "previous failure", 3)
	}

	// Verify DLQ has entries
	count, _ := store.GetDeadLetterQueueCount(context.Background())
	require.Equal(t, int64(3), count)

	cfg := ObservabilityForwarderConfig{
		MaxAttempts:  2,
		RetryBackoff: 10 * time.Millisecond,
	}

	forwarder := NewObservabilityForwarder(store, cfg)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Perform redrive
	response := forwarder.Redrive(ctx)

	require.True(t, response.Success)
	require.Equal(t, 3, response.Processed)
	require.Equal(t, 0, response.Failed)
	require.Contains(t, response.Message, "redrove 3 events")

	// Verify DLQ is empty after successful redrive
	count, _ = store.GetDeadLetterQueueCount(ctx)
	require.Equal(t, int64(0), count)

	// Verify HTTP calls were made
	require.Equal(t, int32(3), atomic.LoadInt32(&successCount))
}

// Test redrive with webhook not configured
func TestObservabilityForwarder_RedriveNotConfigured(t *testing.T) {
	store := newMockObservabilityStore()
	// No webhook config set

	cfg := ObservabilityForwarderConfig{}
	forwarder := NewObservabilityForwarder(store, cfg)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	response := forwarder.Redrive(ctx)

	require.False(t, response.Success)
	require.Contains(t, response.Message, "not configured")
}

// Test redrive with partial failures
func TestObservabilityForwarder_RedrivePartialFailure(t *testing.T) {
	requestCount := int32(0)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count := atomic.AddInt32(&requestCount, 1)
		// Fail every other request (after all retries)
		if count%3 == 0 {
			w.WriteHeader(http.StatusInternalServerError)
		} else {
			w.WriteHeader(http.StatusOK)
		}
	}))
	defer server.Close()

	store := newMockObservabilityStore()
	store.SetWebhookConfig(&types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Enabled: true,
	})

	// Add entries to DLQ
	for i := 0; i < 3; i++ {
		event := &types.ObservabilityEvent{
			EventType:   "test_event",
			EventSource: "test",
			Timestamp:   time.Now().Format(time.RFC3339),
			Data:        map[string]interface{}{"id": i},
		}
		store.AddToDeadLetterQueue(context.Background(), event, "previous failure", 3)
	}

	cfg := ObservabilityForwarderConfig{
		MaxAttempts:  1, // Single attempt per entry
		RetryBackoff: 10 * time.Millisecond,
	}

	forwarder := NewObservabilityForwarder(store, cfg)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	response := forwarder.Redrive(ctx)

	// Some should succeed, some should fail
	require.Equal(t, 3, response.Processed+response.Failed)
}

// Test heartbeat event filtering - node events
func TestObservabilityForwarder_FiltersNodeHeartbeats(t *testing.T) {
	var receivedEvents []types.ObservabilityEvent
	var mu sync.Mutex

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

	store := newMockObservabilityStore()
	store.SetWebhookConfig(&types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Enabled: true,
	})

	cfg := ObservabilityForwarderConfig{
		BatchSize:    10,
		BatchTimeout: 200 * time.Millisecond,
		WorkerCount:  1,
	}

	forwarder := NewObservabilityForwarder(store, cfg)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Publish a mix of events including heartbeats
	events.PublishNodeOnline("node-1", nil)
	events.PublishNodeHeartbeat() // Should be filtered
	events.PublishNodeOffline("node-1", nil)
	events.PublishNodeHeartbeat() // Should be filtered
	events.PublishNodeRegistered("node-2", nil)

	// Wait for batch
	time.Sleep(500 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()

	// Verify no heartbeat events were delivered
	for _, event := range receivedEvents {
		require.NotEqual(t, "node_heartbeat", event.EventType, "heartbeat events should be filtered")
	}
}

// Test heartbeat event filtering - reasoner events
func TestObservabilityForwarder_FiltersReasonerHeartbeats(t *testing.T) {
	var receivedEvents []types.ObservabilityEvent
	var mu sync.Mutex

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

	store := newMockObservabilityStore()
	store.SetWebhookConfig(&types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Enabled: true,
	})

	cfg := ObservabilityForwarderConfig{
		BatchSize:    10,
		BatchTimeout: 200 * time.Millisecond,
		WorkerCount:  1,
	}

	forwarder := NewObservabilityForwarder(store, cfg)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Publish a mix of events including heartbeats
	events.PublishReasonerOnline("reasoner-1", "node-1", nil)
	events.PublishHeartbeat() // Should be filtered
	events.PublishReasonerOffline("reasoner-1", "node-1", nil)
	events.PublishHeartbeat() // Should be filtered

	// Wait for batch
	time.Sleep(500 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()

	// Verify no heartbeat events were delivered
	for _, event := range receivedEvents {
		require.NotEqual(t, "heartbeat", event.EventType, "heartbeat events should be filtered")
	}
}

// Test events not enqueued when webhook disabled
func TestObservabilityForwarder_NoEnqueueWhenDisabled(t *testing.T) {
	store := newMockObservabilityStore()
	// No webhook configured = disabled

	cfg := ObservabilityForwarderConfig{
		BatchSize:    5,
		BatchTimeout: 100 * time.Millisecond,
		WorkerCount:  1,
	}

	forwarder := NewObservabilityForwarder(store, cfg)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Publish events
	events.PublishExecutionCompleted("exec-disabled-1", "wf-1", "agent-1", nil)
	events.PublishExecutionCompleted("exec-disabled-2", "wf-1", "agent-1", nil)

	// Wait
	time.Sleep(300 * time.Millisecond)

	// Queue should be empty since webhook is disabled
	status := forwarder.GetStatus()
	require.Equal(t, 0, status.QueueDepth)
	require.Equal(t, int64(0), status.EventsForwarded)
}

// Test batching behavior
func TestObservabilityForwarder_BatchingBySize(t *testing.T) {
	var batchSizes []int
	var mu sync.Mutex

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

	store := newMockObservabilityStore()
	store.SetWebhookConfig(&types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Enabled: true,
	})

	cfg := ObservabilityForwarderConfig{
		BatchSize:    3,                     // Send every 3 events
		BatchTimeout: 10 * time.Second,      // Long timeout to ensure size-based batching
		WorkerCount:  1,
	}

	forwarder := NewObservabilityForwarder(store, cfg).(*observabilityForwarder)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Wait for forwarder to be fully started
	time.Sleep(100 * time.Millisecond)

	// Send exactly 6 events (should create 2 batches of 3)
	for i := 0; i < 6; i++ {
		forwarder.enqueueEvent(types.ObservabilityEvent{
			EventType:   "execution_created",
			EventSource: "execution",
			Timestamp:   time.Now().Format(time.RFC3339),
			Data:        map[string]interface{}{"index": i},
		})
	}

	// Wait for batches
	time.Sleep(500 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()

	// Should have received batches of size 3
	require.GreaterOrEqual(t, len(batchSizes), 2)
	for _, size := range batchSizes {
		require.LessOrEqual(t, size, 3, "batch size should not exceed configured limit")
	}
}

// Test batching by timeout
func TestObservabilityForwarder_BatchingByTimeout(t *testing.T) {
	var receivedBatches int32

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&receivedBatches, 1)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	store := newMockObservabilityStore()
	store.SetWebhookConfig(&types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     server.URL,
		Enabled: true,
	})

	cfg := ObservabilityForwarderConfig{
		BatchSize:    100,                    // Large batch size
		BatchTimeout: 100 * time.Millisecond, // Short timeout
		WorkerCount:  1,
	}

	forwarder := NewObservabilityForwarder(store, cfg).(*observabilityForwarder)

	ctx := context.Background()
	err := forwarder.Start(ctx)
	require.NoError(t, err)
	defer forwarder.Stop(ctx)

	// Wait for forwarder to be fully started
	time.Sleep(100 * time.Millisecond)

	// Send just 1 event (won't fill batch)
	forwarder.enqueueEvent(types.ObservabilityEvent{
		EventType:   "execution_created",
		EventSource: "execution",
		Timestamp:   time.Now().Format(time.RFC3339),
		Data:        map[string]interface{}{"execution_id": "exec-timeout-1"},
	})

	// Wait for timeout-based flush
	time.Sleep(300 * time.Millisecond)

	// Should have received a batch despite not reaching batch size
	require.GreaterOrEqual(t, atomic.LoadInt32(&receivedBatches), int32(1), "should send batch on timeout")
}
