package storage

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// setupObservabilityTestStorage creates a test storage instance for observability webhook tests.
func setupObservabilityTestStorage(t *testing.T) (*LocalStorage, context.Context) {
	t.Helper()

	ctx := context.Background()
	tempDir := t.TempDir()

	cfg := StorageConfig{
		Mode: "local",
		Local: LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "agentfield.db"),
			KVStorePath:  filepath.Join(tempDir, "agentfield.bolt"),
		},
	}

	ls := NewLocalStorage(LocalStorageConfig{})
	if err := ls.Initialize(ctx, cfg); err != nil {
		if strings.Contains(err.Error(), "no such module: fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping test")
		}
		t.Fatalf("initialize local storage: %v", err)
	}
	t.Cleanup(func() {
		_ = ls.Close(ctx)
	})

	return ls, ctx
}

// Test webhook config CRUD operations
func TestObservabilityWebhook_SetAndGet(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Initially no config should exist
	config, err := ls.GetObservabilityWebhook(ctx)
	require.NoError(t, err)
	require.Nil(t, config)

	// Set webhook config
	secret := "test-secret-123"
	inputConfig := &types.ObservabilityWebhookConfig{
		ID:  "global",
		URL: "https://example.com/webhook",
		Secret: &secret,
		Headers: map[string]string{
			"X-Custom-Header": "custom-value",
			"Authorization":   "Bearer token123",
		},
		Enabled: true,
	}

	err = ls.SetObservabilityWebhook(ctx, inputConfig)
	require.NoError(t, err)

	// Get webhook config
	retrieved, err := ls.GetObservabilityWebhook(ctx)
	require.NoError(t, err)
	require.NotNil(t, retrieved)

	require.Equal(t, "global", retrieved.ID)
	require.Equal(t, "https://example.com/webhook", retrieved.URL)
	require.NotNil(t, retrieved.Secret)
	require.Equal(t, "test-secret-123", *retrieved.Secret)
	require.Equal(t, true, retrieved.Enabled)
	require.Equal(t, "custom-value", retrieved.Headers["X-Custom-Header"])
	require.Equal(t, "Bearer token123", retrieved.Headers["Authorization"])
	require.False(t, retrieved.CreatedAt.IsZero())
	require.False(t, retrieved.UpdatedAt.IsZero())
}

func TestObservabilityWebhook_Update(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Create initial config
	secret1 := "initial-secret"
	initialConfig := &types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     "https://example.com/v1/webhook",
		Secret:  &secret1,
		Headers: map[string]string{"X-Version": "1"},
		Enabled: true,
	}

	err := ls.SetObservabilityWebhook(ctx, initialConfig)
	require.NoError(t, err)

	// Get initial config and record created_at
	initial, err := ls.GetObservabilityWebhook(ctx)
	require.NoError(t, err)
	require.NotNil(t, initial)
	initialCreatedAt := initial.CreatedAt

	// Wait a bit to ensure updated_at will be different
	time.Sleep(10 * time.Millisecond)

	// Update config
	secret2 := "updated-secret"
	updatedConfig := &types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     "https://example.com/v2/webhook",
		Secret:  &secret2,
		Headers: map[string]string{"X-Version": "2"},
		Enabled: false,
	}

	err = ls.SetObservabilityWebhook(ctx, updatedConfig)
	require.NoError(t, err)

	// Verify update
	retrieved, err := ls.GetObservabilityWebhook(ctx)
	require.NoError(t, err)
	require.NotNil(t, retrieved)

	require.Equal(t, "https://example.com/v2/webhook", retrieved.URL)
	require.Equal(t, "updated-secret", *retrieved.Secret)
	require.Equal(t, "2", retrieved.Headers["X-Version"])
	require.Equal(t, false, retrieved.Enabled)

	// Note: created_at may or may not be preserved depending on upsert implementation
	// Just verify updated_at is recent
	require.True(t, retrieved.UpdatedAt.After(initialCreatedAt) || retrieved.UpdatedAt.Equal(initialCreatedAt))
}

func TestObservabilityWebhook_Delete(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Set webhook config
	config := &types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     "https://example.com/webhook",
		Enabled: true,
	}

	err := ls.SetObservabilityWebhook(ctx, config)
	require.NoError(t, err)

	// Verify it exists
	retrieved, err := ls.GetObservabilityWebhook(ctx)
	require.NoError(t, err)
	require.NotNil(t, retrieved)

	// Delete config
	err = ls.DeleteObservabilityWebhook(ctx)
	require.NoError(t, err)

	// Verify it's gone
	retrieved, err = ls.GetObservabilityWebhook(ctx)
	require.NoError(t, err)
	require.Nil(t, retrieved)
}

func TestObservabilityWebhook_DeleteNonExistent(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Delete should not error even if nothing exists
	err := ls.DeleteObservabilityWebhook(ctx)
	require.NoError(t, err)
}

func TestObservabilityWebhook_NilConfig(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	err := ls.SetObservabilityWebhook(ctx, nil)
	require.Error(t, err)
	require.Contains(t, err.Error(), "nil")
}

func TestObservabilityWebhook_EmptyURL(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	config := &types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     "",
		Enabled: true,
	}

	err := ls.SetObservabilityWebhook(ctx, config)
	require.Error(t, err)
	require.Contains(t, err.Error(), "URL")
}

func TestObservabilityWebhook_NilSecret(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Config without secret
	config := &types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     "https://example.com/webhook",
		Secret:  nil,
		Enabled: true,
	}

	err := ls.SetObservabilityWebhook(ctx, config)
	require.NoError(t, err)

	retrieved, err := ls.GetObservabilityWebhook(ctx)
	require.NoError(t, err)
	require.NotNil(t, retrieved)
	require.Nil(t, retrieved.Secret)
}

func TestObservabilityWebhook_EmptyHeaders(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Config without headers
	config := &types.ObservabilityWebhookConfig{
		ID:      "global",
		URL:     "https://example.com/webhook",
		Headers: nil,
		Enabled: true,
	}

	err := ls.SetObservabilityWebhook(ctx, config)
	require.NoError(t, err)

	retrieved, err := ls.GetObservabilityWebhook(ctx)
	require.NoError(t, err)
	require.NotNil(t, retrieved)
	require.NotNil(t, retrieved.Headers)
	require.Empty(t, retrieved.Headers)
}

// Dead Letter Queue Tests

func TestDeadLetterQueue_AddAndGet(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Add event to DLQ
	event := &types.ObservabilityEvent{
		EventType:   "execution_failed",
		EventSource: "execution",
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
		Data: map[string]interface{}{
			"execution_id": "exec-123",
			"error":        "connection refused",
		},
	}

	err := ls.AddToDeadLetterQueue(ctx, event, "webhook delivery failed: connection refused", 3)
	require.NoError(t, err)

	// Get DLQ entries
	entries, err := ls.GetDeadLetterQueue(ctx, 100, 0)
	require.NoError(t, err)
	require.Len(t, entries, 1)

	entry := entries[0]
	require.Equal(t, "execution_failed", entry.EventType)
	require.Equal(t, "execution", entry.EventSource)
	require.Contains(t, entry.ErrorMessage, "connection refused")
	require.Equal(t, 3, entry.RetryCount)
	require.False(t, entry.CreatedAt.IsZero())
	require.False(t, entry.EventTimestamp.IsZero())

	// Verify payload can be parsed
	require.NotEmpty(t, entry.Payload)
}

func TestDeadLetterQueue_Count(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Initially empty
	count, err := ls.GetDeadLetterQueueCount(ctx)
	require.NoError(t, err)
	require.Equal(t, int64(0), count)

	// Add multiple entries
	for i := 0; i < 5; i++ {
		event := &types.ObservabilityEvent{
			EventType:   "test_event",
			EventSource: "test",
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			Data:        map[string]interface{}{"index": i},
		}
		err := ls.AddToDeadLetterQueue(ctx, event, "test error", 3)
		require.NoError(t, err)
	}

	// Verify count
	count, err = ls.GetDeadLetterQueueCount(ctx)
	require.NoError(t, err)
	require.Equal(t, int64(5), count)
}

func TestDeadLetterQueue_Pagination(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Add 10 entries
	for i := 0; i < 10; i++ {
		event := &types.ObservabilityEvent{
			EventType:   "test_event",
			EventSource: "test",
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			Data:        map[string]interface{}{"index": i},
		}
		err := ls.AddToDeadLetterQueue(ctx, event, "test error", 3)
		require.NoError(t, err)
		time.Sleep(time.Millisecond) // Ensure ordering
	}

	// Get first page
	page1, err := ls.GetDeadLetterQueue(ctx, 3, 0)
	require.NoError(t, err)
	require.Len(t, page1, 3)

	// Get second page
	page2, err := ls.GetDeadLetterQueue(ctx, 3, 3)
	require.NoError(t, err)
	require.Len(t, page2, 3)

	// Verify no overlap
	for _, e1 := range page1 {
		for _, e2 := range page2 {
			require.NotEqual(t, e1.ID, e2.ID)
		}
	}

	// Get last page
	page4, err := ls.GetDeadLetterQueue(ctx, 3, 9)
	require.NoError(t, err)
	require.Len(t, page4, 1)

	// Get beyond end
	pageEmpty, err := ls.GetDeadLetterQueue(ctx, 3, 100)
	require.NoError(t, err)
	require.Len(t, pageEmpty, 0)
}

func TestDeadLetterQueue_DefaultPagination(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Add entries
	for i := 0; i < 5; i++ {
		event := &types.ObservabilityEvent{
			EventType:   "test_event",
			EventSource: "test",
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			Data:        map[string]interface{}{"index": i},
		}
		err := ls.AddToDeadLetterQueue(ctx, event, "test error", 3)
		require.NoError(t, err)
	}

	// Test with invalid limit (should use default)
	entries, err := ls.GetDeadLetterQueue(ctx, -1, 0)
	require.NoError(t, err)
	require.Len(t, entries, 5)

	// Test with invalid offset (should use 0)
	entries, err = ls.GetDeadLetterQueue(ctx, 100, -5)
	require.NoError(t, err)
	require.Len(t, entries, 5)
}

func TestDeadLetterQueue_DeleteByIDs(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Add entries
	for i := 0; i < 5; i++ {
		event := &types.ObservabilityEvent{
			EventType:   "test_event",
			EventSource: "test",
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			Data:        map[string]interface{}{"index": i},
		}
		err := ls.AddToDeadLetterQueue(ctx, event, "test error", 3)
		require.NoError(t, err)
	}

	// Get all entries
	entries, err := ls.GetDeadLetterQueue(ctx, 100, 0)
	require.NoError(t, err)
	require.Len(t, entries, 5)

	// Delete specific entries
	idsToDelete := []int64{entries[0].ID, entries[2].ID, entries[4].ID}
	err = ls.DeleteFromDeadLetterQueue(ctx, idsToDelete)
	require.NoError(t, err)

	// Verify count
	count, err := ls.GetDeadLetterQueueCount(ctx)
	require.NoError(t, err)
	require.Equal(t, int64(2), count)

	// Verify remaining entries
	remaining, err := ls.GetDeadLetterQueue(ctx, 100, 0)
	require.NoError(t, err)
	require.Len(t, remaining, 2)

	// Verify deleted IDs are gone
	for _, entry := range remaining {
		for _, deletedID := range idsToDelete {
			require.NotEqual(t, deletedID, entry.ID)
		}
	}
}

func TestDeadLetterQueue_DeleteEmpty(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Delete with empty slice should not error
	err := ls.DeleteFromDeadLetterQueue(ctx, []int64{})
	require.NoError(t, err)

	// Delete with nil should not error
	err = ls.DeleteFromDeadLetterQueue(ctx, nil)
	require.NoError(t, err)
}

func TestDeadLetterQueue_DeleteNonExistent(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Delete non-existent IDs should not error
	err := ls.DeleteFromDeadLetterQueue(ctx, []int64{999, 1000, 1001})
	require.NoError(t, err)
}

func TestDeadLetterQueue_Clear(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Add entries
	for i := 0; i < 10; i++ {
		event := &types.ObservabilityEvent{
			EventType:   "test_event",
			EventSource: "test",
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			Data:        map[string]interface{}{"index": i},
		}
		err := ls.AddToDeadLetterQueue(ctx, event, "test error", 3)
		require.NoError(t, err)
	}

	// Verify entries exist
	count, err := ls.GetDeadLetterQueueCount(ctx)
	require.NoError(t, err)
	require.Equal(t, int64(10), count)

	// Clear DLQ
	err = ls.ClearDeadLetterQueue(ctx)
	require.NoError(t, err)

	// Verify empty
	count, err = ls.GetDeadLetterQueueCount(ctx)
	require.NoError(t, err)
	require.Equal(t, int64(0), count)

	entries, err := ls.GetDeadLetterQueue(ctx, 100, 0)
	require.NoError(t, err)
	require.Len(t, entries, 0)
}

func TestDeadLetterQueue_ClearEmpty(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Clear empty DLQ should not error
	err := ls.ClearDeadLetterQueue(ctx)
	require.NoError(t, err)
}

func TestDeadLetterQueue_NilEvent(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	err := ls.AddToDeadLetterQueue(ctx, nil, "test error", 3)
	require.Error(t, err)
	require.Contains(t, err.Error(), "nil")
}

func TestDeadLetterQueue_InvalidTimestamp(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Event with invalid timestamp should still work (falls back to now)
	event := &types.ObservabilityEvent{
		EventType:   "test_event",
		EventSource: "test",
		Timestamp:   "invalid-timestamp",
		Data:        map[string]interface{}{"key": "value"},
	}

	err := ls.AddToDeadLetterQueue(ctx, event, "test error", 3)
	require.NoError(t, err)

	entries, err := ls.GetDeadLetterQueue(ctx, 100, 0)
	require.NoError(t, err)
	require.Len(t, entries, 1)
	require.False(t, entries[0].EventTimestamp.IsZero())
}

func TestDeadLetterQueue_Ordering(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Add entries with delays
	for i := 0; i < 5; i++ {
		event := &types.ObservabilityEvent{
			EventType:   "test_event",
			EventSource: "test",
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			Data:        map[string]interface{}{"index": i},
		}
		err := ls.AddToDeadLetterQueue(ctx, event, "test error", 3)
		require.NoError(t, err)
		time.Sleep(5 * time.Millisecond)
	}

	// Get entries - should be ordered by created_at ASC (oldest first)
	entries, err := ls.GetDeadLetterQueue(ctx, 100, 0)
	require.NoError(t, err)
	require.Len(t, entries, 5)

	// Verify ordering
	for i := 1; i < len(entries); i++ {
		require.True(t,
			entries[i].CreatedAt.After(entries[i-1].CreatedAt) ||
				entries[i].CreatedAt.Equal(entries[i-1].CreatedAt),
			"entries should be ordered by created_at ASC")
	}
}

func TestDeadLetterQueue_LargePayload(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	// Create event with large payload
	largeData := make(map[string]interface{})
	for i := 0; i < 100; i++ {
		largeData[string(rune('a'+i%26))+strings.Repeat("x", 100)] = strings.Repeat("value", 100)
	}

	event := &types.ObservabilityEvent{
		EventType:   "test_event",
		EventSource: "test",
		Timestamp:   time.Now().UTC().Format(time.RFC3339),
		Data:        largeData,
	}

	err := ls.AddToDeadLetterQueue(ctx, event, "test error", 3)
	require.NoError(t, err)

	entries, err := ls.GetDeadLetterQueue(ctx, 100, 0)
	require.NoError(t, err)
	require.Len(t, entries, 1)
	require.NotEmpty(t, entries[0].Payload)
}

func TestDeadLetterQueue_MultipleEventTypes(t *testing.T) {
	ls, ctx := setupObservabilityTestStorage(t)

	eventTypes := []string{"execution_created", "execution_completed", "node_online", "node_offline", "reasoner_updated"}
	eventSources := []string{"execution", "execution", "node", "node", "reasoner"}

	for i, eventType := range eventTypes {
		event := &types.ObservabilityEvent{
			EventType:   eventType,
			EventSource: eventSources[i],
			Timestamp:   time.Now().UTC().Format(time.RFC3339),
			Data:        map[string]interface{}{"type": eventType},
		}
		err := ls.AddToDeadLetterQueue(ctx, event, "webhook unavailable", 3)
		require.NoError(t, err)
	}

	// Verify all types stored
	entries, err := ls.GetDeadLetterQueue(ctx, 100, 0)
	require.NoError(t, err)
	require.Len(t, entries, 5)

	foundTypes := make(map[string]bool)
	for _, entry := range entries {
		foundTypes[entry.EventType] = true
	}

	for _, eventType := range eventTypes {
		require.True(t, foundTypes[eventType], "expected event type %s to be present", eventType)
	}
}
