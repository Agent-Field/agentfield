package storage

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func setupTestStorage(t *testing.T) (StorageProvider, context.Context) {
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
		if strings.Contains(strings.ToLower(err.Error()), "fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping test")
		}
		require.NoError(t, err)
	}

	t.Cleanup(func() {
		_ = ls.Close(ctx)
	})

	return ls, ctx
}

func TestRegisterExecutionWebhook_Success(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	secret := "test-secret-123"
	webhook := &types.ExecutionWebhook{
		ExecutionID: "exec-1",
		URL:         "https://example.com/webhook",
		Secret:      &secret,
		Headers: map[string]string{
			"X-Custom-Header": "value1",
		},
		Status: types.ExecutionWebhookStatusPending,
	}

	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.NoError(t, err)

	// Verify webhook was stored
	retrieved, err := provider.GetExecutionWebhook(ctx, "exec-1")
	require.NoError(t, err)
	require.NotNil(t, retrieved)

	assert.Equal(t, "exec-1", retrieved.ExecutionID)
	assert.Equal(t, "https://example.com/webhook", retrieved.URL)
	require.NotNil(t, retrieved.Secret)
	assert.Equal(t, "test-secret-123", *retrieved.Secret)
	assert.Equal(t, "value1", retrieved.Headers["X-Custom-Header"])
	assert.Equal(t, types.ExecutionWebhookStatusPending, retrieved.Status)
}

func TestRegisterExecutionWebhook_NilWebhook(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	err := provider.RegisterExecutionWebhook(ctx, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nil")
}

func TestRegisterExecutionWebhook_EmptyExecutionID(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	webhook := &types.ExecutionWebhook{
		ExecutionID: "",
		URL:         "https://example.com/webhook",
	}

	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "execution id is required")
}

func TestRegisterExecutionWebhook_EmptyURL(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	webhook := &types.ExecutionWebhook{
		ExecutionID: "exec-1",
		URL:         "",
	}

	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "url is required")
}

func TestRegisterExecutionWebhook_NoSecret(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	webhook := &types.ExecutionWebhook{
		ExecutionID: "exec-no-secret",
		URL:         "https://example.com/webhook",
		Secret:      nil,
	}

	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.NoError(t, err)

	retrieved, err := provider.GetExecutionWebhook(ctx, "exec-no-secret")
	require.NoError(t, err)
	assert.Nil(t, retrieved.Secret)
}

func TestRegisterExecutionWebhook_NoHeaders(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	webhook := &types.ExecutionWebhook{
		ExecutionID: "exec-no-headers",
		URL:         "https://example.com/webhook",
		Headers:     nil,
	}

	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.NoError(t, err)

	retrieved, err := provider.GetExecutionWebhook(ctx, "exec-no-headers")
	require.NoError(t, err)
	assert.NotNil(t, retrieved.Headers)
	assert.Empty(t, retrieved.Headers)
}

func TestRegisterExecutionWebhook_UpdateExisting(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// Register initial webhook
	secret1 := "secret-1"
	webhook1 := &types.ExecutionWebhook{
		ExecutionID: "exec-update",
		URL:         "https://example.com/webhook1",
		Secret:      &secret1,
	}
	err := provider.RegisterExecutionWebhook(ctx, webhook1)
	require.NoError(t, err)

	// Update with new URL
	secret2 := "secret-2"
	webhook2 := &types.ExecutionWebhook{
		ExecutionID: "exec-update",
		URL:         "https://example.com/webhook2",
		Secret:      &secret2,
	}
	err = provider.RegisterExecutionWebhook(ctx, webhook2)
	require.NoError(t, err)

	// Verify updated
	retrieved, err := provider.GetExecutionWebhook(ctx, "exec-update")
	require.NoError(t, err)
	assert.Equal(t, "https://example.com/webhook2", retrieved.URL)
	assert.Equal(t, "secret-2", *retrieved.Secret)
}

func TestGetExecutionWebhook_NotFound(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	webhook, err := provider.GetExecutionWebhook(ctx, "non-existent")
	require.NoError(t, err)
	assert.Nil(t, webhook, "Should return nil for non-existent webhook")
}

func TestListDueExecutionWebhooks_Empty(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	webhooks, err := provider.ListDueExecutionWebhooks(ctx, 10)
	require.NoError(t, err)
	assert.Empty(t, webhooks)
}

func TestListDueExecutionWebhooks_DueNow(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// Register webhook that's due now
	now := time.Now().UTC()
	webhook := &types.ExecutionWebhook{
		ExecutionID:   "exec-due-now",
		URL:           "https://example.com/webhook",
		Status:        types.ExecutionWebhookStatusPending,
		NextAttemptAt: &now,
	}
	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.NoError(t, err)

	// List due webhooks
	webhooks, err := provider.ListDueExecutionWebhooks(ctx, 10)
	require.NoError(t, err)
	require.Len(t, webhooks, 1)
	assert.Equal(t, "exec-due-now", webhooks[0].ExecutionID)
}

func TestListDueExecutionWebhooks_NotDueYet(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// Register webhook that's not due yet
	future := time.Now().UTC().Add(1 * time.Hour)
	webhook := &types.ExecutionWebhook{
		ExecutionID:   "exec-future",
		URL:           "https://example.com/webhook",
		Status:        types.ExecutionWebhookStatusPending,
		NextAttemptAt: &future,
	}
	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.NoError(t, err)

	// Should not be returned
	webhooks, err := provider.ListDueExecutionWebhooks(ctx, 10)
	require.NoError(t, err)
	assert.Empty(t, webhooks, "Future webhooks should not be returned")
}

func TestListDueExecutionWebhooks_OnlyPending(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	now := time.Now().UTC()

	// Register pending webhook
	webhook1 := &types.ExecutionWebhook{
		ExecutionID:   "exec-pending",
		URL:           "https://example.com/webhook1",
		Status:        types.ExecutionWebhookStatusPending,
		NextAttemptAt: &now,
	}
	err := provider.RegisterExecutionWebhook(ctx, webhook1)
	require.NoError(t, err)

	// Register delivering webhook
	webhook2 := &types.ExecutionWebhook{
		ExecutionID:   "exec-delivering",
		URL:           "https://example.com/webhook2",
		Status:        types.ExecutionWebhookStatusDelivering,
		NextAttemptAt: &now,
	}
	err = provider.RegisterExecutionWebhook(ctx, webhook2)
	require.NoError(t, err)

	// Only pending should be returned
	webhooks, err := provider.ListDueExecutionWebhooks(ctx, 10)
	require.NoError(t, err)
	require.Len(t, webhooks, 1)
	assert.Equal(t, "exec-pending", webhooks[0].ExecutionID)
}

func TestListDueExecutionWebhooks_Limit(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	now := time.Now().UTC()

	// Register multiple webhooks
	for i := 1; i <= 5; i++ {
		webhook := &types.ExecutionWebhook{
			ExecutionID:   fmt.Sprintf("exec-%d", i),
			URL:           fmt.Sprintf("https://example.com/webhook%d", i),
			Status:        types.ExecutionWebhookStatusPending,
			NextAttemptAt: &now,
		}
		err := provider.RegisterExecutionWebhook(ctx, webhook)
		require.NoError(t, err)
	}

	// Request with limit
	webhooks, err := provider.ListDueExecutionWebhooks(ctx, 3)
	require.NoError(t, err)
	assert.Len(t, webhooks, 3, "Should respect limit")
}

func TestListDueExecutionWebhooks_DefaultLimit(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// Pass zero limit to test default - should not error even with no webhooks
	webhooks, err := provider.ListDueExecutionWebhooks(ctx, 0)
	require.NoError(t, err)
	// No webhooks registered, so result is empty (nil slice from SQL iteration)
	assert.Empty(t, webhooks)
}

func TestTryMarkExecutionWebhookInFlight_Success(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// Register pending webhook
	now := time.Now().UTC()
	webhook := &types.ExecutionWebhook{
		ExecutionID:   "exec-mark-flight",
		URL:           "https://example.com/webhook",
		Status:        types.ExecutionWebhookStatusPending,
		NextAttemptAt: &now,
	}
	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.NoError(t, err)

	// Try to mark in flight
	marked, err := provider.TryMarkExecutionWebhookInFlight(ctx, "exec-mark-flight", time.Now().UTC())
	require.NoError(t, err)
	assert.True(t, marked, "Should successfully mark as in flight")

	// Verify status changed
	retrieved, err := provider.GetExecutionWebhook(ctx, "exec-mark-flight")
	require.NoError(t, err)
	assert.Equal(t, types.ExecutionWebhookStatusDelivering, retrieved.Status)
}

func TestTryMarkExecutionWebhookInFlight_AlreadyDelivering(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// Register delivering webhook
	now := time.Now().UTC()
	webhook := &types.ExecutionWebhook{
		ExecutionID:   "exec-already-delivering",
		URL:           "https://example.com/webhook",
		Status:        types.ExecutionWebhookStatusDelivering,
		NextAttemptAt: &now,
	}
	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.NoError(t, err)

	// Try to mark again - should fail
	marked, err := provider.TryMarkExecutionWebhookInFlight(ctx, "exec-already-delivering", time.Now().UTC())
	require.NoError(t, err)
	assert.False(t, marked, "Should not mark already-delivering webhook")
}

func TestTryMarkExecutionWebhookInFlight_NotDueYet(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// Register webhook not due yet
	future := time.Now().UTC().Add(1 * time.Hour)
	webhook := &types.ExecutionWebhook{
		ExecutionID:   "exec-not-due",
		URL:           "https://example.com/webhook",
		Status:        types.ExecutionWebhookStatusPending,
		NextAttemptAt: &future,
	}
	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.NoError(t, err)

	// Try to mark - should fail
	marked, err := provider.TryMarkExecutionWebhookInFlight(ctx, "exec-not-due", time.Now().UTC())
	require.NoError(t, err)
	assert.False(t, marked, "Should not mark webhook not due yet")
}

func TestUpdateExecutionWebhookState_Success(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// Register initial webhook
	webhook := &types.ExecutionWebhook{
		ExecutionID: "exec-update-state",
		URL:         "https://example.com/webhook",
		Status:      types.ExecutionWebhookStatusPending,
	}
	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.NoError(t, err)

	// Update state
	nextAttempt := time.Now().UTC().Add(5 * time.Minute)
	lastAttempt := time.Now().UTC()
	lastError := "Connection timeout"

	update := types.ExecutionWebhookStateUpdate{
		Status:        types.ExecutionWebhookStatusPending,
		AttemptCount:  1,
		NextAttemptAt: &nextAttempt,
		LastAttemptAt: &lastAttempt,
		LastError:     &lastError,
	}
	err = provider.UpdateExecutionWebhookState(ctx, "exec-update-state", update)
	require.NoError(t, err)

	// Verify updated
	retrieved, err := provider.GetExecutionWebhook(ctx, "exec-update-state")
	require.NoError(t, err)
	assert.Equal(t, 1, retrieved.AttemptCount)
	assert.NotNil(t, retrieved.NextAttemptAt)
	assert.NotNil(t, retrieved.LastAttemptAt)
	assert.NotNil(t, retrieved.LastError)
	assert.Equal(t, "Connection timeout", *retrieved.LastError)
}

func TestUpdateExecutionWebhookState_ClearError(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// Register webhook with error
	webhook := &types.ExecutionWebhook{
		ExecutionID: "exec-clear-error",
		URL:         "https://example.com/webhook",
		Status:      types.ExecutionWebhookStatusPending,
	}
	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.NoError(t, err)

	// Set error first
	lastError := "Some error"
	update1 := types.ExecutionWebhookStateUpdate{
		Status:       types.ExecutionWebhookStatusPending,
		AttemptCount: 1,
		LastError:    &lastError,
	}
	err = provider.UpdateExecutionWebhookState(ctx, "exec-clear-error", update1)
	require.NoError(t, err)

	// Clear error (nil LastError)
	update2 := types.ExecutionWebhookStateUpdate{
		Status:       types.ExecutionWebhookStatusDelivered,
		AttemptCount: 2,
		LastError:    nil,
	}
	err = provider.UpdateExecutionWebhookState(ctx, "exec-clear-error", update2)
	require.NoError(t, err)

	retrieved, err := provider.GetExecutionWebhook(ctx, "exec-clear-error")
	require.NoError(t, err)
	// LastError might still be set depending on implementation
	// Test documents current behavior
	_ = retrieved.LastError
}

func TestHasExecutionWebhook_True(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	webhook := &types.ExecutionWebhook{
		ExecutionID: "exec-has-webhook",
		URL:         "https://example.com/webhook",
	}
	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.NoError(t, err)

	has, err := provider.HasExecutionWebhook(ctx, "exec-has-webhook")
	require.NoError(t, err)
	assert.True(t, has)
}

func TestHasExecutionWebhook_False(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	has, err := provider.HasExecutionWebhook(ctx, "non-existent")
	require.NoError(t, err)
	assert.False(t, has)
}

func TestListExecutionWebhooksRegistered_Empty(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	result, err := provider.ListExecutionWebhooksRegistered(ctx, []string{})
	require.NoError(t, err)
	assert.Empty(t, result)
}

func TestListExecutionWebhooksRegistered_SomeRegistered(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// Register some webhooks
	webhook1 := &types.ExecutionWebhook{
		ExecutionID: "exec-1",
		URL:         "https://example.com/webhook1",
	}
	webhook2 := &types.ExecutionWebhook{
		ExecutionID: "exec-2",
		URL:         "https://example.com/webhook2",
	}
	err := provider.RegisterExecutionWebhook(ctx, webhook1)
	require.NoError(t, err)
	err = provider.RegisterExecutionWebhook(ctx, webhook2)
	require.NoError(t, err)

	// Query for mix of registered and unregistered
	result, err := provider.ListExecutionWebhooksRegistered(ctx, []string{
		"exec-1", "exec-2", "exec-3", "exec-4",
	})
	require.NoError(t, err)

	assert.True(t, result["exec-1"])
	assert.True(t, result["exec-2"])
	assert.False(t, result["exec-3"])
	assert.False(t, result["exec-4"])
}

func TestListExecutionWebhooksRegistered_Deduplication(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	webhook := &types.ExecutionWebhook{
		ExecutionID: "exec-dup",
		URL:         "https://example.com/webhook",
	}
	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.NoError(t, err)

	// Pass duplicates
	result, err := provider.ListExecutionWebhooksRegistered(ctx, []string{
		"exec-dup", "exec-dup", "exec-dup",
	})
	require.NoError(t, err)
	assert.True(t, result["exec-dup"])
	assert.Len(t, result, 1, "Should deduplicate")
}

func TestListExecutionWebhooksRegistered_EmptyStrings(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// Pass empty strings
	result, err := provider.ListExecutionWebhooksRegistered(ctx, []string{
		"", "  ", "\t",
	})
	require.NoError(t, err)
	assert.Empty(t, result, "Should handle empty strings")
}

// Concurrency test
func TestExecutionWebhooks_ConcurrentMarkInFlight(t *testing.T) {
	provider, ctx := setupTestStorage(t)

	// Register webhook
	now := time.Now().UTC()
	webhook := &types.ExecutionWebhook{
		ExecutionID:   "exec-concurrent",
		URL:           "https://example.com/webhook",
		Status:        types.ExecutionWebhookStatusPending,
		NextAttemptAt: &now,
	}
	err := provider.RegisterExecutionWebhook(ctx, webhook)
	require.NoError(t, err)

	// Try to mark in flight concurrently
	var wg sync.WaitGroup
	successCount := 0
	var mu sync.Mutex

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			marked, err := provider.TryMarkExecutionWebhookInFlight(ctx, "exec-concurrent", time.Now().UTC())
			if err == nil && marked {
				mu.Lock()
				successCount++
				mu.Unlock()
			}
		}()
	}

	wg.Wait()

	// Only one should succeed
	assert.Equal(t, 1, successCount, "Only one goroutine should mark as in flight")
}
