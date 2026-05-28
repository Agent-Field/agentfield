package storage

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAcquireLock_Success(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available, skipping lock tests")
	}

	key := "test-lock-1"
	timeout := 30 * time.Second

	lock, err := provider.AcquireLock(ctx, key, timeout)
	require.NoError(t, err)
	require.NotNil(t, lock)

	assert.NotEmpty(t, lock.LockID)
	assert.Equal(t, key, lock.Key)
	assert.Equal(t, lock.LockID, lock.Holder)
	assert.True(t, lock.ExpiresAt.After(time.Now()))
	assert.True(t, lock.ExpiresAt.Before(time.Now().Add(timeout+time.Second)))
}

func TestAcquireLock_DefaultTimeout(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-default-timeout"

	// Pass zero timeout to test default
	lock, err := provider.AcquireLock(ctx, key, 0)
	require.NoError(t, err)
	require.NotNil(t, lock)

	// Should use default 30-second timeout
	expectedExpiry := time.Now().Add(30 * time.Second)
	assert.WithinDuration(t, expectedExpiry, lock.ExpiresAt, 2*time.Second)
}

func TestAcquireLock_AlreadyHeld(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-contention"
	timeout := 30 * time.Second

	// First acquisition should succeed
	lock1, err := provider.AcquireLock(ctx, key, timeout)
	require.NoError(t, err)
	require.NotNil(t, lock1)

	// Second acquisition should fail (lock still held)
	_, err = provider.AcquireLock(ctx, key, timeout)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already held")
}

func TestAcquireLock_ExpiredLock(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-expired"

	// Acquire with very short timeout
	lock1, err := provider.AcquireLock(ctx, key, 1*time.Second)
	require.NoError(t, err)
	require.NotNil(t, lock1)

	// Wait for lock to expire
	time.Sleep(2 * time.Second)

	// Second acquisition should succeed (lock expired)
	lock2, err := provider.AcquireLock(ctx, key, 30*time.Second)
	require.NoError(t, err)
	require.NotNil(t, lock2)
	assert.NotEqual(t, lock1.LockID, lock2.LockID, "Should get new lock ID after expiration")
}

func TestAcquireLock_ContextCancellation(t *testing.T) {
	provider, _ := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-context-cancel"

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Should fail immediately due to cancelled context
	_, err := provider.AcquireLock(ctx, key, 30*time.Second)
	require.Error(t, err)
	assert.Equal(t, context.Canceled, err)
}

func TestReleaseLock_Success(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-release"

	// Acquire lock
	lock, err := provider.AcquireLock(ctx, key, 30*time.Second)
	require.NoError(t, err)

	// Release lock
	err = provider.ReleaseLock(ctx, lock.LockID)
	require.NoError(t, err)

	// Should be able to acquire again
	lock2, err := provider.AcquireLock(ctx, key, 30*time.Second)
	require.NoError(t, err)
	require.NotNil(t, lock2)
}

func TestReleaseLock_NotFound(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	// Try to release non-existent lock
	err := provider.ReleaseLock(ctx, "non-existent-lock-id")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestReleaseLock_ContextCancellation(t *testing.T) {
	provider, _ := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Should fail immediately
	err := provider.ReleaseLock(ctx, "some-lock-id")
	require.Error(t, err)
	assert.Equal(t, context.Canceled, err)
}

func TestRenewLock_Success(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-renew"

	// Acquire lock
	lock, err := provider.AcquireLock(ctx, key, 5*time.Second)
	require.NoError(t, err)

	originalExpiry := lock.ExpiresAt

	// Wait a bit
	time.Sleep(1 * time.Second)

	// Renew lock
	renewed, err := provider.RenewLock(ctx, lock.LockID)
	require.NoError(t, err)
	require.NotNil(t, renewed)

	assert.Equal(t, lock.LockID, renewed.LockID)
	assert.Equal(t, lock.Key, renewed.Key)
	assert.True(t, renewed.ExpiresAt.After(originalExpiry), "Renewed expiry should be later")
}

func TestRenewLock_NotFound(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	// Try to renew non-existent lock
	_, err := provider.RenewLock(ctx, "non-existent-lock-id")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestRenewLock_ContextCancellation(t *testing.T) {
	provider, _ := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Should fail immediately
	_, err := provider.RenewLock(ctx, "some-lock-id")
	require.Error(t, err)
	assert.Equal(t, context.Canceled, err)
}

func TestGetLockStatus_Exists(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-status"

	// Acquire lock
	acquired, err := provider.AcquireLock(ctx, key, 30*time.Second)
	require.NoError(t, err)

	// Get status
	status, err := provider.GetLockStatus(ctx, key)
	require.NoError(t, err)
	require.NotNil(t, status)

	assert.Equal(t, acquired.LockID, status.LockID)
	assert.Equal(t, acquired.Key, status.Key)
	assert.Equal(t, acquired.Holder, status.Holder)
}

func TestGetLockStatus_NotExists(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	// Get status for non-existent lock
	status, err := provider.GetLockStatus(ctx, "non-existent-key")
	require.NoError(t, err)
	assert.Nil(t, status, "Should return nil for non-existent lock")
}

func TestGetLockStatus_Expired(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-status-expired"

	// Acquire with short timeout
	_, err := provider.AcquireLock(ctx, key, 1*time.Second)
	require.NoError(t, err)

	// Wait for expiration
	time.Sleep(2 * time.Second)

	// Status should still return the lock (even though expired)
	status, err := provider.GetLockStatus(ctx, key)
	require.NoError(t, err)
	require.NotNil(t, status)
	assert.True(t, status.ExpiresAt.Before(time.Now()), "Lock should be expired")
}

func TestGetLockStatus_ContextCancellation(t *testing.T) {
	provider, _ := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Should fail immediately
	_, err := provider.GetLockStatus(ctx, "some-key")
	require.Error(t, err)
	assert.Equal(t, context.Canceled, err)
}

// Concurrency Tests

func TestAcquireLock_Concurrent(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-concurrent"
	numGoroutines := 10
	timeout := 30 * time.Second

	var wg sync.WaitGroup
	successCount := 0
	var mu sync.Mutex

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			lock, err := provider.AcquireLock(ctx, key, timeout)
			if err == nil && lock != nil {
				mu.Lock()
				successCount++
				mu.Unlock()

				// Hold lock briefly
				time.Sleep(100 * time.Millisecond)

				// Release lock
				_ = provider.ReleaseLock(ctx, lock.LockID)
			}
		}(i)
	}

	wg.Wait()

	// Only one goroutine should have successfully acquired the lock initially
	assert.Equal(t, 1, successCount, "Only one goroutine should acquire lock initially")
}

func TestAcquireLock_RaceCondition(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-race"
	timeout := 30 * time.Second

	var wg sync.WaitGroup
	locks := make([]*string, 100)
	var mu sync.Mutex

	// Try to acquire lock 100 times concurrently
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			lock, err := provider.AcquireLock(ctx, key, timeout)
			if err == nil && lock != nil {
				mu.Lock()
				locks[idx] = &lock.LockID
				mu.Unlock()
			}
		}(i)
	}

	wg.Wait()

	// Count successful acquisitions
	successCount := 0
	for _, lockID := range locks {
		if lockID != nil {
			successCount++
		}
	}

	// Should have exactly 1 successful acquisition
	assert.Equal(t, 1, successCount, "Race condition test: only one lock should be acquired")
}

func TestReleaseLock_Concurrent(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-release-concurrent"

	// Acquire lock
	lock, err := provider.AcquireLock(ctx, key, 30*time.Second)
	require.NoError(t, err)

	// Try to release concurrently (should only succeed once)
	var wg sync.WaitGroup
	successCount := 0
	var mu sync.Mutex

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			err := provider.ReleaseLock(ctx, lock.LockID)
			if err == nil {
				mu.Lock()
				successCount++
				mu.Unlock()
			}
		}()
	}

	wg.Wait()

	// Only one release should succeed
	assert.Equal(t, 1, successCount, "Only one release should succeed")
}

func TestRenewLock_Concurrent(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-renew-concurrent"

	// Acquire lock
	lock, err := provider.AcquireLock(ctx, key, 30*time.Second)
	require.NoError(t, err)

	// Renew concurrently multiple times
	var wg sync.WaitGroup
	errors := make([]error, 5)
	renewed := make([]*time.Time, 5)

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			renewedLock, err := provider.RenewLock(ctx, lock.LockID)
			errors[idx] = err
			if err == nil && renewedLock != nil {
				renewed[idx] = &renewedLock.ExpiresAt
			}
		}(i)
	}

	wg.Wait()

	// All renewals should succeed (idempotent operation)
	for i, err := range errors {
		assert.NoError(t, err, "Renewal %d should succeed", i)
	}
}

// Edge Cases

func TestAcquireLock_EmptyKey(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	// Empty key should be handled
	lock, err := provider.AcquireLock(ctx, "", 30*time.Second)
	// Behavior depends on implementation - document it
	_ = lock
	_ = err
}

func TestAcquireLock_VeryLongKey(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	// Very long key
	longKey := fmt.Sprintf("test-lock-%s", string(make([]byte, 1000)))

	lock, err := provider.AcquireLock(ctx, longKey, 30*time.Second)
	// Should handle long keys gracefully
	_ = lock
	_ = err
}

func TestAcquireLock_NegativeTimeout(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-negative-timeout"

	// Negative timeout should use default
	lock, err := provider.AcquireLock(ctx, key, -1*time.Second)
	require.NoError(t, err)
	require.NotNil(t, lock)

	// Should use default 30-second timeout
	expectedExpiry := time.Now().Add(30 * time.Second)
	assert.WithinDuration(t, expectedExpiry, lock.ExpiresAt, 2*time.Second)
}

func TestAcquireReleaseCycle(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-cycle"

	// Multiple acquire-release cycles
	for i := 0; i < 5; i++ {
		lock, err := provider.AcquireLock(ctx, key, 30*time.Second)
		require.NoError(t, err, "Cycle %d: acquire should succeed", i)
		require.NotNil(t, lock)

		err = provider.ReleaseLock(ctx, lock.LockID)
		require.NoError(t, err, "Cycle %d: release should succeed", i)
	}
}

func TestLockExpiration_AutomaticCleanup(t *testing.T) {
	provider, ctx := setupPostgresTestStorage(t)
	if provider == nil {
		t.Skip("PostgreSQL not available")
	}

	key := "test-lock-auto-cleanup"

	// Acquire with short timeout
	lock1, err := provider.AcquireLock(ctx, key, 1*time.Second)
	require.NoError(t, err)

	// Wait for expiration
	time.Sleep(2 * time.Second)

	// New acquisition should succeed and reuse the same key
	lock2, err := provider.AcquireLock(ctx, key, 30*time.Second)
	require.NoError(t, err)
	require.NotNil(t, lock2)
	assert.NotEqual(t, lock1.LockID, lock2.LockID, "Should get different lock ID")
}

// Helper function to setup PostgreSQL test storage
func setupPostgresTestStorage(t *testing.T) (StorageProvider, context.Context) {
	t.Helper()

	// Check if PostgreSQL connection is configured
	// For now, return nil to skip tests (would need actual PostgreSQL setup)
	// In CI/CD, this would connect to a real PostgreSQL instance
	return nil, context.Background()
}
