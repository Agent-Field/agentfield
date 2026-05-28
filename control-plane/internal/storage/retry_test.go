package storage

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestRetryDatabaseOperation_Success(t *testing.T) {
	ls := &LocalStorage{}

	operationID := "test-operation"
	attempts := 0
	operation := func() error {
		attempts++
		return nil
	}

	err := ls.retryDatabaseOperation(context.Background(), operationID, operation)

	require.NoError(t, err)
	require.Equal(t, 1, attempts)
}

func TestRetryDatabaseOperation_RetryableError(t *testing.T) {
	ls := &LocalStorage{}

	operationID := "test-operation"
	attempts := 0
	operation := func() error {
		attempts++
		if attempts < 3 {
			return errors.New("database is locked")
		}
		return nil
	}

	err := ls.retryDatabaseOperation(context.Background(), operationID, operation)

	require.NoError(t, err)
	require.Equal(t, 3, attempts)
}

func TestRetryDatabaseOperation_NonRetryableError(t *testing.T) {
	ls := &LocalStorage{}

	operationID := "test-operation"
	attempts := 0
	operation := func() error {
		attempts++
		return errors.New("permanent error")
	}

	err := ls.retryDatabaseOperation(context.Background(), operationID, operation)

	require.Error(t, err)
	require.Equal(t, 1, attempts)
	require.Contains(t, err.Error(), "permanent error")
}

func TestRetryDatabaseOperation_MaxRetriesExceeded(t *testing.T) {
	ls := &LocalStorage{}

	operationID := "test-operation"
	attempts := 0
	operation := func() error {
		attempts++
		return errors.New("database is locked")
	}

	err := ls.retryDatabaseOperation(context.Background(), operationID, operation)

	require.Error(t, err)
	require.Equal(t, 4, attempts) // maxRetries + 1 = 4 attempts
	require.Contains(t, err.Error(), "failed after 3 retries")
}

func TestRetryDatabaseOperation_ContextCancellation(t *testing.T) {
	ls := &LocalStorage{}

	operationID := "test-operation"
	ctx, cancel := context.WithCancel(context.Background())

	operation := func() error {
		cancel() // Cancel context during operation
		return errors.New("database is locked")
	}

	err := ls.retryDatabaseOperation(ctx, operationID, operation)

	require.Error(t, err)
	require.Contains(t, err.Error(), "context cancelled")
}

func TestRetryDatabaseOperation_ContextCancellationDuringDelay(t *testing.T) {
	ls := &LocalStorage{}

	operationID := "test-operation"
	ctx, cancel := context.WithCancel(context.Background())

	attempts := 0
	operation := func() error {
		attempts++
		if attempts == 1 {
			// Cancel during retry delay
			go func() {
				time.Sleep(10 * time.Millisecond)
				cancel()
			}()
			return errors.New("database is locked")
		}
		return nil
	}

	err := ls.retryDatabaseOperation(ctx, operationID, operation)

	require.Error(t, err)
	require.Contains(t, err.Error(), "context cancelled")
}

func TestIsRetryableError(t *testing.T) {
	ls := &LocalStorage{}

	tests := []struct {
		name          string
		err           error
		shouldRetry   bool
	}{
		{
			name:        "database locked",
			err:         errors.New("database is locked"),
			shouldRetry: true,
		},
		{
			name:        "SQLITE_BUSY",
			err:         errors.New("SQLITE_BUSY"),
			shouldRetry: true,
		},
		{
			name:        "temporarily unavailable",
			err:         errors.New("database is temporarily unavailable"),
			shouldRetry: false, // This error is not in the retryable list
		},
		{
			name:        "permanent error",
			err:         errors.New("permanent error"),
			shouldRetry: false,
		},
		{
			name:        "nil error",
			err:         nil,
			shouldRetry: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ls.isRetryableError(tt.err)
			require.Equal(t, tt.shouldRetry, result)
		})
	}
}

func TestRetryOnConstraintFailure_Success(t *testing.T) {
	ls := &LocalStorage{}

	attempts := 0
	operation := func() error {
		attempts++
		return nil
	}

	err := ls.retryOnConstraintFailure(context.Background(), operation, 3)

	require.NoError(t, err)
	require.Equal(t, 1, attempts)
}

func TestRetryOnConstraintFailure_TransientError(t *testing.T) {
	ls := &LocalStorage{}

	attempts := 0
	operation := func() error {
		attempts++
		if attempts < 2 {
			return errors.New("database is locked")
		}
		return nil
	}

	err := ls.retryOnConstraintFailure(context.Background(), operation, 3)

	require.NoError(t, err)
	require.Equal(t, 2, attempts)
}

func TestRetryOnConstraintFailure_ValidationError(t *testing.T) {
	ls := &LocalStorage{}

	attempts := 0
	operation := func() error {
		attempts++
		return &ValidationError{
			Field:   "test_field",
			Value:   "test_value",
			Reason:  "validation failed",
			Context: "test_context",
		}
	}

	err := ls.retryOnConstraintFailure(context.Background(), operation, 3)

	require.Error(t, err)
	require.Equal(t, 1, attempts)
	require.IsType(t, &ValidationError{}, err)
}

func TestRetryOnConstraintFailure_ForeignKeyError(t *testing.T) {
	ls := &LocalStorage{}

	attempts := 0
	operation := func() error {
		attempts++
		return &ForeignKeyConstraintError{
			Table:           "test_table",
			Column:          "test_column",
			ReferencedTable: "ref_table",
			ReferencedValue: "ref_value",
			Operation:       "insert",
		}
	}

	err := ls.retryOnConstraintFailure(context.Background(), operation, 3)

	require.Error(t, err)
	require.Equal(t, 1, attempts)
	require.IsType(t, &ForeignKeyConstraintError{}, err)
}

func TestRetryOnConstraintFailure_DuplicateDIDError(t *testing.T) {
	ls := &LocalStorage{}

	attempts := 0
	operation := func() error {
		attempts++
		return &DuplicateDIDError{
			DID:  "did:test:123",
			Type: "agent",
		}
	}

	err := ls.retryOnConstraintFailure(context.Background(), operation, 3)

	require.Error(t, err)
	require.Equal(t, 1, attempts)
	require.IsType(t, &DuplicateDIDError{}, err)
}

func TestRetryOnConstraintFailure_MaxRetries(t *testing.T) {
	ls := &LocalStorage{}

	attempts := 0
	operation := func() error {
		attempts++
		return errors.New("database is locked")
	}

	err := ls.retryOnConstraintFailure(context.Background(), operation, 2)

	require.Error(t, err)
	require.Equal(t, 3, attempts) // maxRetries + 1
	require.Contains(t, err.Error(), "database is locked")
}

func TestRetryOnConstraintFailure_ContextCancellation(t *testing.T) {
	ls := &LocalStorage{}

	ctx, cancel := context.WithCancel(context.Background())

	attempts := 0
	operation := func() error {
		attempts++
		if attempts == 1 {
			cancel()
		}
		return errors.New("database is locked")
	}

	err := ls.retryOnConstraintFailure(ctx, operation, 3)

	require.Error(t, err)
	require.Contains(t, err.Error(), "context cancelled")
}

func TestRetryOnConstraintFailure_ExponentialBackoff(t *testing.T) {
	ls := &LocalStorage{}

	attempts := 0
	startTime := time.Now()
	operation := func() error {
		attempts++
		if attempts < 3 {
			return errors.New("database is locked")
		}
		return nil
	}

	err := ls.retryOnConstraintFailure(context.Background(), operation, 3)
	elapsed := time.Since(startTime)

	require.NoError(t, err)
	require.Equal(t, 3, attempts)
	// Verify exponential backoff: 10ms, 20ms = at least 30ms total
	require.GreaterOrEqual(t, elapsed, 30*time.Millisecond)
}

func TestIsRetryableError_CaseInsensitive(t *testing.T) {
	ls := &LocalStorage{}

	tests := []struct {
		err         error
		shouldRetry bool
	}{
		{errors.New("DATABASE IS LOCKED"), true},
		{errors.New("Database Is Locked"), true},
		{errors.New("sqlite_busy"), true},
		{errors.New("SQLITE_BUSY"), true},
		{errors.New("database is temporarily unavailable"), false}, // Not in retryable list
	}

	for _, tt := range tests {
		result := ls.isRetryableError(tt.err)
		require.Equal(t, tt.shouldRetry, result, "error: %v", tt.err)
	}
}

func TestRetryDatabaseOperation_ExponentialBackoff(t *testing.T) {
	ls := &LocalStorage{}

	operationID := "test-operation"
	attempts := 0
	startTime := time.Now()
	operation := func() error {
		attempts++
		if attempts < 3 {
			return errors.New("database is locked")
		}
		return nil
	}

	err := ls.retryDatabaseOperation(context.Background(), operationID, operation)
	elapsed := time.Since(startTime)

	require.NoError(t, err)
	require.Equal(t, 3, attempts)
	// Verify exponential backoff: 50ms, 100ms = at least 150ms total
	require.GreaterOrEqual(t, elapsed, 150*time.Millisecond)
}

func TestRetryDatabaseOperation_ErrorMessages(t *testing.T) {
	ls := &LocalStorage{}

	// Test various SQLite error messages
	retryableErrors := []string{
		"database is locked",
		"SQLITE_BUSY",
		"sqlite_busy", // Lowercase variant
	}

	for _, errMsg := range retryableErrors {
		t.Run(errMsg, func(t *testing.T) {
			operationID := "test-operation"
			attempts := 0
			operation := func() error {
				attempts++
				if attempts < 2 {
					return errors.New(errMsg)
				}
				return nil
			}

			err := ls.retryDatabaseOperation(context.Background(), operationID, operation)

			// Should retry and eventually succeed
			require.NoError(t, err)
			require.Equal(t, 2, attempts)
		})
	}
}
