package storage

import (
	"github.com/your-org/haxen/control-plane/pkg/types"
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/boltdb/bolt"
	"github.com/google/uuid"
)

const (
	locksBucket = "locks"
)

// AcquireLock attempts to acquire a distributed lock.
func (ls *LocalStorage) AcquireLock(ctx context.Context, key string, timeout time.Duration) (*types.DistributedLock, error) {
	if ls.mode == "postgres" {
		return ls.acquireLockPostgres(ctx, key, timeout)
	}

	// Fast-fail if context is already cancelled
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var lock *types.DistributedLock
	err := ls.kvStore.Update(func(tx *bolt.Tx) error {
		// Implementation will be added here
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to acquire lock: %w", err)
	}
	return lock, nil
}

// ReleaseLock releases a distributed lock.
func (ls *LocalStorage) ReleaseLock(ctx context.Context, lockID string) error {
	if ls.mode == "postgres" {
		return ls.releaseLockPostgres(ctx, lockID)
	}

	// Fast-fail if context is already cancelled
	if err := ctx.Err(); err != nil {
		return err
	}

	return ls.kvStore.Update(func(tx *bolt.Tx) error {
		// Implementation will be added here
		return nil
	})
}

// RenewLock renews a distributed lock to extend its TTL.
func (ls *LocalStorage) RenewLock(ctx context.Context, lockID string) (*types.DistributedLock, error) {
	if ls.mode == "postgres" {
		return ls.renewLockPostgres(ctx, lockID)
	}

	// Fast-fail if context is already cancelled
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var lock *types.DistributedLock
	err := ls.kvStore.Update(func(tx *bolt.Tx) error {
		// Implementation will be added here
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to renew lock: %w", err)
	}
	return lock, nil
}

// GetLockStatus retrieves the status of a distributed lock.
func (ls *LocalStorage) GetLockStatus(ctx context.Context, key string) (*types.DistributedLock, error) {
	if ls.mode == "postgres" {
		return ls.getLockStatusPostgres(ctx, key)
	}

	// Fast-fail if context is already cancelled
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	var lock *types.DistributedLock
	err := ls.kvStore.View(func(tx *bolt.Tx) error {
		// Implementation will be added here
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to get lock status: %w", err)
	}
	return lock, nil
}

func (ls *LocalStorage) acquireLockPostgres(ctx context.Context, key string, timeout time.Duration) (*types.DistributedLock, error) {
	if timeout <= 0 {
		timeout = 30 * time.Second
	}

	expiresAt := time.Now().UTC().Add(timeout)
	lockID := uuid.NewString()

	query := `
        INSERT INTO distributed_locks(lock_id, key, owner, expires_at, created_at, updated_at)
        VALUES (?, ?, ?, ?, NOW(), NOW())
        ON CONFLICT (key) DO UPDATE SET
                lock_id = EXCLUDED.lock_id,
                owner = EXCLUDED.owner,
                expires_at = EXCLUDED.expires_at,
                updated_at = NOW()
        WHERE distributed_locks.expires_at <= NOW();`

	result, err := ls.db.ExecContext(ctx, query, lockID, key, lockID, expiresAt)
	if err != nil {
		return nil, fmt.Errorf("failed to acquire postgres lock: %w", err)
	}

	rows, err := result.RowsAffected()
	if err != nil {
		return nil, fmt.Errorf("failed to determine lock acquisition result: %w", err)
	}

	if rows == 0 {
		existing, err := ls.getLockStatusPostgres(ctx, key)
		if err != nil {
			return nil, err
		}
		if existing != nil && existing.ExpiresAt.After(time.Now()) {
			return nil, fmt.Errorf("lock '%s' is already held", key)
		}
		return nil, fmt.Errorf("failed to acquire lock '%s'", key)
	}

	return &types.DistributedLock{
		LockID:    lockID,
		Key:       key,
		Holder:    lockID,
		ExpiresAt: expiresAt,
		CreatedAt: time.Now().UTC(),
	}, nil
}

func (ls *LocalStorage) releaseLockPostgres(ctx context.Context, lockID string) error {
	query := `DELETE FROM distributed_locks WHERE lock_id = ?`
	result, err := ls.db.ExecContext(ctx, query, lockID)
	if err != nil {
		return fmt.Errorf("failed to release postgres lock: %w", err)
	}
	if rows, err := result.RowsAffected(); err == nil && rows == 0 {
		return fmt.Errorf("lock '%s' not found", lockID)
	}
	return nil
}

func (ls *LocalStorage) renewLockPostgres(ctx context.Context, lockID string) (*types.DistributedLock, error) {
	expiresAt := time.Now().UTC().Add(30 * time.Second)
	query := `
        UPDATE distributed_locks
        SET expires_at = ?, updated_at = NOW()
        WHERE lock_id = ?
        RETURNING key, owner, created_at`

	row := ls.db.QueryRowContext(ctx, query, expiresAt, lockID)
	var (
		key       string
		owner     string
		createdAt time.Time
	)
	if err := row.Scan(&key, &owner, &createdAt); err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("lock '%s' not found", lockID)
		}
		return nil, fmt.Errorf("failed to renew postgres lock: %w", err)
	}

	return &types.DistributedLock{
		LockID:    lockID,
		Key:       key,
		Holder:    owner,
		ExpiresAt: expiresAt,
		CreatedAt: createdAt,
	}, nil
}

func (ls *LocalStorage) getLockStatusPostgres(ctx context.Context, key string) (*types.DistributedLock, error) {
	query := `
        SELECT lock_id, owner, expires_at, created_at
        FROM distributed_locks
        WHERE key = ?`

	row := ls.db.QueryRowContext(ctx, query, key)
	var (
		lockID    string
		owner     string
		expiresAt time.Time
		createdAt time.Time
	)

	if err := row.Scan(&lockID, &owner, &expiresAt, &createdAt); err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to get postgres lock status: %w", err)
	}

	return &types.DistributedLock{
		LockID:    lockID,
		Key:       key,
		Holder:    owner,
		ExpiresAt: expiresAt,
		CreatedAt: createdAt,
	}, nil
}
