package storage

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// newInvariantUoW builds a unitOfWorkImpl with nil DB so that the
// pre-commit registration and lifecycle invariants can be tested without
// a real database.  Commit will fail (nil db) but that is intentional for
// the atomicity / rollback tests.
func newInvariantUoW() *unitOfWorkImpl {
	return &unitOfWorkImpl{
		db:      nil,
		changes: make([]Change, 0),
		active:  true,
		backend: nil,
	}
}

// nopOp is a no-op database operation used when the DB is irrelevant to the invariant being tested.
func nopOp(tx DBTX) error { return nil }

// failOp always returns an error, used to simulate a failing operation.
func failOp(tx DBTX) error { return errors.New("simulated operation failure") }

// ---------------------------------------------------------------------------
// Ordering invariant
// ---------------------------------------------------------------------------

// TestInvariant_UnitOfWork_OrderingPreserved verifies that changes registered
// in order A, B, C are stored (and would be committed) in the same order.
func TestInvariant_UnitOfWork_OrderingPreserved(t *testing.T) {
	uow := newInvariantUoW()

	uow.RegisterNew("entity-A", "table-a", nopOp)
	uow.RegisterDirty("entity-B", "table-b", nopOp)
	uow.RegisterDeleted("entity-C", "table-c", nopOp)

	require.Equal(t, 3, uow.GetChangeCount())

	tables := []string{"table-a", "table-b", "table-c"}
	for i, c := range uow.changes {
		assert.Equal(t, tables[i], c.Table,
			"change at index %d must be for table %s, got %s", i, tables[i], c.Table)
	}
}

// ---------------------------------------------------------------------------
// IsActive lifecycle invariant
// ---------------------------------------------------------------------------

// TestInvariant_UnitOfWork_IsActiveAfterCreation verifies that a newly created
// UoW is active.
func TestInvariant_UnitOfWork_IsActiveAfterCreation(t *testing.T) {
	uow := newInvariantUoW()
	assert.True(t, uow.IsActive(), "unit of work must be active immediately after creation")
}

// TestInvariant_UnitOfWork_IsInactiveAfterRollback verifies that IsActive
// returns false after Rollback is called.
func TestInvariant_UnitOfWork_IsInactiveAfterRollback(t *testing.T) {
	uow := newInvariantUoW()
	uow.RegisterNew("e", "t", nopOp)

	err := uow.Rollback()
	require.NoError(t, err)
	assert.False(t, uow.IsActive(), "unit of work must be inactive after Rollback")
}

// TestInvariant_UnitOfWork_IsInactiveAfterFailedCommit verifies that IsActive
// returns false even when Commit fails (e.g. nil db → cannot begin tx).
func TestInvariant_UnitOfWork_IsInactiveAfterFailedCommit(t *testing.T) {
	uow := newInvariantUoW()
	uow.RegisterNew("e", "t", nopOp)

	err := uow.Commit()
	// A nil DB causes the commit to fail; the UoW must still become inactive.
	require.Error(t, err, "expected commit to fail with nil DB")
	assert.False(t, uow.IsActive(), "unit of work must be inactive after a failed Commit")
}

// ---------------------------------------------------------------------------
// Idempotency of Rollback
// ---------------------------------------------------------------------------

// TestInvariant_UnitOfWork_RollbackIdempotent verifies that calling Rollback
// twice does not panic and does not corrupt state.
func TestInvariant_UnitOfWork_RollbackIdempotent(t *testing.T) {
	uow := newInvariantUoW()
	uow.RegisterNew("e", "t", nopOp)

	require.NotPanics(t, func() {
		_ = uow.Rollback()
		_ = uow.Rollback() // second call — must not panic
	})
	assert.False(t, uow.IsActive())
}

// ---------------------------------------------------------------------------
// HasChanges correctness
// ---------------------------------------------------------------------------

// TestInvariant_UnitOfWork_HasChangesStartsFalse verifies HasChanges is false
// on an empty UoW.
func TestInvariant_UnitOfWork_HasChangesStartsFalse(t *testing.T) {
	uow := newInvariantUoW()
	assert.False(t, uow.HasChanges(), "empty UoW must report HasChanges=false")
}

// TestInvariant_UnitOfWork_HasChangesAfterRegisterNew verifies HasChanges
// becomes true after RegisterNew.
func TestInvariant_UnitOfWork_HasChangesAfterRegisterNew(t *testing.T) {
	uow := newInvariantUoW()
	uow.RegisterNew("entity", "table", nopOp)
	assert.True(t, uow.HasChanges(), "HasChanges must be true after RegisterNew")
}

// TestInvariant_UnitOfWork_HasChangesAfterRegisterDirty verifies HasChanges
// becomes true after RegisterDirty.
func TestInvariant_UnitOfWork_HasChangesAfterRegisterDirty(t *testing.T) {
	uow := newInvariantUoW()
	uow.RegisterDirty("entity", "table", nopOp)
	assert.True(t, uow.HasChanges(), "HasChanges must be true after RegisterDirty")
}

// TestInvariant_UnitOfWork_HasChangesAfterRegisterDeleted verifies HasChanges
// becomes true after RegisterDeleted.
func TestInvariant_UnitOfWork_HasChangesAfterRegisterDeleted(t *testing.T) {
	uow := newInvariantUoW()
	uow.RegisterDeleted("id", "table", nopOp)
	assert.True(t, uow.HasChanges(), "HasChanges must be true after RegisterDeleted")
}

// ---------------------------------------------------------------------------
// Registration ignored when inactive
// ---------------------------------------------------------------------------

// TestInvariant_UnitOfWork_RegisterIgnoredWhenInactive verifies that attempts
// to register changes on an inactive UoW are silently ignored.
func TestInvariant_UnitOfWork_RegisterIgnoredWhenInactive(t *testing.T) {
	uow := newInvariantUoW()
	_ = uow.Rollback() // make inactive

	uow.RegisterNew("e", "t", nopOp)
	uow.RegisterDirty("e", "t", nopOp)
	uow.RegisterDeleted("id", "t", nopOp)

	assert.Equal(t, 0, uow.GetChangeCount(),
		"changes registered on inactive UoW must be ignored")
	assert.False(t, uow.HasChanges())
}

// ---------------------------------------------------------------------------
// Commit on already-inactive UoW
// ---------------------------------------------------------------------------

// TestInvariant_UnitOfWork_CommitOnInactiveReturnsError verifies that calling
// Commit on an already-inactive UoW returns an error rather than panicking.
func TestInvariant_UnitOfWork_CommitOnInactiveReturnsError(t *testing.T) {
	uow := newInvariantUoW()
	_ = uow.Rollback()

	err := uow.Commit()
	require.Error(t, err, "Commit on an inactive UoW must return an error")
}

// ---------------------------------------------------------------------------
// Change type attribution
// ---------------------------------------------------------------------------

// TestInvariant_UnitOfWork_ChangeTypesCorrectlyAttributed verifies that each
// register method stamps the correct ChangeType on the recorded Change.
func TestInvariant_UnitOfWork_ChangeTypesCorrectlyAttributed(t *testing.T) {
	uow := newInvariantUoW()

	uow.RegisterNew("e", "t", nopOp)
	uow.RegisterDirty("e", "t", nopOp)
	uow.RegisterDeleted("id", "t", nopOp)

	require.Equal(t, 3, len(uow.changes))
	assert.Equal(t, ChangeTypeNew, uow.changes[0].Type)
	assert.Equal(t, ChangeTypeDirty, uow.changes[1].Type)
	assert.Equal(t, ChangeTypeDeleted, uow.changes[2].Type)
}
