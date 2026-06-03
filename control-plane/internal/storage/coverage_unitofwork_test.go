package storage

import (
	"context"
	"database/sql"
	"errors"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

type testUnitOfWorkBackend struct {
	workflowCalls  int
	executionCalls int
	sessionCalls   int
	workflowErr    error
	executionErr   error
	sessionErr     error
}

func (b *testUnitOfWorkBackend) executeWorkflowInsertWithTx(_ context.Context, _ DBTX, _ *types.Workflow) error {
	b.workflowCalls++
	return b.workflowErr
}

func (b *testUnitOfWorkBackend) executeWorkflowExecutionInsertWithTx(_ context.Context, _ DBTX, _ *types.WorkflowExecution) error {
	b.executionCalls++
	return b.executionErr
}

func (b *testUnitOfWorkBackend) executeSessionInsertWithTx(_ context.Context, _ DBTX, _ *types.Session) error {
	b.sessionCalls++
	return b.sessionErr
}

func newSQLiteTestDB(t *testing.T) *sqlDatabase {
	t.Helper()

	db, err := sql.Open("sqlite3", ":memory:")
	require.NoError(t, err)
	t.Cleanup(func() {
		_ = db.Close()
	})

	_, err = db.Exec(`CREATE TABLE workflows (workflow_id TEXT PRIMARY KEY, status TEXT, completed_at TEXT, updated_at TEXT)`)
	require.NoError(t, err)

	return newSQLDatabase(db, "local")
}

func TestUnitOfWorkLifecycle(t *testing.T) {
	t.Run("registers changes and commits successfully", func(t *testing.T) {
		db := newSQLiteTestDB(t)
		uow := NewUnitOfWork(db, &testUnitOfWorkBackend{}).(*unitOfWorkImpl)

		executed := 0
		uow.RegisterNew("new", "items", func(DBTX) error { executed++; return nil })
		uow.RegisterDirty("dirty", "items", func(DBTX) error { executed++; return nil })
		uow.RegisterDeleted("gone", "items", func(DBTX) error { executed++; return nil })

		require.True(t, uow.HasChanges())
		require.Equal(t, 3, uow.GetChangeCount())
		require.True(t, uow.IsActive())

		require.NoError(t, uow.Commit())
		require.Equal(t, 3, executed)
		require.False(t, uow.HasChanges())
		require.Equal(t, 0, uow.GetChangeCount())
		require.False(t, uow.IsActive())
		require.Nil(t, uow.tx)

		uow.RegisterNew("ignored", "items", func(DBTX) error { return nil })
		require.Equal(t, 0, uow.GetChangeCount())
		require.EqualError(t, uow.Commit(), "unit of work is not active")
	})

	t.Run("commit with no changes deactivates unit of work", func(t *testing.T) {
		db := newSQLiteTestDB(t)
		uow := NewUnitOfWork(db, &testUnitOfWorkBackend{}).(*unitOfWorkImpl)
		require.NoError(t, uow.Commit())
		require.False(t, uow.IsActive())
	})

	t.Run("commit surfaces operation and begin failures", func(t *testing.T) {
		db := newSQLiteTestDB(t)
		uow := NewUnitOfWork(db, &testUnitOfWorkBackend{}).(*unitOfWorkImpl)
		uow.RegisterNew("bad", "items", func(DBTX) error { return errors.New("boom") })
		require.EqualError(t, uow.Commit(), "failed to execute change 0 for table items: boom")

		nilDBUOW := NewUnitOfWork(nil, &testUnitOfWorkBackend{}).(*unitOfWorkImpl)
		nilDBUOW.RegisterNew("bad", "items", func(DBTX) error { return nil })
		require.EqualError(t, nilDBUOW.Commit(), "failed to begin transaction: sql database is not initialized")
	})

	t.Run("rollback clears state", func(t *testing.T) {
		db := newSQLiteTestDB(t)
		tx, err := db.Begin()
		require.NoError(t, err)

		uow := NewUnitOfWork(db, &testUnitOfWorkBackend{}).(*unitOfWorkImpl)
		uow.tx = tx
		uow.RegisterNew("new", "items", func(DBTX) error { return nil })

		require.NoError(t, uow.Rollback())
		require.False(t, uow.IsActive())
		require.False(t, uow.HasChanges())
		require.Nil(t, uow.tx)
	})

	t.Run("rollback returns transaction error", func(t *testing.T) {
		db := newSQLiteTestDB(t)
		tx, err := db.Begin()
		require.NoError(t, err)
		require.NoError(t, tx.Rollback())

		uow := NewUnitOfWork(db, &testUnitOfWorkBackend{}).(*unitOfWorkImpl)
		uow.tx = tx
		require.EqualError(t, uow.Rollback(), "failed to rollback transaction: sql: transaction has already been committed or rolled back")
	})

	t.Run("identifies retryable errors", func(t *testing.T) {
		uow := NewUnitOfWork(newSQLiteTestDB(t), &testUnitOfWorkBackend{}).(*unitOfWorkImpl)
		require.False(t, uow.isRetryableError(nil))
		require.True(t, uow.isRetryableError(errors.New("database is locked")))
		require.True(t, uow.isRetryableError(errors.New("SQLITE_BUSY: database table is locked")))
		require.False(t, uow.isRetryableError(errors.New("permission denied")))
	})
}

func TestWorkflowUnitOfWorkOperations(t *testing.T) {
	ctx := context.Background()
	workflow := &types.Workflow{WorkflowID: "wf-1"}
	execution := &types.WorkflowExecution{ExecutionID: "exec-1", WorkflowID: "wf-1"}
	session := &types.Session{SessionID: "session-1"}

	t.Run("registers workflow and execution changes", func(t *testing.T) {
		backend := &testUnitOfWorkBackend{}
		wuow := NewWorkflowUnitOfWork(newSQLiteTestDB(t), backend).(*workflowUnitOfWorkImpl)

		require.NoError(t, wuow.StoreWorkflowWithExecution(ctx, workflow, execution))
		require.Equal(t, 2, wuow.GetChangeCount())
		require.NoError(t, wuow.Commit())
		require.Equal(t, 1, backend.workflowCalls)
		require.Equal(t, 1, backend.executionCalls)
	})

	t.Run("registers status update with optional execution", func(t *testing.T) {
		backend := &testUnitOfWorkBackend{}
		wuow := NewWorkflowUnitOfWork(newSQLiteTestDB(t), backend).(*workflowUnitOfWorkImpl)

		require.NoError(t, wuow.UpdateWorkflowStatus(ctx, "wf-1", "running", execution))
		require.Equal(t, 2, wuow.GetChangeCount())
		require.NoError(t, wuow.Commit())
		require.Equal(t, 1, backend.executionCalls)
	})

	t.Run("completes workflow with results and stores session", func(t *testing.T) {
		backend := &testUnitOfWorkBackend{}
		wuow := NewWorkflowUnitOfWork(newSQLiteTestDB(t), backend).(*workflowUnitOfWorkImpl)

		require.NoError(t, wuow.CompleteWorkflowWithResults("wf-1", map[string]interface{}{"ok": true}))
		require.Equal(t, 2, wuow.GetChangeCount())
		require.NoError(t, wuow.Commit())

		wuow = NewWorkflowUnitOfWork(newSQLiteTestDB(t), backend).(*workflowUnitOfWorkImpl)
		require.NoError(t, wuow.StoreWorkflowWithSession(ctx, workflow, session))
		require.NoError(t, wuow.Commit())
		require.Equal(t, 1, backend.workflowCalls)
		require.Equal(t, 1, backend.sessionCalls)
	})

	t.Run("inactive workflow unit of work rejects operations", func(t *testing.T) {
		wuow := NewWorkflowUnitOfWork(newSQLiteTestDB(t), &testUnitOfWorkBackend{}).(*workflowUnitOfWorkImpl)
		require.NoError(t, wuow.Commit())

		require.EqualError(t, wuow.StoreWorkflowWithExecution(ctx, workflow, execution), "workflow unit of work is not active")
		require.EqualError(t, wuow.UpdateWorkflowStatus(ctx, "wf-1", "running", execution), "workflow unit of work is not active")
		require.EqualError(t, wuow.CompleteWorkflowWithResults("wf-1", nil), "workflow unit of work is not active")
		require.EqualError(t, wuow.StoreWorkflowWithSession(ctx, workflow, session), "workflow unit of work is not active")
	})
}
