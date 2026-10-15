package storage

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"path/filepath"
	"sync"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/jmoiron/sqlx"
	"github.com/stretchr/testify/require"
)

type testRollbacker struct {
	err       error
	rollbacks int
}

type prepareCaptureState struct {
	mu      sync.Mutex
	queries []string
}

type prepareCaptureConnector struct {
	state *prepareCaptureState
}

type prepareCaptureDriver struct{}

type prepareCaptureConn struct {
	state *prepareCaptureState
}

type prepareCaptureTx struct{}

type prepareCaptureStmt struct{}

func (prepareCaptureConnector) Driver() driver.Driver {
	return prepareCaptureDriver{}
}

func (c prepareCaptureConnector) Connect(context.Context) (driver.Conn, error) {
	return &prepareCaptureConn{state: c.state}, nil
}

func (prepareCaptureDriver) Open(string) (driver.Conn, error) {
	return nil, errors.New("prepare capture driver requires a connector")
}

func (c *prepareCaptureConn) recordPrepare(query string) driver.Stmt {
	c.state.mu.Lock()
	c.state.queries = append(c.state.queries, query)
	c.state.mu.Unlock()
	return prepareCaptureStmt{}
}

func (c *prepareCaptureConn) Prepare(query string) (driver.Stmt, error) {
	return c.recordPrepare(query), nil
}

func (c *prepareCaptureConn) PrepareContext(_ context.Context, query string) (driver.Stmt, error) {
	return c.recordPrepare(query), nil
}

func (*prepareCaptureConn) Close() error { return nil }

func (*prepareCaptureConn) Begin() (driver.Tx, error) {
	return prepareCaptureTx{}, nil
}

func (prepareCaptureTx) Commit() error   { return nil }
func (prepareCaptureTx) Rollback() error { return nil }

func (prepareCaptureStmt) Close() error  { return nil }
func (prepareCaptureStmt) NumInput() int { return -1 }
func (prepareCaptureStmt) Exec([]driver.Value) (driver.Result, error) {
	return driver.RowsAffected(0), nil
}
func (prepareCaptureStmt) Query([]driver.Value) (driver.Rows, error) {
	return nil, errors.New("prepare capture query not supported")
}

func (s *prepareCaptureState) preparedQueries() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.queries...)
}

func (r *testRollbacker) Rollback() error {
	r.rollbacks++
	return r.err
}

func TestSQLTxPrepareRebindsByMode(t *testing.T) {
	tests := []struct {
		name string
		mode string
		want []string
	}{
		{
			name: "postgres",
			mode: "postgres",
			want: []string{"SELECT $1", "UPDATE items SET name = $1"},
		},
		{
			name: "sqlite",
			mode: "local",
			want: []string{"SELECT ?", "UPDATE items SET name = ?"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			state := &prepareCaptureState{}
			rawDB := sql.OpenDB(prepareCaptureConnector{state: state})
			t.Cleanup(func() { _ = rawDB.Close() })

			rawTx, err := rawDB.Begin()
			require.NoError(t, err)
			tx := newSQLTx(rawTx, test.mode)

			stmt, err := tx.PrepareContext(context.Background(), "SELECT ?")
			require.NoError(t, err)
			require.NoError(t, stmt.Close())

			stmt, err = tx.Prepare("UPDATE items SET name = ?")
			require.NoError(t, err)
			require.NoError(t, stmt.Close())
			require.NoError(t, tx.Rollback())

			require.Equal(t, test.want, state.preparedQueries())
		})
	}
}

func TestStorageHelperCoverage(t *testing.T) {
	t.Run("error types format messages", func(t *testing.T) {
		require.Equal(
			t,
			"duplicate registry DID detected: did:example:123 already exists",
			(&DuplicateDIDError{DID: "did:example:123", Type: "registry"}).Error(),
		)
		require.Equal(
			t,
			"foreign key constraint violation in workflows.workflow_id: referenced executions 'exec-1' does not exist (operation: insert)",
			(&ForeignKeyConstraintError{Table: "workflows", Column: "workflow_id", ReferencedTable: "executions", ReferencedValue: "exec-1", Operation: "insert"}).Error(),
		)
		require.Equal(
			t,
			"validation failed for workflow_id='wf-1': missing value (context: create)",
			(&ValidationError{Field: "workflow_id", Value: "wf-1", Reason: "missing value", Context: "create"}).Error(),
		)
		require.Equal(
			t,
			"invalid execution state transition for exec-1: cannot change from running to pending - transition not allowed",
			(&InvalidExecutionStateTransitionError{ExecutionID: "exec-1", CurrentState: "running", NewState: "pending", Reason: "transition not allowed"}).Error(),
		)
	})

	t.Run("validate execution state transitions", func(t *testing.T) {
		require.NoError(t, validateExecutionStateTransition("unknown", "pending"))
		require.NoError(t, validateExecutionStateTransition("RUNNING", "succeeded"))
		require.NoError(t, validateExecutionStateTransition("failed", "failed"))

		err := validateExecutionStateTransition("running", "bogus")
		var invalid *InvalidExecutionStateTransitionError
		require.ErrorAs(t, err, &invalid)
		require.Equal(t, "running", invalid.CurrentState)
		require.Equal(t, "unknown", invalid.NewState)
	})

	t.Run("sql database and transaction helpers", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		db := newSQLDatabase(rawDB, "postgres")
		require.Equal(t, "postgres", db.Mode())
		require.Equal(t, "SELECT * FROM t WHERE a = $1 AND b = $2", db.rebind("SELECT * FROM t WHERE a = ? AND b = ?"))
		require.Equal(t, "", (*sqlDatabase)(nil).Mode())
		require.Equal(t, "SELECT ?", (*sqlDatabase)(nil).rebind("SELECT ?"))

		nilDB := (*sqlDatabase)(nil)
		_, err = nilDB.Begin()
		require.EqualError(t, err, "sql database is not initialized")
		_, err = nilDB.BeginTx(context.Background(), nil)
		require.EqualError(t, err, "sql database is not initialized")

		tx, err := rawDB.Begin()
		require.NoError(t, err)
		wrappedTx := newSQLTx(tx, "postgres")
		require.Equal(t, "SELECT $1", wrappedTx.rebind("SELECT ?"))
		require.Equal(t, "SELECT ?", (*sqlTx)(nil).rebind("SELECT ?"))
		require.NoError(t, tx.Rollback())
	})

	t.Run("gorm helpers honor context and initialization state", func(t *testing.T) {
		ls := &LocalStorage{}
		require.EqualError(t, ls.initGormDB(), "sql connection is not initialized")

		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		ls = &LocalStorage{db: newSQLDatabase(rawDB, "local"), mode: "local"}
		require.NoError(t, ls.initGormDB())
		require.NotNil(t, ls.gormDB)
		require.NoError(t, ls.initGormDB())

		cancelled, cancel := context.WithCancel(context.Background())
		cancel()
		_, err = ls.gormWithContext(cancelled)
		require.EqualError(t, err, "context cancelled: context canceled")

		dbWithCtx, err := ls.gormWithContext(context.Background())
		require.NoError(t, err)
		require.NotNil(t, dbWithCtx)
	})

	t.Run("vector config and storage factory", func(t *testing.T) {
		enabled := false
		require.True(t, (VectorStoreConfig{}).isEnabled())
		require.False(t, (VectorStoreConfig{Enabled: &enabled}).isEnabled())
		require.Equal(t, "cosine", (VectorStoreConfig{}).normalized().Distance)
		require.Equal(t, "dot", (VectorStoreConfig{Distance: "dot"}).normalized().Distance)

		factory := &StorageFactory{}
		_, _, err := factory.CreateStorage(StorageConfig{Mode: "unsupported"})
		require.EqualError(t, err, "unsupported storage mode: unsupported (supported modes: local, postgres)")

		tempDir := t.TempDir()
		cfg := StorageConfig{
			Mode: "local",
			Local: LocalStorageConfig{
				DatabasePath: filepath.Join(tempDir, "agentfield.db"),
				KVStorePath:  filepath.Join(tempDir, "agentfield.bolt"),
			},
		}

		store, cache, err := factory.CreateStorage(cfg)
		require.NoError(t, err)
		require.NotNil(t, store)
		require.NotNil(t, cache)
		t.Cleanup(func() { _ = store.Close(context.Background()) })

		t.Setenv("AGENTFIELD_STORAGE_MODE", "bogus")
		_, _, err = factory.CreateStorage(cfg)
		require.EqualError(t, err, "unsupported storage mode: bogus (supported modes: local, postgres)")
	})

	t.Run("rollback helper handles nil done and failures", func(t *testing.T) {
		rollbackTx(nil, "noop")

		done := &testRollbacker{err: sql.ErrTxDone}
		rollbackTx(done, "done")
		require.Equal(t, 1, done.rollbacks)

		other := &testRollbacker{err: errors.New("boom")}
		rollbackTx(other, "boom")
		require.Equal(t, 1, other.rollbacks)
	})

	t.Run("require sql db and local wrappers", func(t *testing.T) {
		require.PanicsWithValue(t, "storage database is not initialized", func() {
			(&LocalStorage{}).requireSQLDB()
		})

		ls, ctx := setupLocalStorage(t)
		require.Same(t, ls.db, ls.requireSQLDB())
		require.NotNil(t, ls.NewUnitOfWork())
		require.NotNil(t, ls.NewWorkflowUnitOfWork())

		execution := &types.WorkflowExecution{
			WorkflowID:          "wf-uow",
			ExecutionID:         "exec-uow",
			AgentFieldRequestID: "req-uow",
			AgentNodeID:         "agent-uow",
			ReasonerID:          "reasoner-uow",
			Status:              "pending",
		}
		require.NoError(t, ls.StoreWorkflowExecutionWithUnitOfWork(ctx, execution))
		stored, err := ls.GetWorkflowExecution(ctx, execution.ExecutionID)
		require.NoError(t, err)
		require.NotNil(t, stored)
	})

	t.Run("sqlx rebind expectation stays in sync", func(t *testing.T) {
		require.Equal(t, sqlx.Rebind(sqlx.DOLLAR, "SELECT ?"), newSQLDatabase(&sql.DB{}, "postgres").rebind("SELECT ?"))
	})
}
