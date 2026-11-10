package storage

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/jackc/pgx/v5/pgconn"
	sqlite3 "github.com/mattn/go-sqlite3"
	"github.com/stretchr/testify/require"
)

// newUnmigratedLocalStorage constructs a SQLite-backed LocalStorage that skips
// Initialize and autoMigrateSchema so tests can exercise the lazy-creation guard.
func newUnmigratedLocalStorage(t *testing.T) (*LocalStorage, context.Context) {
	t.Helper()
	ctx := context.Background()
	dsn := fmt.Sprintf("%s?_journal_mode=WAL&_synchronous=NORMAL&_busy_timeout=60000&_txlock=immediate",
		filepath.Join(t.TempDir(), "unmigrated.db"))
	rawDB, err := sql.Open("sqlite3", dsn)
	require.NoError(t, err)
	t.Cleanup(func() { _ = rawDB.Close() })
	return &LocalStorage{
		db:   newSQLDatabase(rawDB, "local"),
		mode: "local",
	}, ctx
}

type sqlHookFunc func(inner driver.Conn, query string) error

type interceptingDriver struct{}

func (interceptingDriver) Open(string) (driver.Conn, error) {
	return nil, errors.New("use sql.OpenDB with interceptingConnector")
}

type interceptingConnector struct {
	dsn     string
	mu      sync.Mutex
	hook    sqlHookFunc
	queries []string
}

func newInterceptingSQLiteDB(t *testing.T, mode string) (*sql.DB, *interceptingConnector, *LocalStorage) {
	t.Helper()
	dsn := fmt.Sprintf("%s?_journal_mode=WAL&_synchronous=NORMAL&_busy_timeout=60000&_txlock=immediate",
		filepath.Join(t.TempDir(), "intercepted.db"))
	conn := &interceptingConnector{dsn: dsn}
	rawDB := sql.OpenDB(conn)
	rawDB.SetMaxOpenConns(1)
	t.Cleanup(func() { _ = rawDB.Close() })
	ls := &LocalStorage{
		db:   newSQLDatabase(rawDB, mode),
		mode: mode,
	}
	return rawDB, conn, ls
}

func (c *interceptingConnector) Driver() driver.Driver {
	return interceptingDriver{}
}

func (c *interceptingConnector) Connect(context.Context) (driver.Conn, error) {
	rawConn, err := (&sqlite3.SQLiteDriver{}).Open(c.dsn)
	if err != nil {
		return nil, err
	}
	return &interceptingConn{inner: rawConn, parent: c}, nil
}

func (c *interceptingConnector) setHook(hook sqlHookFunc) {
	c.mu.Lock()
	c.hook = hook
	c.mu.Unlock()
}

func (c *interceptingConnector) resetQueries() {
	c.mu.Lock()
	c.queries = nil
	c.mu.Unlock()
}

func (c *interceptingConnector) recordedQueries() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]string, len(c.queries))
	copy(out, c.queries)
	return out
}

func (c *interceptingConnector) intercept(inner driver.Conn, query string) error {
	c.mu.Lock()
	c.queries = append(c.queries, query)
	hook := c.hook
	c.mu.Unlock()
	if hook != nil {
		return hook(inner, query)
	}
	return nil
}

type interceptingConn struct {
	inner  driver.Conn
	parent *interceptingConnector
}

func (c *interceptingConn) Prepare(query string) (driver.Stmt, error) {
	if err := c.parent.intercept(c.inner, query); err != nil {
		return nil, err
	}
	return c.inner.Prepare(query)
}

func (c *interceptingConn) PrepareContext(ctx context.Context, query string) (driver.Stmt, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := c.parent.intercept(c.inner, query); err != nil {
		return nil, err
	}
	if pc, ok := c.inner.(driver.ConnPrepareContext); ok {
		return pc.PrepareContext(ctx, query)
	}
	return c.inner.Prepare(query)
}

func (c *interceptingConn) Close() error {
	return c.inner.Close()
}

func (c *interceptingConn) Begin() (driver.Tx, error) {
	return c.inner.Begin()
}

func (c *interceptingConn) BeginTx(ctx context.Context, opts driver.TxOptions) (driver.Tx, error) {
	if bt, ok := c.inner.(driver.ConnBeginTx); ok {
		return bt.BeginTx(ctx, opts)
	}
	return c.inner.Begin()
}

func (c *interceptingConn) CheckNamedValue(nv *driver.NamedValue) error {
	if nvc, ok := c.inner.(driver.NamedValueChecker); ok {
		return nvc.CheckNamedValue(nv)
	}
	return driver.ErrSkip
}

func countCreateTableQueries(queries []string) int {
	n := 0
	for _, q := range queries {
		if strings.Contains(strings.ToUpper(q), "CREATE TABLE") {
			n++
		}
	}
	return n
}

// Contract R1: A fresh store built the normal way (with autoMigrateSchema) has
// execution_relay, run_owners, and execution_approval_questions before any
// relay, owner, or approval call.
func TestRelayTable_R1_StartupMigrationCreatesAllThreeTables(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	for _, table := range []string{
		"execution_relay",
		"run_owners",
		"execution_approval_questions",
	} {
		var count int
		err := ls.db.QueryRowContext(ctx,
			`SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?`, table,
		).Scan(&count)
		require.NoError(t, err)
		require.Equal(t, 1, count, "table %s must exist right after startup schema migration", table)
	}

	require.Equal(t, map[string]bool{
		"execution_id": true,
		"seq":          true,
		"kind":         true,
		"data":         true,
		"recorded_at":  true,
	}, tableColumns(t, ls.db.DB, "execution_relay"))
	require.Equal(t, []string{"execution_id", "seq"}, primaryKeyColumns(t, ls.db.DB, "execution_relay"))

	require.Equal(t, map[string]bool{
		"run_id":        true,
		"owner_node_id": true,
		"created_at":    true,
	}, tableColumns(t, ls.db.DB, "run_owners"))
	require.Equal(t, []string{"run_id"}, primaryKeyColumns(t, ls.db.DB, "run_owners"))

	require.Equal(t, map[string]bool{
		"execution_id":        true,
		"approval_request_id": true,
		"kind":                true,
		"question":            true,
		"body":                true,
		"options":             true,
		"recorded_at":         true,
	}, tableColumns(t, ls.db.DB, "execution_approval_questions"))
	require.Equal(t, []string{"execution_id", "approval_request_id"}, primaryKeyColumns(t, ls.db.DB, "execution_approval_questions"))

	// Re-running autoMigrateSchema against an already-populated database keeps
	// existing rows untouched.
	stored, err := ls.RelayPut(ctx, "exec-r1", []RelayEntry{{Seq: 1, Kind: "assistant"}})
	require.NoError(t, err)
	require.Equal(t, 1, stored)
	require.NoError(t, ls.SetRunOwner(ctx, "run-r1", "workaf"))
	require.NoError(t, ls.PutApprovalQuestion(ctx, types.ApprovalQuestion{
		ExecutionID:       "exec-r1",
		ApprovalRequestID: "apr-r1",
		Kind:              types.ApprovalKindApproval,
		Question:          "Ship it?",
	}))

	require.NoError(t, ls.autoMigrateSchema(ctx))

	held, err := ls.RelayCount(ctx, "exec-r1")
	require.NoError(t, err)
	require.Equal(t, int64(1), held)
	owner, err := ls.RunOwner(ctx, "run-r1")
	require.NoError(t, err)
	require.Equal(t, "workaf", owner)
	gotQ, err := ls.GetApprovalQuestion(ctx, "exec-r1", "apr-r1")
	require.NoError(t, err)
	require.NotNil(t, gotQ)
	require.Equal(t, "Ship it?", gotQ.Question)
}

// Contract R2: A failed creation is not cached; when the first CREATE TABLE
// fails through a wrapped database seam and the fault clears, the next call on
// the same store retries creation and succeeds.
func TestRelayTable_R2_FailedCreateIsNeverCached(t *testing.T) {
	ctx := context.Background()

	t.Run("transient DDL failure is retried and succeeds on next call", func(t *testing.T) {
		_, conn, ls := newInterceptingSQLiteDB(t, "local")

		injected := errors.New("simulated transient DDL error")
		conn.setHook(func(_ driver.Conn, query string) error {
			if strings.Contains(strings.ToUpper(query), "CREATE TABLE") {
				return injected
			}
			return nil
		})

		_, err := ls.RelayPut(ctx, "exec-r2", []RelayEntry{{Seq: 1, Kind: "assistant", Data: json.RawMessage(`{"ok":true}`)}})
		require.ErrorIs(t, err, injected)

		err = ls.SetRunOwner(ctx, "run-r2", "workaf")
		require.ErrorIs(t, err, injected)

		_, err = ls.RunOwner(ctx, "run-r2")
		require.ErrorIs(t, err, injected)

		err = ls.PutApprovalQuestion(ctx, types.ApprovalQuestion{
			ExecutionID:       "exec-r2",
			ApprovalRequestID: "apr-r2",
			Kind:              types.ApprovalKindPlanReview,
			Question:          "Approve plan?",
		})
		require.ErrorIs(t, err, injected)

		// Clear the fault on the same LocalStorage and *sqlDatabase handle.
		// Before the fix, relayTableOnce and approvalQuestionTableOnce returned
		// the cached error forever.
		conn.setHook(nil)

		stored, err := ls.RelayPut(ctx, "exec-r2", []RelayEntry{{Seq: 1, Kind: "assistant", Data: json.RawMessage(`{"ok":true}`)}})
		require.NoError(t, err)
		require.Equal(t, 1, stored)

		entries, err := ls.RelayAfter(ctx, "exec-r2", 0, 10)
		require.NoError(t, err)
		require.Len(t, entries, 1)
		require.JSONEq(t, `{"ok":true}`, string(entries[0].Data))

		require.NoError(t, ls.SetRunOwner(ctx, "run-r2", "workaf"))
		owner, err := ls.RunOwner(ctx, "run-r2")
		require.NoError(t, err)
		require.Equal(t, "workaf", owner)

		require.NoError(t, ls.PutApprovalQuestion(ctx, types.ApprovalQuestion{
			ExecutionID:       "exec-r2",
			ApprovalRequestID: "apr-r2",
			Kind:              types.ApprovalKindPlanReview,
			Question:          "Approve plan?",
		}))
		gotQ, err := ls.GetApprovalQuestion(ctx, "exec-r2", "apr-r2")
		require.NoError(t, err)
		require.NotNil(t, gotQ)
		require.Equal(t, "Approve plan?", gotQ.Question)
	})

	t.Run("postgres 23505 and 42P07 treat confirmed peer creation as success and retry if absent", func(t *testing.T) {
		_, conn, ls := newInterceptingSQLiteDB(t, "postgres")

		// First attempt returns SQLSTATE 23505 while the table still does not
		// exist (peer transaction aborted): must fail and must not cache the error.
		conn.setHook(func(_ driver.Conn, query string) error {
			if strings.Contains(strings.ToUpper(query), "CREATE TABLE") {
				return &pgconn.PgError{Code: "23505", Message: `duplicate key value violates unique constraint "pg_type_typname_nsp_index"`}
			}
			return nil
		})
		_, err := ls.RelayCount(ctx, "")
		require.Error(t, err)

		// Next attempt simulates another Postgres process finishing CREATE TABLE
		// at the same moment and returning 23505 (execution_relay) or 42P07
		// (run_owners and execution_approval_questions).
		conn.setHook(func(inner driver.Conn, query string) error {
			if !strings.Contains(strings.ToUpper(query), "CREATE TABLE") {
				return nil
			}
			stmt, prepErr := inner.Prepare(query)
			if prepErr != nil {
				return prepErr
			}
			_, execErr := stmt.Exec(nil)
			_ = stmt.Close()
			if execErr != nil {
				return execErr
			}
			if strings.Contains(query, "execution_relay") {
				return &pgconn.PgError{Code: "23505", Message: `duplicate key value violates unique constraint "pg_type_typname_nsp_index"`}
			}
			return &pgconn.PgError{Code: "42P07", Message: "relation already exists"}
		})

		n, err := ls.RelayCount(ctx, "")
		require.NoError(t, err)
		require.Equal(t, int64(0), n)

		require.NoError(t, ls.ensureRelayAndApprovalTables(ctx))

		// Subsequent calls hit the positive per-db cache without re-entering DDL.
		conn.resetQueries()
		n, err = ls.RelayCount(ctx, "")
		require.NoError(t, err)
		require.Equal(t, int64(0), n)
		require.Equal(t, 0, countCreateTableQueries(conn.recordedQueries()))
	})
}

// Contract R3: Steady state runs no DDL: after the first success, N calls to
// RunOwner, SetRunOwner, and RelayPut execute no CREATE TABLE statements.
func TestRelayTable_R3_SteadyStateRunsNoDDL(t *testing.T) {
	ctx := context.Background()
	const iterations = 25

	t.Run("after lazy first use on unmigrated store", func(t *testing.T) {
		_, conn, ls := newInterceptingSQLiteDB(t, "local")

		require.NoError(t, ls.SetRunOwner(ctx, "run-warmup", "workaf"))
		owner, err := ls.RunOwner(ctx, "run-warmup")
		require.NoError(t, err)
		require.Equal(t, "workaf", owner)
		stored, err := ls.RelayPut(ctx, "exec-warmup", []RelayEntry{{Seq: 1, Kind: "init"}})
		require.NoError(t, err)
		require.Equal(t, 1, stored)
		require.NoError(t, ls.PutApprovalQuestion(ctx, types.ApprovalQuestion{
			ExecutionID:       "exec-warmup",
			ApprovalRequestID: "apr-warmup",
			Kind:              types.ApprovalKindApproval,
		}))

		// Confirm the wrapper observed the initial lazy CREATE TABLE statements.
		require.Equal(t, 3, countCreateTableQueries(conn.recordedQueries()))
		conn.resetQueries()

		for i := 0; i < iterations; i++ {
			runID := fmt.Sprintf("run-%d", i)
			execID := fmt.Sprintf("exec-%d", i)

			owner, err := ls.RunOwner(ctx, runID)
			require.NoError(t, err)
			require.Equal(t, "", owner)

			require.NoError(t, ls.SetRunOwner(ctx, runID, "workaf"))

			owner, err = ls.RunOwner(ctx, runID)
			require.NoError(t, err)
			require.Equal(t, "workaf", owner)

			n, err := ls.RelayPut(ctx, execID, []RelayEntry{{Seq: 1, Kind: "line"}})
			require.NoError(t, err)
			require.Equal(t, 1, n)

			gotQ, err := ls.GetApprovalQuestion(ctx, execID, "apr-none")
			require.NoError(t, err)
			require.Nil(t, gotQ)
		}

		queries := conn.recordedQueries()
		require.GreaterOrEqual(t, len(queries), iterations*5)
		require.Equal(t, 0, countCreateTableQueries(queries), "steady-state operations must execute zero CREATE TABLE statements")
	})

	t.Run("immediately after autoMigrateSchema on normal store", func(t *testing.T) {
		_, conn, ls := newInterceptingSQLiteDB(t, "local")
		require.NoError(t, ls.autoMigrateSchema(ctx))
		conn.resetQueries()

		for i := 0; i < iterations; i++ {
			runID := fmt.Sprintf("migrated-run-%d", i)
			execID := fmt.Sprintf("migrated-exec-%d", i)

			require.NoError(t, ls.SetRunOwner(ctx, runID, "workaf"))
			owner, err := ls.RunOwner(ctx, runID)
			require.NoError(t, err)
			require.Equal(t, "workaf", owner)

			n, err := ls.RelayPut(ctx, execID, []RelayEntry{{Seq: 1, Kind: "line"}})
			require.NoError(t, err)
			require.Equal(t, 1, n)
		}

		queries := conn.recordedQueries()
		require.GreaterOrEqual(t, len(queries), iterations*3)
		require.Equal(t, 0, countCreateTableQueries(queries), "post-startup calls must execute zero CREATE TABLE statements")
	})
}

// Contract R4: 32 goroutines concurrently calling SetRunOwner and RelayPut on
// a fresh store that skipped startup creation all succeed under -race, and
// each run has exactly one owner row.
func TestRelayTable_R4_ConcurrentFirstUseSerializesCreation(t *testing.T) {
	ls, ctx := newUnmigratedLocalStorage(t)
	ls.db.SetMaxOpenConns(8)

	for _, table := range []string{"execution_relay", "run_owners"} {
		var count int
		require.NoError(t, ls.db.QueryRowContext(ctx,
			`SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?`, table,
		).Scan(&count))
		require.Equal(t, 0, count, "table %s must not exist before concurrent first use", table)
	}

	const workers = 32
	var (
		ready sync.WaitGroup
		done  sync.WaitGroup
		start = make(chan struct{})
		errs  = make([]error, workers)
	)
	ready.Add(workers)
	done.Add(workers)

	for i := 0; i < workers; i++ {
		go func(idx int) {
			defer done.Done()
			runID := fmt.Sprintf("run-%d", idx)
			owner := fmt.Sprintf("owner-%d", idx)
			execID := fmt.Sprintf("exec-%d", idx)
			entries := []RelayEntry{
				{Seq: 1, Kind: "assistant", Data: json.RawMessage(fmt.Sprintf(`{"worker":%d}`, idx))},
			}

			ready.Done()
			<-start

			if err := ls.SetRunOwner(ctx, runID, owner); err != nil {
				errs[idx] = fmt.Errorf("set own run %d: %w", idx, err)
				return
			}
			if err := ls.SetRunOwner(ctx, "run-contested", owner); err != nil {
				errs[idx] = fmt.Errorf("set contested run %d: %w", idx, err)
				return
			}
			stored, err := ls.RelayPut(ctx, execID, entries)
			if err != nil {
				errs[idx] = fmt.Errorf("relay put %d: %w", idx, err)
				return
			}
			if stored != 1 {
				errs[idx] = fmt.Errorf("relay put %d stored %d, want 1", idx, stored)
			}
		}(i)
	}

	ready.Wait()
	close(start)
	done.Wait()

	for i, err := range errs {
		require.NoError(t, err, "worker %d failed", i)
	}

	for i := 0; i < workers; i++ {
		runID := fmt.Sprintf("run-%d", i)
		var rowCount int
		require.NoError(t, ls.db.QueryRowContext(ctx,
			`SELECT COUNT(*) FROM run_owners WHERE run_id = ?`, runID,
		).Scan(&rowCount))
		require.Equal(t, 1, rowCount, "run %s must have exactly one owner row", runID)

		owner, err := ls.RunOwner(ctx, runID)
		require.NoError(t, err)
		require.Equal(t, fmt.Sprintf("owner-%d", i), owner)
	}

	var contestedCount int
	require.NoError(t, ls.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM run_owners WHERE run_id = ?`, "run-contested",
	).Scan(&contestedCount))
	require.Equal(t, 1, contestedCount, "contested run must have exactly one owner row")
	contestedOwner, err := ls.RunOwner(ctx, "run-contested")
	require.NoError(t, err)
	require.NotEmpty(t, contestedOwner)

	held, err := ls.RelayCount(ctx, "")
	require.NoError(t, err)
	require.Equal(t, int64(workers), held)
}

type stubSQLStateError struct {
	code string
	msg  string
}

func (e stubSQLStateError) Error() string    { return e.msg }
func (e stubSQLStateError) SQLState() string { return e.code }

// Branch coverage (migrations.go:292-294): ensureSchemaTable nil-db guard on a
// nil LocalStorage, an uninitialized LocalStorage (ls.db == nil), an empty
// sqlDatabase wrapper (ls.db.DB == nil), and a closed postgres-mode store.
func TestRelayTable_Branch_EnsureSchemaTableUninitializedAndClosedStore(t *testing.T) {
	ctx := context.Background()

	var nilLS *LocalStorage
	_, err := nilLS.RelayPut(ctx, "exec-nil", []RelayEntry{{Seq: 1, Kind: "assistant"}})
	require.EqualError(t, err, "create execution_relay: sql database is not initialized")

	uninitLS := NewLocalStorage(LocalStorageConfig{})
	_, err = uninitLS.RunOwner(ctx, "run-uninit")
	require.EqualError(t, err, "create run_owners: sql database is not initialized")
	_, err = uninitLS.GetApprovalQuestion(ctx, "exec-uninit", "apr-uninit")
	require.EqualError(t, err, "create execution_approval_questions: sql database is not initialized")

	nilInnerDBLS := &LocalStorage{
		db:   &sqlDatabase{mode: "local"},
		mode: "local",
	}
	_, err = nilInnerDBLS.RelayCount(ctx, "")
	require.EqualError(t, err, "create execution_relay: sql database is not initialized")

	rawDB, _, closedPgLS := newInterceptingSQLiteDB(t, "postgres")
	require.NoError(t, rawDB.Close())
	_, err = closedPgLS.RelayCount(ctx, "")
	require.ErrorContains(t, err, "create execution_relay")
}

// Branch coverage (migrations.go:268-274): ensureRelayAndApprovalTables error
// returns when the first (execution_relay), second (run_owners), or third
// (execution_approval_questions) CREATE TABLE fails, followed by clean retry.
func TestRelayTable_Branch_EnsureRelayAndApprovalTablesSecondAndThirdCreateErrors(t *testing.T) {
	ctx := context.Background()

	t.Run("first CREATE fails on execution_relay", func(t *testing.T) {
		_, conn, ls := newInterceptingSQLiteDB(t, "local")
		injected := errors.New("relay DDL failed")
		var creates int
		conn.setHook(func(_ driver.Conn, query string) error {
			if strings.Contains(strings.ToUpper(query), "CREATE TABLE") {
				creates++
				if creates == 1 {
					return injected
				}
			}
			return nil
		})

		err := ls.ensureRelayAndApprovalTables(ctx)
		require.ErrorIs(t, err, injected)
		require.ErrorContains(t, err, "create execution_relay")
		require.Equal(t, 1, creates)
	})

	t.Run("second CREATE fails on run_owners then retry completes remaining tables", func(t *testing.T) {
		_, conn, ls := newInterceptingSQLiteDB(t, "local")
		injected := errors.New("run_owners DDL failed")
		var creates int
		conn.setHook(func(_ driver.Conn, query string) error {
			if strings.Contains(strings.ToUpper(query), "CREATE TABLE") {
				creates++
				if creates == 2 {
					return injected
				}
			}
			return nil
		})

		err := ls.ensureRelayAndApprovalTables(ctx)
		require.ErrorIs(t, err, injected)
		require.ErrorContains(t, err, "create run_owners")
		require.Equal(t, 2, creates)

		conn.setHook(nil)
		require.NoError(t, ls.ensureRelayAndApprovalTables(ctx))
	})

	t.Run("third CREATE fails on execution_approval_questions then retry completes", func(t *testing.T) {
		_, conn, ls := newInterceptingSQLiteDB(t, "local")
		injected := errors.New("approval_questions DDL failed")
		var creates int
		conn.setHook(func(_ driver.Conn, query string) error {
			if strings.Contains(strings.ToUpper(query), "CREATE TABLE") {
				creates++
				if creates == 3 {
					return injected
				}
			}
			return nil
		})

		err := ls.ensureRelayAndApprovalTables(ctx)
		require.ErrorIs(t, err, injected)
		require.ErrorContains(t, err, "create execution_approval_questions")
		require.Equal(t, 3, creates)

		conn.setHook(nil)
		require.NoError(t, ls.ensureRelayAndApprovalTables(ctx))
	})
}

// Branch coverage (migrations.go:254-256): autoMigrateSchema wraps errors from
// ensureRelayAndApprovalTables as "failed to ensure relay and approval tables".
func TestRelayTable_Branch_AutoMigrateSchemaWrapsEnsureTablesError(t *testing.T) {
	ctx := context.Background()

	for _, failTable := range []string{
		"execution_relay",
		"run_owners",
		"execution_approval_questions",
	} {
		t.Run(failTable, func(t *testing.T) {
			_, conn, ls := newInterceptingSQLiteDB(t, "local")
			injected := fmt.Errorf("injected %s failure", failTable)
			conn.setHook(func(_ driver.Conn, query string) error {
				if strings.Contains(strings.ToUpper(query), "CREATE TABLE") && strings.Contains(query, failTable) {
					return injected
				}
				return nil
			})

			err := ls.autoMigrateSchema(ctx)
			require.ErrorIs(t, err, injected)
			require.ErrorContains(t, err, "failed to ensure relay and approval tables")
			require.ErrorContains(t, err, "create "+failTable)

			conn.setHook(nil)
			require.NoError(t, ls.autoMigrateSchema(ctx))
		})
	}
}

// Branch coverage (migrations.go:318-336): isPostgresConcurrentTableCreateError
// nil-error branch, plain non-SQLState error fallback, and SQLState() interface
// branch (both matching 23505/42P07 and non-matching codes) via postgres mode.
func TestRelayTable_Branch_PostgresConcurrentCreateErrorNilPlainAndSQLState(t *testing.T) {
	ctx := context.Background()

	// Nil error branch (migrations.go:323-325).
	require.False(t, isPostgresConcurrentTableCreateError(nil))

	t.Run("plain non-SQLState error on postgres store returns error without caching", func(t *testing.T) {
		_, conn, ls := newInterceptingSQLiteDB(t, "postgres")
		// Leave ls.mode blank to also exercise the ls.db.Mode() == "postgres"
		// fallback in isPostgresMode (migrations.go:319).
		ls.mode = ""

		plainErr := errors.New("plain network reset without sqlstate")
		conn.setHook(func(_ driver.Conn, query string) error {
			if strings.Contains(strings.ToUpper(query), "CREATE TABLE") {
				return plainErr
			}
			return nil
		})

		_, err := ls.RelayPut(ctx, "exec-plain", []RelayEntry{{Seq: 1, Kind: "assistant"}})
		require.ErrorIs(t, err, plainErr)
		require.ErrorContains(t, err, "create execution_relay")

		conn.setHook(nil)
		stored, err := ls.RelayPut(ctx, "exec-plain", []RelayEntry{{Seq: 1, Kind: "assistant"}})
		require.NoError(t, err)
		require.Equal(t, 1, stored)
	})

	t.Run("non-PgError SQLState implementation covers non-matching and 23505/42P07 codes", func(t *testing.T) {
		_, conn, ls := newInterceptingSQLiteDB(t, "postgres")

		syntaxErr := stubSQLStateError{code: "42601", msg: "simulated syntax error"}
		conn.setHook(func(_ driver.Conn, query string) error {
			if strings.Contains(strings.ToUpper(query), "CREATE TABLE") {
				return syntaxErr
			}
			return nil
		})

		err := ls.SetRunOwner(ctx, "run-sqlstate", "workaf")
		require.ErrorIs(t, err, syntaxErr)
		require.ErrorContains(t, err, "create run_owners")

		// Peer creates the table concurrently while this caller receives a
		// non-PgError SQLState() carrier with 23505 or 42P07.
		conn.setHook(func(inner driver.Conn, query string) error {
			if !strings.Contains(strings.ToUpper(query), "CREATE TABLE") {
				return nil
			}
			stmt, prepErr := inner.Prepare(query)
			if prepErr != nil {
				return prepErr
			}
			_, execErr := stmt.Exec(nil)
			_ = stmt.Close()
			if execErr != nil {
				return execErr
			}
			if strings.Contains(query, "run_owners") {
				return stubSQLStateError{code: "23505", msg: "concurrent unique_violation"}
			}
			return stubSQLStateError{code: "42P07", msg: "concurrent duplicate_table"}
		})

		require.NoError(t, ls.SetRunOwner(ctx, "run-sqlstate", "workaf"))
		owner, err := ls.RunOwner(ctx, "run-sqlstate")
		require.NoError(t, err)
		require.Equal(t, "workaf", owner)

		require.NoError(t, ls.PutApprovalQuestion(ctx, types.ApprovalQuestion{
			ExecutionID:       "exec-sqlstate",
			ApprovalRequestID: "apr-sqlstate",
			Kind:              types.ApprovalKindApproval,
			Question:          "Continue?",
		}))
		gotQ, err := ls.GetApprovalQuestion(ctx, "exec-sqlstate", "apr-sqlstate")
		require.NoError(t, err)
		require.NotNil(t, gotQ)
		require.Equal(t, "Continue?", gotQ.Question)
	})
}

// Branch coverage (migrations.go:338-345): tableExistsAfterRace returns false
// when the probe query errors because the table is still absent after a 23505
// or 42P07 error, leaving the failure uncached so a later call succeeds.
func TestRelayTable_Branch_TableExistsAfterRaceFalseWhenTableAbsent(t *testing.T) {
	ctx := context.Background()
	_, conn, ls := newInterceptingSQLiteDB(t, "postgres")

	require.False(t, ls.tableExistsAfterRace(ctx, runOwnersSchemaTable))

	conn.setHook(func(_ driver.Conn, query string) error {
		if strings.Contains(strings.ToUpper(query), "CREATE TABLE") {
			return &pgconn.PgError{Code: "42P07", Message: "relation already exists"}
		}
		return nil
	})

	_, err := ls.RunOwner(ctx, "run-absent")
	require.ErrorContains(t, err, "create run_owners")
	require.False(t, ls.tableExistsAfterRace(ctx, runOwnersSchemaTable))

	conn.setHook(nil)
	require.NoError(t, ls.SetRunOwner(ctx, "run-absent", "workaf"))
	require.True(t, ls.tableExistsAfterRace(ctx, runOwnersSchemaTable))
	owner, err := ls.RunOwner(ctx, "run-absent")
	require.NoError(t, err)
	require.Equal(t, "workaf", owner)
}
