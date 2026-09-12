package storage

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/stdlib"
	"github.com/stretchr/testify/require"
)

// These tests exercise the stale reapers against a real PostgreSQL server.
// They are gated on POSTGRES_TEST_URL, the same variable the rest of the
// storage suite uses, and connect only to a loopback host. Each test creates
// its own throwaway database so rows never leak between tests.

// livePostgresConnConfig reads POSTGRES_TEST_URL and refuses anything that is
// not a loopback server, so a stray staging DSN can never be touched.
func livePostgresConnConfig(t *testing.T) *pgx.ConnConfig {
	t.Helper()

	dsn := strings.TrimSpace(os.Getenv("POSTGRES_TEST_URL"))
	if dsn == "" {
		t.Skip("POSTGRES_TEST_URL not set, skipping live postgres tests")
	}

	cfg, err := pgx.ParseConfig(dsn)
	require.NoError(t, err, "parse POSTGRES_TEST_URL")

	host := cfg.Host
	if host == "" {
		host = "localhost"
	}
	ip := net.ParseIP(host)
	require.Truef(t,
		strings.EqualFold(host, "localhost") || (ip != nil && ip.IsLoopback()),
		"live postgres tests refuse non-loopback host %q", host)
	return cfg
}

// livePostgresStorage spins up an isolated database on the live server and
// returns a Postgres-backed LocalStorage pointed at it.
func livePostgresStorage(t *testing.T) (*LocalStorage, context.Context) {
	t.Helper()

	cfg := livePostgresConnConfig(t)
	ctx := context.Background()

	dbName := fmt.Sprintf("af_reaper_live_%d_%d", time.Now().UnixNano(), os.Getpid())

	admin, err := sql.Open("pgx", cfg.ConnString())
	require.NoError(t, err)
	_, err = admin.ExecContext(ctx, "CREATE DATABASE "+dbName)
	_ = admin.Close()
	require.NoError(t, err, "create throwaway database %s", dbName)

	t.Cleanup(func() {
		drop, dropErr := sql.Open("pgx", cfg.ConnString())
		if dropErr != nil {
			return
		}
		defer drop.Close()
		_, _ = drop.ExecContext(context.Background(), "DROP DATABASE IF EXISTS "+dbName+" WITH (FORCE)")
	})

	storageCfg := StorageConfig{
		Mode: "postgres",
		Postgres: PostgresStorageConfig{
			Host:     cfg.Host,
			Port:     int(cfg.Port),
			User:     cfg.User,
			Password: cfg.Password,
			Database: dbName,
			SSLMode:  "disable",
		},
	}
	ls := NewPostgresStorage(PostgresStorageConfig{})
	require.NoError(t, ls.Initialize(ctx, storageCfg), "initialize postgres storage")
	t.Cleanup(func() { _ = ls.Close(context.Background()) })

	return ls, ctx
}

// preparedSQLRecorder wraps a real PostgreSQL connector and records every SQL
// string handed to the driver's prepare path. That string is the post-rebind
// text, so it is the exact statement PostgreSQL would parse.
type preparedSQLRecorder struct {
	inner driver.Connector
	mu    sync.Mutex
	sql   []string
}

func (r *preparedSQLRecorder) record(query string) {
	r.mu.Lock()
	r.sql = append(r.sql, query)
	r.mu.Unlock()
}

func (r *preparedSQLRecorder) prepared() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.sql...)
}

func (r *preparedSQLRecorder) Connect(ctx context.Context) (driver.Conn, error) {
	conn, err := r.inner.Connect(ctx)
	if err != nil {
		return nil, err
	}
	return &recordingConn{Conn: conn, recorder: r}, nil
}

func (r *preparedSQLRecorder) Driver() driver.Driver { return recordingDriver{recorder: r} }

type recordingDriver struct{ recorder *preparedSQLRecorder }

func (d recordingDriver) Open(string) (driver.Conn, error) {
	return d.recorder.Connect(context.Background())
}

type recordingConn struct {
	driver.Conn
	recorder *preparedSQLRecorder
}

func (c *recordingConn) Prepare(query string) (driver.Stmt, error) {
	c.recorder.record(query)
	return c.Conn.Prepare(query)
}

func (c *recordingConn) PrepareContext(ctx context.Context, query string) (driver.Stmt, error) {
	c.recorder.record(query)
	if pc, ok := c.Conn.(driver.ConnPrepareContext); ok {
		return pc.PrepareContext(ctx, query)
	}
	return c.Conn.Prepare(query)
}

func (c *recordingConn) BeginTx(ctx context.Context, opts driver.TxOptions) (driver.Tx, error) {
	if b, ok := c.Conn.(driver.ConnBeginTx); ok {
		return b.BeginTx(ctx, opts)
	}
	return c.Conn.Begin()
}

// livePreparedRecorder opens a recorder-backed *sql.DB for the same throwaway
// database as ls and swaps it in, returning a restore func.
func livePreparedRecorder(t *testing.T, cfg *pgx.ConnConfig) (*preparedSQLRecorder, *sql.DB) {
	t.Helper()
	connector := stdlib.GetConnector(*cfg)
	recorder := &preparedSQLRecorder{inner: connector}
	db := sql.OpenDB(recorder)
	return recorder, db
}

func liveConfigForDatabase(t *testing.T, dbName string) *pgx.ConnConfig {
	t.Helper()
	cfg := livePostgresConnConfig(t)
	cfg.Database = dbName
	return cfg
}

// livePostgresStorageWithRecorder returns the storage, its database name, and a
// recorder swapped in as the storage SQL handle so prepared statements are
// captured while the reapers run. The original handle is restored on cleanup.
func livePostgresStorageWithRecorder(t *testing.T) (*LocalStorage, context.Context, *preparedSQLRecorder) {
	t.Helper()

	ls, ctx := livePostgresStorage(t)
	dbName := ls.postgresConfig.Database
	require.NotEmpty(t, dbName)

	recorder, recDB := livePreparedRecorder(t, liveConfigForDatabase(t, dbName))
	original := ls.db
	ls.db = newSQLDatabase(recDB, "postgres")
	t.Cleanup(func() {
		ls.db = original
		_ = recDB.Close()
	})
	return ls, ctx, recorder
}

func staleLiveWorkflow(id string, updatedAt time.Time) *types.WorkflowExecution {
	return &types.WorkflowExecution{
		WorkflowID:          "wf-" + id,
		ExecutionID:         id,
		AgentFieldRequestID: "req-" + id,
		AgentNodeID:         "agent-live",
		ReasonerID:          "reasoner-live",
		Status:              "running",
		StartedAt:           updatedAt,
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          0,
		CreatedAt:           updatedAt,
		UpdatedAt:           updatedAt,
	}
}

func liveExecution(id string, startedAt time.Time) *types.Execution {
	return &types.Execution{
		ExecutionID:  id,
		RunID:        "run-" + id,
		AgentNodeID:  "agent-live",
		ReasonerID:   "reasoner-live",
		NodeID:       "agent-live",
		Status:       "running",
		StartedAt:    startedAt,
		CreatedAt:    startedAt,
		UpdatedAt:    startedAt,
		InputPayload: json.RawMessage(`{}`),
	}
}

// TestPostgresStaleReaperRawPlaceholdersFailToPrepare reproduces the original
// defect at the SQL level: the pre-fix code sent `?` placeholders straight to
// PostgreSQL, which rejects them with SQLSTATE 42601.
func TestPostgresStaleReaperRawPlaceholdersFailToPrepare(t *testing.T) {
	cfg := livePostgresConnConfig(t)

	db, err := sql.Open("pgx", cfg.ConnString())
	require.NoError(t, err)
	defer db.Close()

	ctx := context.Background()
	tx, err := db.BeginTx(ctx, nil)
	require.NoError(t, err)
	defer func() { _ = tx.Rollback() }()

	raw := `UPDATE executions AS e
		SET status = ?, error_message = ?, completed_at = ?, duration_ms = ?, updated_at = ?
		WHERE e.execution_id = ?
		  AND e.status IN ('running', 'pending', 'queued')
		  AND COALESCE(e.updated_at, e.created_at, e.started_at) <= ?`

	stmt, err := tx.PrepareContext(ctx, raw)
	if err == nil {
		_, err = stmt.ExecContext(ctx, "timeout", "timed out", time.Now(), 1, time.Now(), "id", time.Now())
		_ = stmt.Close()
	}
	require.Error(t, err, "unrebound ? placeholders must not prepare on PostgreSQL")

	var pgErr *pgconn.PgError
	require.ErrorAs(t, err, &pgErr)
	require.Equal(t, "42601", pgErr.Code)
	t.Logf("pre-fix reproduction: PostgreSQL rejected ? placeholders: SQLSTATE %s: %s", pgErr.Code, pgErr.Message)
}

// TestPostgresStaleReaperPreparedStatementsAreRebound captures the SQL text the
// reapers hand to the driver and asserts every placeholder was rewritten to $n
// before it reached PostgreSQL.
func TestPostgresStaleReaperPreparedStatementsAreRebound(t *testing.T) {
	ls, ctx, recorder := livePostgresStorageWithRecorder(t)
	now := time.Now().UTC()
	suffix := time.Now().UnixNano()

	staleID := fmt.Sprintf("exec-live-prepare-%d", suffix)
	require.NoError(t, ls.StoreWorkflowExecution(ctx, staleLiveWorkflow(staleID, now.Add(-2*time.Hour))))
	require.NoError(t, ls.CreateExecutionRecord(ctx, liveExecution(staleID, now.Add(-2*time.Hour))))
	backdateExecutionUpdatedAt(t, ls, "executions", staleID, now.Add(-2*time.Hour))

	_, err := ls.MarkStaleWorkflowExecutions(ctx, time.Hour, 100)
	require.NoError(t, err)
	_, err = ls.MarkStaleExecutions(ctx, time.Hour, 100)
	require.NoError(t, err)
	_, err = ls.RetryStaleWorkflowExecutions(ctx, time.Hour, 3, 100)
	require.NoError(t, err)

	prepared := recorder.prepared()
	require.NotEmpty(t, prepared, "expected the reapers to prepare statements")

	var updateStmts []string
	for _, query := range prepared {
		normalized := strings.ToLower(strings.TrimSpace(query))
		if strings.HasPrefix(normalized, "update") {
			updateStmts = append(updateStmts, query)
			require.NotContains(t, query, "?", "prepared update still contains ? placeholder")
			require.Contains(t, query, "$1", "prepared update is missing $1 placeholder")
		}
	}
	require.NotEmpty(t, updateStmts, "expected prepared UPDATE statements")

	for _, query := range updateStmts {
		t.Logf("PostgreSQL prepared update:\n%s", strings.Join(strings.Fields(query), " "))
	}
}

// TestPostgresMarkStaleWorkflowExecutionsReapsStaleAndSparesFreshExecution drives
// the workflow reaper end to end: a workflow stale on both clocks is reaped,
// while a stale workflow whose paired execution is still fresh survives.
func TestPostgresMarkStaleWorkflowExecutionsReapsStaleAndSparesFreshExecution(t *testing.T) {
	ls, ctx := livePostgresStorage(t)
	now := time.Now().UTC()
	suffix := time.Now().UnixNano()

	staleID := fmt.Sprintf("exec-live-reap-stale-%d", suffix)
	freshID := fmt.Sprintf("exec-live-reap-fresh-%d", suffix)

	// Stale on both clocks: workflow and paired execution both 2h old.
	require.NoError(t, ls.StoreWorkflowExecution(ctx, staleLiveWorkflow(staleID, now.Add(-2*time.Hour))))
	require.NoError(t, ls.CreateExecutionRecord(ctx, liveExecution(staleID, now.Add(-2*time.Hour))))
	backdateExecutionUpdatedAt(t, ls, "executions", staleID, now.Add(-2*time.Hour))

	// Stale workflow, fresh paired execution: the execution heartbeat must protect it.
	require.NoError(t, ls.StoreWorkflowExecution(ctx, staleLiveWorkflow(freshID, now.Add(-2*time.Hour))))
	require.NoError(t, ls.CreateExecutionRecord(ctx, liveExecution(freshID, now.Add(-2*time.Hour))))
	// CreateExecutionRecord leaves executions.updated_at at "now" (fresh).

	reaped, err := ls.MarkStaleWorkflowExecutions(ctx, time.Hour, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped, "only the workflow stale on both clocks should be reaped")

	staleWorkflow, err := ls.GetWorkflowExecution(ctx, staleID)
	require.NoError(t, err)
	require.Equal(t, "timeout", staleWorkflow.Status)
	t.Logf("reaped (stale on both clocks): %s status=%s", staleID, staleWorkflow.Status)

	staleExecution, err := ls.GetExecutionRecord(ctx, staleID)
	require.NoError(t, err)
	require.Equal(t, "timeout", staleExecution.Status)

	freshWorkflow, err := ls.GetWorkflowExecution(ctx, freshID)
	require.NoError(t, err)
	require.Equal(t, "running", freshWorkflow.Status)
	require.Equal(t, 0, freshWorkflow.RetryCount)
	t.Logf("survived (fresh paired execution): %s status=%s", freshID, freshWorkflow.Status)

	freshExecution, err := ls.GetExecutionRecord(ctx, freshID)
	require.NoError(t, err)
	require.Equal(t, "running", freshExecution.Status)
}

// TestPostgresMarkStaleWorkflowExecutionsReapsWhenPairedExecutionIsTerminal
// covers the new guard's status filter: a fresh but terminal paired execution
// does not shield a stale workflow from being reaped.
func TestPostgresMarkStaleWorkflowExecutionsReapsWhenPairedExecutionIsTerminal(t *testing.T) {
	ls, ctx := livePostgresStorage(t)
	now := time.Now().UTC()
	id := fmt.Sprintf("exec-live-terminal-%d", time.Now().UnixNano())

	require.NoError(t, ls.StoreWorkflowExecution(ctx, staleLiveWorkflow(id, now.Add(-2*time.Hour))))
	exec := liveExecution(id, now.Add(-2*time.Hour))
	exec.Status = "succeeded"
	require.NoError(t, ls.CreateExecutionRecord(ctx, exec))

	reaped, err := ls.MarkStaleWorkflowExecutions(ctx, time.Hour, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped, "a terminal paired execution must not shield a stale workflow")

	workflow, err := ls.GetWorkflowExecution(ctx, id)
	require.NoError(t, err)
	require.Equal(t, "timeout", workflow.Status)
	t.Logf("reaped (terminal paired execution): %s status=%s", id, workflow.Status)
}

// TestPostgresMarkStaleExecutionsReapsStaleAndSparesFreshExecution covers the
// legacy reaper on live PostgreSQL so both prepare call sites are proven.
func TestPostgresMarkStaleExecutionsReapsStaleAndSparesFreshExecution(t *testing.T) {
	ls, ctx := livePostgresStorage(t)
	now := time.Now().UTC()
	suffix := time.Now().UnixNano()

	staleID := fmt.Sprintf("exec-live-legacy-stale-%d", suffix)
	freshID := fmt.Sprintf("exec-live-legacy-fresh-%d", suffix)

	require.NoError(t, ls.CreateExecutionRecord(ctx, liveExecution(staleID, now.Add(-2*time.Hour))))
	backdateExecutionUpdatedAt(t, ls, "executions", staleID, now.Add(-2*time.Hour))
	require.NoError(t, ls.CreateExecutionRecord(ctx, liveExecution(freshID, now.Add(-2*time.Hour))))

	reaped, err := ls.MarkStaleExecutions(ctx, time.Hour, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped)

	staleExecution, err := ls.GetExecutionRecord(ctx, staleID)
	require.NoError(t, err)
	require.Equal(t, "timeout", staleExecution.Status)
	t.Logf("reaped (stale on both clocks): %s status=%s", staleID, staleExecution.Status)

	freshExecution, err := ls.GetExecutionRecord(ctx, freshID)
	require.NoError(t, err)
	require.Equal(t, "running", freshExecution.Status)
	t.Logf("survived (fresh activity): %s status=%s", freshID, freshExecution.Status)
}

// TestPostgresRetryStaleWorkflowExecutionsRepassesStaleAndSparesFreshExecution
// drives the retry reaper end to end. The paired-execution guard is the part of
// the fix that stops a fresh execution from being re-dispatched.
func TestPostgresRetryStaleWorkflowExecutionsRepassesStaleAndSparesFreshExecution(t *testing.T) {
	ls, ctx := livePostgresStorage(t)
	now := time.Now().UTC()
	suffix := time.Now().UnixNano()

	staleID := fmt.Sprintf("exec-live-retry-stale-%d", suffix)
	freshID := fmt.Sprintf("exec-live-retry-fresh-%d", suffix)

	require.NoError(t, ls.StoreWorkflowExecution(ctx, staleLiveWorkflow(staleID, now.Add(-2*time.Hour))))
	require.NoError(t, ls.CreateExecutionRecord(ctx, liveExecution(staleID, now.Add(-2*time.Hour))))
	backdateExecutionUpdatedAt(t, ls, "executions", staleID, now.Add(-2*time.Hour))

	require.NoError(t, ls.StoreWorkflowExecution(ctx, staleLiveWorkflow(freshID, now.Add(-2*time.Hour))))
	require.NoError(t, ls.CreateExecutionRecord(ctx, liveExecution(freshID, now.Add(-2*time.Hour))))

	retried, err := ls.RetryStaleWorkflowExecutions(ctx, time.Hour, 3, 100)
	require.NoError(t, err)
	require.Equal(t, []string{staleID}, retried, "only the workflow stale on both clocks should be retried")

	staleWorkflow, err := ls.GetWorkflowExecution(ctx, staleID)
	require.NoError(t, err)
	require.Equal(t, "pending", staleWorkflow.Status)
	require.Equal(t, 1, staleWorkflow.RetryCount)
	require.Nil(t, staleWorkflow.CompletedAt)
	t.Logf("retried (stale on both clocks): %s status=%s retry_count=%d", staleID, staleWorkflow.Status, staleWorkflow.RetryCount)

	staleExecution, err := ls.GetExecutionRecord(ctx, staleID)
	require.NoError(t, err)
	require.Equal(t, "pending", staleExecution.Status)

	freshWorkflow, err := ls.GetWorkflowExecution(ctx, freshID)
	require.NoError(t, err)
	require.Equal(t, "running", freshWorkflow.Status)
	require.Equal(t, 0, freshWorkflow.RetryCount)
	t.Logf("survived (fresh paired execution): %s status=%s retry_count=%d", freshID, freshWorkflow.Status, freshWorkflow.RetryCount)

	freshExecution, err := ls.GetExecutionRecord(ctx, freshID)
	require.NoError(t, err)
	require.Equal(t, "running", freshExecution.Status)
}

// TestPostgresRetryStaleWorkflowExecutionsHeartbeatBetweenStatementsSparesExecution
// pins down the between-statements heartbeat race on the real engine. A second
// session holds the paired execution's row lock; the retry transaction runs its
// workflow UPDATE and then parks on the execution UPDATE. While it is parked,
// the heartbeat commits, so the execution UPDATE must re-check the staleness
// predicate and leave the freshly heartbeated execution alone.
func TestPostgresRetryStaleWorkflowExecutionsHeartbeatBetweenStatementsSparesExecution(t *testing.T) {
	ls, ctx := livePostgresStorage(t)
	now := time.Now().UTC()
	id := fmt.Sprintf("exec-live-retry-between-%d", time.Now().UnixNano())

	require.NoError(t, ls.StoreWorkflowExecution(ctx, staleLiveWorkflow(id, now.Add(-2*time.Hour))))
	require.NoError(t, ls.CreateExecutionRecord(ctx, liveExecution(id, now.Add(-2*time.Hour))))
	backdateExecutionUpdatedAt(t, ls, "executions", id, now.Add(-2*time.Hour))

	// A second session takes the execution row lock first, so the retry
	// transaction reaches its second UPDATE and blocks there.
	gateCfg := liveConfigForDatabase(t, ls.postgresConfig.Database)
	// ConnConfig.ConnString() keeps the original database, so build the handle
	// from the connector like livePreparedRecorder does.
	gate := sql.OpenDB(stdlib.GetConnector(*gateCfg))
	defer gate.Close()

	gateTx, err := gate.BeginTx(ctx, nil)
	require.NoError(t, err)
	defer func() { _ = gateTx.Rollback() }()

	var lockedID string
	require.NoError(t, gateTx.QueryRowContext(ctx,
		"SELECT execution_id FROM executions WHERE execution_id = $1 FOR UPDATE", id).Scan(&lockedID))

	type retryOutcome struct {
		retried []string
		err     error
	}
	outcomeCh := make(chan retryOutcome, 1)
	go func() {
		retried, err := ls.RetryStaleWorkflowExecutions(ctx, time.Hour, 3, 100)
		outcomeCh <- retryOutcome{retried: retried, err: err}
	}()

	// Wait until the retry transaction has run its workflow UPDATE (it holds
	// the table's ROW EXCLUSIVE lock) and is on or about to reach the paired
	// execution UPDATE.
	deadline := time.Now().Add(10 * time.Second)
	for {
		var workflowUpdated bool
		require.NoError(t, gateTx.QueryRowContext(ctx, `
			SELECT EXISTS (
			    SELECT 1
			    FROM pg_locks
			    WHERE relation = to_regclass('public.workflow_executions')
			      AND mode = 'RowExclusiveLock'
			      AND granted
			)`).Scan(&workflowUpdated))
		if workflowUpdated {
			break
		}
		select {
		case outcome := <-outcomeCh:
			t.Fatalf("retry finished before the heartbeat could interleave: retried=%v err=%v", outcome.retried, outcome.err)
		case <-time.After(10 * time.Millisecond):
		}
		if time.Now().After(deadline) {
			t.Fatal("retry transaction never reached the workflow UPDATE")
		}
	}

	// The heartbeat commits while the retry sits between its two statements.
	heartbeatAt := time.Now().UTC()
	_, err = gateTx.ExecContext(ctx, `
		UPDATE executions
		SET updated_at = $1, status_reason = 'heartbeat-between-statements'
		WHERE execution_id = $2`, heartbeatAt, id)
	require.NoError(t, err)
	require.NoError(t, gateTx.Commit())

	select {
	case outcome := <-outcomeCh:
		require.NoError(t, outcome.err)
		require.Equal(t, []string{id}, outcome.retried)
	case <-time.After(15 * time.Second):
		t.Fatal("retry transaction did not finish after the heartbeat committed")
	}

	execution, err := ls.GetExecutionRecord(ctx, id)
	require.NoError(t, err)
	require.Equal(t, "running", execution.Status,
		"a heartbeat between the retry statements must not be dragged back to pending")
	require.NotNil(t, execution.StatusReason)
	require.Equal(t, "heartbeat-between-statements", *execution.StatusReason)
	require.True(t, execution.UpdatedAt.After(now.Add(-time.Minute)),
		"the heartbeat timestamp must be the surviving one")

	workflow, err := ls.GetWorkflowExecution(ctx, id)
	require.NoError(t, err)
	require.Equal(t, "pending", workflow.Status,
		"the workflow decision was made before the heartbeat landed")
	require.Equal(t, 1, workflow.RetryCount)
}
