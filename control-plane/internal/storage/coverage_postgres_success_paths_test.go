package storage

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/boltdb/bolt"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

type scriptedExecResponse struct {
	err    error
	result driver.Result
}

type scriptedQueryResponse struct {
	err     error
	columns []string
	rows    [][]driver.Value
}

type scriptedSQLState struct {
	mu      sync.Mutex
	execs   []scriptedExecResponse
	queries []scriptedQueryResponse
	begin   []error
	commit  []error
}

func (s *scriptedSQLState) nextExec() (scriptedExecResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.execs) == 0 {
		return scriptedExecResponse{}, errors.New("unexpected exec")
	}
	resp := s.execs[0]
	s.execs = s.execs[1:]
	return resp, nil
}

func (s *scriptedSQLState) nextQuery() (scriptedQueryResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.queries) == 0 {
		return scriptedQueryResponse{}, errors.New("unexpected query")
	}
	resp := s.queries[0]
	s.queries = s.queries[1:]
	return resp, nil
}

func (s *scriptedSQLState) assertConsumed(t *testing.T) {
	t.Helper()

	s.mu.Lock()
	defer s.mu.Unlock()
	require.Empty(t, s.execs)
	require.Empty(t, s.queries)
	require.Empty(t, s.begin)
	require.Empty(t, s.commit)
}

func (s *scriptedSQLState) nextBegin() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.begin) == 0 {
		return nil
	}
	err := s.begin[0]
	s.begin = s.begin[1:]
	return err
}

func (s *scriptedSQLState) nextCommit() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.commit) == 0 {
		return nil
	}
	err := s.commit[0]
	s.commit = s.commit[1:]
	return err
}

type scriptedDriver struct{}

type scriptedConn struct {
	state *scriptedSQLState
}

type scriptedTx struct {
	state *scriptedSQLState
}

type scriptedRows struct {
	columns []string
	rows    [][]driver.Value
	index   int
}

type rowsAffectedWithErr struct {
	rows int64
	err  error
}

var (
	registerScriptedDriverOnce sync.Once
	scriptedDriverCounter      uint64
	scriptedDriverStates       sync.Map
)

func openScriptedSQLDB(t *testing.T, state *scriptedSQLState) *sql.DB {
	t.Helper()

	registerScriptedDriverOnce.Do(func() {
		sql.Register("storage_scripted", scriptedDriver{})
	})

	name := fmt.Sprintf("script-%d", atomic.AddUint64(&scriptedDriverCounter, 1))
	scriptedDriverStates.Store(name, state)
	t.Cleanup(func() { scriptedDriverStates.Delete(name) })

	db, err := sql.Open("storage_scripted", name)
	require.NoError(t, err)
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { _ = db.Close() })
	return db
}

func (scriptedDriver) Open(name string) (driver.Conn, error) {
	raw, ok := scriptedDriverStates.Load(name)
	if !ok {
		return nil, fmt.Errorf("missing scripted state for %s", name)
	}
	return &scriptedConn{state: raw.(*scriptedSQLState)}, nil
}

func (c *scriptedConn) Prepare(string) (driver.Stmt, error) { return nil, errors.New("prepare not supported") }
func (c *scriptedConn) Close() error                        { return nil }
func (c *scriptedConn) Begin() (driver.Tx, error)          { return c.BeginTx(context.Background(), driver.TxOptions{}) }

func (c *scriptedConn) BeginTx(context.Context, driver.TxOptions) (driver.Tx, error) {
	if err := c.state.nextBegin(); err != nil {
		return nil, err
	}
	return &scriptedTx{state: c.state}, nil
}

func (c *scriptedConn) ExecContext(context.Context, string, []driver.NamedValue) (driver.Result, error) {
	resp, err := c.state.nextExec()
	if err != nil {
		return nil, err
	}
	if resp.result == nil {
		resp.result = driver.RowsAffected(1)
	}
	return resp.result, resp.err
}

func (c *scriptedConn) QueryContext(context.Context, string, []driver.NamedValue) (driver.Rows, error) {
	resp, err := c.state.nextQuery()
	if err != nil {
		return nil, err
	}
	if resp.err != nil {
		return nil, resp.err
	}
	return &scriptedRows{columns: resp.columns, rows: resp.rows}, nil
}

func (t *scriptedTx) Commit() error   { return t.state.nextCommit() }
func (t *scriptedTx) Rollback() error { return nil }

func (t *scriptedTx) ExecContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Result, error) {
	return (&scriptedConn{state: t.state}).ExecContext(ctx, query, args)
}

func (t *scriptedTx) QueryContext(ctx context.Context, query string, args []driver.NamedValue) (driver.Rows, error) {
	return (&scriptedConn{state: t.state}).QueryContext(ctx, query, args)
}

func (r *scriptedRows) Columns() []string { return r.columns }
func (r *scriptedRows) Close() error      { return nil }

func (r *scriptedRows) Next(dest []driver.Value) error {
	if r.index >= len(r.rows) {
		return io.EOF
	}
	copy(dest, r.rows[r.index])
	r.index++
	return nil
}

func (r rowsAffectedWithErr) LastInsertId() (int64, error) { return 0, nil }
func (r rowsAffectedWithErr) RowsAffected() (int64, error) { return r.rows, r.err }

func TestPostgresStorageHelperSuccessPaths(t *testing.T) {
	t.Run("schema helpers and vector store initialization succeed", func(t *testing.T) {
		execCount := 1 + 2 + 2 + 11 + 18 + 4 + 4
		state := &scriptedSQLState{execs: make([]scriptedExecResponse, execCount)}
		db := openScriptedSQLDB(t, state)
		enabled := true
		ls := &LocalStorage{
			db:           newSQLDatabase(db, "postgres"),
			mode:         "postgres",
			vectorMetric: VectorDistanceCosine,
			vectorConfig: VectorStoreConfig{Enabled: &enabled},
		}

		require.NoError(t, ls.ensurePostgresKeyValueSchema(context.Background()))
		require.NoError(t, ls.ensurePostgresEventSchema(context.Background()))
		require.NoError(t, ls.ensurePostgresLockSchema(context.Background()))
		require.NoError(t, ls.ensurePostgresWorkflowFTS(context.Background()))
		require.NoError(t, ls.ensurePostgresIndexes(context.Background()))
		require.NoError(t, ls.ensurePostgresVectorSchema(context.Background()))
		require.NoError(t, ls.ensureVectorSchema(context.Background()))
		require.NoError(t, ls.initializeVectorStore())
		require.IsType(t, &postgresVectorStore{}, ls.vectorStore)

		disabled := false
		ls.vectorConfig = VectorStoreConfig{Enabled: &disabled}
		require.NoError(t, ls.initializeVectorStore())
		require.Nil(t, ls.vectorStore)

		state.assertConsumed(t)
	})

	t.Run("run postgres migrations applies once and skips recorded versions", func(t *testing.T) {
		state := &scriptedSQLState{
			execs: []scriptedExecResponse{
				{},
				{},
				{},
				{},
			},
			queries: []scriptedQueryResponse{
				{columns: []string{"count"}, rows: [][]driver.Value{{int64(0)}}},
				{columns: []string{"count"}, rows: [][]driver.Value{{int64(1)}}},
			},
		}
		db := openScriptedSQLDB(t, state)
		ls := &LocalStorage{db: newSQLDatabase(db, "postgres"), mode: "postgres"}

		require.NoError(t, ls.runPostgresMigrations(context.Background()))
		require.NoError(t, ls.runPostgresMigrations(context.Background()))

		state.assertConsumed(t)
	})

	t.Run("postgres composite key migration covers early returns and success path", func(t *testing.T) {
		tests := []struct {
			name    string
			queries []scriptedQueryResponse
			execs   []scriptedExecResponse
		}{
			{
				name: "fresh install returns early",
				queries: []scriptedQueryResponse{
					{columns: []string{"count"}, rows: [][]driver.Value{{int64(0)}}},
				},
			},
			{
				name: "already migrated returns early",
				queries: []scriptedQueryResponse{
					{columns: []string{"count"}, rows: [][]driver.Value{{int64(1)}}},
					{columns: []string{"count"}, rows: [][]driver.Value{{int64(1)}}},
				},
			},
			{
				name: "migration executes all postgres ddl steps",
				queries: []scriptedQueryResponse{
					{columns: []string{"count"}, rows: [][]driver.Value{{int64(1)}}},
					{columns: []string{"count"}, rows: [][]driver.Value{{int64(0)}}},
				},
				execs: make([]scriptedExecResponse, 9),
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				state := &scriptedSQLState{
					queries: append([]scriptedQueryResponse(nil), tc.queries...),
					execs:   append([]scriptedExecResponse(nil), tc.execs...),
				}
				db := openScriptedSQLDB(t, state)
				ls := &LocalStorage{db: newSQLDatabase(db, "postgres"), mode: "postgres"}

				require.NoError(t, ls.migrateAgentNodesCompositePKPostgres(context.Background()))
				state.assertConsumed(t)
			})
		}
	})
}

func TestPostgresVectorStoreSuccessPaths(t *testing.T) {
	now := time.Date(2026, 4, 9, 12, 0, 0, 0, time.UTC)

	t.Run("set get delete and search succeed", func(t *testing.T) {
		state := &scriptedSQLState{
			execs: []scriptedExecResponse{
				{},
				{},
				{result: driver.RowsAffected(3)},
			},
			queries: []scriptedQueryResponse{
				{
					columns: []string{"embedding", "metadata", "created_at", "updated_at"},
					rows: [][]driver.Value{{
						"[1.5, -2, 3]",
						[]byte(`{"team":"ops"}`),
						now,
						now.Add(time.Minute),
					}},
				},
				{
					columns: []string{"scope", "scope_id", "key", "metadata", "created_at", "updated_at", "score", "distance"},
					rows: [][]driver.Value{
						{"session", "scope-1", "doc-1", []byte(`{"kind":"doc"}`), now, now.Add(time.Minute), float64(0.9), float64(0.1)},
						{"session", "scope-1", "doc-2", []byte{}, now, now.Add(2 * time.Minute), float64(0.7), float64(0.3)},
					},
				},
			},
		}
		db := openScriptedSQLDB(t, state)
		store := newPostgresVectorStore(newSQLDatabase(db, "postgres"), VectorDistanceCosine)

		err := store.Set(context.Background(), &types.VectorRecord{
			Scope:     "session",
			ScopeID:   "scope-1",
			Key:       "doc-1",
			Embedding: []float32{1.5, -2, 3},
			Metadata:  map[string]interface{}{"team": "ops"},
		})
		require.NoError(t, err)

		record, err := store.Get(context.Background(), "session", "scope-1", "doc-1")
		require.NoError(t, err)
		require.Equal(t, []float32{1.5, -2, 3}, record.Embedding)
		require.Equal(t, "ops", record.Metadata["team"])

		require.NoError(t, store.Delete(context.Background(), "session", "scope-1", "doc-1"))

		deleted, err := store.DeleteByPrefix(context.Background(), "session", "scope-1", "doc-")
		require.NoError(t, err)
		require.Equal(t, 3, deleted)

		results, err := store.Search(context.Background(), "session", "scope-1", []float32{1, 2, 3}, 0, map[string]interface{}{"kind": "doc"})
		require.NoError(t, err)
		require.Len(t, results, 2)
		require.Equal(t, map[string]interface{}{"kind": "doc"}, results[0].Metadata)
		require.Empty(t, results[1].Metadata)

		state.assertConsumed(t)
	})

	t.Run("get returns nil on no rows and surfaces decode failures", func(t *testing.T) {
		tests := []struct {
			name     string
			response scriptedQueryResponse
			wantErr  string
			wantNil  bool
		}{
			{
				name:     "not found",
				response: scriptedQueryResponse{err: sql.ErrNoRows},
				wantNil:  true,
			},
			{
				name: "bad embedding element",
				response: scriptedQueryResponse{
					columns: []string{"embedding", "metadata", "created_at", "updated_at"},
					rows: [][]driver.Value{{"[bad]", []byte(`{}`), now, now}},
				},
				wantErr: "parse embedding element",
			},
			{
				name: "bad metadata",
				response: scriptedQueryResponse{
					columns: []string{"embedding", "metadata", "created_at", "updated_at"},
					rows: [][]driver.Value{{"[1,2]", []byte(`{`), now, now}},
				},
				wantErr: "unmarshal metadata",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				state := &scriptedSQLState{queries: []scriptedQueryResponse{tc.response}}
				db := openScriptedSQLDB(t, state)
				store := newPostgresVectorStore(newSQLDatabase(db, "postgres"), VectorDistanceDot)

				record, err := store.Get(context.Background(), "session", "scope-1", "doc-1")
				if tc.wantErr != "" {
					require.ErrorContains(t, err, tc.wantErr)
					require.Nil(t, record)
				} else {
					require.NoError(t, err)
					require.Nil(t, record)
				}

				state.assertConsumed(t)
			})
		}
	})

	t.Run("delete by prefix returns rows affected errors", func(t *testing.T) {
		state := &scriptedSQLState{
			execs: []scriptedExecResponse{
				{result: rowsAffectedWithErr{err: errors.New("rows affected failed")}},
			},
		}
		db := openScriptedSQLDB(t, state)
		store := newPostgresVectorStore(newSQLDatabase(db, "postgres"), VectorDistanceL2)

		_, err := store.DeleteByPrefix(context.Background(), "session", "scope-1", "doc-")
		require.EqualError(t, err, "rows affected failed")
		state.assertConsumed(t)
	})
}

func TestSetupWorkflowExecutionFTSErrorPaths(t *testing.T) {
	t.Run("missing base table fails during trigger creation", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		require.NoError(t, rawDB.Close())

		ls := &LocalStorage{db: newSQLDatabase(rawDB, "local"), mode: "local"}
		err = ls.setupWorkflowExecutionFTS()
		require.Error(t, err)
		require.ErrorContains(t, err, "failed to create FTS5 virtual table")
	})
}

func TestCreateSchemaAndSetupFTSWithModernSQLite(t *testing.T) {
	tempDir := t.TempDir()

	rawDB, err := sql.Open("sqlite", filepath.Join(tempDir, "modern.db"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = rawDB.Close() })

	kvStore, err := bolt.Open(filepath.Join(tempDir, "modern.bolt"), 0o600, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = kvStore.Close() })

	ls := &LocalStorage{
		db:      newSQLDatabase(rawDB, "local"),
		kvStore: kvStore,
		mode:    "local",
	}

	require.NoError(t, ls.createSchema(context.Background()))
	require.True(t, ls.ftsEnabled)
	require.NotNil(t, ls.vectorStore)

	workflowName := "Modern FTS"
	require.NoError(t, ls.StoreWorkflowExecution(context.Background(), &types.WorkflowExecution{
		WorkflowID:          "wf-modern",
		ExecutionID:         "exec-modern",
		AgentFieldRequestID: "req-modern",
		AgentNodeID:         "agent-modern",
		ReasonerID:          "reasoner-modern",
		Status:              "pending",
		WorkflowName:        &workflowName,
	}))

	var count int
	require.NoError(t, rawDB.QueryRow(`SELECT COUNT(*) FROM workflow_executions_fts WHERE execution_id = ? AND workflow_name = ?`, "exec-modern", workflowName).Scan(&count))
	require.Equal(t, 1, count)
}

func TestWorkflowCleanupAndTxInsertCoverage(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC()

	session := &types.Session{
		SessionID:       "session-cleanup",
		SessionName:     ptrString("Cleanup Session"),
		TotalWorkflows:  1,
		TotalExecutions: 1,
		StartedAt:       now,
		LastActivityAt:  now,
	}
	require.NoError(t, ls.CreateOrUpdateSession(ctx, session))
	require.NoError(t, ls.CreateOrUpdateSession(ctx, session))

	workflow := &types.Workflow{
		WorkflowID:      "wf-cleanup",
		WorkflowName:    ptrString("Cleanup Workflow"),
		WorkflowTags:    []string{"cleanup"},
		SessionID:       &session.SessionID,
		Status:          "running",
		StartedAt:       now,
		CreatedAt:       now,
		UpdatedAt:       now,
		RootWorkflowID:  ptrString("wf-cleanup"),
		WorkflowDepth:   0,
		TotalExecutions: 1,
	}
	require.NoError(t, ls.CreateOrUpdateWorkflow(ctx, workflow))
	require.NoError(t, ls.CreateOrUpdateWorkflow(ctx, workflow))

	tx, err := ls.db.BeginTx(ctx, nil)
	require.NoError(t, err)
	require.NoError(t, ls.executeSessionInsertWithTx(ctx, tx, &types.Session{
		SessionID:       "session-tx",
		SessionName:     ptrString("TX Session"),
		StartedAt:       now,
		TotalWorkflows:  2,
		TotalExecutions: 3,
	}))
	require.NoError(t, ls.executeWorkflowInsertWithTx(ctx, tx, &types.Workflow{
		WorkflowID:      "wf-tx",
		WorkflowName:    ptrString("TX Workflow"),
		WorkflowTags:    []string{"tx"},
		SessionID:       ptrString("session-tx"),
		Status:          "running",
		StartedAt:       now,
		TotalExecutions: 3,
	}))
	require.NoError(t, tx.Commit())

	run := &types.WorkflowRun{
		RunID:          "run-cleanup",
		RootWorkflowID: workflow.WorkflowID,
		Status:         "running",
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	require.NoError(t, ls.StoreWorkflowRun(ctx, run))

	runID := run.RunID
	require.NoError(t, ls.StoreWorkflowExecution(ctx, &types.WorkflowExecution{
		WorkflowID:          workflow.WorkflowID,
		ExecutionID:         "wfexec-cleanup",
		AgentFieldRequestID: "req-cleanup",
		RunID:               &runID,
		SessionID:           &session.SessionID,
		AgentNodeID:         "agent-cleanup",
		ReasonerID:          "reasoner-cleanup",
		Status:              "running",
		StartedAt:           now,
		CreatedAt:           now,
		UpdatedAt:           now,
		RootWorkflowID:      ptrString(workflow.WorkflowID),
	}))

	_, err = ls.db.ExecContext(ctx, `
		INSERT INTO executions (
			execution_id, run_id, agent_node_id, reasoner_id, node_id, status,
			started_at, notes, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, '[]', ?, ?)
	`, "exec-cleanup", run.RunID, "agent-cleanup", "reasoner-cleanup", "node-cleanup", "running", now, now, now)
	require.NoError(t, err)

	require.NoError(t, ls.RegisterExecutionWebhook(ctx, &types.ExecutionWebhook{
		ExecutionID: "exec-cleanup",
		URL:         "https://example.com/webhook",
		Headers:     map[string]string{"X-Test": "true"},
		Status:      types.ExecutionWebhookStatusPending,
	}))
	require.NoError(t, ls.StoreExecutionWebhookEvent(ctx, &types.ExecutionWebhookEvent{
		ExecutionID: "exec-cleanup",
		EventType:   types.WebhookEventExecutionCompleted,
		Status:      types.ExecutionWebhookStatusDelivered,
		Payload:     []byte(`{"ok":true}`),
		CreatedAt:   now,
	}))

	require.NoError(t, ls.StoreExecutionVC(ctx,
		"vc-cleanup", "exec-cleanup", workflow.WorkflowID, session.SessionID,
		"did:issuer:cleanup", "", "did:caller:cleanup", "input-hash", "output-hash", "pending",
		[]byte(`{"vc":true}`), "sig", "s3://bucket/vc", 128,
	))
	require.NoError(t, ls.StoreWorkflowVC(ctx,
		"wvc-cleanup", workflow.WorkflowID, session.SessionID, []string{"vc-cleanup"}, "pending",
		&now, nil, 1, 0, "s3://bucket/workflow", 64,
	))

	_, err = ls.db.ExecContext(ctx, `
		INSERT INTO workflow_execution_events (
			execution_id, workflow_id, run_id, sequence, previous_sequence,
			event_type, payload, emitted_at, recorded_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, "wfexec-cleanup", workflow.WorkflowID, run.RunID, 1, 0, "execution.started", `{}`, now, now)
	require.NoError(t, err)

	dryRun, err := ls.CleanupWorkflow(ctx, workflow.WorkflowID, true)
	require.NoError(t, err)
	require.True(t, dryRun.Success)
	require.Greater(t, dryRun.DeletedRecords["workflow_runs"], 0)
	require.Greater(t, dryRun.DeletedRecords["executions"], 0)
	require.Greater(t, dryRun.DeletedRecords["execution_webhooks"], 0)
	require.Greater(t, dryRun.DeletedRecords["execution_webhook_events"], 0)
	require.Greater(t, dryRun.DeletedRecords["execution_vcs"], 0)
	require.Greater(t, dryRun.DeletedRecords["workflow_vcs"], 0)
	require.Greater(t, dryRun.DeletedRecords["workflow_executions"], 0)
	require.Greater(t, dryRun.DeletedRecords["workflow_execution_events"], 0)
	require.Greater(t, dryRun.DeletedRecords["workflows"], 0)
	require.NotEmpty(t, makePlaceholders(3))
	require.Empty(t, makePlaceholders(0))

	cleanup, err := ls.CleanupWorkflow(ctx, workflow.WorkflowID, false)
	require.NoError(t, err)
	require.True(t, cleanup.Success)

	assertCount := func(query string, args ...interface{}) {
		t.Helper()
		var count int
		require.NoError(t, ls.db.QueryRowContext(ctx, query, args...).Scan(&count))
		require.Zero(t, count, query)
	}

	assertCount(`SELECT COUNT(*) FROM workflow_runs WHERE run_id = ?`, run.RunID)
	assertCount(`SELECT COUNT(*) FROM executions WHERE execution_id = ?`, "exec-cleanup")
	assertCount(`SELECT COUNT(*) FROM execution_webhooks WHERE execution_id = ?`, "exec-cleanup")
	assertCount(`SELECT COUNT(*) FROM execution_webhook_events WHERE execution_id = ?`, "exec-cleanup")
	assertCount(`SELECT COUNT(*) FROM execution_vcs WHERE workflow_id = ?`, workflow.WorkflowID)
	assertCount(`SELECT COUNT(*) FROM workflow_vcs WHERE workflow_id = ?`, workflow.WorkflowID)
	assertCount(`SELECT COUNT(*) FROM workflow_executions WHERE workflow_id = ?`, workflow.WorkflowID)
	assertCount(`SELECT COUNT(*) FROM workflows WHERE workflow_id = ?`, workflow.WorkflowID)
}

func workflowExecutionDriverRow(executionID, status string, now time.Time) []driver.Value {
	return []driver.Value{
		"wf-scripted",
		executionID,
		"req-scripted",
		nil,
		nil,
		nil,
		"agent-scripted",
		nil,
		nil,
		nil,
		int64(0),
		"reasoner-scripted",
		nil,
		nil,
		int64(0),
		int64(0),
		status,
		now,
		nil,
		nil,
		int64(0),
		int64(0),
		int64(0),
		int64(0),
		nil,
		nil,
		nil,
		nil,
		nil,
		int64(0),
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		[]byte(`[]`),
		[]byte(`[]`),
		now,
		now,
	}
}

func workflowExecutionDriverResponse(executionID, status string, now time.Time) scriptedQueryResponse {
	columns := append([]string(nil), workflowExecutionLifecycleColumns...)
	return scriptedQueryResponse{
		columns: columns,
		rows:    [][]driver.Value{workflowExecutionDriverRow(executionID, status, now)},
	}
}

func TestWorkflowExecutionRetryCoverage(t *testing.T) {
	now := time.Now().UTC()

	t.Run("store workflow execution internal covers begin and commit failures", func(t *testing.T) {
		tests := []struct {
			name    string
			state   *scriptedSQLState
			wantErr string
		}{
			{
				name: "begin failure",
				state: &scriptedSQLState{
					begin: []error{errors.New("database is locked")},
				},
				wantErr: "failed to begin transaction",
			},
			{
				name: "commit failure",
				state: &scriptedSQLState{
					queries: []scriptedQueryResponse{
						{err: sql.ErrNoRows},
					},
					execs:   []scriptedExecResponse{{}},
					commit:  []error{errors.New("database is locked")},
				},
				wantErr: "failed to commit workflow execution transaction",
			},
		}

		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				db := openScriptedSQLDB(t, tc.state)
				ls := &LocalStorage{db: newSQLDatabase(db, "sqlite"), mode: "local"}
				err := ls.storeWorkflowExecutionInternal(context.Background(), &types.WorkflowExecution{
					WorkflowID:          "wf-scripted",
					ExecutionID:         "exec-scripted",
					AgentFieldRequestID: "req-scripted",
					AgentNodeID:         "agent-scripted",
					ReasonerID:          "reasoner-scripted",
					Status:              "running",
					StartedAt:           now,
				})
				require.ErrorContains(t, err, tc.wantErr)
				tc.state.assertConsumed(t)
			})
		}
	})

	t.Run("update workflow execution retries lock errors and surfaces final failure", func(t *testing.T) {
		state := &scriptedSQLState{
			queries: []scriptedQueryResponse{
				workflowExecutionDriverResponse("exec-scripted", "running", now),
				workflowExecutionDriverResponse("exec-scripted", "running", now),
				workflowExecutionDriverResponse("exec-scripted", "running", now),
				workflowExecutionDriverResponse("exec-scripted", "running", now),
				workflowExecutionDriverResponse("exec-scripted", "running", now),
				workflowExecutionDriverResponse("exec-scripted", "running", now),
				workflowExecutionDriverResponse("exec-scripted", "running", now),
				workflowExecutionDriverResponse("exec-scripted", "running", now),
			},
			execs: []scriptedExecResponse{{}, {}, {}, {}},
			commit: []error{
				errors.New("database is locked"),
				errors.New("database is locked"),
				errors.New("database is locked"),
				errors.New("database is locked"),
			},
		}
		db := openScriptedSQLDB(t, state)
		ls := &LocalStorage{db: newSQLDatabase(db, "sqlite"), mode: "local"}

		err := ls.UpdateWorkflowExecution(context.Background(), "exec-scripted", func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error) {
			updated := *execution
			updated.Status = "succeeded"
			return &updated, nil
		})
		require.ErrorContains(t, err, "failed to update workflow execution after 4 attempts")
		state.assertConsumed(t)
	})
}
