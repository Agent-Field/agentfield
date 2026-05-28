package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestAutoMigrateSchemaAdditionalBranches(t *testing.T) {
	t.Run("creates tables for local storage", func(t *testing.T) {
		ls := newSchemaTestStorage(t)
		ctx := context.Background()

		require.NoError(t, ls.autoMigrateSchema(ctx))

		var count int
		require.NoError(t, ls.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'execution_webhooks'`).Scan(&count))
		require.Equal(t, 1, count)
		require.NotNil(t, ls.gormDB)
	})

	t.Run("propagates cancelled context", func(t *testing.T) {
		ls := newSchemaTestStorage(t)
		cancelled, cancel := context.WithCancel(context.Background())
		cancel()

		err := ls.autoMigrateSchema(cancelled)
		require.EqualError(t, err, "failed to initialize gorm for migrations: context cancelled: context canceled")
	})

	t.Run("covers postgres schema setup path until fts trigger creation fails on sqlite", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		ls := &LocalStorage{db: newSQLDatabase(rawDB, "postgres"), mode: "local"}
		require.NoError(t, ls.initGormDB())
		ls.mode = "postgres"

		err = ls.createSchema(context.Background())
		require.Error(t, err)
		require.Contains(t, err.Error(), "syntax error")
	})
}

func TestExecutionWebhookHelperCoverage(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC().Truncate(time.Second)
	secret := "secret-value"
	lastError := "delivery failed"
	lastAttempt := now.Add(-time.Minute)
	nextAttempt := now.Add(time.Minute)

	require.NoError(t, ls.RegisterExecutionWebhook(ctx, &types.ExecutionWebhook{
		ExecutionID: "exec-webhook-helper",
		URL:         "https://example.test/webhook",
		Secret:      &secret,
		Headers: map[string]string{
			"X-Test": "1",
		},
		Status: types.ExecutionWebhookStatusPending,
	}))
	require.NoError(t, ls.UpdateExecutionWebhookState(ctx, "exec-webhook-helper", types.ExecutionWebhookStateUpdate{
		Status:        types.ExecutionWebhookStatusPending,
		AttemptCount:  2,
		NextAttemptAt: &nextAttempt,
		LastAttemptAt: &lastAttempt,
		LastError:     &lastError,
	}))
	require.NoError(t, ls.StoreExecutionWebhookEvent(ctx, &types.ExecutionWebhookEvent{
		ExecutionID:   "exec-webhook-helper",
		EventType:     "delivery_attempt",
		Status:        types.ExecutionWebhookStatusFailed,
		HTTPStatus:    ptr(502),
		Payload:       json.RawMessage(`{"attempt":1}`),
		ResponseBody:  ptrString("bad gateway"),
		ErrorMessage:  ptrString("temporary upstream failure"),
	}))

	t.Run("enriches execution webhook state with and without events", func(t *testing.T) {
		ls.enrichExecutionWebhook(ctx, nil, true)

		execWithWebhook := &types.Execution{ExecutionID: "exec-webhook-helper"}
		ls.enrichExecutionWebhook(ctx, execWithWebhook, false)
		require.True(t, execWithWebhook.WebhookRegistered)
		require.Empty(t, execWithWebhook.WebhookEvents)

		ls.enrichExecutionWebhook(ctx, execWithWebhook, true)
		require.True(t, execWithWebhook.WebhookRegistered)
		require.Len(t, execWithWebhook.WebhookEvents, 1)
		require.Equal(t, ptr(502), execWithWebhook.WebhookEvents[0].HTTPStatus)

		execWithoutWebhook := &types.Execution{ExecutionID: "exec-webhook-missing"}
		ls.enrichExecutionWebhook(ctx, execWithoutWebhook, true)
		require.False(t, execWithoutWebhook.WebhookRegistered)
		require.Empty(t, execWithoutWebhook.WebhookEvents)
	})

	t.Run("populates webhook registration maps", func(t *testing.T) {
		cancelled, cancel := context.WithCancel(ctx)
		cancel()

		cancelledSet := []*types.Execution{
			{ExecutionID: "exec-webhook-helper"},
			{ExecutionID: "exec-webhook-missing"},
		}
		ls.populateWebhookRegistration(cancelled, cancelledSet)
		require.False(t, cancelledSet[0].WebhookRegistered)
		require.False(t, cancelledSet[1].WebhookRegistered)

		executions := []*types.Execution{
			nil,
			{ExecutionID: "exec-webhook-helper"},
			{ExecutionID: "exec-webhook-missing"},
		}
		ls.populateWebhookRegistration(ctx, executions)
		require.True(t, executions[1].WebhookRegistered)
		require.False(t, executions[2].WebhookRegistered)

		ls.populateWebhookRegistration(ctx, nil)
	})

	t.Run("surfaces header decoding errors from getters and due list", func(t *testing.T) {
		_, err := ls.db.ExecContext(ctx, `UPDATE execution_webhooks SET headers = ?, next_attempt_at = NULL WHERE execution_id = ?`, "{", "exec-webhook-helper")
		require.NoError(t, err)

		_, err = ls.GetExecutionWebhook(ctx, "exec-webhook-helper")
		require.EqualError(t, err, "unmarshal webhook headers: unexpected end of JSON input")

		_, err = ls.ListDueExecutionWebhooks(ctx, 1)
		require.EqualError(t, err, "unmarshal webhook headers: unexpected end of JSON input")
	})
}

func TestLocalStorageHealthAndCloseAdditionalBranches(t *testing.T) {
	t.Run("close honors cancelled context", func(t *testing.T) {
		ls, _ := setupLocalStorage(t)
		cancelled, cancel := context.WithCancel(context.Background())
		cancel()

		err := ls.Close(cancelled)
		require.EqualError(t, err, "context cancelled during close: context canceled")

		require.NoError(t, ls.Close(context.Background()))
	})

	t.Run("health check reports closed boltdb", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		require.NoError(t, ls.kvStore.Close())

		err := ls.HealthCheck(ctx)
		require.ErrorContains(t, err, "BoltDB health check failed")

		ls.kvStore = nil
		require.NoError(t, ls.Close(context.Background()))
	})
}

func TestExecutionRecordTimeHelpersAdditionalBranches(t *testing.T) {
	now := time.Now().UTC().Truncate(time.Second)

	tests := []struct {
		name    string
		input   interface{}
		want    time.Time
		wantErr string
	}{
		{
			name:  "sql null time valid",
			input: sql.NullTime{Time: now, Valid: true},
			want:  now,
		},
		{
			name:  "sql null time invalid",
			input: sql.NullTime{},
			want:  time.Time{},
		},
		{
			name:  "sql null string valid",
			input: sql.NullString{String: "2026-04-09T12:34:56", Valid: true},
			want:  time.Date(2026, 4, 9, 12, 34, 56, 0, time.UTC),
		},
		{
			name:  "sql null string invalid",
			input: sql.NullString{},
			want:  time.Time{},
		},
		{
			name:    "unsupported type",
			input:   42,
			wantErr: "unsupported time value type int",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseDBTime(tt.input)
			if tt.wantErr != "" {
				require.EqualError(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}

	t.Run("parses sqlite timestamps missing trailing z", func(t *testing.T) {
		parsed, err := parseTimeString("2026-04-09T12:34:56.123456789")
		require.NoError(t, err)
		require.Equal(t, time.Date(2026, 4, 9, 12, 34, 56, 123456789, time.UTC), parsed)
	})
}

func TestWorkflowCleanupHelperCoverage(t *testing.T) {
	rawDB, err := sql.Open("sqlite3", ":memory:")
	require.NoError(t, err)
	t.Cleanup(func() { _ = rawDB.Close() })

	schema := []string{
		`CREATE TABLE executions (execution_id TEXT PRIMARY KEY, run_id TEXT NOT NULL)`,
		`CREATE TABLE execution_webhooks (execution_id TEXT PRIMARY KEY)`,
		`CREATE TABLE execution_webhook_events (id INTEGER PRIMARY KEY AUTOINCREMENT, execution_id TEXT NOT NULL)`,
		`CREATE TABLE workflow_runs (run_id TEXT PRIMARY KEY, root_workflow_id TEXT)`,
		`CREATE TABLE workflow_executions (execution_id TEXT PRIMARY KEY, workflow_id TEXT, root_workflow_id TEXT, run_id TEXT)`,
		`CREATE TABLE workflow_execution_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, workflow_id TEXT, run_id TEXT)`,
		`CREATE TABLE execution_vcs (vc_id TEXT PRIMARY KEY, workflow_id TEXT)`,
		`CREATE TABLE workflow_vcs (workflow_vc_id TEXT PRIMARY KEY, workflow_id TEXT)`,
		`CREATE TABLE workflows (workflow_id TEXT PRIMARY KEY)`,
	}
	for _, stmt := range schema {
		_, err := rawDB.Exec(stmt)
		require.NoError(t, err)
	}

	const runID = "run-helper"
	const workflowID = "wf-helper"

	inserts := []struct {
		query string
		args  []interface{}
	}{
		{`INSERT INTO executions (execution_id, run_id) VALUES (?, ?)`, []interface{}{"exec-helper", runID}},
		{`INSERT INTO execution_webhooks (execution_id) VALUES (?)`, []interface{}{"exec-helper"}},
		{`INSERT INTO execution_webhook_events (execution_id) VALUES (?)`, []interface{}{"exec-helper"}},
		{`INSERT INTO workflow_runs (run_id, root_workflow_id) VALUES (?, ?)`, []interface{}{runID, workflowID}},
		{`INSERT INTO workflow_executions (execution_id, workflow_id, root_workflow_id, run_id) VALUES (?, ?, ?, ?)`, []interface{}{"wf-exec-helper", workflowID, workflowID, runID}},
		{`INSERT INTO workflow_execution_events (workflow_id, run_id) VALUES (?, ?)`, []interface{}{workflowID, runID}},
		{`INSERT INTO execution_vcs (vc_id, workflow_id) VALUES (?, ?)`, []interface{}{"vc-helper", workflowID}},
		{`INSERT INTO workflow_vcs (workflow_vc_id, workflow_id) VALUES (?, ?)`, []interface{}{"wvc-helper", workflowID}},
		{`INSERT INTO workflows (workflow_id) VALUES (?)`, []interface{}{workflowID}},
	}
	for _, insert := range inserts {
		_, err := rawDB.Exec(insert.query, insert.args...)
		require.NoError(t, err)
	}

	ls := &LocalStorage{db: newSQLDatabase(rawDB, "local"), mode: "local"}
	ctx := context.Background()

	targets, err := ls.resolveWorkflowCleanupTargets(ctx, runID)
	require.NoError(t, err)
	require.Equal(t, workflowID, targets.primaryWorkflowID)
	require.ElementsMatch(t, []string{workflowID, runID}, targets.workflowIDs)
	require.ElementsMatch(t, []string{runID}, targets.runIDs)

	result := &types.WorkflowCleanupResult{DeletedRecords: make(map[string]int)}
	ls.populateWorkflowCleanupCounts(ctx, targets, result)
	require.Equal(t, 1, result.DeletedRecords["workflow_runs"])
	require.Equal(t, 1, result.DeletedRecords["executions"])
	require.Equal(t, 1, result.DeletedRecords["execution_webhooks"])
	require.Equal(t, 1, result.DeletedRecords["execution_webhook_events"])
	require.Equal(t, 1, result.DeletedRecords["execution_vcs"])
	require.Equal(t, 1, result.DeletedRecords["workflow_vcs"])
	require.Equal(t, 1, result.DeletedRecords["workflow_executions"])
	require.Equal(t, 1, result.DeletedRecords["workflow_execution_events"])
	require.Equal(t, 1, result.DeletedRecords["workflows"])

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	tx, err := ls.db.BeginTx(ctx, nil)
	require.NoError(t, err)
	require.EqualError(t, ls.performWorkflowCleanup(cancelled, tx, targets), "context cancelled during workflow cleanup: context canceled")
	require.NoError(t, tx.Rollback())

	tx, err = ls.db.BeginTx(ctx, nil)
	require.NoError(t, err)
	require.NoError(t, ls.performWorkflowCleanup(ctx, tx, targets))
	require.NoError(t, tx.Commit())

	for table, query := range map[string]string{
		"executions":                `SELECT COUNT(*) FROM executions`,
		"execution_webhooks":        `SELECT COUNT(*) FROM execution_webhooks`,
		"execution_webhook_events":  `SELECT COUNT(*) FROM execution_webhook_events`,
		"workflow_runs":             `SELECT COUNT(*) FROM workflow_runs`,
		"workflow_executions":       `SELECT COUNT(*) FROM workflow_executions`,
		"execution_vcs":             `SELECT COUNT(*) FROM execution_vcs`,
		"workflow_vcs":              `SELECT COUNT(*) FROM workflow_vcs`,
		"workflows":                 `SELECT COUNT(*) FROM workflows`,
	} {
		var count int
		require.NoError(t, rawDB.QueryRow(query).Scan(&count))
		require.Zero(t, count, table)
	}

	var workflowExecutionEventCount int
	require.NoError(t, rawDB.QueryRow(`SELECT COUNT(*) FROM workflow_execution_events`).Scan(&workflowExecutionEventCount))
	require.Equal(t, 1, workflowExecutionEventCount)

	require.Nil(t, setToSlice(map[string]struct{}{}))
	require.Empty(t, makePlaceholders(0))
	require.Empty(t, stringsToInterfaces(nil))
}

func TestDIDStorageValidationBranches(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC()

	t.Run("validates agentfield server DID input", func(t *testing.T) {
		err := ls.StoreAgentFieldServerDID(ctx, "", "did:root:missing-id", []byte("seed"), now, now)
		require.EqualError(t, err, "validation failed for agentfield_server_id='': af server ID cannot be empty (context: StoreAgentFieldServerDID)")

		err = ls.StoreAgentFieldServerDID(ctx, "srv-missing-root", "", []byte("seed"), now, now)
		require.EqualError(t, err, "validation failed for root_did='': root DID cannot be empty (context: StoreAgentFieldServerDID)")

		err = ls.StoreAgentFieldServerDID(ctx, "srv-missing-seed", "did:root:missing-seed", nil, now, now)
		require.EqualError(t, err, "validation failed for master_seed='<encrypted>': master seed cannot be empty (context: StoreAgentFieldServerDID)")
	})

	t.Run("rejects missing server and duplicate agent did entries", func(t *testing.T) {
		err := ls.StoreAgentDIDWithComponents(ctx, "agent-missing-server", "did:agent:missing", "srv-does-not-exist", `{"kid":"missing"}`, 1, nil)
		require.EqualError(t, err, "pre-storage validation failed: foreign key constraint violation in agent_dids.agentfield_server_id: referenced did_registry 'srv-does-not-exist' does not exist (operation: INSERT)")

		require.NoError(t, ls.StoreAgentFieldServerDID(ctx, "srv-dup-agent", "did:root:dup-agent", []byte("seed"), now, now))
		require.NoError(t, ls.StoreAgentDIDWithComponents(ctx, "agent-dup", "did:agent:dup", "srv-dup-agent", `{"kid":"dup-1"}`, 2, nil))

		err = ls.StoreAgentDIDWithComponents(ctx, "agent-dup-2", "did:agent:dup", "srv-dup-agent", `{"kid":"dup-2"}`, 3, nil)
		require.EqualError(t, err, "duplicate agent DID detected: agent:agent-dup-2@srv-dup-agent already exists")
	})
}
