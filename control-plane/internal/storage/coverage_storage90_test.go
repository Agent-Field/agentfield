package storage

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
	_ "modernc.org/sqlite"
)

func TestStorage90QueryWorkflowExecutionsAdditionalBranches(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	now := time.Now().UTC().Truncate(time.Second)
	runID := "run-storage90"
	sessionID := "session-storage90"
	actorID := "actor-storage90"
	parentExecutionID := "parent-storage90"
	approvalRequestID := "approval-storage90"
	approvalRequestURL := "https://example.test/approval/storage90"
	approvalStatus := "pending"
	approvalResponse := "waiting"
	approvalCallbackURL := "https://example.test/callback/storage90"
	pendingTerminal := "cancelled"
	statusReason := "waiting_for_approval"
	leaseOwner := "worker-storage90"
	errorMessage := "still waiting"
	workflowName := "Approval Workflow"
	workflowTags := []string{"ops", "review"}
	approvalRequestedAt := now.Add(-2 * time.Minute)
	approvalRespondedAt := now.Add(-1 * time.Minute)
	approvalExpiresAt := now.Add(10 * time.Minute)
	leaseExpiresAt := now.Add(5 * time.Minute)
	durationMS := int64(2500)
	startedAt := now.Add(-3 * time.Minute)
	completedAt := now.Add(-30 * time.Second)

	require.NoError(t, ls.StoreWorkflowExecution(ctx, &types.WorkflowExecution{
		WorkflowID:             "wf-storage90",
		ExecutionID:            "exec-storage90",
		AgentFieldRequestID:    "req-storage90",
		RunID:                  &runID,
		SessionID:              &sessionID,
		ActorID:                &actorID,
		AgentNodeID:            "agent-storage90",
		ParentExecutionID:      &parentExecutionID,
		ReasonerID:             "reasoner-storage90",
		Status:                 "waiting",
		StartedAt:              startedAt,
		CompletedAt:            &completedAt,
		DurationMS:             &durationMS,
		WorkflowName:           &workflowName,
		WorkflowTags:           workflowTags,
		PendingTerminalStatus:  &pendingTerminal,
		StatusReason:           &statusReason,
		LeaseOwner:             &leaseOwner,
		LeaseExpiresAt:         &leaseExpiresAt,
		ErrorMessage:           &errorMessage,
		ApprovalRequestID:      &approvalRequestID,
		ApprovalRequestURL:     &approvalRequestURL,
		ApprovalStatus:         &approvalStatus,
		ApprovalResponse:       &approvalResponse,
		ApprovalRequestedAt:    &approvalRequestedAt,
		ApprovalRespondedAt:    &approvalRespondedAt,
		ApprovalCallbackURL:    &approvalCallbackURL,
		ApprovalExpiresAt:      &approvalExpiresAt,
		InputData:              []byte(`{"step":"input"}`),
		OutputData:             []byte(`{"step":"output"}`),
		CreatedAt:              now,
		UpdatedAt:              now,
	}))

	t.Run("like search path and nullable fields are hydrated", func(t *testing.T) {
		ls.ftsEnabled = false
		search := "Approval*"
		sortBy := "status"
		sortOrder := "ASC"

		results, err := ls.QueryWorkflowExecutions(ctx, types.WorkflowExecutionFilters{
			ParentExecutionID: &parentExecutionID,
			ApprovalRequestID: &approvalRequestID,
			Search:            &search,
			SortBy:            &sortBy,
			SortOrder:         &sortOrder,
		})
		require.NoError(t, err)
		require.Len(t, results, 1)
		require.Equal(t, "exec-storage90", results[0].ExecutionID)
		require.Equal(t, workflowTags, results[0].WorkflowTags)
		require.Equal(t, `{"step":"input"}`, string(results[0].InputData))
		require.Equal(t, `{"step":"output"}`, string(results[0].OutputData))
		require.Equal(t, pendingTerminal, *results[0].PendingTerminalStatus)
		require.Equal(t, statusReason, *results[0].StatusReason)
		require.Equal(t, leaseOwner, *results[0].LeaseOwner)
		require.Equal(t, approvalRequestID, *results[0].ApprovalRequestID)
		require.Equal(t, approvalRequestURL, *results[0].ApprovalRequestURL)
		require.Equal(t, approvalStatus, *results[0].ApprovalStatus)
		require.Equal(t, approvalResponse, *results[0].ApprovalResponse)
		require.Equal(t, approvalCallbackURL, *results[0].ApprovalCallbackURL)
		require.WithinDuration(t, leaseExpiresAt, *results[0].LeaseExpiresAt, time.Second)
		require.WithinDuration(t, approvalRequestedAt, *results[0].ApprovalRequestedAt, time.Second)
		require.WithinDuration(t, approvalRespondedAt, *results[0].ApprovalRespondedAt, time.Second)
		require.WithinDuration(t, approvalExpiresAt, *results[0].ApprovalExpiresAt, time.Second)
	})

	t.Run("invalid workflow tags surface unmarshal error", func(t *testing.T) {
		_, err := ls.db.ExecContext(ctx, `UPDATE workflow_executions SET workflow_tags = ? WHERE execution_id = ?`, "{", "exec-storage90")
		require.NoError(t, err)

		_, err = ls.QueryWorkflowExecutions(ctx, types.WorkflowExecutionFilters{})
		require.EqualError(t, err, "failed to unmarshal workflow tags: unexpected end of JSON input")

		_, err = ls.db.ExecContext(ctx, `UPDATE workflow_executions SET workflow_tags = ? WHERE execution_id = ?`, `["ops","review"]`, "exec-storage90")
		require.NoError(t, err)
	})
}

func TestStorage90StoreAgentDIDWithComponentsDuplicateRollback(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC()
	require.NoError(t, ls.StoreAgentFieldServerDID(ctx, "srv-storage90", "did:root:storage90", []byte("seed"), now, now))

	initial := []ComponentDIDRequest{
		{ComponentDID: "did:component:storage90:seed", ComponentType: "reasoner", ComponentName: "seed", PublicKeyJWK: `{"kid":"seed"}`, DerivationIndex: 1},
	}
	require.NoError(t, ls.StoreAgentDIDWithComponents(ctx, "agent-seed", "did:agent:seed", "srv-storage90", `{"kid":"agent-seed"}`, 1, initial))

	duplicateComponents := []ComponentDIDRequest{
		{ComponentDID: "did:component:storage90:seed", ComponentType: "reasoner", ComponentName: "dup", PublicKeyJWK: `{"kid":"dup"}`, DerivationIndex: 2},
	}

	err := ls.StoreAgentDIDWithComponents(ctx, "agent-rollback", "did:agent:rollback", "srv-storage90", `{"kid":"agent-rollback"}`, 2, duplicateComponents)
	require.EqualError(t, err, "duplicate component DID detected: component:reasoner/dup@did:agent:rollback already exists")

	agentInfo, err := ls.GetAgentDID(ctx, "agent-rollback")
	require.EqualError(t, err, "agent DID for agent-rollback not found")
	require.Nil(t, agentInfo)

	components, err := ls.ListComponentDIDs(ctx, "did:agent:rollback")
	require.NoError(t, err)
	require.Len(t, components, 0)
}

func TestStorage90PostgresHelperBranches(t *testing.T) {
	rawDB := openSQLiteNowDB(t)

	db := newSQLDatabase(rawDB, "postgres")
	ls := &LocalStorage{
		db:           db,
		mode:         "postgres",
		vectorConfig: VectorStoreConfig{},
		vectorMetric: VectorDistanceCosine,
	}
	ctx := context.Background()

	tableDDLs := []string{
		`CREATE TABLE agent_configurations (agent_id TEXT, package_id TEXT)`,
		`CREATE TABLE workflow_runs (status TEXT, root_workflow_id TEXT, created_at TIMESTAMP, updated_at TIMESTAMP)`,
		`CREATE TABLE workflow_steps (run_id TEXT, execution_id TEXT, status TEXT, not_before TIMESTAMP, parent_step_id TEXT, created_at TIMESTAMP, updated_at TIMESTAMP, agent_node_id TEXT, priority INTEGER)`,
		`CREATE TABLE workflow_executions (workflow_id TEXT, execution_id TEXT, session_id TEXT, actor_id TEXT, agent_node_id TEXT, started_at TIMESTAMP, parent_execution_id TEXT, parent_workflow_id TEXT, root_workflow_id TEXT, status TEXT)`,
		`CREATE TABLE agent_nodes (id TEXT, group_id TEXT)`,
	}
	for _, ddl := range tableDDLs {
		_, err := rawDB.Exec(ddl)
		require.NoError(t, err)
	}

	t.Run("schema helpers hit success and error branches", func(t *testing.T) {
		err := ls.ensurePostgresEventSchema(ctx)
		require.Error(t, err)

		err = ls.ensurePostgresLockSchema(ctx)
		require.Error(t, err)

		require.NoError(t, ls.ensurePostgresIndexes(ctx))

		err = ls.ensurePostgresKeyValueSchema(ctx)
		require.Error(t, err)

		err = ls.ensurePostgresWorkflowFTS(ctx)
		require.Error(t, err)

		err = ls.ensurePostgresVectorSchema(ctx)
		require.Error(t, err)

		err = ls.ensureVectorSchema(ctx)
		require.Error(t, err)
	})

	t.Run("migration helper runs on sqlite with now function", func(t *testing.T) {
		_, err := rawDB.Exec(`INSERT INTO agent_nodes (id, group_id) VALUES ('agent-1', '')`)
		require.NoError(t, err)

		err = ls.runPostgresMigrations(ctx)
		require.Error(t, err)
		require.Contains(t, err.Error(), "failed to create schema_migrations table")
	})

	t.Run("postgres composite pk migration hits sqlite-specific failure after setup", func(t *testing.T) {
		_, err := rawDB.Exec(`ATTACH DATABASE ':memory:' AS information_schema`)
		require.NoError(t, err)
		t.Cleanup(func() { _, _ = rawDB.Exec(`DETACH DATABASE information_schema`) })

		_, err = rawDB.Exec(`CREATE TABLE information_schema.tables (table_schema TEXT, table_name TEXT)`)
		require.NoError(t, err)
		_, err = rawDB.Exec(`CREATE TABLE information_schema.columns (table_name TEXT, column_name TEXT)`)
		require.NoError(t, err)
		_, err = rawDB.Exec(`INSERT INTO information_schema.tables (table_schema, table_name) VALUES ('public', 'agent_nodes')`)
		require.NoError(t, err)

		err = ls.migrateAgentNodesCompositePKPostgres(ctx)
		require.Error(t, err)
		require.Contains(t, err.Error(), "postgres ensure column failed")
	})
}

func TestStorage90InitializePostgresBranches(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	t.Run("validates missing config before connecting", func(t *testing.T) {
		ls := &LocalStorage{}
		require.EqualError(t, ls.initializePostgres(ctx), "postgres configuration requires either a connection string or host information")

		ls.postgresConfig = PostgresStorageConfig{Host: "127.0.0.1"}
		require.EqualError(t, ls.initializePostgres(ctx), "postgres configuration requires a user when host is specified")

		ls.postgresConfig = PostgresStorageConfig{Host: "127.0.0.1", User: "agentfield"}
		require.EqualError(t, ls.initializePostgres(ctx), "postgres configuration requires a database when host is specified")
	})

	t.Run("dsn creation and ping failures are wrapped", func(t *testing.T) {
		ls := &LocalStorage{
			postgresConfig: PostgresStorageConfig{
				Host:     "127.0.0.1",
				Port:     1,
				User:     "agentfield",
				Password: "secret",
				Database: "agentfield",
				SSLMode:  "disable",
			},
		}

		err := ls.initializePostgres(ctx)
		require.Error(t, err)
		require.Contains(t, err.Error(), "failed to ping PostgreSQL database")
		require.Contains(t, ls.postgresConfig.DSN, "postgres://agentfield:secret@127.0.0.1:1/agentfield?sslmode=disable")
	})

	t.Run("database creation helper validates dsn parsing before connecting", func(t *testing.T) {
		ls := &LocalStorage{}

		err := ls.ensurePostgresDatabaseExists(ctx, PostgresStorageConfig{})
		require.EqualError(t, err, "postgres DSN is required to create database")

		err = ls.ensurePostgresDatabaseExists(ctx, PostgresStorageConfig{DSN: "://bad"})
		require.Error(t, err)
		require.Contains(t, err.Error(), "failed to parse postgres DSN")

		err = ls.ensurePostgresDatabaseExists(ctx, PostgresStorageConfig{DSN: "postgres://agentfield@127.0.0.1"})
		require.EqualError(t, err, "postgres DSN must specify a database name")

		err = ls.ensurePostgresDatabaseExists(ctx, PostgresStorageConfig{DSN: "postgres://agentfield@127.0.0.1:1/agentfield", AdminDatabase: "postgres"})
		require.Error(t, err)
		require.Contains(t, err.Error(), "failed to ping postgres admin database")
	})
}

func TestStorage90SQLiteAndWorkflowVCBranches(t *testing.T) {
	t.Run("initialize sqlite validates paths and env fallbacks", func(t *testing.T) {
		ls := NewLocalStorage(LocalStorageConfig{})
		require.EqualError(t, ls.initializeSQLite(context.Background()), "database path is empty - please set a valid database path in configuration")

		tempDir := t.TempDir()
		wd, err := os.Getwd()
		require.NoError(t, err)
		require.NoError(t, os.Chdir(tempDir))
		t.Cleanup(func() { _ = os.Chdir(wd) })

		t.Setenv("AGENTFIELD_SQLITE_BUSY_TIMEOUT_MS", "bad")
		t.Setenv("AGENTFIELD_SQLITE_MAX_OPEN_CONNS", "0")
		t.Setenv("AGENTFIELD_SQLITE_MAX_IDLE_CONNS", "-1")

		ls = NewLocalStorage(LocalStorageConfig{
			DatabasePath: filepath.Join("relative", "agentfield.db"),
			KVStorePath:  filepath.Join(tempDir, "agentfield.bolt"),
		})
		ls.mode = "local"
		require.NoError(t, ls.initializeSQLite(context.Background()))
		t.Cleanup(func() { _ = ls.Close(context.Background()) })
		require.NotNil(t, ls.db)
		require.FileExists(t, filepath.Join(tempDir, "relative", "agentfield.db"))
	})

	t.Run("workflow vc lookup and agent post processing error branches", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		now := time.Now().UTC().Truncate(time.Second)

		cancelled, cancel := context.WithCancel(ctx)
		cancel()
		_, err := ls.GetWorkflowVC(cancelled, "wvc-cancelled")
		require.EqualError(t, err, "context cancelled during get workflow VC: context canceled")

		_, err = ls.GetWorkflowVC(ctx, "wvc-missing")
		require.EqualError(t, err, "workflow VC wvc-missing not found")

		require.NoError(t, ls.StoreWorkflowVC(ctx, "wvc-bad-json", "wf-bad-json", "session-bad-json", []string{"vc-1"}, "running", &now, nil, 1, 0, "", 0))
		_, err = ls.db.ExecContext(ctx, `UPDATE workflow_vcs SET component_vc_ids = ? WHERE workflow_vc_id = ?`, "{", "wvc-bad-json")
		require.NoError(t, err)
		_, err = ls.GetWorkflowVC(ctx, "wvc-bad-json")
		require.EqualError(t, err, "failed to unmarshal component VC IDs: unexpected end of JSON input")

		serverless := &types.AgentNode{
			BaseURL: "https://agent.example/",
			Metadata: types.AgentMetadata{
				Custom: map[string]interface{}{"serverless": true},
			},
		}
		ls.postProcessAgentNode(serverless, string(types.HealthStatusActive), string(types.AgentStatusReady), sql.NullString{},
			nil, nil, nil, nil, nil, nil, nil)
		require.Equal(t, "serverless", serverless.DeploymentType)
		require.NotNil(t, serverless.InvocationURL)
		require.Equal(t, "https://agent.example/execute", *serverless.InvocationURL)

		longRunning := &types.AgentNode{}
		ls.postProcessAgentNode(longRunning, string(types.HealthStatusInactive), string(types.AgentStatusStarting), sql.NullString{},
			nil, nil, nil, nil, nil, nil, nil)
		require.Equal(t, "long_running", longRunning.DeploymentType)
		require.Equal(t, types.HealthStatusInactive, longRunning.HealthStatus)
		require.Equal(t, types.AgentStatusStarting, longRunning.LifecycleStatus)
	})
}
