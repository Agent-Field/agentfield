package storage

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/stretchr/testify/require"
)

func TestPostgresInitializationValidationCoverage(t *testing.T) {
	t.Run("initialize postgres validates required fields", func(t *testing.T) {
		ls := &LocalStorage{}

		err := ls.initializePostgres(context.Background())
		require.EqualError(t, err, "postgres configuration requires either a connection string or host information")

		ls.postgresConfig = PostgresStorageConfig{Host: "127.0.0.1"}
		err = ls.initializePostgres(context.Background())
		require.EqualError(t, err, "postgres configuration requires a user when host is specified")

		ls.postgresConfig = PostgresStorageConfig{Host: "127.0.0.1", User: "agentfield"}
		err = ls.initializePostgres(context.Background())
		require.EqualError(t, err, "postgres configuration requires a database when host is specified")
	})

	t.Run("initialize postgres builds dsn before ping failure", func(t *testing.T) {
		ls := &LocalStorage{
			postgresConfig: PostgresStorageConfig{
				Host:     "127.0.0.1",
				Port:     1,
				User:     "agentfield",
				Password: "secret",
				Database: "control",
			},
		}

		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		defer cancel()

		err := ls.initializePostgres(ctx)
		require.Error(t, err)
		require.Contains(t, err.Error(), "failed to ping PostgreSQL database")
		require.Equal(t, "postgres://agentfield:secret@127.0.0.1:1/control?sslmode=disable", ls.postgresConfig.DSN)
		require.Equal(t, ls.postgresConfig.DSN, ls.postgresConfig.URL)
	})

	t.Run("ensure postgres database exists validates dsn branches", func(t *testing.T) {
		ls := &LocalStorage{}

		require.EqualError(t, ls.ensurePostgresDatabaseExists(context.Background(), PostgresStorageConfig{}), "postgres DSN is required to create database")

		err := ls.ensurePostgresDatabaseExists(context.Background(), PostgresStorageConfig{DSN: "://bad-dsn"})
		require.EqualError(t, err, "failed to parse postgres DSN: parse \"://bad-dsn\": missing protocol scheme")

		err = ls.ensurePostgresDatabaseExists(context.Background(), PostgresStorageConfig{DSN: "postgres://agentfield@127.0.0.1"})
		require.EqualError(t, err, "postgres DSN must specify a database name")

		ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
		defer cancel()
		err = ls.ensurePostgresDatabaseExists(ctx, PostgresStorageConfig{
			DSN:           "postgres://agentfield:secret@127.0.0.1:1/control?sslmode=disable",
			AdminDatabase: "postgres",
		})
		require.Error(t, err)
		require.Contains(t, err.Error(), "failed to ping postgres admin database")
	})

	t.Run("postgres helper predicates and quoting cover both branches", func(t *testing.T) {
		require.False(t, isPostgresDatabaseMissingError(nil))
		require.True(t, isPostgresDatabaseMissingError(&pgconn.PgError{Code: "3D000"}))
		require.True(t, isPostgresDatabaseMissingError(errors.New("database does not exist")))
		require.False(t, isPostgresDatabaseMissingError(errors.New("other")))

		require.False(t, isPostgresDatabaseAlreadyExistsError(nil))
		require.True(t, isPostgresDatabaseAlreadyExistsError(&pgconn.PgError{Code: "42P04"}))
		require.True(t, isPostgresDatabaseAlreadyExistsError(errors.New("already exists")))
		require.False(t, isPostgresDatabaseAlreadyExistsError(errors.New("other")))

		require.Equal(t, `"simple"`, quotePostgresIdentifier("simple"))
		require.Equal(t, `"a""b"`, quotePostgresIdentifier(`a"b`))
	})
}

func TestAgentRetrievalErrorBranchesAndDefaults(t *testing.T) {
	tests := []struct {
		name    string
		column  string
		value   string
		wantErr string
	}{
		{name: "reasoners", column: "reasoners", value: "{", wantErr: "failed to unmarshal agent reasoners"},
		{name: "skills", column: "skills", value: "{", wantErr: "failed to unmarshal agent skills"},
		{name: "communication_config", column: "communication_config", value: "{", wantErr: "failed to unmarshal agent communication config"},
		{name: "features", column: "features", value: "{", wantErr: "failed to unmarshal agent features"},
		{name: "metadata", column: "metadata", value: "{", wantErr: "failed to unmarshal agent metadata"},
		{name: "proposed_tags", column: "proposed_tags", value: "{", wantErr: "failed to unmarshal agent proposed tags"},
		{name: "approved_tags", column: "approved_tags", value: "{", wantErr: "failed to unmarshal agent approved tags"},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ls, ctx := setupLocalStorage(t)
			agent := makeTestAgent("agent-"+tc.name, "")
			require.NoError(t, ls.RegisterAgent(ctx, agent))

			_, err := ls.db.ExecContext(ctx, "UPDATE agent_nodes SET "+tc.column+" = ? WHERE id = ? AND version = ''", tc.value, agent.ID)
			require.NoError(t, err)

			_, err = ls.GetAgent(ctx, agent.ID)
			require.ErrorContains(t, err, tc.wantErr)
		})
	}

	t.Run("deployment type defaults and reconstructed invocation url", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		agent := makeTestAgent("agent-serverless", "")
		agent.BaseURL = "https://agent.example/base/"
		agent.DeploymentType = "long_running"
		require.NoError(t, ls.RegisterAgent(ctx, agent))

		_, err := ls.db.ExecContext(ctx, `
			UPDATE agent_nodes
			SET deployment_type = '', invocation_url = NULL, metadata = ?, proposed_tags = '[]', approved_tags = '[]'
			WHERE id = ? AND version = ''
		`, `{"custom":{"serverless":"true"}}`, agent.ID)
		require.NoError(t, err)

		got, err := ls.GetAgent(ctx, agent.ID)
		require.NoError(t, err)
		require.Equal(t, "serverless", got.DeploymentType)
		require.NotNil(t, got.InvocationURL)
		require.Equal(t, "https://agent.example/base/execute", *got.InvocationURL)
	})

	t.Run("get agent version reports missing row", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		_, err := ls.GetAgentVersion(ctx, "missing-agent", "v0")
		require.EqualError(t, err, "agent node with ID '' version '' not found")
	})

	t.Run("atomic health update reports both zero-row branches", func(t *testing.T) {
		ls, ctx := setupLocalStorage(t)
		agent := makeTestAgent("agent-health", "")
		require.NoError(t, ls.RegisterAgent(ctx, agent))

		err := ls.UpdateAgentHealthAtomic(ctx, "missing-agent", types.HealthStatusActive, nil)
		require.EqualError(t, err, "agent node with ID 'missing-agent' not found")

		expected := time.Now().UTC()
		err = ls.UpdateAgentHealthAtomic(ctx, agent.ID, types.HealthStatusInactive, &expected)
		require.EqualError(t, err, "no rows updated for agent ID 'agent-health' - possible concurrent modification or node not found")
	})
}

func TestAgentConfigurationHappyPathCoverage(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC()

	pkg := testAgentPackage(now)
	require.NoError(t, ls.StoreAgentPackage(ctx, pkg))

	_, err := ls.db.ExecContext(ctx, `CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_config_agent_package_manual ON agent_configurations(agent_id, package_id)`)
	require.NoError(t, err)

	cfg := testAgentConfiguration(now)
	require.NoError(t, ls.StoreAgentConfiguration(ctx, cfg))
	require.NotZero(t, cfg.ID)

	stored, err := ls.GetAgentConfiguration(ctx, cfg.AgentID, cfg.PackageID)
	require.NoError(t, err)
	require.Equal(t, 1, stored.Version)
	require.Equal(t, "abc", stored.Configuration["token"])

	cfg.Configuration["retries"] = 7
	cfg.Status = types.ConfigurationStatusActive
	require.NoError(t, ls.StoreAgentConfiguration(ctx, cfg))

	updated, err := ls.GetAgentConfiguration(ctx, cfg.AgentID, cfg.PackageID)
	require.NoError(t, err)
	require.Equal(t, 2, updated.Version)
	require.Equal(t, types.ConfigurationStatusActive, updated.Status)
	require.Equal(t, float64(7), updated.Configuration["retries"])

	start := now.Add(-time.Minute)
	end := now.Add(time.Minute)
	queryResults, err := ls.QueryAgentConfigurations(ctx, types.ConfigurationFilters{
		AgentID:   &cfg.AgentID,
		PackageID: &cfg.PackageID,
		StartTime: &start,
		EndTime:   &end,
		Limit:     1,
		Offset:    0,
	})
	require.NoError(t, err)
	require.Len(t, queryResults, 1)

	_, err = ls.db.ExecContext(ctx, `UPDATE agent_configurations SET configuration = ? WHERE agent_id = ? AND package_id = ?`, "{", cfg.AgentID, cfg.PackageID)
	require.NoError(t, err)
	_, err = ls.QueryAgentConfigurations(ctx, types.ConfigurationFilters{AgentID: &cfg.AgentID})
	require.ErrorContains(t, err, "failed to unmarshal configuration")
}

func TestWorkflowVCAndDIDAdditionalBranches(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC().Truncate(time.Second)

	require.NoError(t, ls.StoreExecutionVC(ctx, "vc-extra", "exec-extra", "wf-extra", "session-extra", "issuer", "target", "caller", "in", "out", string(types.ExecutionStatusSucceeded), []byte(`{"ok":true}`), "sig", "", 0))
	_, err := ls.ListWorkflowVCStatusSummaries(ctx, []string{"wf-extra"})
	require.ErrorContains(t, err, "failed to scan workflow VC status summary")

	require.NoError(t, ls.StoreWorkflowVC(ctx, "wvc-extra", "wf-extra", "session-extra", []string{"vc-extra"}, "running", &now, nil, 1, 0, "", 0))
	_, err = ls.db.ExecContext(ctx, `UPDATE workflow_vcs SET component_vc_ids = ? WHERE workflow_vc_id = ?`, "{", "wvc-extra")
	require.NoError(t, err)

	_, err = ls.GetWorkflowVC(ctx, "wvc-extra")
	require.ErrorContains(t, err, "failed to unmarshal component VC IDs")

	_, err = ls.ListWorkflowVCs(ctx, "wf-extra")
	require.ErrorContains(t, err, "failed to unmarshal component VC IDs")

	_, _, err = ls.GetFullExecutionVC("missing-vc")
	require.EqualError(t, err, "execution VC missing-vc not found")

	record := &types.DIDDocumentRecord{
		DID:          "did:web:agentfield:test",
		AgentID:      "agent-did-extra",
		DIDDocument:  json.RawMessage(`{"id":"did:web:agentfield:test"}`),
		PublicKeyJWK: `{"kty":"OKP"}`,
		CreatedAt:    now,
		UpdatedAt:    now,
	}
	require.NoError(t, ls.StoreDIDDocument(ctx, record))

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	require.EqualError(t, ls.StoreDIDDocument(cancelled, record), "context cancelled during store DID document: context canceled")
	_, err = ls.GetDIDDocument(cancelled, record.DID)
	require.EqualError(t, err, "context cancelled during get DID document: context canceled")
	_, err = ls.GetDIDDocumentByAgentID(cancelled, record.AgentID)
	require.EqualError(t, err, "context cancelled during get DID document by agent ID: context canceled")
	require.EqualError(t, ls.RevokeDIDDocument(cancelled, record.DID), "context cancelled during revoke DID document: context canceled")
	_, err = ls.ListDIDDocuments(cancelled)
	require.EqualError(t, err, "context cancelled during list DID documents: context canceled")
}

func TestCountHelpersReturnZeroOnQueryErrors(t *testing.T) {
	rawDB, err := sql.Open("sqlite3", ":memory:")
	require.NoError(t, err)
	require.NoError(t, rawDB.Close())

	ls := &LocalStorage{db: newSQLDatabase(rawDB, "local"), mode: "local"}
	ctx := context.Background()

	require.Zero(t, ls.countWorkflowRuns(ctx, "wf-primary", []string{"wf-1"}, []string{"run-1"}))
	require.Zero(t, ls.countExecutions(ctx, []string{"run-1"}))
	require.Zero(t, ls.countExecutionWebhooks(ctx, []string{"run-1"}))
	require.Zero(t, ls.countExecutionWebhookEvents(ctx, []string{"run-1"}))
	require.Zero(t, ls.countExecutionVCs(ctx, []string{"wf-1"}))
	require.Zero(t, ls.countWorkflowVCs(ctx, []string{"wf-1"}))
	require.Zero(t, ls.countWorkflowExecutions(ctx, []string{"wf-1"}, []string{"run-1"}))
	require.Zero(t, ls.countWorkflowExecutionEvents(ctx, []string{"wf-1"}, []string{"run-1"}))
	require.Zero(t, ls.countWorkflows(ctx, []string{"wf-1"}))
}

func TestCreateSchemaNilKVStoreBranch(t *testing.T) {
	ls := newSchemaTestStorage(t)
	ls.kvStore = nil

	require.Panics(t, func() {
		_ = ls.createSchema(context.Background())
	})
}

func TestWorkflowSessionAndExecutionQueryBranches(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC().Truncate(time.Second)

	workflowName := "Workflow Alpha"
	workflow := &types.Workflow{
		WorkflowID:           "wf-query-extra",
		WorkflowName:         &workflowName,
		WorkflowTags:         []string{"alpha", "beta"},
		SessionID:            ptrString("session-extra"),
		ActorID:              ptrString("actor-extra"),
		RootWorkflowID:       ptrString("wf-query-extra"),
		WorkflowDepth:        1,
		TotalExecutions:      2,
		SuccessfulExecutions: 1,
		FailedExecutions:     1,
		TotalDurationMS:      120,
		Status:               "running",
		StartedAt:            now,
		CreatedAt:            now,
		UpdatedAt:            now,
	}
	require.NoError(t, ls.CreateOrUpdateWorkflow(ctx, workflow))

	workflow.Status = "succeeded"
	workflow.TotalExecutions = 3
	workflow.UpdatedAt = now.Add(time.Minute)
	require.NoError(t, ls.CreateOrUpdateWorkflow(ctx, workflow))

	gotWorkflow, err := ls.GetWorkflow(ctx, workflow.WorkflowID)
	require.NoError(t, err)
	require.Equal(t, "succeeded", gotWorkflow.Status)
	require.Equal(t, []string{"alpha", "beta"}, gotWorkflow.WorkflowTags)

	sortBy := "status"
	sortOrder := "asc"
	start := now.Add(-time.Minute)
	end := now.Add(time.Minute)
	workflows, err := ls.QueryWorkflows(ctx, types.WorkflowFilters{
		SessionID: workflow.SessionID,
		ActorID:   workflow.ActorID,
		Status:    ptrString("succeeded"),
		StartTime: &start,
		EndTime:   &end,
		SortBy:    &sortBy,
		SortOrder: &sortOrder,
		Limit:     1,
		Offset:    0,
	})
	require.NoError(t, err)
	require.Len(t, workflows, 1)

	_, err = ls.db.ExecContext(ctx, `UPDATE workflows SET workflow_tags = ? WHERE workflow_id = ?`, "{", workflow.WorkflowID)
	require.NoError(t, err)
	_, err = ls.GetWorkflow(ctx, workflow.WorkflowID)
	require.ErrorContains(t, err, "failed to unmarshal workflow tags")
	_, err = ls.QueryWorkflows(ctx, types.WorkflowFilters{})
	require.ErrorContains(t, err, "failed to unmarshal workflow tags")

	sessionName := "Session Extra"
	session := &types.Session{
		SessionID:       "session-extra",
		ActorID:         ptrString("actor-extra"),
		SessionName:     &sessionName,
		RootSessionID:   ptrString("session-extra"),
		TotalWorkflows:  1,
		TotalExecutions: 2,
		TotalDurationMS: 120,
		StartedAt:       now,
		LastActivityAt:  now,
		CreatedAt:       now,
		UpdatedAt:       now,
	}
	require.NoError(t, ls.CreateOrUpdateSession(ctx, session))

	session.TotalExecutions = 3
	session.UpdatedAt = now.Add(time.Minute)
	require.NoError(t, ls.CreateOrUpdateSession(ctx, session))

	gotSession, err := ls.GetSession(ctx, session.SessionID)
	require.NoError(t, err)
	require.Equal(t, 3, gotSession.TotalExecutions)

	sessions, err := ls.QuerySessions(ctx, types.SessionFilters{
		ActorID:   session.ActorID,
		StartTime: &start,
		EndTime:   &end,
		Limit:     1,
		Offset:    0,
	})
	require.NoError(t, err)
	require.Len(t, sessions, 1)

	userID := "user-extra"
	teamID := "team-extra"
	execOne := &types.AgentExecution{
		WorkflowID:   workflow.WorkflowID,
		SessionID:    &session.SessionID,
		AgentNodeID:  "agent-extra",
		ReasonerID:   "reasoner-extra",
		InputData:    json.RawMessage(`{"input":1}`),
		OutputData:   json.RawMessage(`{"output":1}`),
		InputSize:    1,
		OutputSize:   1,
		DurationMS:   10,
		Status:       "running",
		UserID:       &userID,
		NodeID:       &teamID,
		Metadata:     types.ExecutionMetadata{Custom: map[string]interface{}{"trace": "one"}},
		CreatedAt:    now,
	}
	require.NoError(t, ls.StoreExecution(ctx, execOne))
	require.NotZero(t, execOne.ID)

	execTwo := &types.AgentExecution{
		WorkflowID:   "wf-other-extra",
		AgentNodeID:  "agent-other",
		ReasonerID:   "reasoner-other",
		InputData:    json.RawMessage(`{"input":2}`),
		OutputData:   json.RawMessage(`{"output":2}`),
		InputSize:    2,
		OutputSize:   2,
		DurationMS:   20,
		Status:       "failed",
		CreatedAt:    now.Add(time.Minute),
	}
	require.NoError(t, ls.StoreExecution(ctx, execTwo))

	gotExec, err := ls.GetExecution(ctx, execOne.ID)
	require.NoError(t, err)
	require.Equal(t, teamID, *gotExec.NodeID)
	require.Equal(t, "one", gotExec.Metadata.Custom["trace"])

	executions, err := ls.QueryExecutions(ctx, types.ExecutionFilters{
		WorkflowID:  &workflow.WorkflowID,
		SessionID:   &session.SessionID,
		AgentNodeID: ptrString("agent-extra"),
		ReasonerID:  ptrString("reasoner-extra"),
		Status:      ptrString("running"),
		UserID:      &userID,
		TeamID:      &teamID,
		StartTime:   &start,
		EndTime:     &end,
		Limit:       1,
		Offset:      0,
	})
	require.NoError(t, err)
	require.Len(t, executions, 1)

	_, err = ls.db.ExecContext(ctx, `UPDATE agent_executions SET metadata = ? WHERE id = ?`, "{", execOne.ID)
	require.NoError(t, err)
	_, err = ls.GetExecution(ctx, execOne.ID)
	require.ErrorContains(t, err, "failed to unmarshal execution metadata")
	_, err = ls.QueryExecutions(ctx, types.ExecutionFilters{})
	require.ErrorContains(t, err, "failed to unmarshal execution metadata")
}

func TestWorkflowExecutionAdditionalBranches(t *testing.T) {
	ls, ctx := setupLocalStorage(t)
	now := time.Now().UTC().Truncate(time.Second)
	runID := "run-dag-extra"

	require.NoError(t, ls.StoreWorkflowRun(ctx, &types.WorkflowRun{
		RunID:          runID,
		RootWorkflowID: "wf-dag-extra",
		Status:         string(types.ExecutionStatusRunning),
		CreatedAt:      now,
		UpdatedAt:      now,
	}))

	root := &types.WorkflowExecution{
		WorkflowID:          "wf-dag-extra",
		ExecutionID:         "exec-root-extra",
		AgentFieldRequestID: "req-root-extra",
		RunID:               &runID,
		SessionID:           ptrString("session-dag"),
		ActorID:             ptrString("actor-dag"),
		AgentNodeID:         "agent-dag",
		ReasonerID:          "reasoner-dag",
		InputData:           json.RawMessage(`{"in":1}`),
		OutputData:          json.RawMessage(`{"out":1}`),
		InputSize:           1,
		OutputSize:          1,
		WorkflowName:        ptrString("WF DAG"),
		WorkflowTags:        []string{"dag"},
		Status:              string(types.ExecutionStatusRunning),
		StartedAt:           now,
		CreatedAt:           now,
		UpdatedAt:           now,
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, root))

	parentID := root.ExecutionID
	child := &types.WorkflowExecution{
		WorkflowID:          "wf-dag-extra",
		ExecutionID:         "exec-child-extra",
		AgentFieldRequestID: "req-child-extra",
		RunID:               &runID,
		ParentExecutionID:   &parentID,
		RootWorkflowID:      ptrString("wf-dag-extra"),
		WorkflowDepth:       1,
		AgentNodeID:         "agent-dag",
		ReasonerID:          "reasoner-dag",
		Status:              string(types.ExecutionStatusSucceeded),
		StartedAt:           now.Add(time.Minute),
		CreatedAt:           now.Add(time.Minute),
		UpdatedAt:           now.Add(time.Minute),
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, child))

	dag, err := ls.QueryWorkflowDAG(ctx, "wf-dag-extra")
	require.NoError(t, err)
	require.Len(t, dag, 2)

	_, err = ls.db.ExecContext(ctx, `UPDATE workflow_executions SET workflow_tags = ? WHERE execution_id = ?`, "{", child.ExecutionID)
	require.NoError(t, err)
	_, err = ls.QueryWorkflowDAG(ctx, "wf-dag-extra")
	require.ErrorContains(t, err, "failed to unmarshal workflow tags")

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	require.EqualError(t, ls.StoreWorkflowExecution(cancelled, root), "context cancelled during store workflow execution: context canceled")
	_, err = ls.QueryWorkflowDAG(cancelled, "wf-dag-extra")
	require.EqualError(t, err, "context cancelled during query workflow DAG: context canceled")
}
