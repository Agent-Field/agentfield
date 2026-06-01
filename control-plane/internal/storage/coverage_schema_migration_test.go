package storage

import (
	"context"
	"database/sql"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/boltdb/bolt"
	"github.com/stretchr/testify/require"
)

func newSchemaTestStorage(t *testing.T) *LocalStorage {
	t.Helper()

	tempDir := t.TempDir()

	rawDB, err := sql.Open("sqlite3", filepath.Join(tempDir, "storage.db"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = rawDB.Close() })

	kvStore, err := bolt.Open(filepath.Join(tempDir, "storage.bolt"), 0o600, nil)
	require.NoError(t, err)
	t.Cleanup(func() { _ = kvStore.Close() })

	return &LocalStorage{
		db:      newSQLDatabase(rawDB, "local"),
		kvStore: kvStore,
		mode:    "local",
	}
}

func TestCreateSchemaCreatesLocalSupportStructures(t *testing.T) {
	ls := newSchemaTestStorage(t)
	ctx := context.Background()

	err := ls.createSchema(ctx)
	if err != nil && strings.Contains(err.Error(), "no such module: fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)
	require.NotNil(t, ls.vectorStore)

	err = ls.kvStore.View(func(tx *bolt.Tx) error {
		for _, scope := range []string{"workflow", "session", "actor", "reasoner", "global"} {
			require.NotNil(t, tx.Bucket([]byte(scope)))
		}
		return nil
	})
	require.NoError(t, err)

	workflowName := "Schema Coverage Workflow"
	require.NoError(t, ls.StoreWorkflowExecution(ctx, &types.WorkflowExecution{
		WorkflowID:          "wf-schema",
		ExecutionID:         "exec-schema",
		AgentFieldRequestID: "req-schema",
		AgentNodeID:         "agent-schema",
		ReasonerID:          "reasoner-schema",
		Status:              "pending",
		WorkflowName:        &workflowName,
	}))

	if !ls.ftsEnabled {
		return
	}

	var indexedName string
	require.NoError(t, ls.db.QueryRowContext(ctx,
		`SELECT workflow_name FROM workflow_executions_fts WHERE execution_id = ?`,
		"exec-schema",
	).Scan(&indexedName))
	require.Equal(t, workflowName, indexedName)
}

func TestSetupWorkflowExecutionFTSPopulatesExistingRows(t *testing.T) {
	rawDB, err := sql.Open("sqlite3", ":memory:")
	require.NoError(t, err)
	t.Cleanup(func() { _ = rawDB.Close() })

	_, err = rawDB.Exec(`
		CREATE TABLE workflow_executions (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			execution_id TEXT,
			workflow_id TEXT,
			agent_node_id TEXT,
			session_id TEXT,
			workflow_name TEXT
		)
	`)
	require.NoError(t, err)
	_, err = rawDB.Exec(`
		INSERT INTO workflow_executions (execution_id, workflow_id, agent_node_id, session_id, workflow_name)
		VALUES ('exec-fts', 'wf-fts', 'agent-fts', 'session-fts', 'FTS Ready')
	`)
	require.NoError(t, err)

	ls := &LocalStorage{db: newSQLDatabase(rawDB, "local"), mode: "local"}
	err = ls.setupWorkflowExecutionFTS()
	if err != nil && strings.Contains(err.Error(), "no such module: fts5") {
		t.Skip("sqlite3 compiled without FTS5")
	}
	require.NoError(t, err)

	var count int
	require.NoError(t, rawDB.QueryRow(`
		SELECT COUNT(*) FROM workflow_executions_fts WHERE execution_id = 'exec-fts' AND workflow_name = 'FTS Ready'
	`).Scan(&count))
	require.Equal(t, 1, count)

	_, err = rawDB.Exec(`UPDATE workflow_executions SET workflow_name = 'FTS Updated' WHERE execution_id = 'exec-fts'`)
	require.NoError(t, err)
	require.NoError(t, rawDB.QueryRow(`
		SELECT COUNT(*) FROM workflow_executions_fts WHERE execution_id = 'exec-fts' AND workflow_name = 'FTS Updated'
	`).Scan(&count))
	require.Equal(t, 1, count)

	_, err = rawDB.Exec(`DELETE FROM workflow_executions WHERE execution_id = 'exec-fts'`)
	require.NoError(t, err)
	require.NoError(t, rawDB.QueryRow(`SELECT COUNT(*) FROM workflow_executions_fts WHERE execution_id = 'exec-fts'`).Scan(&count))
	require.Equal(t, 0, count)
}

func TestMigrateAgentNodesCompositePKBranches(t *testing.T) {
	t.Run("migrates legacy schema and preserves rows", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		_, err = rawDB.Exec(`
			CREATE TABLE agent_nodes (
				id TEXT PRIMARY KEY,
				team_id TEXT NOT NULL DEFAULT '',
				base_url TEXT NOT NULL DEFAULT '',
				deployment_type TEXT DEFAULT 'long_running',
				invocation_url TEXT,
				reasoners BLOB,
				skills BLOB,
				communication_config BLOB,
				health_status TEXT NOT NULL DEFAULT 'unknown',
				lifecycle_status TEXT DEFAULT 'starting',
				last_heartbeat TIMESTAMP,
				registered_at TIMESTAMP,
				features BLOB,
				metadata BLOB
			)
		`)
		require.NoError(t, err)
		_, err = rawDB.Exec(`
			INSERT INTO agent_nodes (id, team_id, base_url, deployment_type, health_status, lifecycle_status)
			VALUES ('agent-1', 'team-1', 'https://agent.example', 'long_running', 'healthy', 'ready')
		`)
		require.NoError(t, err)

		ls := &LocalStorage{db: newSQLDatabase(rawDB, "local"), mode: "local"}
		require.NoError(t, ls.migrateAgentNodesCompositePK(context.Background()))

		var version, groupID string
		var trafficWeight int
		require.NoError(t, rawDB.QueryRow(`
			SELECT version, group_id, traffic_weight FROM agent_nodes WHERE id = 'agent-1'
		`).Scan(&version, &groupID, &trafficWeight))
		require.Equal(t, "", version)
		require.Equal(t, "agent-1", groupID)
		require.Equal(t, 100, trafficWeight)

		var columnCount int
		require.NoError(t, rawDB.QueryRow(`
			SELECT COUNT(*) FROM pragma_table_info('agent_nodes') WHERE name IN ('traffic_weight', 'version', 'proposed_tags', 'approved_tags')
		`).Scan(&columnCount))
		require.Equal(t, 4, columnCount)
	})

	t.Run("skips fresh installs and already migrated schemas", func(t *testing.T) {
		rawDB, err := sql.Open("sqlite3", ":memory:")
		require.NoError(t, err)
		t.Cleanup(func() { _ = rawDB.Close() })

		ls := &LocalStorage{db: newSQLDatabase(rawDB, "local"), mode: "local"}
		require.NoError(t, ls.migrateAgentNodesCompositePK(context.Background()))

		_, err = rawDB.Exec(`
			CREATE TABLE agent_nodes (
				id TEXT NOT NULL,
				version TEXT NOT NULL DEFAULT '',
				group_id TEXT NOT NULL DEFAULT '',
				traffic_weight INTEGER NOT NULL DEFAULT 100,
				PRIMARY KEY (id, version)
			)
		`)
		require.NoError(t, err)

		require.NoError(t, ls.migrateAgentNodesCompositePK(context.Background()))
	})
}
