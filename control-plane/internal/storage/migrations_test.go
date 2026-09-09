package storage

import (
	"context"
	"database/sql"
	"testing"

	"github.com/stretchr/testify/require"
)

// legacyAgentNodesDDL is the pre-composite-PK schema: a single-column primary
// key on id, no version / traffic_weight / tag columns. Seeded via raw SQL so
// the test exercises the real migration path rather than GORM's current models.
const legacyAgentNodesDDL = `
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
)`

// newMigrationTestDB opens an in-memory SQLite database wired into a
// LocalStorage in local mode. The driver is registered at package import time.
func newMigrationTestDB(t *testing.T) (*sql.DB, *LocalStorage) {
	t.Helper()
	rawDB, err := sql.Open("sqlite3", ":memory:")
	require.NoError(t, err)
	t.Cleanup(func() { _ = rawDB.Close() })

	ls := &LocalStorage{db: newSQLDatabase(rawDB, "local"), mode: "local"}
	return rawDB, ls
}

// tableColumns returns the set of column names on the given table via
// pragma_table_info.
func tableColumns(t *testing.T, db *sql.DB, table string) map[string]bool {
	t.Helper()
	rows, err := db.Query(`SELECT name FROM pragma_table_info(?)`, table)
	require.NoError(t, err)
	defer rows.Close()

	cols := map[string]bool{}
	for rows.Next() {
		var name string
		require.NoError(t, rows.Scan(&name))
		cols[name] = true
	}
	require.NoError(t, rows.Err())
	return cols
}

// primaryKeyColumns returns the ordered primary-key column names via
// pragma_table_info (pk > 0 marks membership; the value is the 1-based
// position in the key).
func primaryKeyColumns(t *testing.T, db *sql.DB, table string) []string {
	t.Helper()
	rows, err := db.Query(`SELECT name, pk FROM pragma_table_info(?) WHERE pk > 0 ORDER BY pk`, table)
	require.NoError(t, err)
	defer rows.Close()

	var pk []string
	for rows.Next() {
		var name string
		var pos int
		require.NoError(t, rows.Scan(&name, &pos))
		pk = append(pk, name)
	}
	require.NoError(t, rows.Err())
	return pk
}

func TestMigrateAgentNodesCompositePK(t *testing.T) {
	ctx := context.Background()

	t.Run("fresh install skips legacy migration", func(t *testing.T) {
		_, ls := newMigrationTestDB(t)
		// No agent_nodes table exists: the migration must be a no-op success so
		// GORM can create the table with the composite PK afterward.
		require.NoError(t, ls.migrateAgentNodesCompositePK(ctx))
	})

	t.Run("already-migrated schema is a no-op and preserves data", func(t *testing.T) {
		db, ls := newMigrationTestDB(t)
		_, err := db.Exec(`
			CREATE TABLE agent_nodes (
				id TEXT NOT NULL,
				version TEXT NOT NULL DEFAULT '',
				group_id TEXT NOT NULL DEFAULT '',
				traffic_weight INTEGER NOT NULL DEFAULT 100,
				PRIMARY KEY (id, version)
			)`)
		require.NoError(t, err)
		_, err = db.Exec(`INSERT INTO agent_nodes (id, version, group_id, traffic_weight) VALUES ('a', 'v1', 'g1', 42)`)
		require.NoError(t, err)

		require.NoError(t, ls.migrateAgentNodesCompositePK(ctx))

		// Data untouched (no table rewrite occurred).
		var groupID string
		var weight int
		require.NoError(t, db.QueryRow(`SELECT group_id, traffic_weight FROM agent_nodes WHERE id = 'a' AND version = 'v1'`).Scan(&groupID, &weight))
		require.Equal(t, "g1", groupID)
		require.Equal(t, 42, weight)
	})

	t.Run("legacy table recreated with composite PK and traffic_weight", func(t *testing.T) {
		db, ls := newMigrationTestDB(t)
		_, err := db.Exec(legacyAgentNodesDDL)
		require.NoError(t, err)
		_, err = db.Exec(`
			INSERT INTO agent_nodes (id, team_id, base_url, deployment_type, health_status, lifecycle_status)
			VALUES ('agent-1', 'team-1', 'https://agent.example', 'long_running', 'healthy', 'ready')`)
		require.NoError(t, err)

		require.NoError(t, ls.migrateAgentNodesCompositePK(ctx))

		require.Equal(t, []string{"id", "version"}, primaryKeyColumns(t, db, "agent_nodes"))

		cols := tableColumns(t, db, "agent_nodes")
		require.True(t, cols["traffic_weight"], "traffic_weight column must exist")
		require.True(t, cols["version"], "version column must exist")

		var weight int
		require.NoError(t, db.QueryRow(`SELECT traffic_weight FROM agent_nodes WHERE id = 'agent-1'`).Scan(&weight))
		require.Equal(t, 100, weight)
	})

	t.Run("missing feature columns are backfilled before copy", func(t *testing.T) {
		db, ls := newMigrationTestDB(t)
		// Legacy DDL omits proposed_tags / approved_tags / version entirely.
		_, err := db.Exec(legacyAgentNodesDDL)
		require.NoError(t, err)
		_, err = db.Exec(`
			INSERT INTO agent_nodes (id, team_id, base_url, health_status, lifecycle_status)
			VALUES ('agent-2', 'team-2', 'https://two.example', 'unknown', 'starting')`)
		require.NoError(t, err)

		require.NoError(t, ls.migrateAgentNodesCompositePK(ctx))

		cols := tableColumns(t, db, "agent_nodes")
		require.True(t, cols["proposed_tags"], "proposed_tags must be present after migration")
		require.True(t, cols["approved_tags"], "approved_tags must be present after migration")

		// Row copied through with defaulted tag columns (NULL blobs).
		var proposed, approved sql.NullString
		require.NoError(t, db.QueryRow(`SELECT proposed_tags, approved_tags FROM agent_nodes WHERE id = 'agent-2'`).Scan(&proposed, &approved))
		require.False(t, proposed.Valid)
		require.False(t, approved.Valid)
	})

	t.Run("group_id backfill uses id during copy", func(t *testing.T) {
		db, ls := newMigrationTestDB(t)
		// Legacy table has no group_id column at all; the copy query sources
		// group_id from id, so every migrated row gets group_id = id.
		_, err := db.Exec(legacyAgentNodesDDL)
		require.NoError(t, err)
		_, err = db.Exec(`
			INSERT INTO agent_nodes (id, team_id, base_url, health_status, lifecycle_status)
			VALUES ('agent-3', 'team-3', 'https://three.example', 'healthy', 'ready')`)
		require.NoError(t, err)

		require.NoError(t, ls.migrateAgentNodesCompositePK(ctx))

		var groupID string
		require.NoError(t, db.QueryRow(`SELECT group_id FROM agent_nodes WHERE id = 'agent-3'`).Scan(&groupID))
		require.Equal(t, "agent-3", groupID)
	})
}

func TestAutoMigrateSchemaRunsCompositePKBeforeModels(t *testing.T) {
	ctx := context.Background()

	db, ls := newMigrationTestDB(t)
	// Seed the legacy schema so autoMigrateSchema must run the composite-PK
	// migration first, then GORM AutoMigrate over the models.
	_, err := db.Exec(legacyAgentNodesDDL)
	require.NoError(t, err)
	_, err = db.Exec(`
		INSERT INTO agent_nodes (id, team_id, base_url, health_status, lifecycle_status)
		VALUES ('agent-4', 'team-4', 'https://four.example', 'healthy', 'ready')`)
	require.NoError(t, err)

	require.NoError(t, ls.autoMigrateSchema(ctx))

	// Composite PK migration ran: agent_nodes now keys on (id, version).
	require.Equal(t, []string{"id", "version"}, primaryKeyColumns(t, db, "agent_nodes"))

	cols := tableColumns(t, db, "agent_nodes")
	require.True(t, cols["traffic_weight"])

	// Original row survived the migration.
	var count int
	require.NoError(t, db.QueryRow(`SELECT COUNT(*) FROM agent_nodes WHERE id = 'agent-4'`).Scan(&count))
	require.Equal(t, 1, count)

	// GORM models were migrated afterward: a representative target table exists.
	require.NoError(t, db.QueryRow(`SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'execution_webhooks'`).Scan(&count))
	require.Equal(t, 1, count)

	require.NotNil(t, ls.gormDB)
}
