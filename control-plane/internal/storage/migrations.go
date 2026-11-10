package storage

import (
	"context"
	"errors"
	"fmt"

	"github.com/jackc/pgx/v5/pgconn"
)

// migrateAgentNodesCompositePK recreates the agent_nodes table with a composite
// primary key (id, version) and adds the traffic_weight column. This is needed
// because SQLite does not support ALTER TABLE ... DROP PRIMARY KEY.
func (ls *LocalStorage) migrateAgentNodesCompositePK(ctx context.Context) error {
	// Check if migration is needed by looking for the traffic_weight column
	var count int
	err := ls.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM pragma_table_info('agent_nodes') WHERE name = 'traffic_weight'`).Scan(&count)
	if err != nil {
		// Table might not exist yet (fresh install); GORM will create it with composite PK
		return nil
	}
	if count > 0 {
		// Already migrated
		return nil
	}

	// Check the table exists at all
	var tableCount int
	err = ls.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='agent_nodes'`).Scan(&tableCount)
	if err != nil || tableCount == 0 {
		return nil // Fresh install, GORM will create the table
	}

	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin migration transaction: %w", err)
	}
	defer rollbackTx(tx, "migrateAgentNodesCompositePK")

	// Ensure columns from recent features exist before table recreation.
	// These may be absent if the DB predates those features being merged.
	columnsToEnsure := []struct {
		name string
		ddl  string
	}{
		{"version", `ALTER TABLE agent_nodes ADD COLUMN version TEXT NOT NULL DEFAULT ''`},
		{"proposed_tags", `ALTER TABLE agent_nodes ADD COLUMN proposed_tags BLOB`},
		{"approved_tags", `ALTER TABLE agent_nodes ADD COLUMN approved_tags BLOB`},
	}
	for _, col := range columnsToEnsure {
		var colExists int
		if err := tx.QueryRowContext(ctx, `SELECT COUNT(*) FROM pragma_table_info('agent_nodes') WHERE name = ?`, col.name).Scan(&colExists); err != nil {
			return fmt.Errorf("failed to check for column %s: %w", col.name, err)
		}
		if colExists == 0 {
			if _, err := tx.ExecContext(ctx, col.ddl); err != nil {
				return fmt.Errorf("failed to add column %s: %w", col.name, err)
			}
		}
	}

	migrations := []string{
		`CREATE TABLE agent_nodes_new (
			id TEXT NOT NULL,
			version TEXT NOT NULL DEFAULT '',
			group_id TEXT NOT NULL DEFAULT '',
			team_id TEXT NOT NULL DEFAULT '',
			base_url TEXT NOT NULL DEFAULT '',
			traffic_weight INTEGER NOT NULL DEFAULT 100,
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
			metadata BLOB,
			proposed_tags BLOB,
			approved_tags BLOB,
			PRIMARY KEY (id, version)
		)`,
		`INSERT INTO agent_nodes_new (
			id, version, group_id, team_id, base_url, deployment_type, invocation_url,
			reasoners, skills, communication_config, health_status, lifecycle_status,
			last_heartbeat, registered_at, features, metadata, proposed_tags, approved_tags
		) SELECT
			id, version, id, team_id, base_url, deployment_type, invocation_url,
			reasoners, skills, communication_config, health_status, lifecycle_status,
			last_heartbeat, registered_at, features, metadata, proposed_tags, approved_tags
		FROM agent_nodes`,
		`DROP TABLE agent_nodes`,
		`ALTER TABLE agent_nodes_new RENAME TO agent_nodes`,
		`CREATE INDEX IF NOT EXISTS idx_agent_nodes_team_id ON agent_nodes(team_id)`,
		`CREATE INDEX IF NOT EXISTS idx_agent_nodes_group_id ON agent_nodes(group_id)`,
	}

	for _, m := range migrations {
		if _, err := tx.ExecContext(ctx, m); err != nil {
			return fmt.Errorf("composite PK migration failed: %w", err)
		}
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit composite PK migration: %w", err)
	}

	fmt.Println("[migration] agent_nodes table migrated to composite PK (id, version) with traffic_weight")
	return nil
}

// migrateAgentNodesCompositePKPostgres handles the composite PK migration for PostgreSQL.
func (ls *LocalStorage) migrateAgentNodesCompositePKPostgres(ctx context.Context) error {
	// Check if the table exists at all — on a fresh database GORM will create it
	// with the composite PK already in place, so we skip this migration entirely.
	var tableExists int
	if err := ls.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'agent_nodes'`,
	).Scan(&tableExists); err != nil || tableExists == 0 {
		return nil // Fresh install, GORM will create the table
	}

	var count int
	err := ls.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'agent_nodes' AND column_name = 'traffic_weight'`,
	).Scan(&count)
	if err != nil {
		return nil // Unexpected error querying schema
	}
	if count > 0 {
		return nil // Already migrated
	}

	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin postgres migration transaction: %w", err)
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback()
		}
	}()

	// Ensure columns from recent features exist before altering PK.
	ensureColumns := []string{
		`ALTER TABLE agent_nodes ADD COLUMN IF NOT EXISTS version TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE agent_nodes ADD COLUMN IF NOT EXISTS group_id TEXT NOT NULL DEFAULT ''`,
		`ALTER TABLE agent_nodes ADD COLUMN IF NOT EXISTS proposed_tags BYTEA`,
		`ALTER TABLE agent_nodes ADD COLUMN IF NOT EXISTS approved_tags BYTEA`,
	}
	for _, ddl := range ensureColumns {
		if _, err = tx.ExecContext(ctx, ddl); err != nil {
			return fmt.Errorf("postgres ensure column failed: %w", err)
		}
	}
	// Backfill group_id with id where empty
	if _, err = tx.ExecContext(ctx, `UPDATE agent_nodes SET group_id = id WHERE group_id = '' OR group_id IS NULL`); err != nil {
		return fmt.Errorf("postgres backfill group_id failed: %w", err)
	}

	migrations := []string{
		`ALTER TABLE agent_nodes DROP CONSTRAINT IF EXISTS agent_nodes_pkey`,
		`ALTER TABLE agent_nodes ALTER COLUMN version SET DEFAULT ''`,
		`ALTER TABLE agent_nodes ADD PRIMARY KEY (id, version)`,
		`ALTER TABLE agent_nodes ADD COLUMN IF NOT EXISTS traffic_weight INTEGER NOT NULL DEFAULT 100`,
	}

	for _, m := range migrations {
		if _, err = tx.ExecContext(ctx, m); err != nil {
			return fmt.Errorf("postgres composite PK migration failed: %w", err)
		}
	}

	if err = tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit postgres composite PK migration: %w", err)
	}

	fmt.Println("[migration] agent_nodes table migrated to composite PK (id, version) with traffic_weight [postgres]")
	return nil
}

func (ls *LocalStorage) autoMigrateSchema(ctx context.Context) error {
	// Run composite PK migration before GORM auto-migrate
	if ls.mode == "local" {
		if err := ls.migrateAgentNodesCompositePK(ctx); err != nil {
			return fmt.Errorf("agent_nodes composite PK migration failed: %w", err)
		}
	} else {
		if err := ls.migrateAgentNodesCompositePKPostgres(ctx); err != nil {
			return fmt.Errorf("agent_nodes composite PK migration (postgres) failed: %w", err)
		}
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return fmt.Errorf("failed to initialize gorm for migrations: %w", err)
	}

	if ls.mode == "local" {
		if err := gormDB.Exec("PRAGMA foreign_keys = OFF").Error; err != nil {
			return fmt.Errorf("failed to disable foreign keys: %w", err)
		}
		defer func() {
			if err := gormDB.Exec("PRAGMA foreign_keys = ON").Error; err != nil {
				fmt.Printf("failed to re-enable foreign keys: %v\n", err)
			}
		}()
	}

	models := []interface{}{
		&ExecutionRecordModel{},
		&AgentExecutionModel{},
		&AgentNodeModel{},
		&AgentConfigurationModel{},
		&AgentPackageModel{},
		&WorkflowExecutionModel{},
		&WorkflowExecutionEventModel{},
		&ExecutionLogEntryModel{},
		&WorkflowRunEventModel{},
		&WorkflowRunModel{},
		&WorkflowStepModel{},
		&WorkflowModel{},
		&SessionModel{},
		&DIDRegistryModel{},
		&AgentDIDModel{},
		&ComponentDIDModel{},
		&ExecutionVCModel{},
		&WorkflowVCModel{},
		&SchemaMigrationModel{},
		&ExecutionWebhookEventModel{},
		&ExecutionWebhookModel{},
		&ObservabilityWebhookModel{},
		&ObservabilityDeadLetterQueueModel{},
		// VC Authorization models
		&DIDDocumentModel{},
		&AccessPolicyModel{},
		&AgentTagVCModel{},
		&ConfigStorageModel{},
		&TriggerModel{},
		&InboundEventModel{},
		&ExecutionUsageModel{},
	}

	if err := gormDB.WithContext(ctx).AutoMigrate(models...); err != nil {
		return fmt.Errorf("failed to auto-migrate schema: %w", err)
	}

	// THESE TABLES ARE CREATED WITH THE SCHEMA, NOT ON FIRST USE. Running the
	// same guarded DDL helpers here creates execution_relay, run_owners, and
	// execution_approval_questions at startup for both SQLite and PostgreSQL
	// and marks ls.db ready so steady-state operations issue no DDL.
	if err := ls.ensureRelayAndApprovalTables(ctx); err != nil {
		return fmt.Errorf("failed to ensure relay and approval tables: %w", err)
	}

	return nil
}

type guardedSchemaTable struct {
	name        string
	ddl         string
	existsQuery string
}

func (ls *LocalStorage) ensureRelayAndApprovalTables(ctx context.Context) error {
	if err := ls.relayTable(ctx); err != nil {
		return err
	}
	if err := ls.runOwnerTable(ctx); err != nil {
		return err
	}
	return ls.approvalQuestionTable(ctx)
}

// ensureSchemaTable creates t on ls.db if it has not yet been confirmed on
// this database handle.
//
// THESE TABLES ARE CREATED WITH THE SCHEMA, NOT ON FIRST USE, via
// autoMigrateSchema; this guard remains only so stores constructed in tests
// without autoMigrateSchema still work on first call. Creation is serialized
// per *sqlDatabase with tableCreateMu so goroutines in one process never race
// CREATE TABLE IF NOT EXISTS. A FAILED CREATE IS NEVER CACHED: tablesCreated
// records only confirmed tables, so a transient failure is retried on the next
// call instead of breaking the store until restart. On PostgreSQL, another
// process running CREATE TABLE IF NOT EXISTS at the same moment can trip
// pg_type_typname_nsp_index (SQLSTATE 23505) or pg_class (SQLSTATE 42P07); if
// the table is present afterward, that peer finished creating it and the guard
// records success.
func (ls *LocalStorage) ensureSchemaTable(ctx context.Context, t guardedSchemaTable) error {
	if ls == nil || ls.db == nil || ls.db.DB == nil {
		return fmt.Errorf("create %s: sql database is not initialized", t.name)
	}
	if _, ok := ls.db.tablesCreated.Load(t.name); ok {
		return nil
	}

	ls.db.tableCreateMu.Lock()
	defer ls.db.tableCreateMu.Unlock()

	if _, ok := ls.db.tablesCreated.Load(t.name); ok {
		return nil
	}

	if _, err := ls.db.ExecContext(ctx, t.ddl); err != nil {
		if ls.isPostgresMode() && isPostgresConcurrentTableCreateError(err) && ls.tableExistsAfterRace(ctx, t) {
			ls.db.tablesCreated.Store(t.name, struct{}{})
			return nil
		}
		return fmt.Errorf("create %s: %w", t.name, err)
	}

	ls.db.tablesCreated.Store(t.name, struct{}{})
	return nil
}

func (ls *LocalStorage) isPostgresMode() bool {
	return ls != nil && (ls.mode == "postgres" || ls.db.Mode() == "postgres")
}

func isPostgresConcurrentTableCreateError(err error) bool {
	if err == nil {
		return false
	}
	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		return pgErr.Code == "23505" || pgErr.Code == "42P07"
	}
	var stateErr interface{ SQLState() string }
	if errors.As(err, &stateErr) {
		code := stateErr.SQLState()
		return code == "23505" || code == "42P07"
	}
	return false
}

func (ls *LocalStorage) tableExistsAfterRace(ctx context.Context, t guardedSchemaTable) bool {
	rows, err := ls.db.QueryContext(ctx, t.existsQuery)
	if err != nil {
		return false
	}
	defer rows.Close()
	return rows.Err() == nil
}
