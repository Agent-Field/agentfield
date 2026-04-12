package storage

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/boltdb/bolt"
	"github.com/jackc/pgx/v5/pgconn"
	_ "github.com/jackc/pgx/v5/stdlib" // Import pgx driver for PostgreSQL
	_ "github.com/mattn/go-sqlite3"    // Import sqlite3 driver
	"gorm.io/gorm"
	"gorm.io/gorm/clause"
)

// Custom error types for data integrity issues
type DuplicateDIDError struct {
	DID  string
	Type string // "registry", "agent", or "component"
}

func (e *DuplicateDIDError) Error() string {
	return fmt.Sprintf("duplicate %s DID detected: %s already exists", e.Type, e.DID)
}

// ForeignKeyConstraintError represents a foreign key constraint violation
type ForeignKeyConstraintError struct {
	Table           string
	Column          string
	ReferencedTable string
	ReferencedValue string
	Operation       string
}

func (e *ForeignKeyConstraintError) Error() string {
	return fmt.Sprintf("foreign key constraint violation in %s.%s: referenced %s '%s' does not exist (operation: %s)",
		e.Table, e.Column, e.ReferencedTable, e.ReferencedValue, e.Operation)
}

// ValidationError represents a pre-storage validation failure
type ValidationError struct {
	Field   string
	Value   string
	Reason  string
	Context string
}

func (e *ValidationError) Error() string {
	return fmt.Sprintf("validation failed for %s='%s': %s (context: %s)",
		e.Field, e.Value, e.Reason, e.Context)
}

// getWorkflowExecutionByID is a helper function that retrieves a workflow execution using DBTX interface
func (ls *LocalStorage) getWorkflowExecutionByID(ctx context.Context, q DBTX, executionID string) (*types.WorkflowExecution, error) {
	query := `
		SELECT workflow_id, execution_id, agentfield_request_id, run_id, session_id, actor_id,
		       agent_node_id, parent_workflow_id, parent_execution_id, root_workflow_id, workflow_depth,
		       reasoner_id, input_data, output_data, input_size, output_size,
		       status, started_at, completed_at, duration_ms,
		       state_version, last_event_sequence, active_children, pending_children,
		       pending_terminal_status, status_reason, lease_owner, lease_expires_at,
		       error_message, retry_count,
		       approval_request_id, approval_request_url, approval_status, approval_response,
		       approval_requested_at, approval_responded_at, approval_callback_url, approval_expires_at,
		       workflow_name, workflow_tags, notes, created_at, updated_at
		FROM workflow_executions WHERE execution_id = ?`

	row := q.QueryRowContext(ctx, query, executionID)
	execution := &types.WorkflowExecution{}

	var workflowTagsJSON, notesJSON []byte
	var inputData, outputData sql.NullString
	var runID sql.NullString
	var pendingTerminal sql.NullString
	var statusReason sql.NullString
	var leaseOwner sql.NullString
	var leaseExpires sql.NullTime
	var approvalRequestID, approvalRequestURL, approvalStatus, approvalResponse, approvalCallbackURL sql.NullString
	var approvalRequestedAt, approvalRespondedAt, approvalExpiresAt sql.NullTime
	err := row.Scan(
		&execution.WorkflowID, &execution.ExecutionID, &execution.AgentFieldRequestID,
		&runID, &execution.SessionID, &execution.ActorID, &execution.AgentNodeID,
		&execution.ParentWorkflowID, &execution.ParentExecutionID, &execution.RootWorkflowID, &execution.WorkflowDepth,
		&execution.ReasonerID, &inputData, &outputData,
		&execution.InputSize, &execution.OutputSize, &execution.Status,
		&execution.StartedAt, &execution.CompletedAt, &execution.DurationMS,
		&execution.StateVersion, &execution.LastEventSequence, &execution.ActiveChildren, &execution.PendingChildren,
		&pendingTerminal, &statusReason,
		&leaseOwner, &leaseExpires,
		&execution.ErrorMessage, &execution.RetryCount,
		&approvalRequestID, &approvalRequestURL, &approvalStatus, &approvalResponse,
		&approvalRequestedAt, &approvalRespondedAt, &approvalCallbackURL, &approvalExpiresAt,
		&execution.WorkflowName,
		&workflowTagsJSON, &notesJSON, &execution.CreatedAt, &execution.UpdatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			// "Not found" is a valid case for an upsert operation, so we return nil without an error.
			// The caller is responsible for handling the nil execution record.
			return nil, nil
		}
		return nil, fmt.Errorf("failed to get workflow execution: %w", err)
	}

	// Handle nullable JSON fields
	if runID.Valid {
		execution.RunID = &runID.String
	}
	if inputData.Valid {
		execution.InputData = safeJSONRawMessage(inputData.String, "{}", fmt.Sprintf("execution %s input_data", execution.ExecutionID))
	} else {
		execution.InputData = json.RawMessage("{}")
	}

	if outputData.Valid {
		execution.OutputData = safeJSONRawMessage(outputData.String, "{}", fmt.Sprintf("execution %s output_data", execution.ExecutionID))
	} else {
		execution.OutputData = json.RawMessage("{}")
	}
	if pendingTerminal.Valid {
		execution.PendingTerminalStatus = &pendingTerminal.String
	}
	if statusReason.Valid {
		execution.StatusReason = &statusReason.String
	}
	if leaseOwner.Valid {
		execution.LeaseOwner = &leaseOwner.String
	}
	if leaseExpires.Valid {
		t := leaseExpires.Time
		execution.LeaseExpiresAt = &t
	}
	if approvalRequestID.Valid {
		execution.ApprovalRequestID = &approvalRequestID.String
	}
	if approvalRequestURL.Valid {
		execution.ApprovalRequestURL = &approvalRequestURL.String
	}
	if approvalStatus.Valid {
		execution.ApprovalStatus = &approvalStatus.String
	}
	if approvalResponse.Valid {
		execution.ApprovalResponse = &approvalResponse.String
	}
	if approvalRequestedAt.Valid {
		t := approvalRequestedAt.Time
		execution.ApprovalRequestedAt = &t
	}
	if approvalRespondedAt.Valid {
		t := approvalRespondedAt.Time
		execution.ApprovalRespondedAt = &t
	}
	if approvalCallbackURL.Valid {
		execution.ApprovalCallbackURL = &approvalCallbackURL.String
	}
	if approvalExpiresAt.Valid {
		t := approvalExpiresAt.Time
		execution.ApprovalExpiresAt = &t
	}

	// Unmarshal workflow tags
	if len(workflowTagsJSON) > 0 {
		if err := json.Unmarshal(workflowTagsJSON, &execution.WorkflowTags); err != nil {
			return nil, fmt.Errorf("failed to unmarshal workflow tags: %w", err)
		}
	}

	// Unmarshal notes
	if len(notesJSON) > 0 {
		if err := json.Unmarshal(notesJSON, &execution.Notes); err != nil {
			return nil, fmt.Errorf("failed to unmarshal notes: %w", err)
		}
	} else {
		execution.Notes = []types.ExecutionNote{}
	}

	return execution, nil
}

func (ls *LocalStorage) StoreWorkflowRun(ctx context.Context, run *types.WorkflowRun) error {
	if run == nil {
		return fmt.Errorf("workflow run cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	db := ls.requireSQLDB()
	createdAt := run.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}
	updatedAt := run.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}
	metadata := "{}"
	if len(run.Metadata) > 0 {
		metadata = string(run.Metadata)
	}

	query := `
		INSERT INTO workflow_runs (
			run_id, root_workflow_id, root_execution_id, status, total_steps,
			completed_steps, failed_steps, state_version, last_event_sequence,
			metadata, created_at, updated_at, completed_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(run_id) DO UPDATE SET
			root_workflow_id=excluded.root_workflow_id,
			root_execution_id=excluded.root_execution_id,
			status=excluded.status,
			total_steps=excluded.total_steps,
			completed_steps=excluded.completed_steps,
			failed_steps=excluded.failed_steps,
			state_version=excluded.state_version,
			last_event_sequence=excluded.last_event_sequence,
			metadata=excluded.metadata,
			updated_at=excluded.updated_at,
			completed_at=excluded.completed_at
	`

	_, err := db.ExecContext(
		ctx,
		query,
		run.RunID,
		run.RootWorkflowID,
		run.RootExecutionID,
		run.Status,
		run.TotalSteps,
		run.CompletedSteps,
		run.FailedSteps,
		run.StateVersion,
		run.LastEventSequence,
		metadata,
		createdAt.UTC(),
		updatedAt.UTC(),
		run.CompletedAt,
	)
	return err
}

func (ls *LocalStorage) GetWorkflowRun(ctx context.Context, runID string) (*types.WorkflowRun, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if strings.TrimSpace(runID) == "" {
		return nil, fmt.Errorf("run_id cannot be empty")
	}

	db := ls.requireSQLDB()
	query := `
		SELECT run_id, root_workflow_id, root_execution_id, status, total_steps,
		       completed_steps, failed_steps, state_version, last_event_sequence,
		       metadata, created_at, updated_at, completed_at
		FROM workflow_runs
		WHERE run_id = ?
	`

	row := db.QueryRowContext(ctx, query, runID)

	var (
		rootExecutionID sql.NullString
		metadata        sql.NullString
		completedAt     sql.NullTime
		run             types.WorkflowRun
	)

	if err := row.Scan(
		&run.RunID,
		&run.RootWorkflowID,
		&rootExecutionID,
		&run.Status,
		&run.TotalSteps,
		&run.CompletedSteps,
		&run.FailedSteps,
		&run.StateVersion,
		&run.LastEventSequence,
		&metadata,
		&run.CreatedAt,
		&run.UpdatedAt,
		&completedAt,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, err
	}

	if rootExecutionID.Valid {
		run.RootExecutionID = &rootExecutionID.String
	}
	if completedAt.Valid {
		ts := completedAt.Time
		run.CompletedAt = &ts
	}

	if metadata.Valid && strings.TrimSpace(metadata.String) != "" {
		run.Metadata = json.RawMessage(metadata.String)
	} else {
		run.Metadata = json.RawMessage(`{}`)
	}

	return &run, nil
}

func (ls *LocalStorage) StoreWorkflowRunEvent(ctx context.Context, event *types.WorkflowRunEvent) error {
	if event == nil {
		return fmt.Errorf("workflow run event cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	db := ls.requireSQLDB()

	payload := "{}"
	if len(event.Payload) > 0 {
		payload = string(event.Payload)
	}

	recordedAt := event.RecordedAt
	if recordedAt.IsZero() {
		recordedAt = time.Now().UTC()
	}

	query := `
		INSERT INTO workflow_run_events (
			run_id, sequence, previous_sequence, event_type,
			status, status_reason, payload, emitted_at, recorded_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
	`

	_, err := db.ExecContext(
		ctx,
		query,
		event.RunID,
		event.Sequence,
		event.PreviousSequence,
		event.EventType,
		event.Status,
		event.StatusReason,
		payload,
		event.EmittedAt.UTC(),
		recordedAt.UTC(),
	)
	return err
}

func (ls *LocalStorage) StoreWorkflowStep(ctx context.Context, step *types.WorkflowStep) error {
	if step == nil {
		return fmt.Errorf("workflow step cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	db := ls.requireSQLDB()
	metadata := "{}"
	if len(step.Metadata) > 0 {
		metadata = string(step.Metadata)
	}

	notBefore := step.NotBefore
	if notBefore.IsZero() {
		notBefore = time.Now().UTC()
	}

	createdAt := step.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}

	updatedAt := step.UpdatedAt
	if updatedAt.IsZero() {
		updatedAt = createdAt
	}

	query := `
		INSERT INTO workflow_steps (
			step_id, run_id, parent_step_id, execution_id, agent_node_id,
			target, status, attempt, priority, not_before, input_uri, result_uri,
			error_message, metadata, started_at, completed_at, leased_at,
			lease_timeout, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(step_id) DO UPDATE SET
			run_id=excluded.run_id,
			parent_step_id=excluded.parent_step_id,
			execution_id=excluded.execution_id,
			agent_node_id=excluded.agent_node_id,
			target=excluded.target,
			status=excluded.status,
			attempt=excluded.attempt,
			priority=excluded.priority,
			not_before=excluded.not_before,
			input_uri=excluded.input_uri,
			result_uri=excluded.result_uri,
			error_message=excluded.error_message,
			metadata=excluded.metadata,
			started_at=excluded.started_at,
			completed_at=excluded.completed_at,
			leased_at=excluded.leased_at,
			lease_timeout=excluded.lease_timeout,
			updated_at=excluded.updated_at
	`

	_, err := db.ExecContext(
		ctx,
		query,
		step.StepID,
		step.RunID,
		step.ParentStepID,
		step.ExecutionID,
		step.AgentNodeID,
		step.Target,
		step.Status,
		step.Attempt,
		step.Priority,
		notBefore.UTC(),
		step.InputURI,
		step.ResultURI,
		step.ErrorMessage,
		metadata,
		step.StartedAt,
		step.CompletedAt,
		step.LeasedAt,
		step.LeaseTimeout,
		createdAt.UTC(),
		updatedAt.UTC(),
	)
	return err
}

// DBTX interface for operations that can run on a db or tx
type DBTX interface {
	ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error)
	Exec(query string, args ...interface{}) (sql.Result, error)
	QueryRowContext(ctx context.Context, query string, args ...interface{}) *sql.Row
	QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error)
	Query(query string, args ...interface{}) (*sql.Rows, error)
	QueryRow(query string, args ...interface{}) *sql.Row
}

// LocalStorage implements the StorageProvider and CacheProvider interfaces
// using SQLite for structured data and BoltDB for key-value data (memory).
//
// CONCURRENCY MODEL:
// - SQLite is configured with WAL (Write-Ahead Logging) mode for optimal concurrency
// - Read-only operations (SELECT queries) do NOT acquire writeMutex - they run concurrently
// - Write operations (INSERT/UPDATE/DELETE) acquire writeMutex for serialization
// - WAL mode allows multiple concurrent readers with a single writer without blocking
// - This eliminates the performance bottleneck where analytics queries blocked all writes
type LocalStorage struct {
	db                        *sqlDatabase
	gormDB                    *gorm.DB                                  // GORM handle for ORM operations
	kvStore                   *bolt.DB                                  // BoltDB for key-value (memory)
	cache                     *sync.Map                                 // In-memory cache for hot data
	subscribers               map[string][]chan types.MemoryChangeEvent // Local pub/sub
	mu                        sync.RWMutex
	mode                      string
	config                    LocalStorageConfig
	postgresConfig            PostgresStorageConfig
	vectorConfig              VectorStoreConfig
	vectorMetric              VectorDistanceMetric
	vectorStore               vectorStore
	eventBus                  *events.ExecutionEventBus // Event bus for real-time updates
	workflowExecutionEventBus *events.EventBus[*types.WorkflowExecutionEvent]
	executionLogEventBus      *events.EventBus[*types.ExecutionLogEntry]
	ftsEnabled                bool
}

// NewLocalStorage creates a new instance of LocalStorage.
func NewLocalStorage(config LocalStorageConfig) *LocalStorage {
	return &LocalStorage{
		mode:                      "local",
		config:                    config,
		vectorMetric:              VectorDistanceCosine,
		cache:                     &sync.Map{},
		subscribers:               make(map[string][]chan types.MemoryChangeEvent),
		eventBus:                  events.NewExecutionEventBus(),
		workflowExecutionEventBus: events.NewEventBus[*types.WorkflowExecutionEvent](),
		executionLogEventBus:      events.NewEventBus[*types.ExecutionLogEntry](),
	}
}

// NewPostgresStorage creates a new instance configured for PostgreSQL.
func NewPostgresStorage(config PostgresStorageConfig) *LocalStorage {
	return &LocalStorage{
		mode:                      "postgres",
		postgresConfig:            config,
		vectorMetric:              VectorDistanceCosine,
		cache:                     &sync.Map{},
		subscribers:               make(map[string][]chan types.MemoryChangeEvent),
		eventBus:                  events.NewExecutionEventBus(),
		workflowExecutionEventBus: events.NewEventBus[*types.WorkflowExecutionEvent](),
		executionLogEventBus:      events.NewEventBus[*types.ExecutionLogEntry](),
	}
}

// Initialize sets up the SQLite and BoltDB databases.
func (ls *LocalStorage) Initialize(ctx context.Context, config StorageConfig) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during initialization: %w", err)
	}

	mode := config.Mode
	if mode == "" {
		mode = "local"
	}

	ls.mode = mode
	ls.config = config.Local
	ls.postgresConfig = config.Postgres
	ls.vectorConfig = config.Vector.normalized()
	ls.vectorMetric = parseDistanceMetric(ls.vectorConfig.Distance)

	switch mode {
	case "local":
		return ls.initializeSQLite(ctx)
	case "postgres":
		return ls.initializePostgres(ctx)
	default:
		return fmt.Errorf("unsupported storage mode: %s", mode)
	}
}

func (ls *LocalStorage) initializeSQLite(ctx context.Context) error {
	// Validate that the database path is absolute to prevent files being created in random directories
	if ls.config.DatabasePath == "" {
		return fmt.Errorf("database path is empty - please set a valid database path in configuration")
	}

	// Convert to absolute path if it's relative
	dbPath := ls.config.DatabasePath
	if !filepath.IsAbs(dbPath) {
		absPath, err := filepath.Abs(dbPath)
		if err != nil {
			return fmt.Errorf("failed to convert database path to absolute path: %w", err)
		}
		log.Printf("⚠️  WARNING: Database path was relative (%s), converted to absolute path: %s", dbPath, absPath)
		dbPath = absPath
	}

	// Ensure the directory exists
	dbDir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dbDir, 0755); err != nil {
		return fmt.Errorf("failed to create database directory %s: %w", dbDir, err)
	}

	log.Printf("📁 Initializing SQLite database at: %s", dbPath)

	busyTimeout := resolveEnvInt("AGENTFIELD_SQLITE_BUSY_TIMEOUT_MS", 60000)
	if busyTimeout <= 0 {
		busyTimeout = 60000
	}

	dsn := fmt.Sprintf("%s?_journal_mode=WAL&_synchronous=NORMAL&_cache_size=10000&_foreign_keys=ON&_busy_timeout=%d&_wal_autocheckpoint=1000&_temp_store=MEMORY&_mmap_size=268435456",
		dbPath, busyTimeout)

	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return fmt.Errorf("failed to open SQLite database: %w", err)
	}

	ls.db = newSQLDatabase(db, "local")

	maxOpen := resolveEnvInt("AGENTFIELD_SQLITE_MAX_OPEN_CONNS", 1)
	if maxOpen <= 0 {
		maxOpen = 1
	}
	ls.db.SetMaxOpenConns(maxOpen)
	idleConns := resolveEnvInt("AGENTFIELD_SQLITE_MAX_IDLE_CONNS", 1)
	if idleConns < 0 {
		idleConns = 0
	}
	ls.db.SetMaxIdleConns(idleConns)
	ls.db.SetConnMaxLifetime(15 * time.Minute)
	ls.db.SetConnMaxIdleTime(5 * time.Minute)

	pragmas := []string{
		"PRAGMA wal_autocheckpoint=1000",
		"PRAGMA journal_size_limit=67108864",
		"PRAGMA optimize",
	}

	for _, pragma := range pragmas {
		if _, err := ls.db.Exec(pragma); err != nil {
			return fmt.Errorf("failed to set pragma %s: %w", pragma, err)
		}
	}

	if err := ls.initGormDB(); err != nil {
		return fmt.Errorf("failed to initialize gorm: %w", err)
	}

	go ls.periodicWALCheckpoint()

	kvStore, err := bolt.Open(ls.config.KVStorePath, 0600, &bolt.Options{Timeout: 1 * time.Second})
	if err != nil {
		return fmt.Errorf("failed to open BoltDB database: %w", err)
	}
	ls.kvStore = kvStore

	if err := ls.createSchema(ctx); err != nil {
		return fmt.Errorf("failed to create local storage schema: %w", err)
	}

	return nil
}

func resolveEnvInt(key string, fallback int) int {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		log.Printf("⚠️  Invalid integer for %s=%s, using fallback %d", key, raw, fallback)
		return fallback
	}
	return value
}

func (ls *LocalStorage) initializePostgres(ctx context.Context) error {
	cfg := ls.postgresConfig
	dsn := strings.TrimSpace(cfg.DSN)
	if dsn == "" {
		dsn = strings.TrimSpace(cfg.URL)
	}

	if dsn == "" {
		if cfg.Host == "" {
			return fmt.Errorf("postgres configuration requires either a connection string or host information")
		}

		if cfg.Port == 0 {
			cfg.Port = 5432
		}

		if cfg.User == "" {
			return fmt.Errorf("postgres configuration requires a user when host is specified")
		}

		if cfg.Database == "" {
			return fmt.Errorf("postgres configuration requires a database when host is specified")
		}

		if cfg.SSLMode == "" {
			cfg.SSLMode = "disable"
		}

		pgURL := &url.URL{
			Scheme: "postgres",
			Host:   fmt.Sprintf("%s:%d", cfg.Host, cfg.Port),
			Path:   "/" + strings.TrimPrefix(cfg.Database, "/"),
		}

		if cfg.Password != "" {
			pgURL.User = url.UserPassword(cfg.User, cfg.Password)
		} else {
			pgURL.User = url.User(cfg.User)
		}

		query := pgURL.Query()
		if cfg.SSLMode != "" {
			query.Set("sslmode", cfg.SSLMode)
		}
		if len(query) > 0 {
			pgURL.RawQuery = query.Encode()
		}

		dsn = pgURL.String()
	}

	if cfg.SSLMode == "" {
		cfg.SSLMode = "disable"
	}

	cfg.DSN = dsn
	cfg.URL = dsn
	ls.postgresConfig = cfg

	db, err := sql.Open("pgx", dsn)
	if err != nil {
		return fmt.Errorf("failed to open PostgreSQL database: %w", err)
	}

	sqlDB := newSQLDatabase(db, "postgres")
	ls.applyPostgresConnectionSettings(sqlDB, cfg)

	if err := sqlDB.PingContext(ctx); err != nil {
		if isPostgresDatabaseMissingError(err) {
			_ = sqlDB.Close()
			if err := ls.ensurePostgresDatabaseExists(ctx, cfg); err != nil {
				return fmt.Errorf("failed to create postgres database: %w", err)
			}

			db, err = sql.Open("pgx", cfg.DSN)
			if err != nil {
				return fmt.Errorf("failed to open PostgreSQL database after creation: %w", err)
			}

			sqlDB = newSQLDatabase(db, "postgres")
			ls.applyPostgresConnectionSettings(sqlDB, cfg)

			if err := sqlDB.PingContext(ctx); err != nil {
				_ = sqlDB.Close()
				return fmt.Errorf("failed to ping PostgreSQL database after creation: %w", err)
			}
		} else {
			_ = sqlDB.Close()
			return fmt.Errorf("failed to ping PostgreSQL database: %w", err)
		}
	}

	ls.db = sqlDB

	if err := ls.initGormDB(); err != nil {
		return fmt.Errorf("failed to initialize gorm for postgres: %w", err)
	}

	if err := ls.createSchema(ctx); err != nil {
		return fmt.Errorf("failed to create postgres storage schema: %w", err)
	}

	return nil
}

func (ls *LocalStorage) applyPostgresConnectionSettings(db *sqlDatabase, cfg PostgresStorageConfig) {
	if db == nil {
		return
	}

	maxOpen := cfg.MaxOpenConns
	if maxOpen <= 0 {
		maxOpen = 25
	}
	maxIdle := cfg.MaxIdleConns
	if maxIdle <= 0 {
		maxIdle = 5
	}

	db.SetMaxOpenConns(maxOpen)
	db.SetMaxIdleConns(maxIdle)

	if cfg.ConnMaxLifetime > 0 {
		db.SetConnMaxLifetime(cfg.ConnMaxLifetime)
	} else {
		db.SetConnMaxLifetime(30 * time.Minute)
	}
}

func isPostgresDatabaseMissingError(err error) bool {
	if err == nil {
		return false
	}

	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		return pgErr.Code == "3D000" // undefined_database
	}

	return strings.Contains(err.Error(), "does not exist")
}

func isPostgresDatabaseAlreadyExistsError(err error) bool {
	if err == nil {
		return false
	}

	var pgErr *pgconn.PgError
	if errors.As(err, &pgErr) {
		return pgErr.Code == "42P04" // duplicate_database
	}

	return strings.Contains(err.Error(), "already exists")
}

func (ls *LocalStorage) ensurePostgresDatabaseExists(ctx context.Context, cfg PostgresStorageConfig) error {
	dsn := strings.TrimSpace(cfg.DSN)
	if dsn == "" {
		return fmt.Errorf("postgres DSN is required to create database")
	}

	parsed, err := url.Parse(dsn)
	if err != nil {
		return fmt.Errorf("failed to parse postgres DSN: %w", err)
	}

	dbName := strings.TrimPrefix(parsed.Path, "/")
	dbName = strings.TrimSpace(dbName)
	if dbName == "" {
		return fmt.Errorf("postgres DSN must specify a database name")
	}

	adminDatabase := strings.TrimSpace(cfg.AdminDatabase)
	if adminDatabase == "" {
		adminDatabase = "postgres"
	}

	parsed.Path = "/" + adminDatabase
	adminDSN := parsed.String()

	adminDB, err := sql.Open("pgx", adminDSN)
	if err != nil {
		return fmt.Errorf("failed to open postgres admin connection: %w", err)
	}
	defer adminDB.Close()

	if err := adminDB.PingContext(ctx); err != nil {
		return fmt.Errorf("failed to ping postgres admin database: %w", err)
	}

	createStmt := fmt.Sprintf("CREATE DATABASE %s", quotePostgresIdentifier(dbName))
	if _, err := adminDB.ExecContext(ctx, createStmt); err != nil {
		if isPostgresDatabaseAlreadyExistsError(err) {
			return nil
		}
		return fmt.Errorf("failed to create postgres database '%s': %w", dbName, err)
	}

	return nil
}

func quotePostgresIdentifier(identifier string) string {
	escaped := strings.ReplaceAll(identifier, "\"", "\"\"")
	return "\"" + escaped + "\""
}

// periodicWALCheckpoint runs periodic WAL checkpoints to prevent WAL file buildup
// SQLite WAL mode handles write coordination internally, so no additional locking needed
func (ls *LocalStorage) periodicWALCheckpoint() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		if ls.db != nil {
			// Perform checkpoint - SQLite WAL mode handles coordination internally
			_, err := ls.db.Exec("PRAGMA wal_checkpoint(PASSIVE)")
			if err != nil {
				// Log error but don't stop the checkpoint process
				fmt.Printf("WAL checkpoint failed: %v\n", err)
			}
		}
	}
}

// createSchema ensures the SQLite schema, indexes, and supporting buckets exist.
func (ls *LocalStorage) createSchema(ctx context.Context) error {
	if err := ls.autoMigrateSchema(ctx); err != nil {
		return fmt.Errorf("auto migrate schema: %w", err)
	}

	if ls.mode == "postgres" {
		if err := ls.ensurePostgresKeyValueSchema(ctx); err != nil {
			return err
		}
		if err := ls.ensurePostgresEventSchema(ctx); err != nil {
			return err
		}
		if err := ls.ensurePostgresLockSchema(ctx); err != nil {
			return err
		}
		if err := ls.ensurePostgresWorkflowFTS(ctx); err != nil {
			return err
		}
		if err := ls.ensurePostgresIndexes(ctx); err != nil {
			return err
		}
		if err := ls.runPostgresMigrations(ctx); err != nil {
			return fmt.Errorf("failed to run postgres migrations: %w", err)
		}
		if ls.vectorConfig.isEnabled() {
			if err := ls.ensureVectorSchema(ctx); err != nil {
				return err
			}
			if err := ls.initializeVectorStore(); err != nil {
				return err
			}
		}
		return nil
	}

	if err := ls.initializeMemoryBuckets(); err != nil {
		return err
	}

	if err := ls.ensureExecutionVCSchema(); err != nil {
		return err
	}

	if err := ls.ensureWorkflowVCSchema(); err != nil {
		return err
	}

	if err := ls.runMigrations(); err != nil {
		return fmt.Errorf("failed to run migrations: %w", err)
	}

	if err := ls.setupWorkflowExecutionFTS(); err != nil {
		if strings.Contains(err.Error(), "no such module: fts5") {
			ls.ftsEnabled = false
			log.Printf("FTS5 module not available, full-text search will be degraded")
		} else {
			return err
		}
	} else {
		ls.ftsEnabled = true
	}

	if err := ls.ensureSQLiteIndexes(); err != nil {
		return err
	}

	if ls.vectorConfig.isEnabled() {
		if err := ls.ensureVectorSchema(ctx); err != nil {
			return err
		}
		if err := ls.initializeVectorStore(); err != nil {
			return err
		}
	}

	return nil
}

func (ls *LocalStorage) initializeMemoryBuckets() error {
	if err := ls.kvStore.Update(func(tx *bolt.Tx) error {
		scopes := []string{"workflow", "session", "actor", "reasoner", "global"}
		for _, scope := range scopes {
			if _, err := tx.CreateBucketIfNotExists([]byte(scope)); err != nil {
				return fmt.Errorf("failed to create BoltDB bucket '%s': %w", scope, err)
			}
		}
		return nil
	}); err != nil {
		return err
	}
	return nil
}

func (ls *LocalStorage) ensurePostgresKeyValueSchema(ctx context.Context) error {
	createTable := `
        CREATE TABLE IF NOT EXISTS kv_store (
                scope TEXT NOT NULL,
                scope_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value JSONB NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (scope, scope_id, key)
        );`

	_, err := ls.db.Exec(createTable)
	return err
}

func (ls *LocalStorage) ensurePostgresEventSchema(ctx context.Context) error {
	statements := []string{
		`CREATE TABLE IF NOT EXISTS memory_events (
                        id BIGSERIAL PRIMARY KEY,
                        scope TEXT NOT NULL,
                        scope_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        event_type TEXT,
                        action TEXT,
                        data JSONB,
                        previous_data JSONB,
                        metadata JSONB,
                        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );`,
		`CREATE INDEX IF NOT EXISTS idx_memory_events_scope ON memory_events(scope, scope_id);`,
	}

	for _, stmt := range statements {
		if _, err := ls.db.Exec(stmt); err != nil {
			return err
		}
	}
	return nil
}

func (ls *LocalStorage) ensurePostgresLockSchema(ctx context.Context) error {
	statements := []string{
		`CREATE TABLE IF NOT EXISTS distributed_locks (
                        lock_id TEXT PRIMARY KEY,
                        key TEXT NOT NULL UNIQUE,
                        owner TEXT NOT NULL,
                        expires_at TIMESTAMPTZ NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );`,
		`CREATE INDEX IF NOT EXISTS idx_distributed_locks_expires ON distributed_locks(expires_at);`,
	}

	for _, stmt := range statements {
		if _, err := ls.db.Exec(stmt); err != nil {
			return err
		}
	}
	return nil
}

func (ls *LocalStorage) ensurePostgresWorkflowFTS(ctx context.Context) error {
	statements := []string{
		`CREATE TABLE IF NOT EXISTS workflow_executions_fts (
                        execution_id TEXT PRIMARY KEY,
                        workflow_id TEXT,
                        agent_node_id TEXT,
                        session_id TEXT,
                        workflow_name TEXT,
                        search_vector TSVECTOR
                );`,
		`CREATE OR REPLACE FUNCTION workflow_executions_fts_upsert() RETURNS trigger AS $$
                BEGIN
                        INSERT INTO workflow_executions_fts(execution_id, workflow_id, agent_node_id, session_id, workflow_name, search_vector)
                        VALUES (NEW.execution_id, NEW.workflow_id, NEW.agent_node_id, NEW.session_id, NEW.workflow_name,
                                to_tsvector('simple', coalesce(NEW.workflow_name, '') || ' ' || coalesce(NEW.execution_id, '') || ' ' || coalesce(NEW.workflow_id, '')))
                        ON CONFLICT (execution_id) DO UPDATE SET
                                workflow_id = EXCLUDED.workflow_id,
                                agent_node_id = EXCLUDED.agent_node_id,
                                session_id = EXCLUDED.session_id,
                                workflow_name = EXCLUDED.workflow_name,
                                search_vector = EXCLUDED.search_vector;
                        RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;`,
		`CREATE OR REPLACE FUNCTION workflow_executions_fts_delete() RETURNS trigger AS $$
                BEGIN
                        DELETE FROM workflow_executions_fts WHERE execution_id = OLD.execution_id;
                        RETURN OLD;
                END;
                $$ LANGUAGE plpgsql;`,
		`DROP TRIGGER IF EXISTS workflow_executions_fts_insert ON workflow_executions;`,
		`DROP TRIGGER IF EXISTS workflow_executions_fts_update ON workflow_executions;`,
		`DROP TRIGGER IF EXISTS workflow_executions_fts_delete ON workflow_executions;`,
		`CREATE TRIGGER workflow_executions_fts_insert
                        AFTER INSERT ON workflow_executions
                        FOR EACH ROW EXECUTE FUNCTION workflow_executions_fts_upsert();`,
		`CREATE TRIGGER workflow_executions_fts_update
                        AFTER UPDATE ON workflow_executions
                        FOR EACH ROW EXECUTE FUNCTION workflow_executions_fts_upsert();`,
		`CREATE TRIGGER workflow_executions_fts_delete
                        AFTER DELETE ON workflow_executions
                        FOR EACH ROW EXECUTE FUNCTION workflow_executions_fts_delete();`,
		`INSERT INTO workflow_executions_fts(execution_id, workflow_id, agent_node_id, session_id, workflow_name, search_vector)
                        SELECT execution_id, workflow_id, agent_node_id, session_id, workflow_name,
                               to_tsvector('simple', coalesce(workflow_name, '') || ' ' || coalesce(execution_id, '') || ' ' || coalesce(workflow_id, ''))
                        FROM workflow_executions
                        ON CONFLICT (execution_id) DO NOTHING;`,
		`CREATE INDEX IF NOT EXISTS idx_workflow_executions_fts_vector ON workflow_executions_fts USING GIN(search_vector);`,
	}

	for _, stmt := range statements {
		if _, err := ls.db.Exec(stmt); err != nil {
			return err
		}
	}
	return nil
}

func (ls *LocalStorage) ensurePostgresIndexes(ctx context.Context) error {
	indexStatements := []string{
		"CREATE INDEX IF NOT EXISTS idx_agent_config_agent_package ON agent_configurations(agent_id, package_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_runs_status ON workflow_runs(status)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_runs_root ON workflow_runs(root_workflow_id)",
		"CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_steps_run_execution ON workflow_steps(run_id, execution_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_steps_run_status ON workflow_steps(run_id, status)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_steps_status_not_before ON workflow_steps(status, not_before)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_steps_parent ON workflow_steps(parent_step_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON workflow_executions(workflow_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_execution_id ON workflow_executions(execution_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_session_id ON workflow_executions(session_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_actor_id ON workflow_executions(actor_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_agent_node ON workflow_executions(agent_node_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_started_at ON workflow_executions(started_at)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_parent_execution_id ON workflow_executions(parent_execution_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_parent_workflow_id ON workflow_executions(parent_workflow_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_root_workflow_id ON workflow_executions(root_workflow_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status)",
		"CREATE INDEX IF NOT EXISTS idx_agent_nodes_group_id ON agent_nodes(group_id)",
	}

	for _, stmt := range indexStatements {
		if _, err := ls.db.Exec(stmt); err != nil {
			return err
		}
	}

	return nil
}

func (ls *LocalStorage) setupWorkflowExecutionFTS() error {
	createFTSTable := `
        CREATE VIRTUAL TABLE IF NOT EXISTS workflow_executions_fts USING fts5(
                execution_id,
                workflow_id,
                agent_node_id,
                session_id,
                workflow_name
        );`

	if _, err := ls.db.Exec(createFTSTable); err != nil {
		return fmt.Errorf("failed to create FTS5 virtual table: %w", err)
	}

	createFTSTriggers := []string{
		`CREATE TRIGGER IF NOT EXISTS workflow_executions_fts_insert AFTER INSERT ON workflow_executions BEGIN
                        INSERT INTO workflow_executions_fts(rowid, execution_id, workflow_id, agent_node_id, session_id, workflow_name)
                        VALUES (new.id, new.execution_id, new.workflow_id, new.agent_node_id, new.session_id, new.workflow_name);
                END;`,
		`CREATE TRIGGER IF NOT EXISTS workflow_executions_fts_update AFTER UPDATE ON workflow_executions BEGIN
                        UPDATE workflow_executions_fts SET
                                execution_id = new.execution_id,
                                workflow_id = new.workflow_id,
                                agent_node_id = new.agent_node_id,
                                session_id = new.session_id,
                                workflow_name = new.workflow_name
                        WHERE rowid = new.id;
                END;`,
		`CREATE TRIGGER IF NOT EXISTS workflow_executions_fts_delete AFTER DELETE ON workflow_executions BEGIN
                        DELETE FROM workflow_executions_fts WHERE rowid = old.id;
                END;`,
	}

	for _, triggerSQL := range createFTSTriggers {
		if _, err := ls.db.Exec(triggerSQL); err != nil {
			return fmt.Errorf("failed to create FTS5 trigger: %w", err)
		}
	}

	populateFTS := `
        INSERT INTO workflow_executions_fts(rowid, execution_id, workflow_id, agent_node_id, session_id, workflow_name)
        SELECT id, execution_id, workflow_id, agent_node_id, session_id, workflow_name
        FROM workflow_executions
        WHERE NOT EXISTS (SELECT 1 FROM workflow_executions_fts WHERE rowid = workflow_executions.id);`

	if _, err := ls.db.Exec(populateFTS); err != nil {
		return fmt.Errorf("failed to populate FTS5 table: %w", err)
	}

	return nil
}

func (ls *LocalStorage) ensureSQLiteIndexes() error {
	indexStatements := []string{
		"CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_config_agent_package ON agent_configurations(agent_id, package_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_runs_status ON workflow_runs(status)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_runs_root ON workflow_runs(root_workflow_id)",
		"CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_steps_run_execution ON workflow_steps(run_id, execution_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_steps_run_status ON workflow_steps(run_id, status)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_steps_status_not_before ON workflow_steps(status, not_before)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_steps_parent ON workflow_steps(parent_step_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_id ON workflow_executions(workflow_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_execution_id ON workflow_executions(execution_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_session_id ON workflow_executions(session_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_actor_id ON workflow_executions(actor_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_agent_node ON workflow_executions(agent_node_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_started_at ON workflow_executions(started_at)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_parent_execution_id ON workflow_executions(parent_execution_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_parent_workflow_id ON workflow_executions(parent_workflow_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_root_workflow_id ON workflow_executions(root_workflow_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_agent_node_status ON workflow_executions(agent_node_id, status)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_session_status ON workflow_executions(session_id, status)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_actor_status ON workflow_executions(actor_id, status)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow_status ON workflow_executions(workflow_id, status)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_runs_created_at ON workflow_runs(created_at)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_runs_updated_at ON workflow_runs(updated_at)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_steps_created_at ON workflow_steps(created_at)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_steps_updated_at ON workflow_steps(updated_at)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_steps_agent_not_before ON workflow_steps(agent_node_id, status, not_before)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_steps_run_priority ON workflow_steps(run_id, priority DESC, not_before)",
		"CREATE INDEX IF NOT EXISTS idx_workflows_session ON workflows(session_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflows_actor ON workflows(actor_id)",
		"CREATE INDEX IF NOT EXISTS idx_sessions_actor ON sessions(actor_id)",
		"CREATE INDEX IF NOT EXISTS idx_sessions_root ON sessions(root_session_id)",
		"CREATE INDEX IF NOT EXISTS idx_agent_nodes_team ON agent_nodes(team_id)",
		"CREATE INDEX IF NOT EXISTS idx_agent_nodes_health ON agent_nodes(health_status)",
		"CREATE INDEX IF NOT EXISTS idx_agent_nodes_lifecycle ON agent_nodes(lifecycle_status)",
		"CREATE INDEX IF NOT EXISTS idx_agent_nodes_group_id ON agent_nodes(group_id)",
		"CREATE INDEX IF NOT EXISTS idx_agent_dids_agent_node ON agent_dids(agent_node_id)",
		"CREATE INDEX IF NOT EXISTS idx_agent_dids_agentfield_server ON agent_dids(agentfield_server_id)",
		"CREATE INDEX IF NOT EXISTS idx_component_dids_agent_did ON component_dids(agent_did)",
		"CREATE INDEX IF NOT EXISTS idx_component_dids_type ON component_dids(component_type)",
		"CREATE INDEX IF NOT EXISTS idx_execution_vcs_execution_id ON execution_vcs(execution_id)",
		"CREATE INDEX IF NOT EXISTS idx_execution_vcs_workflow_id ON execution_vcs(workflow_id)",
		"CREATE INDEX IF NOT EXISTS idx_execution_vcs_session_id ON execution_vcs(session_id)",
		"CREATE INDEX IF NOT EXISTS idx_execution_vcs_issuer_did ON execution_vcs(issuer_did)",
		"CREATE INDEX IF NOT EXISTS idx_execution_vcs_target_did ON execution_vcs(target_did)",
		"CREATE INDEX IF NOT EXISTS idx_execution_vcs_caller_did ON execution_vcs(caller_did)",
		"CREATE INDEX IF NOT EXISTS idx_execution_vcs_status ON execution_vcs(status)",
		"CREATE INDEX IF NOT EXISTS idx_execution_vcs_parent_vc_id ON execution_vcs(parent_vc_id)",
		"CREATE INDEX IF NOT EXISTS idx_execution_vcs_created_at ON execution_vcs(created_at)",
		"CREATE UNIQUE INDEX IF NOT EXISTS idx_execution_vcs_execution_unique ON execution_vcs(execution_id, issuer_did, target_did)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_vcs_workflow_id ON workflow_vcs(workflow_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_vcs_session_id ON workflow_vcs(session_id)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_vcs_status ON workflow_vcs(status)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_vcs_start_time ON workflow_vcs(start_time)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_vcs_end_time ON workflow_vcs(end_time)",
		"CREATE INDEX IF NOT EXISTS idx_workflow_vcs_created_at ON workflow_vcs(created_at)",
		"CREATE UNIQUE INDEX IF NOT EXISTS idx_workflow_vcs_workflow_session ON workflow_vcs(workflow_id, session_id)",
	}

	for _, stmt := range indexStatements {
		if _, err := ls.db.Exec(stmt); err != nil {
			return fmt.Errorf("failed to create index '%s': %w", stmt, err)
		}
	}

	return nil
}

func (ls *LocalStorage) ensureVectorSchema(ctx context.Context) error {
	switch ls.mode {
	case "postgres":
		return ls.ensurePostgresVectorSchema(ctx)
	default:
		return ls.ensureSQLiteVectorSchema()
	}
}

func (ls *LocalStorage) ensureSQLiteVectorSchema() error {
	statements := []string{
		`CREATE TABLE IF NOT EXISTS memory_vectors (
			scope TEXT NOT NULL,
			scope_id TEXT NOT NULL,
			key TEXT NOT NULL,
			dimension INTEGER NOT NULL,
			embedding BLOB NOT NULL,
			metadata JSON DEFAULT '{}',
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			PRIMARY KEY(scope, scope_id, key)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_memory_vectors_scope ON memory_vectors(scope, scope_id);`,
		`CREATE INDEX IF NOT EXISTS idx_memory_vectors_updated ON memory_vectors(scope, scope_id, updated_at);`,
	}

	for _, stmt := range statements {
		if _, err := ls.db.Exec(stmt); err != nil {
			return fmt.Errorf("failed to ensure sqlite vector schema: %w", err)
		}
	}
	return nil
}

func (ls *LocalStorage) ensurePostgresVectorSchema(ctx context.Context) error {
	statements := []string{
		`CREATE EXTENSION IF NOT EXISTS vector;`,
		`CREATE TABLE IF NOT EXISTS memory_vectors (
			scope TEXT NOT NULL,
			scope_id TEXT NOT NULL,
			key TEXT NOT NULL,
			embedding vector NOT NULL,
			metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
			created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
			PRIMARY KEY(scope, scope_id, key)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_memory_vectors_scope ON memory_vectors(scope, scope_id);`,
		`CREATE INDEX IF NOT EXISTS idx_memory_vectors_metadata ON memory_vectors USING GIN(metadata);`,
	}

	for _, stmt := range statements {
		if _, err := ls.db.ExecContext(ctx, stmt); err != nil {
			return fmt.Errorf("failed to ensure postgres vector schema: %w", err)
		}
	}
	return nil
}

func (ls *LocalStorage) initializeVectorStore() error {
	if !ls.vectorConfig.isEnabled() {
		ls.vectorStore = nil
		return nil
	}

	switch ls.mode {
	case "postgres":
		ls.vectorStore = newPostgresVectorStore(ls.db, ls.vectorMetric)
	default:
		ls.vectorStore = newSQLiteVectorStore(ls.db, ls.vectorMetric)
	}
	return nil
}

func (ls *LocalStorage) runPostgresMigrations(ctx context.Context) error {
	_, err := ls.db.Exec(`
                CREATE TABLE IF NOT EXISTS schema_migrations (
                        version TEXT PRIMARY KEY,
                        applied_at TIMESTAMPTZ DEFAULT NOW(),
                        description TEXT
                );`)
	if err != nil {
		return fmt.Errorf("failed to create schema_migrations table: %w", err)
	}

	migrations := []struct {
		version     string
		description string
		sql         string
	}{
		{
			version:     "015",
			description: "Backfill group_id on agent_nodes with id",
			sql:         `UPDATE agent_nodes SET group_id = id WHERE group_id = '' OR group_id IS NULL;`,
		},
	}

	for _, m := range migrations {
		var count int
		err := ls.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM schema_migrations WHERE version = $1`, m.version).Scan(&count)
		if err != nil {
			return fmt.Errorf("failed to check migration %s: %w", m.version, err)
		}
		if count > 0 {
			continue
		}
		if _, err := ls.db.ExecContext(ctx, m.sql); err != nil {
			return fmt.Errorf("failed to apply migration %s: %w", m.version, err)
		}
		if _, err := ls.db.ExecContext(ctx, `INSERT INTO schema_migrations (version, description) VALUES ($1, $2)`, m.version, m.description); err != nil {
			return fmt.Errorf("failed to record migration %s: %w", m.version, err)
		}
		log.Printf("Applied postgres migration %s: %s", m.version, m.description)
	}

	return nil
}

// buildExecutionVCTableSQL returns the CREATE TABLE statement for execution VC storage.
func buildExecutionVCTableSQL(tableName string, includeIfNotExists bool) string {
	keyword := "CREATE TABLE"
	if includeIfNotExists {
		keyword += " IF NOT EXISTS"
	}
	keyword += " "

	return fmt.Sprintf(`%s%s (
		vc_id TEXT PRIMARY KEY,
		execution_id TEXT NOT NULL,
		workflow_id TEXT NOT NULL,
		session_id TEXT NOT NULL,
		issuer_did TEXT NOT NULL,
		target_did TEXT,
		caller_did TEXT NOT NULL,
		vc_document TEXT NOT NULL,
		signature TEXT NOT NULL,
		storage_uri TEXT DEFAULT '',
		document_size_bytes INTEGER DEFAULT 0,
		input_hash TEXT NOT NULL,
		output_hash TEXT NOT NULL,
		status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('unknown', 'pending', 'queued', 'running', 'waiting', 'paused', 'succeeded', 'failed', 'cancelled', 'timeout', 'revoked')),
		parent_vc_id TEXT,
		child_vc_ids TEXT DEFAULT '[]',
		created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
		updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
		FOREIGN KEY (parent_vc_id) REFERENCES %s(vc_id) ON DELETE SET NULL
	);`, keyword, tableName, tableName)
}

func buildWorkflowVCTableSQL(tableName string, includeIfNotExists bool) string {
	keyword := "CREATE TABLE"
	if includeIfNotExists {
		keyword += " IF NOT EXISTS"
	}
	keyword += " "

	return fmt.Sprintf(`%s%s (
		workflow_vc_id TEXT PRIMARY KEY,
		workflow_id TEXT NOT NULL,
		session_id TEXT NOT NULL,
		component_vc_ids TEXT DEFAULT '[]',
		status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('unknown', 'pending', 'in_progress', 'running', 'waiting', 'paused', 'succeeded', 'failed', 'cancelled', 'timeout')),
		start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
		end_time TIMESTAMP,
		total_steps INTEGER DEFAULT 0,
		completed_steps INTEGER DEFAULT 0,
		storage_uri TEXT DEFAULT '',
		document_size_bytes INTEGER DEFAULT 0,
		created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
		updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
	);`, keyword, tableName)
}

// ensureExecutionVCSchema removes outdated foreign key constraints that prevented
// execution verifiable credentials from persisting when referencing non-component DIDs.
func (ls *LocalStorage) ensureExecutionVCSchema() error {
	var tableCount int
	if err := ls.db.QueryRow("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='execution_vcs'").Scan(&tableCount); err != nil {
		return fmt.Errorf("failed to inspect execution_vcs table: %w", err)
	}
	if tableCount == 0 {
		return nil
	}

	needsMigration := false

	rows, err := ls.db.Query("PRAGMA foreign_key_list('execution_vcs')")
	if err != nil {
		return fmt.Errorf("failed to inspect execution_vcs foreign keys: %w", err)
	}
	for rows.Next() {
		var (
			id, seq   int
			tableName string
			fromCol   string
			toCol     string
			onUpdate  string
			onDelete  string
			match     string
		)
		if err := rows.Scan(&id, &seq, &tableName, &fromCol, &toCol, &onUpdate, &onDelete, &match); err != nil {
			rows.Close()
			return fmt.Errorf("failed to scan execution_vcs foreign key info: %w", err)
		}
		if tableName == "component_dids" {
			needsMigration = true
			break
		}
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return fmt.Errorf("failed during execution_vcs foreign key inspection: %w", err)
	}
	rows.Close()

	if !needsMigration {
		var createSQL string
		if err := ls.db.QueryRow("SELECT sql FROM sqlite_master WHERE type='table' AND name='execution_vcs'").Scan(&createSQL); err != nil {
			return fmt.Errorf("failed to inspect execution_vcs schema: %w", err)
		}
		if strings.Contains(createSQL, "status IN ('pending', 'completed', 'failed', 'revoked')") {
			needsMigration = true
		}
	}

	if !needsMigration {
		return nil
	}

	log.Printf("Migrating execution_vcs table to remove component_dids foreign keys for VC persistence")

	tx, err := ls.db.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin execution_vcs migration: %w", err)
	}
	committed := false
	defer func() {
		if !committed {
			rollbackTx(tx, "migrate_execution_vcs")
		}
	}()

	createNewSQL := buildExecutionVCTableSQL("execution_vcs_new", false)
	if _, err := tx.Exec(createNewSQL); err != nil {
		return fmt.Errorf("failed to create execution_vcs_new table: %w", err)
	}

	copySQL := `INSERT INTO execution_vcs_new (
		vc_id, execution_id, workflow_id, session_id, issuer_did, target_did, caller_did,
		vc_document, signature, storage_uri, document_size_bytes, input_hash, output_hash, status,
		parent_vc_id, child_vc_ids, created_at, updated_at
	) SELECT
		vc_id, execution_id, workflow_id, session_id, issuer_did, target_did, caller_did,
		vc_document, signature, COALESCE(storage_uri, ''), COALESCE(document_size_bytes, 0),
		input_hash, output_hash, status, parent_vc_id, COALESCE(child_vc_ids, '[]'), created_at, updated_at
	FROM execution_vcs;`

	if _, err := tx.Exec(copySQL); err != nil {
		return fmt.Errorf("failed to copy data into execution_vcs_new: %w", err)
	}

	if _, err := tx.Exec("DROP TABLE execution_vcs;"); err != nil {
		return fmt.Errorf("failed to drop old execution_vcs table: %w", err)
	}

	if _, err := tx.Exec("ALTER TABLE execution_vcs_new RENAME TO execution_vcs;"); err != nil {
		return fmt.Errorf("failed to rename execution_vcs_new table: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit execution_vcs schema migration: %w", err)
	}
	committed = true

	return nil
}

func (ls *LocalStorage) ensureWorkflowVCSchema() error {
	var tableCount int
	if err := ls.db.QueryRow("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='workflow_vcs'").Scan(&tableCount); err != nil {
		return fmt.Errorf("failed to inspect workflow_vcs table: %w", err)
	}
	if tableCount == 0 {
		return nil
	}

	var createSQL string
	if err := ls.db.QueryRow("SELECT sql FROM sqlite_master WHERE type='table' AND name='workflow_vcs'").Scan(&createSQL); err != nil {
		return fmt.Errorf("failed to inspect workflow_vcs schema: %w", err)
	}

	if !strings.Contains(createSQL, "status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled')") {
		return nil
	}

	log.Printf("Migrating workflow_vcs table to update status constraint")

	tx, err := ls.db.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin workflow_vcs migration: %w", err)
	}
	committed := false
	defer func() {
		if !committed {
			rollbackTx(tx, "migrate_workflow_vcs")
		}
	}()

	createNewSQL := buildWorkflowVCTableSQL("workflow_vcs_new", false)
	if _, err := tx.Exec(createNewSQL); err != nil {
		return fmt.Errorf("failed to create workflow_vcs_new table: %w", err)
	}

	copySQL := `INSERT INTO workflow_vcs_new (
		workflow_vc_id, workflow_id, session_id, component_vc_ids, status,
		start_time, end_time, total_steps, completed_steps, storage_uri,
		document_size_bytes, created_at, updated_at
	) SELECT
		workflow_vc_id, workflow_id, session_id, component_vc_ids, status,
		start_time, end_time, total_steps, completed_steps, storage_uri,
		document_size_bytes, created_at, updated_at
	FROM workflow_vcs;`

	if _, err := tx.Exec(copySQL); err != nil {
		return fmt.Errorf("failed to copy data into workflow_vcs_new: %w", err)
	}

	if _, err := tx.Exec("DROP TABLE workflow_vcs;"); err != nil {
		return fmt.Errorf("failed to drop old workflow_vcs table: %w", err)
	}

	if _, err := tx.Exec("ALTER TABLE workflow_vcs_new RENAME TO workflow_vcs;"); err != nil {
		return fmt.Errorf("failed to rename workflow_vcs_new table: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit workflow_vcs schema migration: %w", err)
	}
	committed = true

	return nil
}

// runMigrations handles database schema migrations for existing databases
func (ls *LocalStorage) runMigrations() error {
	// Create migrations tracking table if it doesn't exist
	createMigrationsTable := `
		CREATE TABLE IF NOT EXISTS schema_migrations (
			version TEXT PRIMARY KEY,
			applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
			description TEXT
		);`

	_, err := ls.db.Exec(createMigrationsTable)
	if err != nil {
		return fmt.Errorf("failed to create schema_migrations table: %w", err)
	}

	// Define all migrations with their SQL content
	migrations := []struct {
		version     string
		description string
		sql         string
	}{
		{
			version:     "007",
			description: "Add parent_execution_id column",
			sql:         `ALTER TABLE workflow_executions ADD COLUMN parent_execution_id TEXT;`,
		},
		{
			version:     "008",
			description: "Create FTS5 search table",
			sql: `
				-- Check if FTS table exists before creating
				CREATE VIRTUAL TABLE IF NOT EXISTS workflow_executions_fts USING fts5(
					execution_id,
					workflow_id,
					agent_node_id,
					session_id,
					workflow_name
				);

				-- Drop existing triggers if they exist to avoid conflicts
				DROP TRIGGER IF EXISTS workflow_executions_fts_insert;
				DROP TRIGGER IF EXISTS workflow_executions_fts_update;
				DROP TRIGGER IF EXISTS workflow_executions_fts_delete;

				-- Create triggers
				CREATE TRIGGER workflow_executions_fts_insert AFTER INSERT ON workflow_executions BEGIN
					INSERT INTO workflow_executions_fts(rowid, execution_id, workflow_id, agent_node_id, session_id, workflow_name)
					VALUES (new.id, new.execution_id, new.workflow_id, new.agent_node_id, new.session_id, new.workflow_name);
				END;

				CREATE TRIGGER workflow_executions_fts_update AFTER UPDATE ON workflow_executions BEGIN
					UPDATE workflow_executions_fts SET
						execution_id = new.execution_id,
						workflow_id = new.workflow_id,
						agent_node_id = new.agent_node_id,
						session_id = new.session_id,
						workflow_name = new.workflow_name
					WHERE rowid = new.id;
				END;

				CREATE TRIGGER workflow_executions_fts_delete AFTER DELETE ON workflow_executions BEGIN
					DELETE FROM workflow_executions_fts WHERE rowid = old.id;
				END;

				-- Populate FTS table with existing data (ignore duplicates)
				INSERT OR IGNORE INTO workflow_executions_fts(rowid, execution_id, workflow_id, agent_node_id, session_id, workflow_name)
				SELECT id, execution_id, workflow_id, agent_node_id, session_id, workflow_name
				FROM workflow_executions
				WHERE NOT EXISTS (SELECT 1 FROM workflow_executions_fts WHERE rowid = workflow_executions.id);`,
		},
		{
			version:     "009",
			description: "Add notes column to workflow_executions",
			sql:         `ALTER TABLE workflow_executions ADD COLUMN notes TEXT DEFAULT '[]';`,
		},
		{
			version:     "010",
			description: "Add composite indexes for workflow execution filtering performance",
			sql: `
				-- Composite index for session + status + time queries
				CREATE INDEX IF NOT EXISTS idx_workflow_executions_session_status_time ON workflow_executions(session_id, status, started_at);

				-- Composite index for actor + status + time queries
				CREATE INDEX IF NOT EXISTS idx_workflow_executions_actor_status_time ON workflow_executions(actor_id, status, started_at);

				-- Composite index for agent + status + time queries
				CREATE INDEX IF NOT EXISTS idx_workflow_executions_agent_status_time ON workflow_executions(agent_node_id, status, started_at);

				-- Composite index for status + time queries
				CREATE INDEX IF NOT EXISTS idx_workflow_executions_status_time ON workflow_executions(status, started_at);

				-- Composite index for session + time queries (without status filter)
				CREATE INDEX IF NOT EXISTS idx_workflow_executions_session_time ON workflow_executions(session_id, started_at);

				-- Composite index for actor + time queries (without status filter)
				CREATE INDEX IF NOT EXISTS idx_workflow_executions_actor_time ON workflow_executions(actor_id, started_at);`,
		},
		{
			version:     "011",
			description: "Add storage URI column to execution_vcs",
			sql:         `ALTER TABLE execution_vcs ADD COLUMN storage_uri TEXT DEFAULT '';`,
		},
		{
			version:     "012",
			description: "Add document size column to execution_vcs",
			sql:         `ALTER TABLE execution_vcs ADD COLUMN document_size_bytes INTEGER DEFAULT 0;`,
		},
		{
			version:     "013",
			description: "Add storage URI column to workflow_vcs",
			sql:         `ALTER TABLE workflow_vcs ADD COLUMN storage_uri TEXT DEFAULT '';`,
		},
		{
			version:     "014",
			description: "Add document size column to workflow_vcs",
			sql:         `ALTER TABLE workflow_vcs ADD COLUMN document_size_bytes INTEGER DEFAULT 0;`,
		},
		{
			version:     "015",
			description: "Backfill group_id on agent_nodes with id",
			sql:         `UPDATE agent_nodes SET group_id = id WHERE group_id = '' OR group_id IS NULL;`,
		},
	}

	// Apply each migration if not already applied
	for _, migration := range migrations {
		// Check if migration has already been applied
		var count int
		checkQuery := `SELECT COUNT(*) FROM schema_migrations WHERE version = ?`
		err := ls.db.QueryRow(checkQuery, migration.version).Scan(&count)
		if err != nil {
			return fmt.Errorf("failed to check migration status for version %s: %w", migration.version, err)
		}

		if count > 0 {
			// Migration already applied, skip
			continue
		}

		// Apply the migration
		log.Printf("Applying migration %s: %s", migration.version, migration.description)

		// Execute the migration SQL
		_, err = ls.db.Exec(migration.sql)
		if err != nil {
			// For ALTER TABLE operations, check if column already exists
			if strings.Contains(err.Error(), "duplicate column name") {
				log.Printf("Column already exists for migration %s, marking as applied", migration.version)
			} else if strings.Contains(err.Error(), "no such module: fts5") {
				log.Printf("FTS5 module not available, skipping migration %s (search will be degraded)", migration.version)
			} else {
				return fmt.Errorf("failed to apply migration %s: %w", migration.version, err)
			}
		}

		// Record that the migration has been applied
		insertQuery := `INSERT INTO schema_migrations (version, description) VALUES (?, ?)`
		_, err = ls.db.Exec(insertQuery, migration.version, migration.description)
		if err != nil {
			return fmt.Errorf("failed to record migration %s: %w", migration.version, err)
		}

		log.Printf("Successfully applied migration %s", migration.version)
	}

	return nil
}

// sanitizeFTS5Query sanitizes user input for FTS5 MATCH queries to prevent syntax errors
func sanitizeFTS5Query(query string) string {
	if query == "" {
		return ""
	}

	// Remove or escape FTS5 special characters that can cause syntax errors
	// FTS5 special characters: " * ( ) AND OR NOT
	specialChars := regexp.MustCompile(`[*"()]+`)
	sanitized := specialChars.ReplaceAllString(query, " ")

	// Replace FTS5 operators with spaces to avoid syntax errors
	operatorPattern := regexp.MustCompile(`(?i)\b(AND|OR|NOT)\b`)
	sanitized = operatorPattern.ReplaceAllString(sanitized, " ")

	// Clean up multiple spaces and trim
	spacePattern := regexp.MustCompile(`\s+`)
	sanitized = spacePattern.ReplaceAllString(sanitized, " ")
	sanitized = strings.TrimSpace(sanitized)

	// If the sanitized query is empty, return empty string
	if sanitized == "" {
		return ""
	}

	// Wrap in quotes for phrase search to avoid further syntax issues
	return `"` + sanitized + `"`
}

// Close closes the SQLite and BoltDB connections.
func (ls *LocalStorage) Close(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during close: %w", err)
	}

	if ls.db != nil {
		if err := ls.db.Close(); err != nil {
			return fmt.Errorf("failed to close database: %w", err)
		}
	}
	ls.gormDB = nil
	if ls.kvStore != nil {
		if err := ls.kvStore.Close(); err != nil {
			return fmt.Errorf("failed to close BoltDB database: %w", err)
		}
	}
	return nil
}

// HealthCheck checks the health of the local storage including database integrity.
func (ls *LocalStorage) HealthCheck(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during health check: %w", err)
	}

	if ls.db == nil {
		return fmt.Errorf("database connection is not initialized")
	}

	if err := ls.db.PingContext(ctx); err != nil {
		return fmt.Errorf("database is unhealthy: %w", err)
	}

	switch ls.mode {
	case "postgres":
		if err := ls.db.QueryRowContext(ctx, "SELECT 1").Scan(new(int)); err != nil {
			return fmt.Errorf("postgres health check failed: %w", err)
		}
	default:
		var result string
		if err := ls.db.QueryRowContext(ctx, "PRAGMA integrity_check").Scan(&result); err != nil {
			return fmt.Errorf("database integrity check failed: %w", err)
		}
		if result != "ok" {
			return fmt.Errorf("database integrity compromised: %s", result)
		}
	}

	if ls.kvStore != nil {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("context cancelled before BoltDB health check: %w", err)
		}
		if err := ls.kvStore.View(func(tx *bolt.Tx) error {
			if tx == nil {
				return fmt.Errorf("BoltDB transaction is nil")
			}
			return nil
		}); err != nil {
			return fmt.Errorf("BoltDB health check failed: %w", err)
		}
	}
	return nil
}

// StoreExecution stores an agent execution record in SQLite.
func (ls *LocalStorage) StoreExecution(ctx context.Context, execution *types.AgentExecution) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store execution: %w", err)
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	model, err := agentExecutionToModel(execution)
	if err != nil {
		return err
	}

	result := gormDB.Create(model)
	if result.Error != nil {
		return fmt.Errorf("failed to store agent execution: %w", result.Error)
	}

	execution.ID = model.ID
	return nil
}

// GetExecution retrieves an agent execution record from SQLite by ID.
func (ls *LocalStorage) GetExecution(ctx context.Context, id int64) (*types.AgentExecution, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get execution: %w", err)
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	model := &AgentExecutionModel{}
	if err := gormDB.Where("id = ?", id).Take(model).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, fmt.Errorf("execution with ID %d not found", id)
		}
		return nil, fmt.Errorf("failed to get execution with ID %d: %w", id, err)
	}

	return agentExecutionFromModel(model)
}

// QueryExecutions retrieves agent execution records based on filters using GORM.
func (ls *LocalStorage) QueryExecutions(ctx context.Context, filters types.ExecutionFilters) ([]*types.AgentExecution, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during query executions: %w", err)
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	query := gormDB.Model(&AgentExecutionModel{})

	if filters.WorkflowID != nil {
		query = query.Where("workflow_id = ?", *filters.WorkflowID)
	}
	if filters.SessionID != nil {
		query = query.Where("session_id = ?", *filters.SessionID)
	}
	if filters.AgentNodeID != nil {
		query = query.Where("agent_node_id = ?", *filters.AgentNodeID)
	}
	if filters.ReasonerID != nil {
		query = query.Where("reasoner_id = ?", *filters.ReasonerID)
	}
	if filters.Status != nil {
		query = query.Where("status = ?", *filters.Status)
	}
	if filters.UserID != nil {
		query = query.Where("user_id = ?", *filters.UserID)
	}
	if filters.TeamID != nil {
		query = query.Where("team_id = ?", *filters.TeamID)
	}
	if filters.StartTime != nil {
		query = query.Where("created_at >= ?", filters.StartTime.UTC())
	}
	if filters.EndTime != nil {
		query = query.Where("created_at <= ?", filters.EndTime.UTC())
	}

	query = query.Order("created_at DESC")
	if filters.Limit > 0 {
		query = query.Limit(filters.Limit)
	}
	if filters.Offset > 0 {
		query = query.Offset(filters.Offset)
	}

	var models []AgentExecutionModel
	if err := query.Find(&models).Error; err != nil {
		return nil, fmt.Errorf("failed to query agent executions: %w", err)
	}

	executions := make([]*types.AgentExecution, 0, len(models))
	for i := range models {
		exec, err := agentExecutionFromModel(&models[i])
		if err != nil {
			return nil, err
		}
		executions = append(executions, exec)
	}

	return executions, nil
}
func agentExecutionToModel(exec *types.AgentExecution) (*AgentExecutionModel, error) {
	metadataJSON, err := json.Marshal(exec.Metadata)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal execution metadata: %w", err)
	}

	model := &AgentExecutionModel{
		ID:           exec.ID,
		WorkflowID:   exec.WorkflowID,
		SessionID:    exec.SessionID,
		AgentNodeID:  exec.AgentNodeID,
		ReasonerID:   exec.ReasonerID,
		InputData:    []byte(exec.InputData),
		OutputData:   []byte(exec.OutputData),
		InputSize:    exec.InputSize,
		OutputSize:   exec.OutputSize,
		DurationMS:   exec.DurationMS,
		Status:       exec.Status,
		ErrorMessage: exec.ErrorMessage,
		UserID:       exec.UserID,
		TeamID:       exec.NodeID,
		Metadata:     metadataJSON,
		CreatedAt:    exec.CreatedAt,
	}

	return model, nil
}

func agentExecutionFromModel(model *AgentExecutionModel) (*types.AgentExecution, error) {
	exec := &types.AgentExecution{
		ID:           model.ID,
		WorkflowID:   model.WorkflowID,
		SessionID:    model.SessionID,
		AgentNodeID:  model.AgentNodeID,
		ReasonerID:   model.ReasonerID,
		InputData:    json.RawMessage(append([]byte(nil), model.InputData...)),
		OutputData:   json.RawMessage(append([]byte(nil), model.OutputData...)),
		InputSize:    model.InputSize,
		OutputSize:   model.OutputSize,
		DurationMS:   model.DurationMS,
		Status:       model.Status,
		ErrorMessage: model.ErrorMessage,
		UserID:       model.UserID,
		NodeID:       model.TeamID,
		CreatedAt:    model.CreatedAt,
	}

	if len(model.Metadata) > 0 {
		if err := json.Unmarshal(model.Metadata, &exec.Metadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal execution metadata: %w", err)
		}
	}

	return exec, nil
}

// StoreWorkflowExecution stores a workflow execution record in SQLite with UPSERT capability
// Uses transactions to prevent database corruption - SQLite WAL mode handles write coordination
func (ls *LocalStorage) StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store workflow execution: %w", err)
	}

	// 🔧 FIX: Add retry logic for database lock errors
	return ls.retryDatabaseOperation(ctx, execution.ExecutionID, func() error {
		return ls.storeWorkflowExecutionInternal(ctx, execution)
	})
}

// storeWorkflowExecutionInternal performs the actual storage operation
func (ls *LocalStorage) storeWorkflowExecutionInternal(ctx context.Context, execution *types.WorkflowExecution) error {
	// DIAGNOSTIC: Log concurrent transaction attempt
	log.Printf("🔒 CONCURRENCY_DEBUG: Starting transaction for execution %s", execution.ExecutionID)

	// Begin transaction for atomic operation
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		// DIAGNOSTIC: Log database lock errors
		if ls.isRetryableError(err) {
			log.Printf("🚨 DATABASE_LOCK: Failed to begin transaction for execution %s: %v", execution.ExecutionID, err)
		}
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer rollbackTx(tx, "storeWorkflowExecution:"+execution.ExecutionID)

	// Execute the workflow insert using the transaction
	if err := ls.executeWorkflowInsert(ctx, tx, execution); err != nil {
		// DIAGNOSTIC: Log insert/update failures
		if ls.isRetryableError(err) {
			log.Printf("🚨 DATABASE_LOCK: Failed to execute workflow insert for execution %s: %v", execution.ExecutionID, err)
		}
		return err
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		// DIAGNOSTIC: Log commit failures
		if ls.isRetryableError(err) {
			log.Printf("🚨 DATABASE_LOCK: Failed to commit transaction for execution %s: %v", execution.ExecutionID, err)
		}
		return fmt.Errorf("failed to commit workflow execution transaction: %w", err)
	}

	log.Printf("✅ CONCURRENCY_DEBUG: Successfully committed transaction for execution %s", execution.ExecutionID)
	return nil
}

// isRetryableError determines if a database error is retryable
func (ls *LocalStorage) isRetryableError(err error) bool {
	if err == nil {
		return false
	}
	errStr := strings.ToLower(err.Error())
	// Common retryable SQLite errors
	retryableErrors := []string{
		"database is locked",
		"database disk image is malformed",
		"disk i/o error",
		"attempt to write a readonly database",
		"busy",
		"sqlite_busy",
		"sqlite_locked",
		"cannot start a transaction within a transaction",
		"database table is locked",
	}

	for _, retryable := range retryableErrors {
		if strings.Contains(errStr, retryable) {
			return true
		}
	}
	return false
}

// retryDatabaseOperation implements exponential backoff retry for database operations
func (ls *LocalStorage) retryDatabaseOperation(ctx context.Context, operationID string, operation func() error) error {
	const maxRetries = 3
	const baseDelay = 50 * time.Millisecond

	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		// Check context cancellation before each attempt
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("context cancelled during retry attempt %d: %w", attempt, err)
		}

		err := operation()
		if err == nil {
			if attempt > 0 {
				log.Printf("🔄 RETRY_SUCCESS: Operation succeeded on attempt %d for %s", attempt+1, operationID)
			}
			return nil
		}

		lastErr = err

		// Check if error is retryable
		if !ls.isRetryableError(err) {
			log.Printf("❌ NON_RETRYABLE: Operation failed with non-retryable error for %s: %v", operationID, err)
			return err
		}

		// Don't retry on the last attempt
		if attempt == maxRetries {
			break
		}

		// Calculate delay with exponential backoff
		delay := time.Duration(1<<uint(attempt)) * baseDelay
		log.Printf("🔄 RETRY_ATTEMPT: Retrying operation for %s in %v (attempt %d/%d): %v", operationID, delay, attempt+1, maxRetries, err)

		// Wait with context cancellation support
		select {
		case <-ctx.Done():
			return fmt.Errorf("context cancelled during retry delay: %w", ctx.Err())
		case <-time.After(delay):
			// Continue to next attempt
		}
	}

	log.Printf("🚨 RETRY_EXHAUSTED: All retry attempts failed for %s: %v", operationID, lastErr)
	return fmt.Errorf("operation failed after %d retries: %w", maxRetries, lastErr)
}

// sqliteWorkflowExecutionInsertQuery captures the column order for workflow execution inserts.
const sqliteWorkflowExecutionInsertQuery = `INSERT INTO workflow_executions (
	workflow_id, execution_id, agentfield_request_id, run_id, session_id, actor_id,
	agent_node_id, parent_workflow_id, parent_execution_id, root_workflow_id, workflow_depth,
	reasoner_id, input_data, output_data, input_size, output_size,
	status, started_at, completed_at, duration_ms,
	state_version, last_event_sequence, active_children, pending_children,
	pending_terminal_status, status_reason, lease_owner, lease_expires_at,
	error_message, retry_count,
	approval_request_id, approval_request_url, approval_status, approval_response,
	approval_requested_at, approval_responded_at, approval_callback_url, approval_expires_at,
	workflow_name, workflow_tags, notes, created_at, updated_at
) VALUES (
	?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
	?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
	?, ?, ?, ?, ?, ?, ?, ?,
	?, ?, ?, ?, ?
)`

// executeWorkflowInsert performs the actual database insert/update operation
func (ls *LocalStorage) executeWorkflowInsert(ctx context.Context, q DBTX, execution *types.WorkflowExecution) error {
	// First, check if execution already exists to validate state transitions
	existingExecution, err := ls.getWorkflowExecutionByID(ctx, q, execution.ExecutionID)
	if err != nil && !strings.Contains(err.Error(), "not found") {
		return fmt.Errorf("failed to check existing execution: %w", err)
	}

	// If execution exists, validate the state transition
	if existingExecution != nil {
		if err := validateExecutionStateTransition(existingExecution.Status, execution.Status); err != nil {
			log.Printf("Invalid workflow execution state transition blocked: execution_id=%s, current=%s, new=%s",
				execution.ExecutionID, existingExecution.Status, execution.Status)

			// Add execution ID to the error for better context
			if stateErr, ok := err.(*InvalidExecutionStateTransitionError); ok {
				stateErr.ExecutionID = execution.ExecutionID
				return stateErr
			}
			return err
		}

		// Valid transition - perform UPDATE
		// Serialize notes to JSON for storage
		notesJSON, err := json.Marshal(execution.Notes)
		if err != nil {
			return fmt.Errorf("failed to marshal notes: %w", err)
		}

		updateQuery := `
			UPDATE workflow_executions SET
				status = ?, completed_at = ?, duration_ms = ?,
				state_version = ?, last_event_sequence = ?, active_children = ?, pending_children = ?,
				pending_terminal_status = ?, status_reason = ?, lease_owner = ?, lease_expires_at = ?,
				output_data = ?, output_size = ?, error_message = ?,
				approval_request_id = ?, approval_request_url = ?, approval_status = ?,
				approval_response = ?, approval_requested_at = ?, approval_responded_at = ?,
				approval_callback_url = ?, approval_expires_at = ?,
				notes = ?, updated_at = ?
			WHERE execution_id = ?`

		_, err = q.ExecContext(ctx, updateQuery,
			execution.Status, execution.CompletedAt, execution.DurationMS,
			execution.StateVersion, execution.LastEventSequence, execution.ActiveChildren, execution.PendingChildren,
			execution.PendingTerminalStatus, execution.StatusReason, execution.LeaseOwner, execution.LeaseExpiresAt,
			execution.OutputData, execution.OutputSize, execution.ErrorMessage,
			execution.ApprovalRequestID, execution.ApprovalRequestURL, execution.ApprovalStatus,
			execution.ApprovalResponse, execution.ApprovalRequestedAt, execution.ApprovalRespondedAt,
			execution.ApprovalCallbackURL, execution.ApprovalExpiresAt,
			notesJSON, time.Now(), execution.ExecutionID)

		if err != nil {
			return fmt.Errorf("failed to update workflow execution: %w", err)
		}

		log.Printf("Successfully updated workflow execution: execution_id=%s, status=%s", execution.ExecutionID, execution.Status)
		return nil
	}

	// New execution - perform INSERT
	insertQuery := sqliteWorkflowExecutionInsertQuery

	workflowTagsJSON, err := json.Marshal(execution.WorkflowTags)
	if err != nil {
		return fmt.Errorf("failed to marshal workflow tags: %w", err)
	}

	// Serialize notes to JSON for storage
	notesJSON, err := json.Marshal(execution.Notes)
	if err != nil {
		return fmt.Errorf("failed to marshal notes: %w", err)
	}

	// Set default timestamps if not provided
	if execution.CreatedAt.IsZero() {
		execution.CreatedAt = time.Now()
	}
	if execution.UpdatedAt.IsZero() {
		execution.UpdatedAt = time.Now()
	}

	// Execute INSERT query using the DBTX interface
	_, err = q.ExecContext(ctx, insertQuery,
		execution.WorkflowID, execution.ExecutionID, execution.AgentFieldRequestID, execution.RunID,
		execution.SessionID, execution.ActorID, execution.AgentNodeID,
		execution.ParentWorkflowID, execution.ParentExecutionID, execution.RootWorkflowID, execution.WorkflowDepth,
		execution.ReasonerID, execution.InputData, execution.OutputData,
		execution.InputSize, execution.OutputSize,
		execution.Status, execution.StartedAt, execution.CompletedAt, execution.DurationMS,
		execution.StateVersion, execution.LastEventSequence, execution.ActiveChildren, execution.PendingChildren,
		execution.PendingTerminalStatus, execution.StatusReason, execution.LeaseOwner, execution.LeaseExpiresAt,
		execution.ErrorMessage, execution.RetryCount,
		execution.ApprovalRequestID, execution.ApprovalRequestURL, execution.ApprovalStatus,
		execution.ApprovalResponse, execution.ApprovalRequestedAt, execution.ApprovalRespondedAt,
		execution.ApprovalCallbackURL, execution.ApprovalExpiresAt,
		execution.WorkflowName,
		workflowTagsJSON, notesJSON, execution.CreatedAt, execution.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to insert workflow execution: %w", err)
	}

	log.Printf("Successfully inserted new workflow execution: execution_id=%s, status=%s", execution.ExecutionID, execution.Status)
	return nil
}

// UpdateWorkflowExecution atomically updates a workflow execution using a user-provided update function
// This eliminates the read-modify-write race condition by performing the entire operation within a single transaction
func (ls *LocalStorage) UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during update workflow execution: %w", err)
	}

	// Implement retry logic for database lock errors
	maxRetries := 3
	baseDelay := 50 * time.Millisecond

	for attempt := 0; attempt <= maxRetries; attempt++ {
		err := ls.attemptWorkflowExecutionUpdate(ctx, executionID, updateFunc)

		// If successful or non-retryable error, return immediately
		if err == nil || !isDatabaseLockError(err) {
			return err
		}

		// If this was the last attempt, return the error
		if attempt == maxRetries {
			return fmt.Errorf("failed to update workflow execution after %d attempts: %w", maxRetries+1, err)
		}

		// Wait before retrying with exponential backoff
		delay := baseDelay * time.Duration(1<<attempt) // 50ms, 100ms, 200ms
		log.Printf("🔄 RETRY: Database locked, retrying workflow update for %s in %v (attempt %d/%d)", executionID, delay, attempt+1, maxRetries+1)

		select {
		case <-time.After(delay):
			// Continue to next attempt
		case <-ctx.Done():
			return fmt.Errorf("context cancelled during retry delay: %w", ctx.Err())
		}
	}

	return nil // Should never reach here
}

// attemptWorkflowExecutionUpdate performs a single attempt at updating a workflow execution
func (ls *LocalStorage) attemptWorkflowExecutionUpdate(ctx context.Context, executionID string, updateFunc func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	// Begin transaction for atomic operation with shorter timeout
	txCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()

	tx, err := ls.db.BeginTx(txCtx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer rollbackTx(tx, "attemptWorkflowExecutionUpdate:"+executionID)

	// Read the current execution within the transaction
	currentExecution, err := ls.getWorkflowExecutionWithTx(txCtx, tx, executionID)
	if err != nil {
		return fmt.Errorf("failed to get workflow execution %s: %w", executionID, err)
	}

	// Apply the user-provided update function
	updatedExecution, err := updateFunc(currentExecution)
	if err != nil {
		return fmt.Errorf("update function failed for execution %s: %w", executionID, err)
	}

	// Validate that the execution ID hasn't changed
	if updatedExecution.ExecutionID != executionID {
		return fmt.Errorf("update function cannot change execution ID: expected %s, got %s", executionID, updatedExecution.ExecutionID)
	}

	// Store the updated execution using the existing transaction-aware method
	if err := ls.executeWorkflowInsert(txCtx, tx, updatedExecution); err != nil {
		return fmt.Errorf("failed to store updated workflow execution: %w", err)
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit workflow execution update transaction: %w", err)
	}

	return nil
}

// isDatabaseLockError checks if an error is a SQLite database lock error
func isDatabaseLockError(err error) bool {
	if err == nil {
		return false
	}
	errStr := strings.ToLower(err.Error())
	return strings.Contains(errStr, "database is locked") ||
		strings.Contains(errStr, "database locked") ||
		strings.Contains(errStr, "sqlite_busy")
}

// getWorkflowExecutionWithTx retrieves a workflow execution within an existing transaction
// This is a helper method for atomic operations that need to read and write within the same transaction
func (ls *LocalStorage) getWorkflowExecutionWithTx(ctx context.Context, tx DBTX, executionID string) (*types.WorkflowExecution, error) {
	return ls.getWorkflowExecutionByID(ctx, tx, executionID)
}

// executeWorkflowExecutionInsertWithTx performs workflow execution insert within an existing transaction
func (ls *LocalStorage) executeWorkflowExecutionInsertWithTx(ctx context.Context, tx DBTX, execution *types.WorkflowExecution) error {
	return ls.executeWorkflowInsert(ctx, tx, execution)
}

// executeWorkflowInsertWithTx performs workflow insert within an existing transaction
func (ls *LocalStorage) executeWorkflowInsertWithTx(ctx context.Context, tx DBTX, workflow *types.Workflow) error {
	query := `
		INSERT INTO workflows (
			workflow_id, workflow_name, workflow_tags, session_id, actor_id,
			parent_workflow_id, root_workflow_id, workflow_depth,
			total_executions, successful_executions, failed_executions, total_duration_ms,
			status, started_at, completed_at, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(workflow_id) DO UPDATE SET
			workflow_name = excluded.workflow_name,
			workflow_tags = excluded.workflow_tags,
			status = excluded.status,
			completed_at = excluded.completed_at,
			total_executions = excluded.total_executions,
			successful_executions = excluded.successful_executions,
			failed_executions = excluded.failed_executions,
			total_duration_ms = excluded.total_duration_ms,
			updated_at = excluded.updated_at`

	// Set default timestamps if not provided
	if workflow.CreatedAt.IsZero() {
		workflow.CreatedAt = time.Now()
	}
	if workflow.UpdatedAt.IsZero() {
		workflow.UpdatedAt = time.Now()
	}

	// Marshal workflow tags
	tagsJSON, err := json.Marshal(workflow.WorkflowTags)
	if err != nil {
		return fmt.Errorf("failed to marshal workflow tags: %w", err)
	}

	// Execute query within transaction with context
	_, err = tx.ExecContext(ctx, query,
		workflow.WorkflowID, workflow.WorkflowName, tagsJSON, workflow.SessionID, workflow.ActorID,
		workflow.ParentWorkflowID, workflow.RootWorkflowID, workflow.WorkflowDepth,
		workflow.TotalExecutions, workflow.SuccessfulExecutions, workflow.FailedExecutions, workflow.TotalDurationMS,
		workflow.Status, workflow.StartedAt, workflow.CompletedAt, workflow.CreatedAt, workflow.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to execute workflow insert query: %w", err)
	}

	return nil
}

// executeSessionInsertWithTx performs session insert within an existing transaction
func (ls *LocalStorage) executeSessionInsertWithTx(ctx context.Context, tx DBTX, session *types.Session) error {
	query := `
		INSERT INTO sessions (
			session_id, actor_id, session_name, parent_session_id, root_session_id,
			total_workflows, total_executions, total_duration_ms,
			started_at, last_activity_at, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(session_id) DO UPDATE SET
			actor_id = excluded.actor_id,
			session_name = excluded.session_name,
			total_workflows = excluded.total_workflows,
			total_executions = excluded.total_executions,
			total_duration_ms = excluded.total_duration_ms,
			last_activity_at = excluded.last_activity_at,
			updated_at = excluded.updated_at`

	// Set default timestamps if not provided
	if session.CreatedAt.IsZero() {
		session.CreatedAt = time.Now()
	}
	if session.UpdatedAt.IsZero() {
		session.UpdatedAt = time.Now()
	}
	if session.LastActivityAt.IsZero() {
		session.LastActivityAt = time.Now()
	}

	// Execute query within transaction with context
	_, err := tx.ExecContext(ctx, query,
		session.SessionID, session.ActorID, session.SessionName, session.ParentSessionID, session.RootSessionID,
		session.TotalWorkflows, session.TotalExecutions, session.TotalDurationMS,
		session.StartedAt, session.LastActivityAt, session.CreatedAt, session.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to execute session insert query: %w", err)
	}

	return nil
}

// requireSQLDB returns the underlying *sql.DB, panicking if the storage
// connection has not been initialized. The storage initialization flow always
// sets the sqlDatabase before exposing the provider, so this guards against
// incorrect usage during future refactors.
func (ls *LocalStorage) requireSQLDB() *sqlDatabase {
	if ls.db == nil {
		panic("storage database is not initialized")
	}
	return ls.db
}

// NewUnitOfWork creates a new unit of work instance for this storage
func (ls *LocalStorage) NewUnitOfWork() UnitOfWork {
	return NewUnitOfWork(ls.requireSQLDB(), ls)
}

// NewWorkflowUnitOfWork creates a new workflow-specific unit of work instance for this storage
func (ls *LocalStorage) NewWorkflowUnitOfWork() WorkflowUnitOfWork {
	return NewWorkflowUnitOfWork(ls.requireSQLDB(), ls)
}

// StoreWorkflowExecutionWithUnitOfWork demonstrates using Unit of Work for atomic operations
func (ls *LocalStorage) StoreWorkflowExecutionWithUnitOfWork(ctx context.Context, execution *types.WorkflowExecution) error {
	uow := ls.NewUnitOfWork()

	// Register the workflow execution operation
	executionOp := func(tx DBTX) error {
		return ls.executeWorkflowInsert(ctx, tx, execution)
	}
	uow.RegisterNew(execution, "workflow_executions", executionOp)

	// Commit the unit of work
	return uow.Commit()
}

// GetWorkflowExecution retrieves a workflow execution record from SQLite by ID
func (ls *LocalStorage) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	return ls.getWorkflowExecutionByID(ctx, ls.db, executionID)
}

// QueryWorkflowExecutions retrieves workflow execution records from SQLite based on filters
func (ls *LocalStorage) QueryWorkflowExecutions(ctx context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during query workflow executions: %w", err)
	}

	// Build base query
	baseQuery := `
		SELECT
			workflow_executions.id, workflow_executions.workflow_id, workflow_executions.execution_id,
			workflow_executions.agentfield_request_id, workflow_executions.run_id, workflow_executions.session_id, workflow_executions.actor_id,
			workflow_executions.agent_node_id, workflow_executions.parent_workflow_id, workflow_executions.parent_execution_id,
			workflow_executions.root_workflow_id, workflow_executions.workflow_depth,
			workflow_executions.reasoner_id, workflow_executions.input_data, workflow_executions.output_data,
			workflow_executions.input_size, workflow_executions.output_size,
			workflow_executions.status, workflow_executions.started_at, workflow_executions.completed_at,
			workflow_executions.duration_ms,
		workflow_executions.state_version, workflow_executions.last_event_sequence,
		workflow_executions.active_children, workflow_executions.pending_children,
		workflow_executions.pending_terminal_status, workflow_executions.status_reason,
		workflow_executions.lease_owner, workflow_executions.lease_expires_at,
		workflow_executions.error_message,
			workflow_executions.retry_count, workflow_executions.workflow_name, workflow_executions.workflow_tags,
			workflow_executions.notes, workflow_executions.created_at, workflow_executions.updated_at,
			workflow_executions.approval_request_id, workflow_executions.approval_request_url,
			workflow_executions.approval_status, workflow_executions.approval_response,
			workflow_executions.approval_requested_at, workflow_executions.approval_responded_at,
			workflow_executions.approval_callback_url, workflow_executions.approval_expires_at
		FROM workflow_executions`

	var conditions []string
	var args []interface{}

	// Check if we need search
	var ftsJoin string
	if filters.Search != nil && *filters.Search != "" {
		sanitizedSearch := sanitizeFTS5Query(*filters.Search)
		if sanitizedSearch != "" {
			if ls.ftsEnabled {
				// Use FTS5 MATCH for efficient full-text search when available.
				ftsJoin = " INNER JOIN workflow_executions_fts ON workflow_executions.id = workflow_executions_fts.rowid"
				conditions = append(conditions, "workflow_executions_fts MATCH ?")
				args = append(args, sanitizedSearch)
			} else {
				searchTerm := strings.Trim(strings.TrimSpace(sanitizedSearch), "\"")
				if searchTerm == "" {
					searchTerm = strings.TrimSpace(*filters.Search)
				}
				like := "%" + searchTerm + "%"
				conditions = append(conditions, `(workflow_executions.execution_id LIKE ? OR workflow_executions.workflow_id LIKE ? OR workflow_executions.agent_node_id LIKE ? OR workflow_executions.session_id LIKE ? OR workflow_executions.workflow_name LIKE ?)`)
				args = append(args, like, like, like, like, like)
			}
		}
	}

	// Build complete query with optional FTS join
	query := baseQuery + ftsJoin

	// Add filters
	if filters.WorkflowID != nil {
		conditions = append(conditions, "workflow_executions.workflow_id = ?")
		args = append(args, *filters.WorkflowID)
	}
	if filters.SessionID != nil {
		conditions = append(conditions, "workflow_executions.session_id = ?")
		args = append(args, *filters.SessionID)
	}
	if filters.ActorID != nil {
		conditions = append(conditions, "workflow_executions.actor_id = ?")
		args = append(args, *filters.ActorID)
	}
	if filters.AgentNodeID != nil {
		conditions = append(conditions, "workflow_executions.agent_node_id = ?")
		args = append(args, *filters.AgentNodeID)
	}
	if filters.ParentExecutionID != nil {
		conditions = append(conditions, "workflow_executions.parent_execution_id = ?")
		args = append(args, *filters.ParentExecutionID)
	}
	if filters.Status != nil {
		conditions = append(conditions, "workflow_executions.status = ?")
		args = append(args, *filters.Status)
	}
	if filters.ApprovalRequestID != nil {
		conditions = append(conditions, "workflow_executions.approval_request_id = ?")
		args = append(args, *filters.ApprovalRequestID)
	}
	if filters.StartTime != nil {
		conditions = append(conditions, "workflow_executions.started_at >= ?")
		args = append(args, *filters.StartTime)
	}
	if filters.EndTime != nil {
		conditions = append(conditions, "workflow_executions.started_at <= ?")
		args = append(args, *filters.EndTime)
	}

	// Add WHERE clause if there are conditions
	if len(conditions) > 0 {
		query += " WHERE " + strings.Join(conditions, " AND ")
	}

	// Add dynamic ordering
	orderBy := "started_at"
	if filters.SortBy != nil {
		switch *filters.SortBy {
		case "time":
			orderBy = "started_at"
		case "duration":
			orderBy = "duration_ms"
		case "status":
			orderBy = "status"
		default:
			orderBy = "started_at"
		}
	}

	sortOrder := "DESC"
	if filters.SortOrder != nil && strings.ToUpper(*filters.SortOrder) == "ASC" {
		sortOrder = "ASC"
	}

	query += fmt.Sprintf(" ORDER BY %s %s", orderBy, sortOrder)

	// Add pagination
	if filters.Limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", filters.Limit)
	}
	if filters.Offset > 0 {
		query += fmt.Sprintf(" OFFSET %d", filters.Offset)
	}

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query workflow executions: %w", err)
	}
	defer rows.Close()

	executions := []*types.WorkflowExecution{}
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during workflow execution query iteration: %w", err)
		}

		execution := &types.WorkflowExecution{}
		var workflowTagsJSON, notesJSON []byte
		var inputData, outputData sql.NullString
		var pendingTerminal sql.NullString
		var statusReason sql.NullString
		var runID sql.NullString
		var leaseOwner sql.NullString
		var leaseExpires sql.NullTime
		var approvalRequestID, approvalRequestURL, approvalStatus, approvalResponse, approvalCallbackURL sql.NullString
		var approvalRequestedAt, approvalRespondedAt, approvalExpiresAt sql.NullTime

		err := rows.Scan(
			&execution.ID, &execution.WorkflowID, &execution.ExecutionID,
			&execution.AgentFieldRequestID, &runID, &execution.SessionID, &execution.ActorID,
			&execution.AgentNodeID, &execution.ParentWorkflowID, &execution.ParentExecutionID, &execution.RootWorkflowID,
			&execution.WorkflowDepth, &execution.ReasonerID, &inputData,
			&outputData, &execution.InputSize, &execution.OutputSize,
			&execution.Status, &execution.StartedAt, &execution.CompletedAt,
			&execution.DurationMS,
			&execution.StateVersion, &execution.LastEventSequence, &execution.ActiveChildren, &execution.PendingChildren,
			&pendingTerminal, &statusReason,
			&leaseOwner, &leaseExpires,
			&execution.ErrorMessage, &execution.RetryCount,
			&execution.WorkflowName, &workflowTagsJSON, &notesJSON, &execution.CreatedAt,
			&execution.UpdatedAt,
			&approvalRequestID, &approvalRequestURL,
			&approvalStatus, &approvalResponse,
			&approvalRequestedAt, &approvalRespondedAt,
			&approvalCallbackURL, &approvalExpiresAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan workflow execution row: %w", err)
		}

		// Handle nullable input/output data
		if runID.Valid {
			execution.RunID = &runID.String
		}
		if inputData.Valid {
			execution.InputData = safeJSONRawMessage(inputData.String, "{}", fmt.Sprintf("execution %s input_data", execution.ExecutionID))
		} else {
			execution.InputData = json.RawMessage("{}")
		}
		if outputData.Valid {
			execution.OutputData = safeJSONRawMessage(outputData.String, "{}", fmt.Sprintf("execution %s output_data", execution.ExecutionID))
		} else {
			execution.OutputData = json.RawMessage("{}")
		}
		if pendingTerminal.Valid {
			execution.PendingTerminalStatus = &pendingTerminal.String
		}
		if statusReason.Valid {
			execution.StatusReason = &statusReason.String
		}
		if leaseOwner.Valid {
			execution.LeaseOwner = &leaseOwner.String
		}
		if leaseExpires.Valid {
			t := leaseExpires.Time
			execution.LeaseExpiresAt = &t
		}
		if approvalRequestID.Valid {
			execution.ApprovalRequestID = &approvalRequestID.String
		}
		if approvalRequestURL.Valid {
			execution.ApprovalRequestURL = &approvalRequestURL.String
		}
		if approvalStatus.Valid {
			execution.ApprovalStatus = &approvalStatus.String
		}
		if approvalResponse.Valid {
			execution.ApprovalResponse = &approvalResponse.String
		}
		if approvalRequestedAt.Valid {
			t := approvalRequestedAt.Time
			execution.ApprovalRequestedAt = &t
		}
		if approvalRespondedAt.Valid {
			t := approvalRespondedAt.Time
			execution.ApprovalRespondedAt = &t
		}
		if approvalCallbackURL.Valid {
			execution.ApprovalCallbackURL = &approvalCallbackURL.String
		}
		if approvalExpiresAt.Valid {
			t := approvalExpiresAt.Time
			execution.ApprovalExpiresAt = &t
		}

		if len(workflowTagsJSON) > 0 {
			if err := json.Unmarshal(workflowTagsJSON, &execution.WorkflowTags); err != nil {
				return nil, fmt.Errorf("failed to unmarshal workflow tags: %w", err)
			}
		}

		executions = append(executions, execution)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error after querying workflow executions: %w", err)
	}

	return executions, nil
}

// QueryWorkflowDAG retrieves a complete workflow DAG using recursive CTE for optimal performance
func (ls *LocalStorage) QueryWorkflowDAG(ctx context.Context, rootWorkflowID string) ([]*types.WorkflowExecution, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during query workflow DAG: %w", err)
	}

	// Recursive CTE query to build the complete DAG hierarchy in a single query
	// This eliminates the N+1 query problem by using database-level recursion
	query := `
		WITH RECURSIVE workflow_dag AS (
			-- Base case: Find the root execution(s)
			SELECT
				id, workflow_id, execution_id, agentfield_request_id, run_id, session_id, actor_id,
				agent_node_id, parent_workflow_id, parent_execution_id, root_workflow_id,
				workflow_depth, reasoner_id, input_data, output_data, input_size, output_size,
				status, started_at, completed_at, duration_ms,
				state_version, last_event_sequence, active_children, pending_children,
				pending_terminal_status, status_reason,
				error_message, retry_count,
				workflow_name, workflow_tags, notes, created_at, updated_at,
				0 as dag_depth,  -- Track depth for cycle detection
				execution_id as path  -- Track path for cycle detection
			FROM workflow_executions
			WHERE (workflow_id = ? OR root_workflow_id = ?)
			  AND parent_execution_id IS NULL

			UNION ALL

			-- Recursive case: Find children of current level
			SELECT
				we.id, we.workflow_id, we.execution_id, we.agentfield_request_id, we.run_id, we.session_id, we.actor_id,
				we.agent_node_id, we.parent_workflow_id, we.parent_execution_id, we.root_workflow_id,
				we.workflow_depth, we.reasoner_id, we.input_data, we.output_data, we.input_size, we.output_size,
				we.status, we.started_at, we.completed_at, we.duration_ms,
				we.state_version, we.last_event_sequence, we.active_children, we.pending_children,
				we.pending_terminal_status, we.status_reason,
				we.error_message, we.retry_count,
				we.workflow_name, we.workflow_tags, we.notes, we.created_at, we.updated_at,
				wd.dag_depth + 1,  -- Increment depth
				wd.path || ',' || we.execution_id  -- Append to path for cycle detection
			FROM workflow_executions we
			INNER JOIN workflow_dag wd ON we.parent_execution_id = wd.execution_id
			WHERE wd.dag_depth < 100  -- Prevent infinite recursion (max depth limit)
			  AND wd.path NOT LIKE '%' || we.execution_id || '%'  -- Cycle detection
		)
		SELECT
			id, workflow_id, execution_id, agentfield_request_id, run_id, session_id, actor_id,
			agent_node_id, parent_workflow_id, parent_execution_id, root_workflow_id,
			workflow_depth, reasoner_id, input_data, output_data, input_size, output_size,
			status, started_at, completed_at, duration_ms,
			state_version, last_event_sequence, active_children, pending_children,
			pending_terminal_status, status_reason,
			error_message, retry_count,
			workflow_name, workflow_tags, notes, created_at, updated_at
		FROM workflow_dag
		ORDER BY dag_depth, started_at`

	rows, err := ls.db.QueryContext(ctx, query, rootWorkflowID, rootWorkflowID)
	if err != nil {
		return nil, fmt.Errorf("failed to query workflow DAG: %w", err)
	}
	defer rows.Close()

	executions := []*types.WorkflowExecution{}
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during workflow DAG query iteration: %w", err)
		}

		execution := &types.WorkflowExecution{}
		var workflowTagsJSON, notesJSON []byte
		var inputData, outputData sql.NullString
		var pendingTerminal sql.NullString
		var statusReason sql.NullString
		var runID sql.NullString

		err := rows.Scan(
			&execution.ID, &execution.WorkflowID, &execution.ExecutionID,
			&execution.AgentFieldRequestID, &runID, &execution.SessionID, &execution.ActorID,
			&execution.AgentNodeID, &execution.ParentWorkflowID, &execution.ParentExecutionID, &execution.RootWorkflowID,
			&execution.WorkflowDepth, &execution.ReasonerID, &inputData,
			&outputData, &execution.InputSize, &execution.OutputSize,
			&execution.Status, &execution.StartedAt, &execution.CompletedAt,
			&execution.DurationMS,
			&execution.StateVersion, &execution.LastEventSequence, &execution.ActiveChildren, &execution.PendingChildren,
			&pendingTerminal, &statusReason,
			&execution.ErrorMessage, &execution.RetryCount,
			&execution.WorkflowName, &workflowTagsJSON, &notesJSON, &execution.CreatedAt,
			&execution.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan workflow DAG row: %w", err)
		}

		if runID.Valid {
			execution.RunID = &runID.String
		}
		// Handle nullable input/output data
		if inputData.Valid {
			execution.InputData = safeJSONRawMessage(inputData.String, "{}", fmt.Sprintf("DAG execution %s input_data", execution.ExecutionID))
		} else {
			execution.InputData = json.RawMessage("{}")
		}
		if outputData.Valid {
			execution.OutputData = safeJSONRawMessage(outputData.String, "{}", fmt.Sprintf("DAG execution %s output_data", execution.ExecutionID))
		} else {
			execution.OutputData = json.RawMessage("{}")
		}
		if pendingTerminal.Valid {
			execution.PendingTerminalStatus = &pendingTerminal.String
		}
		if statusReason.Valid {
			execution.StatusReason = &statusReason.String
		}

		if len(workflowTagsJSON) > 0 {
			if err := json.Unmarshal(workflowTagsJSON, &execution.WorkflowTags); err != nil {
				return nil, fmt.Errorf("failed to unmarshal workflow tags: %w", err)
			}
		}

		// Parse notes JSON
		if len(notesJSON) > 0 {
			if err := json.Unmarshal(notesJSON, &execution.Notes); err != nil {
				return nil, fmt.Errorf("failed to unmarshal notes: %w", err)
			}
		}

		executions = append(executions, execution)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error after querying workflow DAG: %w", err)
	}

	return executions, nil
}

// CleanupOldExecutions removes old completed workflow executions based on retention period
func (ls *LocalStorage) CleanupOldExecutions(ctx context.Context, retentionPeriod time.Duration, batchSize int) (int, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return 0, fmt.Errorf("context cancelled during cleanup old executions: %w", err)
	}

	// Calculate cutoff time
	cutoffTime := time.Now().UTC().Add(-retentionPeriod)

	// Query to find old completed executions to delete
	// Only delete executions that are completed or failed and older than retention period
	query := `
		SELECT execution_id
		FROM workflow_executions
		WHERE (status = 'completed' OR status = 'failed')
		  AND completed_at IS NOT NULL
		  AND completed_at < ?
		ORDER BY completed_at ASC
		LIMIT ?`

	rows, err := ls.db.QueryContext(ctx, query, cutoffTime, batchSize)
	if err != nil {
		return 0, fmt.Errorf("failed to query old executions for cleanup: %w", err)
	}
	defer rows.Close()

	var executionIDs []string
	for rows.Next() {
		var executionID string
		if err := rows.Scan(&executionID); err != nil {
			return 0, fmt.Errorf("failed to scan execution ID for cleanup: %w", err)
		}
		executionIDs = append(executionIDs, executionID)
	}

	if err := rows.Err(); err != nil {
		return 0, fmt.Errorf("error after querying old executions for cleanup: %w", err)
	}

	// If no executions to clean up, return early
	if len(executionIDs) == 0 {
		return 0, nil
	}

	// Begin transaction for atomic cleanup
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("failed to begin cleanup transaction: %w", err)
	}
	defer rollbackTx(tx, "cleanupOldExecutions")

	// Delete executions in batch
	// Use placeholders for safe deletion
	placeholders := make([]string, len(executionIDs))
	args := make([]interface{}, len(executionIDs))
	for i, id := range executionIDs {
		placeholders[i] = "?"
		args[i] = id
	}

	deleteQuery := fmt.Sprintf(`
		DELETE FROM workflow_executions
		WHERE execution_id IN (%s)`, strings.Join(placeholders, ","))

	result, err := tx.ExecContext(ctx, deleteQuery, args...)
	if err != nil {
		return 0, fmt.Errorf("failed to delete old executions: %w", err)
	}

	deletedCount, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("failed to get deleted rows count: %w", err)
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("failed to commit cleanup transaction: %w", err)
	}

	return int(deletedCount), nil
}

// CleanupWorkflow deletes all data related to a specific workflow ID or workflow run identifier
func (ls *LocalStorage) CleanupWorkflow(ctx context.Context, identifier string, dryRun bool) (*types.WorkflowCleanupResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during workflow cleanup: %w", err)
	}

	startTime := time.Now()
	trimmedID := strings.TrimSpace(identifier)
	result := &types.WorkflowCleanupResult{
		WorkflowID:      trimmedID,
		DryRun:          dryRun,
		DeletedRecords:  make(map[string]int),
		FreedSpaceBytes: 0,
		Success:         false,
	}

	if trimmedID == "" {
		errMsg := "workflow ID cannot be empty"
		result.ErrorMessage = &errMsg
		return result, errors.New(errMsg)
	}

	targets, err := ls.resolveWorkflowCleanupTargets(ctx, trimmedID)
	if err != nil {
		errMsg := fmt.Sprintf("failed to resolve workflow cleanup targets: %v", err)
		result.ErrorMessage = &errMsg
		return result, errors.New(errMsg)
	}

	if targets.primaryWorkflowID != "" {
		result.WorkflowID = targets.primaryWorkflowID
	}

	ls.populateWorkflowCleanupCounts(ctx, targets, result)

	total := 0
	for _, count := range result.DeletedRecords {
		total += count
	}
	result.DeletedRecords["total"] = total

	if dryRun {
		result.Success = true
		result.DurationMS = time.Since(startTime).Milliseconds()
		return result, nil
	}

	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		errMsg := fmt.Sprintf("failed to begin cleanup transaction: %v", err)
		result.ErrorMessage = &errMsg
		return result, errors.New(errMsg)
	}
	defer rollbackTx(tx, "CleanupWorkflow:"+trimmedID)

	if err := ls.performWorkflowCleanup(ctx, tx, targets); err != nil {
		errMsg := fmt.Sprintf("failed to cleanup workflow: %v", err)
		result.ErrorMessage = &errMsg
		return result, errors.New(errMsg)
	}

	if err := tx.Commit(); err != nil {
		errMsg := fmt.Sprintf("failed to commit cleanup transaction: %v", err)
		result.ErrorMessage = &errMsg
		return result, errors.New(errMsg)
	}

	result.Success = true
	result.DurationMS = time.Since(startTime).Milliseconds()
	return result, nil
}

// workflowCleanupTargets captures identifiers needed for cleanup operations
// primaryWorkflowID is the canonical workflow identifier (root workflow ID when available).
// workflowIDs contains all identifiers stored in workflow-scoped tables (includes run IDs when the system stored them as workflow IDs).
// runIDs includes all workflow run identifiers that should be purged.
type workflowCleanupTargets struct {
	primaryWorkflowID string
	workflowIDs       []string
	runIDs            []string
}

func (ls *LocalStorage) resolveWorkflowCleanupTargets(ctx context.Context, identifier string) (*workflowCleanupTargets, error) {
	workflowSet := map[string]struct{}{}
	runSet := map[string]struct{}{}
	addWorkflow := func(id string) {
		id = strings.TrimSpace(id)
		if id != "" {
			workflowSet[id] = struct{}{}
		}
	}
	addRun := func(id string) {
		id = strings.TrimSpace(id)
		if id != "" {
			runSet[id] = struct{}{}
		}
	}

	addWorkflow(identifier)
	addRun(identifier)

	primaryWorkflowID := identifier

	rows, err := ls.db.QueryContext(ctx, `SELECT run_id, root_workflow_id FROM workflow_runs WHERE run_id = ? OR root_workflow_id = ?`, identifier, identifier)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	for rows.Next() {
		var runID sql.NullString
		var rootID sql.NullString
		if err := rows.Scan(&runID, &rootID); err != nil {
			return nil, err
		}
		if runID.Valid {
			addRun(runID.String)
		}
		if rootID.Valid && rootID.String != "" {
			primaryWorkflowID = rootID.String
			addWorkflow(rootID.String)
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	executionRunRows, err := ls.db.QueryContext(ctx, `SELECT DISTINCT run_id FROM executions WHERE run_id = ?`, identifier)
	if err != nil {
		return nil, err
	}
	defer executionRunRows.Close()
	for executionRunRows.Next() {
		var runID sql.NullString
		if err := executionRunRows.Scan(&runID); err != nil {
			return nil, err
		}
		if runID.Valid {
			addRun(runID.String)
		}
	}
	if err := executionRunRows.Err(); err != nil {
		return nil, err
	}

	workflowExecutionRunRows, err := ls.db.QueryContext(
		ctx,
		`SELECT DISTINCT run_id FROM workflow_executions WHERE run_id IS NOT NULL AND (run_id = ? OR workflow_id = ? OR root_workflow_id = ?)`,
		identifier,
		identifier,
		identifier,
	)
	if err != nil {
		return nil, err
	}
	defer workflowExecutionRunRows.Close()
	for workflowExecutionRunRows.Next() {
		var runID sql.NullString
		if err := workflowExecutionRunRows.Scan(&runID); err != nil {
			return nil, err
		}
		if runID.Valid {
			addRun(runID.String)
		}
	}
	if err := workflowExecutionRunRows.Err(); err != nil {
		return nil, err
	}

	if primaryWorkflowID != "" && primaryWorkflowID != identifier {
		addWorkflow(primaryWorkflowID)
		extraRuns, err := ls.db.QueryContext(ctx, `SELECT run_id FROM workflow_runs WHERE root_workflow_id = ?`, primaryWorkflowID)
		if err != nil {
			return nil, err
		}
		defer extraRuns.Close()
		for extraRuns.Next() {
			var runID string
			if err := extraRuns.Scan(&runID); err != nil {
				return nil, err
			}
			addRun(runID)
		}
		if err := extraRuns.Err(); err != nil {
			return nil, err
		}
	}

	for runID := range runSet {
		addWorkflow(runID)
	}

	return &workflowCleanupTargets{
		primaryWorkflowID: strings.TrimSpace(primaryWorkflowID),
		workflowIDs:       setToSlice(workflowSet),
		runIDs:            setToSlice(runSet),
	}, nil
}

func setToSlice(input map[string]struct{}) []string {
	if len(input) == 0 {
		return nil
	}
	out := make([]string, 0, len(input))
	for value := range input {
		out = append(out, value)
	}
	return out
}

func (ls *LocalStorage) populateWorkflowCleanupCounts(ctx context.Context, targets *workflowCleanupTargets, result *types.WorkflowCleanupResult) {
	primaryWorkflowID := targets.primaryWorkflowID
	workflowIDs := targets.workflowIDs
	runIDs := targets.runIDs
	result.DeletedRecords["workflow_runs"] = ls.countWorkflowRuns(ctx, primaryWorkflowID, workflowIDs, runIDs)
	result.DeletedRecords["executions"] = ls.countExecutions(ctx, runIDs)
	result.DeletedRecords["execution_webhooks"] = ls.countExecutionWebhooks(ctx, runIDs)
	result.DeletedRecords["execution_webhook_events"] = ls.countExecutionWebhookEvents(ctx, runIDs)
	result.DeletedRecords["execution_vcs"] = ls.countExecutionVCs(ctx, workflowIDs)
	result.DeletedRecords["workflow_vcs"] = ls.countWorkflowVCs(ctx, workflowIDs)
	result.DeletedRecords["workflow_executions"] = ls.countWorkflowExecutions(ctx, workflowIDs, runIDs)
	result.DeletedRecords["workflow_execution_events"] = ls.countWorkflowExecutionEvents(ctx, workflowIDs, runIDs)
	result.DeletedRecords["workflows"] = ls.countWorkflows(ctx, workflowIDs)
	result.DeletedRecords["workflow_runs"] = ls.countWorkflowRuns(ctx, targets.primaryWorkflowID, workflowIDs, runIDs)
}

func (ls *LocalStorage) performWorkflowCleanup(ctx context.Context, tx DBTX, targets *workflowCleanupTargets) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during workflow cleanup: %w", err)
	}

	primaryWorkflowID := targets.primaryWorkflowID
	workflowIDs := targets.workflowIDs
	runIDs := targets.runIDs

	if _, err := ls.deleteExecutionVCs(ctx, tx, workflowIDs); err != nil {
		return fmt.Errorf("failed to delete execution VCs: %w", err)
	}
	if _, err := ls.deleteWorkflowVCs(ctx, tx, workflowIDs); err != nil {
		return fmt.Errorf("failed to delete workflow VCs: %w", err)
	}
	if _, err := ls.deleteExecutionWebhookEvents(ctx, tx, runIDs); err != nil {
		return fmt.Errorf("failed to delete execution webhook events: %w", err)
	}
	if _, err := ls.deleteExecutionWebhooks(ctx, tx, runIDs); err != nil {
		return fmt.Errorf("failed to delete execution webhooks: %w", err)
	}
	if _, err := ls.deleteExecutions(ctx, tx, runIDs); err != nil {
		return fmt.Errorf("failed to delete executions: %w", err)
	}
	if _, err := ls.deleteWorkflowExecutions(ctx, tx, workflowIDs, runIDs); err != nil {
		return fmt.Errorf("failed to delete workflow executions: %w", err)
	}
	if _, err := ls.deleteWorkflowRuns(ctx, tx, primaryWorkflowID, workflowIDs, runIDs); err != nil {
		return fmt.Errorf("failed to delete workflow runs: %w", err)
	}
	if _, err := ls.deleteWorkflows(ctx, tx, workflowIDs); err != nil {
		return fmt.Errorf("failed to delete workflow definitions: %w", err)
	}

	return nil
}

func makePlaceholders(n int) string {
	if n <= 0 {
		return ""
	}
	return strings.TrimRight(strings.Repeat("?,", n), ",")
}

func stringsToInterfaces(values []string) []interface{} {
	args := make([]interface{}, len(values))
	for i, v := range values {
		args[i] = v
	}
	return args
}

func (ls *LocalStorage) countWorkflowRuns(ctx context.Context, primaryWorkflowID string, workflowIDs, runIDs []string) int {
	conditions := []string{}
	args := []interface{}{}

	if primaryWorkflowID != "" {
		conditions = append(conditions, "root_workflow_id = ?")
		args = append(args, primaryWorkflowID)
		conditions = append(conditions, "run_id = ?")
		args = append(args, primaryWorkflowID)
	}
	if len(workflowIDs) > 0 {
		placeholders := makePlaceholders(len(workflowIDs))
		conditions = append(conditions, fmt.Sprintf("root_workflow_id IN (%s)", placeholders))
		args = append(args, stringsToInterfaces(workflowIDs)...)
	}
	if len(runIDs) > 0 {
		placeholders := makePlaceholders(len(runIDs))
		conditions = append(conditions, fmt.Sprintf("run_id IN (%s)", placeholders))
		args = append(args, stringsToInterfaces(runIDs)...)
	}

	if len(conditions) == 0 {
		return 0
	}

	query := "SELECT COUNT(*) FROM workflow_runs WHERE " + strings.Join(conditions, " OR ")
	var count int
	if err := ls.db.QueryRowContext(ctx, query, args...).Scan(&count); err != nil {
		return 0
	}
	return count
}

func (ls *LocalStorage) countExecutions(ctx context.Context, runIDs []string) int {
	if len(runIDs) == 0 {
		return 0
	}
	query := fmt.Sprintf(`SELECT COUNT(*) FROM executions WHERE run_id IN (%s)`, makePlaceholders(len(runIDs)))
	var count int
	if err := ls.db.QueryRowContext(ctx, query, stringsToInterfaces(runIDs)...).Scan(&count); err != nil {
		return 0
	}
	return count
}

func (ls *LocalStorage) countExecutionWebhooks(ctx context.Context, runIDs []string) int {
	if len(runIDs) == 0 {
		return 0
	}
	query := fmt.Sprintf(
		`SELECT COUNT(*) FROM execution_webhooks WHERE execution_id IN (SELECT execution_id FROM executions WHERE run_id IN (%s))`,
		makePlaceholders(len(runIDs)),
	)
	var count int
	if err := ls.db.QueryRowContext(ctx, query, stringsToInterfaces(runIDs)...).Scan(&count); err != nil {
		return 0
	}
	return count
}

func (ls *LocalStorage) countExecutionWebhookEvents(ctx context.Context, runIDs []string) int {
	if len(runIDs) == 0 {
		return 0
	}
	query := fmt.Sprintf(
		`SELECT COUNT(*) FROM execution_webhook_events WHERE execution_id IN (SELECT execution_id FROM executions WHERE run_id IN (%s))`,
		makePlaceholders(len(runIDs)),
	)
	var count int
	if err := ls.db.QueryRowContext(ctx, query, stringsToInterfaces(runIDs)...).Scan(&count); err != nil {
		return 0
	}
	return count
}

func (ls *LocalStorage) countExecutionVCs(ctx context.Context, workflowIDs []string) int {
	if len(workflowIDs) == 0 {
		return 0
	}
	query := fmt.Sprintf(`SELECT COUNT(*) FROM execution_vcs WHERE workflow_id IN (%s)`, makePlaceholders(len(workflowIDs)))
	var count int
	if err := ls.db.QueryRowContext(ctx, query, stringsToInterfaces(workflowIDs)...).Scan(&count); err != nil {
		return 0
	}
	return count
}

func (ls *LocalStorage) countWorkflowVCs(ctx context.Context, workflowIDs []string) int {
	if len(workflowIDs) == 0 {
		return 0
	}
	query := fmt.Sprintf(`SELECT COUNT(*) FROM workflow_vcs WHERE workflow_id IN (%s)`, makePlaceholders(len(workflowIDs)))
	var count int
	if err := ls.db.QueryRowContext(ctx, query, stringsToInterfaces(workflowIDs)...).Scan(&count); err != nil {
		return 0
	}
	return count
}

func (ls *LocalStorage) countWorkflowExecutions(ctx context.Context, workflowIDs, runIDs []string) int {
	conditions := []string{}
	args := []interface{}{}

	if len(workflowIDs) > 0 {
		placeholders := makePlaceholders(len(workflowIDs))
		conditions = append(conditions, fmt.Sprintf("workflow_id IN (%s)", placeholders))
		args = append(args, stringsToInterfaces(workflowIDs)...)
		conditions = append(conditions, fmt.Sprintf("root_workflow_id IN (%s)", placeholders))
		args = append(args, stringsToInterfaces(workflowIDs)...)
	}
	if len(runIDs) > 0 {
		placeholders := makePlaceholders(len(runIDs))
		conditions = append(conditions, fmt.Sprintf("run_id IN (%s)", placeholders))
		args = append(args, stringsToInterfaces(runIDs)...)
	}

	if len(conditions) == 0 {
		return 0
	}

	query := "SELECT COUNT(*) FROM workflow_executions WHERE " + strings.Join(conditions, " OR ")
	var count int
	if err := ls.db.QueryRowContext(ctx, query, args...).Scan(&count); err != nil {
		return 0
	}
	return count
}

func (ls *LocalStorage) countWorkflowExecutionEvents(ctx context.Context, workflowIDs, runIDs []string) int {
	conditions := []string{}
	args := []interface{}{}

	if len(workflowIDs) > 0 {
		placeholders := makePlaceholders(len(workflowIDs))
		conditions = append(conditions, fmt.Sprintf("workflow_id IN (%s)", placeholders))
		args = append(args, stringsToInterfaces(workflowIDs)...)
	}
	if len(runIDs) > 0 {
		placeholders := makePlaceholders(len(runIDs))
		conditions = append(conditions, fmt.Sprintf("run_id IN (%s)", placeholders))
		args = append(args, stringsToInterfaces(runIDs)...)
	}

	if len(conditions) == 0 {
		return 0
	}

	query := "SELECT COUNT(*) FROM workflow_execution_events WHERE " + strings.Join(conditions, " OR ")
	var count int
	if err := ls.db.QueryRowContext(ctx, query, args...).Scan(&count); err != nil {
		return 0
	}
	return count
}

func (ls *LocalStorage) countWorkflows(ctx context.Context, workflowIDs []string) int {
	if len(workflowIDs) == 0 {
		return 0
	}
	query := fmt.Sprintf(`SELECT COUNT(*) FROM workflows WHERE workflow_id IN (%s)`, makePlaceholders(len(workflowIDs)))
	var count int
	if err := ls.db.QueryRowContext(ctx, query, stringsToInterfaces(workflowIDs)...).Scan(&count); err != nil {
		return 0
	}
	return count
}

func (ls *LocalStorage) deleteExecutionVCs(ctx context.Context, tx DBTX, workflowIDs []string) (int, error) {
	if len(workflowIDs) == 0 {
		return 0, nil
	}
	query := fmt.Sprintf(`DELETE FROM execution_vcs WHERE workflow_id IN (%s)`, makePlaceholders(len(workflowIDs)))
	result, err := tx.ExecContext(ctx, query, stringsToInterfaces(workflowIDs)...)
	if err != nil {
		return 0, err
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return 0, err
	}
	return int(rows), nil
}

func (ls *LocalStorage) deleteWorkflowVCs(ctx context.Context, tx DBTX, workflowIDs []string) (int, error) {
	if len(workflowIDs) == 0 {
		return 0, nil
	}
	query := fmt.Sprintf(`DELETE FROM workflow_vcs WHERE workflow_id IN (%s)`, makePlaceholders(len(workflowIDs)))
	result, err := tx.ExecContext(ctx, query, stringsToInterfaces(workflowIDs)...)
	if err != nil {
		return 0, err
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return 0, err
	}
	return int(rows), nil
}

func (ls *LocalStorage) deleteExecutionWebhookEvents(ctx context.Context, tx DBTX, runIDs []string) (int, error) {
	if len(runIDs) == 0 {
		return 0, nil
	}
	query := fmt.Sprintf(
		`DELETE FROM execution_webhook_events WHERE execution_id IN (SELECT execution_id FROM executions WHERE run_id IN (%s))`,
		makePlaceholders(len(runIDs)),
	)
	result, err := tx.ExecContext(ctx, query, stringsToInterfaces(runIDs)...)
	if err != nil {
		return 0, err
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return 0, err
	}
	return int(rows), nil
}

func (ls *LocalStorage) deleteExecutionWebhooks(ctx context.Context, tx DBTX, runIDs []string) (int, error) {
	if len(runIDs) == 0 {
		return 0, nil
	}
	query := fmt.Sprintf(
		`DELETE FROM execution_webhooks WHERE execution_id IN (SELECT execution_id FROM executions WHERE run_id IN (%s))`,
		makePlaceholders(len(runIDs)),
	)
	result, err := tx.ExecContext(ctx, query, stringsToInterfaces(runIDs)...)
	if err != nil {
		return 0, err
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return 0, err
	}
	return int(rows), nil
}

func (ls *LocalStorage) deleteExecutions(ctx context.Context, tx DBTX, runIDs []string) (int, error) {
	if len(runIDs) == 0 {
		return 0, nil
	}
	query := fmt.Sprintf(`DELETE FROM executions WHERE run_id IN (%s)`, makePlaceholders(len(runIDs)))
	result, err := tx.ExecContext(ctx, query, stringsToInterfaces(runIDs)...)
	if err != nil {
		return 0, err
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return 0, err
	}
	return int(rows), nil
}

func (ls *LocalStorage) deleteWorkflowExecutions(ctx context.Context, tx DBTX, workflowIDs, runIDs []string) (int, error) {
	conditions := []string{}
	args := []interface{}{}

	if len(workflowIDs) > 0 {
		placeholders := makePlaceholders(len(workflowIDs))
		workflowClause := fmt.Sprintf("workflow_id IN (%s)", placeholders)
		rootClause := fmt.Sprintf("root_workflow_id IN (%s)", placeholders)
		conditions = append(conditions, workflowClause, rootClause)
		workflowArgs := stringsToInterfaces(workflowIDs)
		args = append(args, workflowArgs...)
		args = append(args, workflowArgs...)
	}
	if len(runIDs) > 0 {
		placeholders := makePlaceholders(len(runIDs))
		conditions = append(conditions, fmt.Sprintf("run_id IN (%s)", placeholders))
		args = append(args, stringsToInterfaces(runIDs)...)
	}

	if len(conditions) == 0 {
		return 0, nil
	}

	query := "DELETE FROM workflow_executions WHERE " + strings.Join(conditions, " OR ")
	result, err := tx.ExecContext(ctx, query, args...)
	if err != nil {
		return 0, err
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return 0, err
	}
	return int(rows), nil
}

func (ls *LocalStorage) deleteWorkflowRuns(ctx context.Context, tx DBTX, primaryWorkflowID string, workflowIDs, runIDs []string) (int, error) {
	conditions := []string{}
	args := []interface{}{}

	if primaryWorkflowID != "" {
		conditions = append(conditions, "root_workflow_id = ?")
		args = append(args, primaryWorkflowID)
		conditions = append(conditions, "run_id = ?")
		args = append(args, primaryWorkflowID)
	}
	if len(workflowIDs) > 0 {
		placeholders := makePlaceholders(len(workflowIDs))
		conditions = append(conditions, fmt.Sprintf("root_workflow_id IN (%s)", placeholders))
		args = append(args, stringsToInterfaces(workflowIDs)...)
	}
	if len(runIDs) > 0 {
		placeholders := makePlaceholders(len(runIDs))
		conditions = append(conditions, fmt.Sprintf("run_id IN (%s)", placeholders))
		args = append(args, stringsToInterfaces(runIDs)...)
	}

	if len(conditions) == 0 {
		return 0, nil
	}

	query := "DELETE FROM workflow_runs WHERE " + strings.Join(conditions, " OR ")
	result, err := tx.ExecContext(ctx, query, args...)
	if err != nil {
		return 0, err
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return 0, err
	}
	return int(rows), nil
}

func (ls *LocalStorage) deleteWorkflows(ctx context.Context, tx DBTX, workflowIDs []string) (int, error) {
	if len(workflowIDs) == 0 {
		return 0, nil
	}
	query := fmt.Sprintf(`DELETE FROM workflows WHERE workflow_id IN (%s)`, makePlaceholders(len(workflowIDs)))
	result, err := tx.ExecContext(ctx, query, stringsToInterfaces(workflowIDs)...)
	if err != nil {
		return 0, err
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return 0, err
	}
	return int(rows), nil
}

// CreateOrUpdateWorkflow creates or updates a workflow record in SQLite
func (ls *LocalStorage) CreateOrUpdateWorkflow(ctx context.Context, workflow *types.Workflow) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during create or update workflow: %w", err)
	}

	query := `
		INSERT INTO workflows (
			workflow_id, workflow_name, workflow_tags, session_id, actor_id,
			parent_workflow_id, root_workflow_id, workflow_depth,
			total_executions, successful_executions, failed_executions,
			total_duration_ms, status, started_at, completed_at,
			created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(workflow_id) DO UPDATE SET
			workflow_name = excluded.workflow_name,
			workflow_tags = excluded.workflow_tags,
			session_id = excluded.session_id,
			actor_id = excluded.actor_id,
			parent_workflow_id = excluded.parent_workflow_id,
			root_workflow_id = excluded.root_workflow_id,
			workflow_depth = excluded.workflow_depth,
			total_executions = excluded.total_executions,
			successful_executions = excluded.successful_executions,
			failed_executions = excluded.failed_executions,
			total_duration_ms = excluded.total_duration_ms,
			status = excluded.status,
			completed_at = excluded.completed_at,
			updated_at = excluded.updated_at;`

	workflowTagsJSON, err := json.Marshal(workflow.WorkflowTags)
	if err != nil {
		return fmt.Errorf("failed to marshal workflow tags: %w", err)
	}

	_, err = ls.db.ExecContext(ctx, query,
		workflow.WorkflowID, workflow.WorkflowName, workflowTagsJSON,
		workflow.SessionID, workflow.ActorID, workflow.ParentWorkflowID,
		workflow.RootWorkflowID, workflow.WorkflowDepth,
		workflow.TotalExecutions, workflow.SuccessfulExecutions,
		workflow.FailedExecutions, workflow.TotalDurationMS,
		workflow.Status, workflow.StartedAt, workflow.CompletedAt,
		workflow.CreatedAt, workflow.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to create or update workflow: %w", err)
	}

	return nil
}

// GetWorkflow retrieves a workflow record from SQLite by ID
func (ls *LocalStorage) GetWorkflow(ctx context.Context, workflowID string) (*types.Workflow, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get workflow: %w", err)
	}

	query := `
		SELECT
			workflow_id, workflow_name, workflow_tags, session_id, actor_id,
			parent_workflow_id, root_workflow_id, workflow_depth,
			total_executions, successful_executions, failed_executions,
			total_duration_ms, status, started_at, completed_at,
			created_at, updated_at
		FROM workflows WHERE workflow_id = ?`

	row := ls.db.QueryRowContext(ctx, query, workflowID)

	workflow := &types.Workflow{}
	var workflowTagsJSON []byte

	err := row.Scan(
		&workflow.WorkflowID, &workflow.WorkflowName, &workflowTagsJSON,
		&workflow.SessionID, &workflow.ActorID, &workflow.ParentWorkflowID,
		&workflow.RootWorkflowID, &workflow.WorkflowDepth,
		&workflow.TotalExecutions, &workflow.SuccessfulExecutions,
		&workflow.FailedExecutions, &workflow.TotalDurationMS,
		&workflow.Status, &workflow.StartedAt, &workflow.CompletedAt,
		&workflow.CreatedAt, &workflow.UpdatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("workflow with ID %s not found", workflowID)
		}
		return nil, fmt.Errorf("failed to get workflow: %w", err)
	}

	if len(workflowTagsJSON) > 0 {
		if err := json.Unmarshal(workflowTagsJSON, &workflow.WorkflowTags); err != nil {
			return nil, fmt.Errorf("failed to unmarshal workflow tags: %w", err)
		}
	}

	return workflow, nil
}

// QueryWorkflows retrieves workflow records from SQLite based on filters
func (ls *LocalStorage) QueryWorkflows(ctx context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during query workflows: %w", err)
	}
	// Build query with filters
	query := `
		SELECT
			workflow_id, workflow_name, workflow_tags, session_id, actor_id,
			parent_workflow_id, root_workflow_id, workflow_depth,
			total_executions, successful_executions, failed_executions,
			total_duration_ms, status, started_at, completed_at,
			created_at, updated_at
		FROM workflows`

	var conditions []string
	var args []interface{}

	// Add filters
	if filters.SessionID != nil {
		conditions = append(conditions, "session_id = ?")
		args = append(args, *filters.SessionID)
	}
	if filters.ActorID != nil {
		conditions = append(conditions, "actor_id = ?")
		args = append(args, *filters.ActorID)
	}
	if filters.Status != nil {
		conditions = append(conditions, "status = ?")
		args = append(args, *filters.Status)
	}
	if filters.StartTime != nil {
		conditions = append(conditions, "started_at >= ?")
		args = append(args, *filters.StartTime)
	}
	if filters.EndTime != nil {
		conditions = append(conditions, "started_at <= ?")
		args = append(args, *filters.EndTime)
	}

	// Add WHERE clause if there are conditions
	if len(conditions) > 0 {
		query += " WHERE " + strings.Join(conditions, " AND ")
	}

	// Add ordering and pagination
	// Determine order by clause
	sortColumn := "updated_at"
	if filters.SortBy != nil {
		switch *filters.SortBy {
		case "started_at", "started", "time":
			sortColumn = "started_at"
		case "total_executions":
			sortColumn = "total_executions"
		case "duration", "duration_ms":
			sortColumn = "total_duration_ms"
		case "display_name", "workflow_name":
			sortColumn = "workflow_name"
		case "status":
			sortColumn = "status"
		}
	}
	sortDirection := "DESC"
	if filters.SortOrder != nil && strings.EqualFold(*filters.SortOrder, "asc") {
		sortDirection = "ASC"
	}
	query += fmt.Sprintf(" ORDER BY %s %s", sortColumn, sortDirection)
	if filters.Limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", filters.Limit)
	}
	if filters.Offset > 0 {
		query += fmt.Sprintf(" OFFSET %d", filters.Offset)
	}

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query workflows: %w", err)
	}
	defer rows.Close()

	workflows := []*types.Workflow{}
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during workflow query iteration: %w", err)
		}

		workflow := &types.Workflow{}
		var workflowTagsJSON []byte

		err := rows.Scan(
			&workflow.WorkflowID, &workflow.WorkflowName, &workflowTagsJSON,
			&workflow.SessionID, &workflow.ActorID, &workflow.ParentWorkflowID,
			&workflow.RootWorkflowID, &workflow.WorkflowDepth,
			&workflow.TotalExecutions, &workflow.SuccessfulExecutions,
			&workflow.FailedExecutions, &workflow.TotalDurationMS,
			&workflow.Status, &workflow.StartedAt, &workflow.CompletedAt,
			&workflow.CreatedAt, &workflow.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan workflow row: %w", err)
		}

		if len(workflowTagsJSON) > 0 {
			if err := json.Unmarshal(workflowTagsJSON, &workflow.WorkflowTags); err != nil {
				return nil, fmt.Errorf("failed to unmarshal workflow tags: %w", err)
			}
		}

		workflows = append(workflows, workflow)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error after querying workflows: %w", err)
	}

	return workflows, nil
}

// CreateOrUpdateSession creates or updates a session record in SQLite
func (ls *LocalStorage) CreateOrUpdateSession(ctx context.Context, session *types.Session) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during create or update session: %w", err)
	}

	query := `
		INSERT INTO sessions (
			session_id, actor_id, session_name, parent_session_id, root_session_id,
			total_workflows, total_executions, total_duration_ms,
			started_at, last_activity_at, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(session_id) DO UPDATE SET
			actor_id = excluded.actor_id,
			session_name = excluded.session_name,
			parent_session_id = excluded.parent_session_id,
			root_session_id = excluded.root_session_id,
			total_workflows = excluded.total_workflows,
			total_executions = excluded.total_executions,
			total_duration_ms = excluded.total_duration_ms,
			last_activity_at = excluded.last_activity_at,
			updated_at = excluded.updated_at;`

	_, err := ls.db.ExecContext(ctx, query,
		session.SessionID, session.ActorID, session.SessionName,
		session.ParentSessionID, session.RootSessionID,
		session.TotalWorkflows, session.TotalExecutions, session.TotalDurationMS,
		session.StartedAt, session.LastActivityAt, session.CreatedAt, session.UpdatedAt,
	)

	if err != nil {
		return fmt.Errorf("failed to create or update session: %w", err)
	}

	return nil
}

// GetSession retrieves a session record from SQLite by ID
func (ls *LocalStorage) GetSession(ctx context.Context, sessionID string) (*types.Session, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get session: %w", err)
	}

	query := `
		SELECT
			session_id, actor_id, session_name, parent_session_id, root_session_id,
			total_workflows, total_executions, total_duration_ms,
			started_at, last_activity_at, created_at, updated_at
		FROM sessions WHERE session_id = ?`

	row := ls.db.QueryRowContext(ctx, query, sessionID)

	session := &types.Session{}

	err := row.Scan(
		&session.SessionID, &session.ActorID, &session.SessionName,
		&session.ParentSessionID, &session.RootSessionID,
		&session.TotalWorkflows, &session.TotalExecutions, &session.TotalDurationMS,
		&session.StartedAt, &session.LastActivityAt, &session.CreatedAt, &session.UpdatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("session with ID %s not found", sessionID)
		}
		return nil, fmt.Errorf("failed to get session: %w", err)
	}

	return session, nil
}

// QuerySessions retrieves session records from SQLite based on filters
func (ls *LocalStorage) QuerySessions(ctx context.Context, filters types.SessionFilters) ([]*types.Session, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during query sessions: %w", err)
	}
	// Build query with filters
	query := `
		SELECT
			session_id, actor_id, session_name, parent_session_id, root_session_id,
			total_workflows, total_executions, total_duration_ms,
			started_at, last_activity_at, created_at, updated_at
		FROM sessions`

	var conditions []string
	var args []interface{}

	// Add filters
	if filters.ActorID != nil {
		conditions = append(conditions, "actor_id = ?")
		args = append(args, *filters.ActorID)
	}
	if filters.StartTime != nil {
		conditions = append(conditions, "started_at >= ?")
		args = append(args, *filters.StartTime)
	}
	if filters.EndTime != nil {
		conditions = append(conditions, "started_at <= ?")
		args = append(args, *filters.EndTime)
	}

	// Add WHERE clause if there are conditions
	if len(conditions) > 0 {
		query += " WHERE " + strings.Join(conditions, " AND ")
	}

	// Add ordering and pagination
	query += " ORDER BY started_at DESC"
	if filters.Limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", filters.Limit)
	}
	if filters.Offset > 0 {
		query += fmt.Sprintf(" OFFSET %d", filters.Offset)
	}

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query sessions: %w", err)
	}
	defer rows.Close()

	sessions := []*types.Session{}
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during session query iteration: %w", err)
		}

		session := &types.Session{}

		err := rows.Scan(
			&session.SessionID, &session.ActorID, &session.SessionName,
			&session.ParentSessionID, &session.RootSessionID,
			&session.TotalWorkflows, &session.TotalExecutions, &session.TotalDurationMS,
			&session.StartedAt, &session.LastActivityAt, &session.CreatedAt, &session.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan session row: %w", err)
		}

		sessions = append(sessions, session)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error after querying sessions: %w", err)
	}

	return sessions, nil
}

// SetMemory stores a memory record in BoltDB.
func (ls *LocalStorage) SetMemory(ctx context.Context, memory *types.Memory) error {
	if ls.mode == "postgres" {
		return ls.setMemoryPostgres(ctx, memory)
	}

	// Fast-fail check for BoltDB operations since BoltDB doesn't support mid-flight cancellation
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled before BoltDB SetMemory operation: %w", err)
	}

	return ls.kvStore.Update(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(memory.Scope))
		if bucket == nil {
			return fmt.Errorf("BoltDB bucket '%s' not found", memory.Scope)
		}

		key := fmt.Sprintf("%s:%s", memory.ScopeID, memory.Key)
		data, err := json.Marshal(memory)
		if err != nil {
			return fmt.Errorf("failed to marshal memory: %w", err)
		}

		// Store in BoltDB
		if err := bucket.Put([]byte(key), data); err != nil {
			return fmt.Errorf("failed to put memory in BoltDB: %w", err)
		}

		// Update cache
		ls.cache.Store(fmt.Sprintf("%s:%s", memory.Scope, key), memory)

		return nil
	})
}

// GetMemory retrieves a memory record from BoltDB or cache.
func (ls *LocalStorage) GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error) {
	if ls.mode == "postgres" {
		return ls.getMemoryPostgres(ctx, scope, scopeID, key)
	}

	// Fast-fail check for BoltDB operations since BoltDB doesn't support mid-flight cancellation
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled before BoltDB GetMemory operation: %w", err)
	}

	cacheKey := fmt.Sprintf("%s:%s:%s", scope, scopeID, key)
	if val, ok := ls.cache.Load(cacheKey); ok {
		if memory, ok := val.(*types.Memory); ok {
			return memory, nil
		}
	}

	var memory *types.Memory
	err := ls.kvStore.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(scope))
		if bucket == nil {
			return fmt.Errorf("BoltDB bucket '%s' not found", scope)
		}

		boltKey := fmt.Sprintf("%s:%s", scopeID, key)
		data := bucket.Get([]byte(boltKey))
		if data == nil {
			return fmt.Errorf("memory with key '%s' not found in scope '%s' for ID '%s'", key, scope, scopeID)
		}

		memory = &types.Memory{}
		if err := json.Unmarshal(data, memory); err != nil {
			return fmt.Errorf("failed to unmarshal memory from BoltDB: %w", err)
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	// Store in cache
	ls.cache.Store(cacheKey, memory)

	return memory, nil
}

// DeleteMemory deletes a memory record from BoltDB and cache.
func (ls *LocalStorage) DeleteMemory(ctx context.Context, scope, scopeID, key string) error {
	if ls.mode == "postgres" {
		return ls.deleteMemoryPostgres(ctx, scope, scopeID, key)
	}

	// Fast-fail check for BoltDB operations since BoltDB doesn't support mid-flight cancellation
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled before BoltDB DeleteMemory operation: %w", err)
	}

	return ls.kvStore.Update(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(scope))
		if bucket == nil {
			return fmt.Errorf("BoltDB bucket '%s' not found", scope)
		}

		boltKey := fmt.Sprintf("%s:%s", scopeID, key)
		if err := bucket.Delete([]byte(boltKey)); err != nil {
			return fmt.Errorf("failed to delete memory from BoltDB: %w", err)
		}

		// Delete from cache
		cacheKey := fmt.Sprintf("%s:%s:%s", scope, scopeID, key)
		ls.cache.Delete(cacheKey)

		return nil
	})
}

// ListMemory retrieves all memory records for a given scope and scope ID from BoltDB.
func (ls *LocalStorage) ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error) {
	if ls.mode == "postgres" {
		return ls.listMemoryPostgres(ctx, scope, scopeID)
	}

	// Fast-fail check for BoltDB operations since BoltDB doesn't support mid-flight cancellation
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled before BoltDB ListMemory operation: %w", err)
	}

	memories := []*types.Memory{}
	err := ls.kvStore.View(func(tx *bolt.Tx) error {
		bucket := tx.Bucket([]byte(scope))
		if bucket == nil {
			return fmt.Errorf("BoltDB bucket '%s' not found", scope)
		}

		c := bucket.Cursor()

		prefix := []byte(scopeID + ":")
		for k, v := c.Seek(prefix); k != nil && bytes.HasPrefix(k, prefix); k, v = c.Next() {
			memory := &types.Memory{}
			if err := json.Unmarshal(v, memory); err != nil {
				return fmt.Errorf("failed to unmarshal memory from BoltDB: %w", err)
			}
			memories = append(memories, memory)
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	return memories, nil
}

func (ls *LocalStorage) requireVectorStore() error {
	if !ls.vectorConfig.isEnabled() {
		return fmt.Errorf("vector store is disabled")
	}
	if ls.vectorStore == nil {
		return fmt.Errorf("vector store is not initialized")
	}
	return nil
}

// SetVector stores or updates a vector embedding for the specified scope/key.
func (ls *LocalStorage) SetVector(ctx context.Context, record *types.VectorRecord) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := ls.requireVectorStore(); err != nil {
		return err
	}
	return ls.vectorStore.Set(ctx, record)
}

// GetVector retrieves a vector embedding by key.
func (ls *LocalStorage) GetVector(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := ls.requireVectorStore(); err != nil {
		return nil, err
	}
	return ls.vectorStore.Get(ctx, scope, scopeID, key)
}

// DeleteVector removes a stored vector embedding.
func (ls *LocalStorage) DeleteVector(ctx context.Context, scope, scopeID, key string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := ls.requireVectorStore(); err != nil {
		return err
	}
	return ls.vectorStore.Delete(ctx, scope, scopeID, key)
}

// DeleteVectorsByPrefix deletes all vectors whose key starts with the given prefix.
func (ls *LocalStorage) DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, err
	}
	if err := ls.requireVectorStore(); err != nil {
		return 0, err
	}
	return ls.vectorStore.DeleteByPrefix(ctx, scope, scopeID, prefix)
}

// SimilaritySearch performs a similarity search within a scope using the configured vector backend.
func (ls *LocalStorage) SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := ls.requireVectorStore(); err != nil {
		return nil, err
	}
	return ls.vectorStore.Search(ctx, scope, scopeID, queryEmbedding, topK, filters)
}

func (ls *LocalStorage) setMemoryPostgres(ctx context.Context, memory *types.Memory) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled before postgres SetMemory operation: %w", err)
	}

	payload, err := json.Marshal(memory)
	if err != nil {
		return fmt.Errorf("failed to marshal memory payload: %w", err)
	}

	query := `
        INSERT INTO kv_store(scope, scope_id, key, value, updated_at)
        VALUES (?, ?, ?, ?, NOW())
        ON CONFLICT(scope, scope_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = NOW();`

	if _, err := ls.db.ExecContext(ctx, query, memory.Scope, memory.ScopeID, memory.Key, payload); err != nil {
		return fmt.Errorf("failed to upsert memory in postgres: %w", err)
	}

	cacheKey := fmt.Sprintf("%s:%s:%s", memory.Scope, memory.ScopeID, memory.Key)
	ls.cache.Store(cacheKey, memory)

	return nil
}

func (ls *LocalStorage) getMemoryPostgres(ctx context.Context, scope, scopeID, key string) (*types.Memory, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled before postgres GetMemory operation: %w", err)
	}

	cacheKey := fmt.Sprintf("%s:%s:%s", scope, scopeID, key)
	if val, ok := ls.cache.Load(cacheKey); ok {
		if memory, ok := val.(*types.Memory); ok {
			return memory, nil
		}
	}

	query := `SELECT value FROM kv_store WHERE scope = ? AND scope_id = ? AND key = ?`
	row := ls.db.QueryRowContext(ctx, query, scope, scopeID, key)

	var payload []byte
	if err := row.Scan(&payload); err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("memory with key '%s' not found in scope '%s' for ID '%s'", key, scope, scopeID)
		}
		return nil, fmt.Errorf("failed to load memory from postgres: %w", err)
	}

	memory := &types.Memory{}
	if err := json.Unmarshal(payload, memory); err != nil {
		return nil, fmt.Errorf("failed to unmarshal postgres memory payload: %w", err)
	}

	ls.cache.Store(cacheKey, memory)
	return memory, nil
}

func (ls *LocalStorage) deleteMemoryPostgres(ctx context.Context, scope, scopeID, key string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled before postgres DeleteMemory operation: %w", err)
	}

	query := `DELETE FROM kv_store WHERE scope = ? AND scope_id = ? AND key = ?`
	result, err := ls.db.ExecContext(ctx, query, scope, scopeID, key)
	if err != nil {
		return fmt.Errorf("failed to delete memory from postgres: %w", err)
	}
	if rows, err := result.RowsAffected(); err == nil && rows == 0 {
		return fmt.Errorf("memory with key '%s' not found in scope '%s' for ID '%s'", key, scope, scopeID)
	}

	cacheKey := fmt.Sprintf("%s:%s:%s", scope, scopeID, key)
	ls.cache.Delete(cacheKey)

	return nil
}

func (ls *LocalStorage) listMemoryPostgres(ctx context.Context, scope, scopeID string) ([]*types.Memory, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled before postgres ListMemory operation: %w", err)
	}

	query := `SELECT value FROM kv_store WHERE scope = ? AND scope_id = ?`
	rows, err := ls.db.QueryContext(ctx, query, scope, scopeID)
	if err != nil {
		return nil, fmt.Errorf("failed to list memory from postgres: %w", err)
	}
	defer rows.Close()

	memories := []*types.Memory{}
	for rows.Next() {
		var payload []byte
		if err := rows.Scan(&payload); err != nil {
			return nil, fmt.Errorf("failed to scan postgres memory payload: %w", err)
		}

		memory := &types.Memory{}
		if err := json.Unmarshal(payload, memory); err != nil {
			return nil, fmt.Errorf("failed to unmarshal postgres memory payload: %w", err)
		}

		memories = append(memories, memory)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating postgres memory rows: %w", err)
	}

	return memories, nil
}

// Set implements the CacheProvider Set method using the in-memory cache.
func (ls *LocalStorage) Set(key string, value interface{}, ttl time.Duration) error {
	// TODO: Implement TTL for in-memory cache if needed, or rely on BoltDB TTL
	ls.cache.Store(key, value)
	return nil
}

// Get implements the CacheProvider Get method using the in-memory cache.
func (ls *LocalStorage) Get(key string, dest interface{}) error {
	if val, ok := ls.cache.Load(key); ok {
		// Attempt to unmarshal if dest is a pointer to a struct
		if destPtr := reflect.ValueOf(dest); destPtr.Kind() == reflect.Ptr && destPtr.Elem().Kind() == reflect.Struct {
			valBytes, err := json.Marshal(val)
			if err != nil {
				return fmt.Errorf("failed to marshal cached value for unmarshalling: %w", err)
			}
			if err := json.Unmarshal(valBytes, dest); err != nil {
				return fmt.Errorf("failed to unmarshal cached value into destination: %w", err)
			}
			return nil
		}
		// Otherwise, return the value directly if types match
		if reflect.TypeOf(val) == reflect.TypeOf(dest).Elem() {
			reflect.ValueOf(dest).Elem().Set(reflect.ValueOf(val))
			return nil
		}
		return fmt.Errorf("cached value type mismatch")
	}
	return fmt.Errorf("key '%s' not found in cache", key)
}

// Delete implements the CacheProvider Delete method using the in-memory cache.
func (ls *LocalStorage) Delete(key string) error {
	ls.cache.Delete(key)
	return nil
}

// Exists implements the CacheProvider Exists method using the in-memory cache.
func (ls *LocalStorage) Exists(key string) bool {
	_, ok := ls.cache.Load(key)
	return ok
}

// Subscribe implements the CacheProvider Subscribe method using local pub/sub.
func (ls *LocalStorage) Subscribe(channel string) (<-chan CacheMessage, error) {
	ls.mu.Lock()
	defer ls.mu.Unlock()

	// Create a new channel for this subscriber
	subChannel := make(chan types.MemoryChangeEvent, 100) // Buffered channel

	// Store the subscriber channel
	ls.subscribers[channel] = append(ls.subscribers[channel], subChannel)

	// Convert MemoryChangeEvent to CacheMessage for the return channel
	cacheMsgChannel := make(chan CacheMessage, 100)
	go func() {
		for event := range subChannel {
			payload, _ := json.Marshal(event) // Marshal event to bytes
			cacheMsgChannel <- CacheMessage{
				Channel: channel,
				Payload: payload,
			}
		}
		close(cacheMsgChannel)
	}()

	return cacheMsgChannel, nil
}

// Publish implements the CacheProvider Publish method using local pub/sub.
func (ls *LocalStorage) Publish(channel string, message interface{}) error {
	ls.mu.RLock()
	defer ls.mu.RUnlock()

	// Send message to all subscribers of the channel
	if subscribers, ok := ls.subscribers[channel]; ok {
		for _, subChannel := range subscribers {
			// Non-blocking send
			select {
			case subChannel <- message.(types.MemoryChangeEvent): // Assuming message is always MemoryChangeEvent for this channel
				// Sent successfully
			default:
				// Subscriber channel is full, drop the message or log a warning
				fmt.Printf("Warning: Subscriber channel for '%s' is full, dropping message.\n", channel)
			}
		}
	}

	return nil
}

// publishMemoryChange is an internal helper to publish memory change events.
func subscriberKey(scope, scopeID string) string {
	if scope == "" {
		scope = "*"
	}
	if scopeID == "" {
		scopeID = "*"
	}
	return fmt.Sprintf("memory_changes:%s:%s", scope, scopeID)
}

func (ls *LocalStorage) publishMemoryChange(event types.MemoryChangeEvent) {
	targets := map[string]struct{}{}
	keys := []string{
		subscriberKey(event.Scope, event.ScopeID),
		subscriberKey(event.Scope, "*"),
		subscriberKey("*", event.ScopeID),
		subscriberKey("*", "*"),
	}
	for _, key := range keys {
		targets[key] = struct{}{}
	}

	// Use a goroutine to avoid blocking the main thread
	go func() {
		ls.mu.RLock()
		defer ls.mu.RUnlock()

		for key := range targets {
			if subscribers, ok := ls.subscribers[key]; ok {
				for _, subChannel := range subscribers {
					// Non-blocking send
					select {
					case subChannel <- event:
						// Sent successfully
					default:
						fmt.Printf("Warning: Memory change subscriber channel for '%s' is full, dropping event.\n", key)
					}
				}
			}
		}
	}()
}

// RegisterAgent stores an agent node record in SQLite.
func (ls *LocalStorage) RegisterAgent(ctx context.Context, agent *types.AgentNode) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during register agent: %w", err)
	}

	if strings.TrimSpace(agent.DeploymentType) == "" {
		agent.DeploymentType = "long_running"
	}

	// Begin transaction for atomic operation
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction for agent registration: %w", err)
	}
	defer rollbackTx(tx, "RegisterAgent:"+agent.ID)

	// Execute the agent registration using the transaction
	if err := ls.executeRegisterAgent(ctx, tx, agent); err != nil {
		return err
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit agent registration transaction: %w", err)
	}

	return nil
}

// executeRegisterAgent performs the actual agent registration using DBTX interface
func (ls *LocalStorage) executeRegisterAgent(ctx context.Context, q DBTX, agent *types.AgentNode) error {
	query := `
		INSERT INTO agent_nodes (
			id, version, group_id, team_id, base_url, traffic_weight, deployment_type, invocation_url, reasoners, skills,
			communication_config, health_status, lifecycle_status, last_heartbeat,
			registered_at, features, metadata, proposed_tags, approved_tags
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id, version) DO UPDATE SET
			group_id = excluded.group_id,
			team_id = excluded.team_id,
			base_url = excluded.base_url,
			traffic_weight = excluded.traffic_weight,
			deployment_type = excluded.deployment_type,
			invocation_url = excluded.invocation_url,
			reasoners = excluded.reasoners,
			skills = excluded.skills,
			communication_config = excluded.communication_config,
			health_status = excluded.health_status,
			lifecycle_status = excluded.lifecycle_status,
			last_heartbeat = excluded.last_heartbeat,
			features = excluded.features,
			metadata = excluded.metadata,
			proposed_tags = excluded.proposed_tags,
			approved_tags = excluded.approved_tags;`

	reasonersJSON, err := json.Marshal(agent.Reasoners)
	if err != nil {
		return fmt.Errorf("failed to marshal reasoners: %w", err)
	}
	skillsJSON, err := json.Marshal(agent.Skills)
	if err != nil {
		return fmt.Errorf("failed to marshal skills: %w", err)
	}
	commConfigJSON, err := json.Marshal(agent.CommunicationConfig)
	if err != nil {
		return fmt.Errorf("failed to marshal communication config: %w", err)
	}
	featuresJSON, err := json.Marshal(agent.Features)
	if err != nil {
		return fmt.Errorf("failed to marshal agent features: %w", err)
	}
	metadataJSON, err := json.Marshal(agent.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal agent metadata: %w", err)
	}
	proposedTagsJSON, err := json.Marshal(agent.ProposedTags)
	if err != nil {
		return fmt.Errorf("failed to marshal proposed tags: %w", err)
	}
	approvedTagsJSON, err := json.Marshal(agent.ApprovedTags)
	if err != nil {
		return fmt.Errorf("failed to marshal approved tags: %w", err)
	}

	trafficWeight := agent.TrafficWeight
	if trafficWeight == 0 {
		trafficWeight = 100
	}

	_, err = q.ExecContext(ctx, query,
		agent.ID, agent.Version, agent.GroupID, agent.TeamID, agent.BaseURL, trafficWeight, agent.DeploymentType, agent.InvocationURL,
		reasonersJSON, skillsJSON, commConfigJSON, agent.HealthStatus, agent.LifecycleStatus,
		agent.LastHeartbeat, agent.RegisteredAt, featuresJSON, metadataJSON, proposedTagsJSON, approvedTagsJSON,
	)

	if err != nil {
		return fmt.Errorf("failed to register agent node: %w", err)
	}

	return nil
}

// GetAgent retrieves the default (unversioned) agent node record by ID.
// It filters for version = ” to return only the default agent.
// Use GetAgentVersion for a specific version, or ListAgentVersions for all versions.
func (ls *LocalStorage) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get agent: %w", err)
	}

	query := `
		SELECT
			id, version, group_id, team_id, base_url, traffic_weight, deployment_type, invocation_url, reasoners, skills,
			communication_config, health_status, lifecycle_status, last_heartbeat,
			registered_at, features, metadata, proposed_tags, approved_tags
		FROM agent_nodes WHERE id = ?
		ORDER BY CASE WHEN version = '' THEN 0 ELSE 1 END, version ASC
		LIMIT 1`

	row := ls.db.QueryRowContext(ctx, query, id)

	agent := &types.AgentNode{}
	var reasonersJSON, skillsJSON, commConfigJSON, featuresJSON, metadataJSON []byte
	var proposedTagsJSON, approvedTagsJSON []byte
	var healthStatusStr, lifecycleStatusStr string
	var invocationURL sql.NullString
	var lastHeartbeat, registeredAt sql.NullTime

	err := row.Scan(
		&agent.ID, &agent.Version, &agent.GroupID, &agent.TeamID, &agent.BaseURL, &agent.TrafficWeight, &agent.DeploymentType, &invocationURL,
		&reasonersJSON, &skillsJSON, &commConfigJSON, &healthStatusStr, &lifecycleStatusStr,
		&lastHeartbeat, &registeredAt, &featuresJSON, &metadataJSON,
		&proposedTagsJSON, &approvedTagsJSON,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("agent node with ID '%s' not found", id)
		}
		return nil, fmt.Errorf("failed to get agent node with ID '%s': %w", id, err)
	}

	if lastHeartbeat.Valid {
		agent.LastHeartbeat = lastHeartbeat.Time
	}
	if registeredAt.Valid {
		agent.RegisteredAt = registeredAt.Time
	}
	agent.HealthStatus = types.HealthStatus(healthStatusStr)
	agent.LifecycleStatus = types.AgentLifecycleStatus(lifecycleStatusStr)
	if invocationURL.Valid && strings.TrimSpace(invocationURL.String) != "" {
		url := strings.TrimSpace(invocationURL.String)
		agent.InvocationURL = &url
	}

	if len(reasonersJSON) > 0 {
		if err := json.Unmarshal(reasonersJSON, &agent.Reasoners); err != nil {
			return nil, fmt.Errorf("failed to unmarshal agent reasoners: %w", err)
		}
	}
	if len(skillsJSON) > 0 {
		if err := json.Unmarshal(skillsJSON, &agent.Skills); err != nil {
			return nil, fmt.Errorf("failed to unmarshal agent skills: %w", err)
		}
	}
	if len(commConfigJSON) > 0 {
		if err := json.Unmarshal(commConfigJSON, &agent.CommunicationConfig); err != nil {
			return nil, fmt.Errorf("failed to unmarshal agent communication config: %w", err)
		}
	}
	if len(featuresJSON) > 0 {
		if err := json.Unmarshal(featuresJSON, &agent.Features); err != nil {
			return nil, fmt.Errorf("failed to unmarshal agent features: %w", err)
		}
	}
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &agent.Metadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal agent metadata: %w", err)
		}
	}
	if len(proposedTagsJSON) > 0 {
		if err := json.Unmarshal(proposedTagsJSON, &agent.ProposedTags); err != nil {
			return nil, fmt.Errorf("failed to unmarshal agent proposed tags: %w", err)
		}
	}
	if len(approvedTagsJSON) > 0 {
		if err := json.Unmarshal(approvedTagsJSON, &agent.ApprovedTags); err != nil {
			return nil, fmt.Errorf("failed to unmarshal agent approved tags: %w", err)
		}
	}
	if strings.TrimSpace(agent.DeploymentType) == "" {
		if agent.InvocationURL != nil && strings.TrimSpace(*agent.InvocationURL) != "" {
			agent.DeploymentType = "serverless"
		} else if agent.Metadata.Custom != nil {
			if v, ok := agent.Metadata.Custom["serverless"]; ok && fmt.Sprint(v) == "true" {
				agent.DeploymentType = "serverless"
			}
		}
		if strings.TrimSpace(agent.DeploymentType) == "" {
			agent.DeploymentType = "long_running"
		}
	}
	if agent.DeploymentType == "serverless" && (agent.InvocationURL == nil || strings.TrimSpace(*agent.InvocationURL) == "") {
		if trimmed := strings.TrimSpace(agent.BaseURL); trimmed != "" {
			execURL := strings.TrimSuffix(trimmed, "/") + "/execute"
			agent.InvocationURL = &execURL
		}
	}

	// Reconstruct agent-level ProposedTags and ApprovedTags from per-component fields.
	// These fields are not stored in dedicated columns but are derived from the
	// reasoners/skills JSON blobs.
	reconstructAgentLevelTags(agent)

	return agent, nil
}

// GetAgentVersion retrieves a specific (id, version) agent node.
func (ls *LocalStorage) GetAgentVersion(ctx context.Context, id string, version string) (*types.AgentNode, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get agent version: %w", err)
	}

	query := `
		SELECT
			id, version, group_id, team_id, base_url, traffic_weight, deployment_type, invocation_url, reasoners, skills,
			communication_config, health_status, lifecycle_status, last_heartbeat,
			registered_at, features, metadata, proposed_tags, approved_tags
		FROM agent_nodes WHERE id = ? AND version = ?`

	row := ls.db.QueryRowContext(ctx, query, id, version)
	return ls.scanAgentNode(row)
}

// DeleteAgentVersion deletes a specific agent version row from the agent_nodes table.
func (ls *LocalStorage) DeleteAgentVersion(ctx context.Context, id string, version string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during delete agent version: %w", err)
	}

	_, err := ls.db.ExecContext(ctx, `DELETE FROM agent_nodes WHERE id = ? AND version = ?`, id, version)
	if err != nil {
		return fmt.Errorf("failed to delete agent version id='%s' version='%s': %w", id, version, err)
	}
	return nil
}

// ListAgentVersions returns all versioned agents with the given ID (version != ”).
func (ls *LocalStorage) ListAgentVersions(ctx context.Context, id string) ([]*types.AgentNode, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list agent versions: %w", err)
	}

	query := `
		SELECT
			id, version, group_id, team_id, base_url, traffic_weight, deployment_type, invocation_url, reasoners, skills,
			communication_config, health_status, lifecycle_status, last_heartbeat,
			registered_at, features, metadata, proposed_tags, approved_tags
		FROM agent_nodes WHERE id = ? AND version != '' ORDER BY registered_at DESC`

	rows, err := ls.db.QueryContext(ctx, query, id)
	if err != nil {
		return nil, fmt.Errorf("failed to list agent versions for '%s': %w", id, err)
	}
	defer rows.Close()

	return ls.scanAgentNodes(ctx, rows)
}

// scanAgentNode scans a single row into an AgentNode, applying post-processing.
func (ls *LocalStorage) scanAgentNode(row *sql.Row) (*types.AgentNode, error) {
	agent := &types.AgentNode{}
	var reasonersJSON, skillsJSON, commConfigJSON, featuresJSON, metadataJSON []byte
	var proposedTagsJSON, approvedTagsJSON []byte
	var healthStatusStr, lifecycleStatusStr string
	var invocationURL sql.NullString
	var lastHeartbeat, registeredAt sql.NullTime

	err := row.Scan(
		&agent.ID, &agent.Version, &agent.GroupID, &agent.TeamID, &agent.BaseURL, &agent.TrafficWeight, &agent.DeploymentType, &invocationURL,
		&reasonersJSON, &skillsJSON, &commConfigJSON, &healthStatusStr, &lifecycleStatusStr,
		&lastHeartbeat, &registeredAt, &featuresJSON, &metadataJSON,
		&proposedTagsJSON, &approvedTagsJSON,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("agent node with ID '%s' version '%s' not found", agent.ID, agent.Version)
		}
		return nil, fmt.Errorf("failed to scan agent node: %w", err)
	}

	if lastHeartbeat.Valid {
		agent.LastHeartbeat = lastHeartbeat.Time
	}
	if registeredAt.Valid {
		agent.RegisteredAt = registeredAt.Time
	}
	ls.postProcessAgentNode(agent, healthStatusStr, lifecycleStatusStr, invocationURL,
		reasonersJSON, skillsJSON, commConfigJSON, featuresJSON, metadataJSON, proposedTagsJSON, approvedTagsJSON)
	return agent, nil
}

// scanAgentNodes scans multiple rows into AgentNode slices, applying post-processing.
func (ls *LocalStorage) scanAgentNodes(ctx context.Context, rows *sql.Rows) ([]*types.AgentNode, error) {
	agents := []*types.AgentNode{}
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during agent list iteration: %w", err)
		}

		agent := &types.AgentNode{}
		var reasonersJSON, skillsJSON, commConfigJSON, featuresJSON, metadataJSON []byte
		var proposedTagsJSON, approvedTagsJSON []byte
		var healthStatusStr, lifecycleStatusStr string
		var invocationURL sql.NullString
		var lastHeartbeat, registeredAt sql.NullTime

		err := rows.Scan(
			&agent.ID, &agent.Version, &agent.GroupID, &agent.TeamID, &agent.BaseURL, &agent.TrafficWeight, &agent.DeploymentType, &invocationURL,
			&reasonersJSON, &skillsJSON, &commConfigJSON, &healthStatusStr, &lifecycleStatusStr,
			&lastHeartbeat, &registeredAt, &featuresJSON, &metadataJSON,
			&proposedTagsJSON, &approvedTagsJSON,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan agent node row: %w", err)
		}

		if lastHeartbeat.Valid {
			agent.LastHeartbeat = lastHeartbeat.Time
		}
		if registeredAt.Valid {
			agent.RegisteredAt = registeredAt.Time
		}
		ls.postProcessAgentNode(agent, healthStatusStr, lifecycleStatusStr, invocationURL,
			reasonersJSON, skillsJSON, commConfigJSON, featuresJSON, metadataJSON, proposedTagsJSON, approvedTagsJSON)
		agents = append(agents, agent)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error after listing agent nodes: %w", err)
	}
	return agents, nil
}

// postProcessAgentNode applies common post-processing to a scanned AgentNode.
func (ls *LocalStorage) postProcessAgentNode(agent *types.AgentNode, healthStatusStr, lifecycleStatusStr string, invocationURL sql.NullString,
	reasonersJSON, skillsJSON, commConfigJSON, featuresJSON, metadataJSON, proposedTagsJSON, approvedTagsJSON []byte) {

	agent.HealthStatus = types.HealthStatus(healthStatusStr)
	agent.LifecycleStatus = types.AgentLifecycleStatus(lifecycleStatusStr)
	if invocationURL.Valid && strings.TrimSpace(invocationURL.String) != "" {
		url := strings.TrimSpace(invocationURL.String)
		agent.InvocationURL = &url
	}

	if len(reasonersJSON) > 0 {
		_ = json.Unmarshal(reasonersJSON, &agent.Reasoners)
	}
	if len(skillsJSON) > 0 {
		_ = json.Unmarshal(skillsJSON, &agent.Skills)
	}
	if len(commConfigJSON) > 0 {
		_ = json.Unmarshal(commConfigJSON, &agent.CommunicationConfig)
	}
	if len(featuresJSON) > 0 {
		_ = json.Unmarshal(featuresJSON, &agent.Features)
	}
	if len(metadataJSON) > 0 {
		_ = json.Unmarshal(metadataJSON, &agent.Metadata)
	}
	if len(proposedTagsJSON) > 0 {
		_ = json.Unmarshal(proposedTagsJSON, &agent.ProposedTags)
	}
	if len(approvedTagsJSON) > 0 {
		_ = json.Unmarshal(approvedTagsJSON, &agent.ApprovedTags)
	}

	if strings.TrimSpace(agent.DeploymentType) == "" {
		if agent.InvocationURL != nil && strings.TrimSpace(*agent.InvocationURL) != "" {
			agent.DeploymentType = "serverless"
		} else if agent.Metadata.Custom != nil {
			if v, ok := agent.Metadata.Custom["serverless"]; ok && fmt.Sprint(v) == "true" {
				agent.DeploymentType = "serverless"
			}
		}
		if strings.TrimSpace(agent.DeploymentType) == "" {
			agent.DeploymentType = "long_running"
		}
	}
	if agent.DeploymentType == "serverless" && (agent.InvocationURL == nil || strings.TrimSpace(*agent.InvocationURL) == "") {
		if trimmed := strings.TrimSpace(agent.BaseURL); trimmed != "" {
			execURL := strings.TrimSuffix(trimmed, "/") + "/execute"
			agent.InvocationURL = &execURL
		}
	}

	reconstructAgentLevelTags(agent)
}

// ListAgents retrieves agent node records from SQLite based on filters.
func (ls *LocalStorage) ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list agents: %w", err)
	}

	query := `
		SELECT
			id, version, group_id, team_id, base_url, traffic_weight, deployment_type, invocation_url, reasoners, skills,
			communication_config, health_status, lifecycle_status, last_heartbeat,
			registered_at, features, metadata, proposed_tags, approved_tags
		FROM agent_nodes`

	var conditions []string
	var args []interface{}

	if filters.HealthStatus != nil {
		conditions = append(conditions, "health_status = ?")
		args = append(args, string(*filters.HealthStatus))
	}
	if filters.TeamID != nil {
		conditions = append(conditions, "team_id = ?")
		args = append(args, *filters.TeamID)
	}
	if filters.GroupID != nil {
		conditions = append(conditions, "group_id = ?")
		args = append(args, *filters.GroupID)
	}

	if len(conditions) > 0 {
		query += " WHERE " + conditions[0]
		for i := 1; i < len(conditions); i++ {
			query += " AND " + conditions[i]
		}
	}

	query += " ORDER BY registered_at DESC"

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list agent nodes: %w", err)
	}
	defer rows.Close()

	return ls.scanAgentNodes(ctx, rows)
}

// ListAgentsByGroup returns all agents belonging to a specific group.
func (ls *LocalStorage) ListAgentsByGroup(ctx context.Context, groupID string) ([]*types.AgentNode, error) {
	return ls.ListAgents(ctx, types.AgentFilters{GroupID: &groupID})
}

// ListAgentGroups returns distinct agent groups with summary info for a team.
func (ls *LocalStorage) ListAgentGroups(ctx context.Context, teamID string) ([]types.AgentGroupSummary, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list agent groups: %w", err)
	}

	var query string
	if ls.mode == "postgres" {
		query = `
			SELECT group_id, team_id, COUNT(*) as node_count, STRING_AGG(DISTINCT version, ',') as versions
			FROM agent_nodes
			WHERE team_id = $1
			GROUP BY group_id, team_id
			ORDER BY group_id`
	} else {
		query = `
			SELECT group_id, team_id, COUNT(*) as node_count, GROUP_CONCAT(DISTINCT version) as versions
			FROM agent_nodes
			WHERE team_id = ?
			GROUP BY group_id, team_id
			ORDER BY group_id`
	}

	rows, err := ls.db.QueryContext(ctx, query, teamID)
	if err != nil {
		return nil, fmt.Errorf("failed to list agent groups: %w", err)
	}
	defer rows.Close()

	var groups []types.AgentGroupSummary
	for rows.Next() {
		var g types.AgentGroupSummary
		var versionsStr sql.NullString
		if err := rows.Scan(&g.GroupID, &g.TeamID, &g.NodeCount, &versionsStr); err != nil {
			return nil, fmt.Errorf("failed to scan agent group row: %w", err)
		}
		if versionsStr.Valid && versionsStr.String != "" {
			g.Versions = strings.Split(versionsStr.String, ",")
		}
		groups = append(groups, g)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error after listing agent groups: %w", err)
	}

	return groups, nil
}

// UpdateAgentHealth updates the health status of an agent node in SQLite.
// IMPORTANT: This method ONLY updates health_status, never last_heartbeat (only heartbeat endpoint should do that)
func (ls *LocalStorage) UpdateAgentHealth(ctx context.Context, id string, status types.HealthStatus) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during update agent health: %w", err)
	}

	// Begin transaction for atomic operation
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction for agent health update: %w", err)
	}
	defer rollbackTx(tx, "UpdateAgentHealth:"+id)

	// Execute the health update using the transaction
	if err := ls.executeUpdateAgentHealth(ctx, tx, id, status); err != nil {
		return err
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit agent health status transaction: %w", err)
	}

	return nil
}

// executeUpdateAgentHealth performs the actual health status update using DBTX interface
func (ls *LocalStorage) executeUpdateAgentHealth(ctx context.Context, q DBTX, id string, status types.HealthStatus) error {
	query := `
		UPDATE agent_nodes
		SET health_status = ?
		WHERE id = ?;`

	_, err := q.ExecContext(ctx, query, status, id)
	if err != nil {
		return fmt.Errorf("failed to update agent health status for ID '%s': %w", id, err)
	}

	return nil
}

// UpdateAgentHealthAtomic updates the health status of an agent node atomically with optimistic locking.
// If expectedLastHeartbeat is provided, the update will only succeed if the current last_heartbeat matches.
// This prevents race conditions between health monitor and heartbeat updates.
// IMPORTANT: This method ONLY updates health_status, never last_heartbeat (only heartbeat endpoint should do that)
func (ls *LocalStorage) UpdateAgentHealthAtomic(ctx context.Context, id string, status types.HealthStatus, expectedLastHeartbeat *time.Time) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during update agent health atomic: %w", err)
	}

	var query string
	var args []interface{}

	if expectedLastHeartbeat != nil {
		// Atomic update with optimistic locking - only update health_status if last_heartbeat hasn't changed
		// DO NOT update last_heartbeat here - that creates phantom heartbeats!
		query = `
			UPDATE agent_nodes
			SET health_status = ?
			WHERE id = ? AND last_heartbeat = ?;`
		args = []interface{}{status, id, expectedLastHeartbeat.UTC().Format(time.RFC3339Nano)}
	} else {
		// Standard atomic update without timestamp check - only update health_status
		query = `
			UPDATE agent_nodes
			SET health_status = ?
			WHERE id = ?;`
		args = []interface{}{status, id}
	}

	result, err := ls.db.ExecContext(ctx, query, args...)
	if err != nil {
		return fmt.Errorf("failed to update agent health status atomically for ID '%s': %w", id, err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected for agent health update ID '%s': %w", id, err)
	}

	if rowsAffected == 0 {
		if expectedLastHeartbeat != nil {
			return fmt.Errorf("no rows updated for agent ID '%s' - possible concurrent modification or node not found", id)
		} else {
			return fmt.Errorf("agent node with ID '%s' not found", id)
		}
	}

	return nil
}

// UpdateAgentHeartbeat updates only the heartbeat timestamp of an agent node in SQLite.
// If version is empty, it updates the default (unversioned) agent.
func (ls *LocalStorage) UpdateAgentHeartbeat(ctx context.Context, id string, version string, heartbeatTime time.Time) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during update agent heartbeat: %w", err)
	}

	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction for agent heartbeat update: %w", err)
	}
	defer rollbackTx(tx, "UpdateAgentHeartbeat:"+id)

	if err := ls.executeUpdateAgentHeartbeat(ctx, tx, id, version, heartbeatTime); err != nil {
		return err
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit agent heartbeat transaction: %w", err)
	}

	return nil
}

// executeUpdateAgentHeartbeat performs the actual heartbeat timestamp update using DBTX interface
func (ls *LocalStorage) executeUpdateAgentHeartbeat(ctx context.Context, q DBTX, id string, version string, heartbeatTime time.Time) error {
	query := `
		UPDATE agent_nodes
		SET last_heartbeat = ?
		WHERE id = ? AND version = ?;`

	_, err := q.ExecContext(ctx, query, heartbeatTime.UTC().Format(time.RFC3339Nano), id, version)
	if err != nil {
		return fmt.Errorf("failed to update agent heartbeat for ID '%s' version '%s': %w", id, version, err)
	}

	return nil
}

// UpdateAgentLifecycleStatus updates the lifecycle status of an agent node in SQLite.
func (ls *LocalStorage) UpdateAgentLifecycleStatus(ctx context.Context, id string, status types.AgentLifecycleStatus) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during update agent lifecycle status: %w", err)
	}

	// Begin transaction for atomic operation
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction for agent lifecycle update: %w", err)
	}
	defer rollbackTx(tx, "UpdateAgentLifecycleStatus:"+id)

	// Execute the lifecycle status update using the transaction
	if err := ls.executeUpdateAgentLifecycleStatus(ctx, tx, id, status); err != nil {
		return err
	}

	// Commit transaction
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit agent lifecycle status transaction: %w", err)
	}

	return nil
}

// executeUpdateAgentLifecycleStatus performs the actual lifecycle status update using DBTX interface
func (ls *LocalStorage) executeUpdateAgentLifecycleStatus(ctx context.Context, q DBTX, id string, status types.AgentLifecycleStatus) error {
	query := `
		UPDATE agent_nodes
		SET lifecycle_status = ?
		WHERE id = ?;`

	_, err := q.ExecContext(ctx, query, status, id)
	if err != nil {
		fmt.Printf("❌ DEBUG: Database update failed for node %s: %v\n", id, err)
		return fmt.Errorf("failed to update agent lifecycle status for ID '%s': %w", id, err)
	}

	return nil
}

// UpdateAgentVersion updates only the version field for an agent node.
func (ls *LocalStorage) UpdateAgentVersion(ctx context.Context, id string, version string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during update agent version: %w", err)
	}

	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction for agent version update: %w", err)
	}
	defer rollbackTx(tx, "UpdateAgentVersion:"+id)

	query := `UPDATE agent_nodes SET version = ? WHERE id = ?;`
	if _, err := tx.ExecContext(ctx, query, version, id); err != nil {
		return fmt.Errorf("failed to update agent version for ID '%s': %w", id, err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit agent version transaction: %w", err)
	}

	return nil
}

// UpdateAgentTrafficWeight sets the traffic_weight for a specific (id, version) pair.
func (ls *LocalStorage) UpdateAgentTrafficWeight(ctx context.Context, id string, version string, weight int) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during update traffic weight: %w", err)
	}

	result, err := ls.db.ExecContext(ctx,
		`UPDATE agent_nodes SET traffic_weight = ? WHERE id = ? AND version = ?`,
		weight, id, version)
	if err != nil {
		return fmt.Errorf("failed to update traffic weight: %w", err)
	}
	rows, _ := result.RowsAffected()
	if rows == 0 {
		return fmt.Errorf("agent (id=%s, version=%s) not found", id, version)
	}
	return nil
}

// SetConfig upserts a configuration entry in the database.
// On conflict (duplicate key), it increments the version and updates the value.
func (ls *LocalStorage) SetConfig(ctx context.Context, key string, value string, updatedBy string) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	db := ls.requireSQLDB()
	now := time.Now().UTC()

	if ls.mode == "postgres" {
		_, err := db.ExecContext(ctx, `
			INSERT INTO config_storage (key, value, version, created_by, updated_by, created_at, updated_at)
			VALUES ($1, $2, 1, $3, $3, $4, $4)
			ON CONFLICT (key) DO UPDATE SET
				value = EXCLUDED.value,
				version = config_storage.version + 1,
				updated_by = EXCLUDED.updated_by,
				updated_at = EXCLUDED.updated_at`,
			key, value, updatedBy, now)
		return err
	}

	// SQLite
	_, err := db.ExecContext(ctx, `
		INSERT INTO config_storage (key, value, version, created_by, updated_by, created_at, updated_at)
		VALUES (?, ?, 1, ?, ?, ?, ?)
		ON CONFLICT (key) DO UPDATE SET
			value = excluded.value,
			version = config_storage.version + 1,
			updated_by = excluded.updated_by,
			updated_at = excluded.updated_at`,
		key, value, updatedBy, updatedBy, now, now)
	return err
}

// GetConfig retrieves a configuration entry by key.
func (ls *LocalStorage) GetConfig(ctx context.Context, key string) (*ConfigEntry, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	db := ls.requireSQLDB()
	var entry ConfigEntry

	var placeholder string
	if ls.mode == "postgres" {
		placeholder = "$1"
	} else {
		placeholder = "?"
	}

	row := db.QueryRowContext(ctx,
		fmt.Sprintf(`SELECT key, value, version, COALESCE(created_by, ''), COALESCE(updated_by, ''), created_at, updated_at
		FROM config_storage WHERE key = %s`, placeholder), key)

	err := row.Scan(&entry.Key, &entry.Value, &entry.Version,
		&entry.CreatedBy, &entry.UpdatedBy, &entry.CreatedAt, &entry.UpdatedAt)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to get config %q: %w", key, err)
	}
	return &entry, nil
}

// ListConfigs returns all stored configuration entries.
func (ls *LocalStorage) ListConfigs(ctx context.Context) ([]*ConfigEntry, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	db := ls.requireSQLDB()
	rows, err := db.QueryContext(ctx,
		`SELECT key, value, version, COALESCE(created_by, ''), COALESCE(updated_by, ''), created_at, updated_at
		FROM config_storage ORDER BY key`)
	if err != nil {
		return nil, fmt.Errorf("failed to list configs: %w", err)
	}
	defer rows.Close()

	var entries []*ConfigEntry
	for rows.Next() {
		var entry ConfigEntry
		if err := rows.Scan(&entry.Key, &entry.Value, &entry.Version,
			&entry.CreatedBy, &entry.UpdatedBy, &entry.CreatedAt, &entry.UpdatedAt); err != nil {
			return nil, fmt.Errorf("failed to scan config row: %w", err)
		}
		entries = append(entries, &entry)
	}
	return entries, rows.Err()
}

// DeleteConfig removes a configuration entry by key.
func (ls *LocalStorage) DeleteConfig(ctx context.Context, key string) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	db := ls.requireSQLDB()
	var placeholder string
	if ls.mode == "postgres" {
		placeholder = "$1"
	} else {
		placeholder = "?"
	}

	result, err := db.ExecContext(ctx,
		fmt.Sprintf(`DELETE FROM config_storage WHERE key = %s`, placeholder), key)
	if err != nil {
		return fmt.Errorf("failed to delete config %q: %w", key, err)
	}
	rows, _ := result.RowsAffected()
	if rows == 0 {
		return fmt.Errorf("config %q not found", key)
	}
	return nil
}

// SubscribeToMemoryChanges implements the StorageProvider SubscribeToMemoryChanges method using local pub/sub.
func (ls *LocalStorage) SubscribeToMemoryChanges(ctx context.Context, scope, scopeID string) (<-chan types.MemoryChangeEvent, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during subscribe to memory changes: %w", err)
	}

	channel := subscriberKey(scope, scopeID)
	ls.mu.Lock()
	defer ls.mu.Unlock()

	// Create a new channel for this subscriber
	subChannel := make(chan types.MemoryChangeEvent, 100) // Buffered channel

	// Store the subscriber channel
	ls.subscribers[channel] = append(ls.subscribers[channel], subChannel)

	return subChannel, nil
}

// PublishMemoryChange implements the StorageProvider PublishMemoryChange method using local pub/sub.
func (ls *LocalStorage) PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during publish memory change: %w", err)
	}

	ls.publishMemoryChange(event)
	return nil
}

// Transaction represents a database transaction.
type Transaction interface {
	StorageProvider
	Commit() error
	Rollback() error
}

// Agent Configuration Management Methods

func agentConfigurationToModel(cfg *types.AgentConfiguration) (*AgentConfigurationModel, error) {
	configJSON, err := json.Marshal(cfg.Configuration)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal configuration: %w", err)
	}

	encryptedFieldsJSON, err := json.Marshal(cfg.EncryptedFields)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal encrypted fields: %w", err)
	}

	return &AgentConfigurationModel{
		ID:              cfg.ID,
		AgentID:         cfg.AgentID,
		PackageID:       cfg.PackageID,
		Configuration:   configJSON,
		EncryptedFields: encryptedFieldsJSON,
		Status:          string(cfg.Status),
		Version:         cfg.Version,
		CreatedAt:       cfg.CreatedAt,
		UpdatedAt:       cfg.UpdatedAt,
		CreatedBy:       cfg.CreatedBy,
		UpdatedBy:       cfg.UpdatedBy,
	}, nil
}

func agentConfigurationFromModel(model *AgentConfigurationModel) (*types.AgentConfiguration, error) {
	cfg := &types.AgentConfiguration{
		ID:        model.ID,
		AgentID:   model.AgentID,
		PackageID: model.PackageID,
		Status:    types.ConfigurationStatus(model.Status),
		Version:   model.Version,
		CreatedAt: model.CreatedAt,
		UpdatedAt: model.UpdatedAt,
		CreatedBy: model.CreatedBy,
		UpdatedBy: model.UpdatedBy,
	}

	if len(model.Configuration) > 0 {
		if err := json.Unmarshal(model.Configuration, &cfg.Configuration); err != nil {
			return nil, fmt.Errorf("failed to unmarshal configuration: %w", err)
		}
	} else {
		cfg.Configuration = map[string]interface{}{}
	}

	if len(model.EncryptedFields) > 0 {
		if err := json.Unmarshal(model.EncryptedFields, &cfg.EncryptedFields); err != nil {
			return nil, fmt.Errorf("failed to unmarshal encrypted fields: %w", err)
		}
	}

	return cfg, nil
}

// StoreAgentConfiguration stores an agent configuration record in SQLite
func (ls *LocalStorage) StoreAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store agent configuration: %w", err)
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	model, err := agentConfigurationToModel(config)
	if err != nil {
		return err
	}

	result := gormDB.Clauses(clause.OnConflict{
		Columns: []clause.Column{{Name: "agent_id"}, {Name: "package_id"}},
		DoUpdates: clause.Assignments(map[string]interface{}{
			"configuration":    gorm.Expr("excluded.configuration"),
			"encrypted_fields": gorm.Expr("excluded.encrypted_fields"),
			"status":           gorm.Expr("excluded.status"),
			"version":          gorm.Expr("agent_configurations.version + 1"),
			"updated_at":       gorm.Expr("excluded.updated_at"),
			"updated_by":       gorm.Expr("excluded.updated_by"),
		}),
	}).Create(model)

	if result.Error != nil {
		return fmt.Errorf("failed to store agent configuration: %w", result.Error)
	}

	config.ID = model.ID
	return nil
}

// GetAgentConfiguration retrieves an agent configuration record from SQLite
func (ls *LocalStorage) GetAgentConfiguration(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error) {
	// Fast-fail if context is already cancelled
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	model := &AgentConfigurationModel{}
	if err := gormDB.Where("agent_id = ? AND package_id = ?", agentID, packageID).Take(model).Error; err != nil {
		if errors.Is(err, gorm.ErrRecordNotFound) {
			return nil, fmt.Errorf("configuration for agent '%s' and package '%s' not found", agentID, packageID)
		}
		return nil, fmt.Errorf("failed to get agent configuration: %w", err)
	}

	return agentConfigurationFromModel(model)
}

// QueryAgentConfigurations retrieves agent configuration records from SQLite based on filters
func (ls *LocalStorage) QueryAgentConfigurations(ctx context.Context, filters types.ConfigurationFilters) ([]*types.AgentConfiguration, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during query agent configurations: %w", err)
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	query := gormDB.Model(&AgentConfigurationModel{})

	if filters.AgentID != nil {
		query = query.Where("agent_id = ?", *filters.AgentID)
	}
	if filters.PackageID != nil {
		query = query.Where("package_id = ?", *filters.PackageID)
	}
	if filters.Status != nil {
		query = query.Where("status = ?", *filters.Status)
	}
	if filters.CreatedBy != nil {
		query = query.Where("created_by = ?", *filters.CreatedBy)
	}
	if filters.StartTime != nil {
		query = query.Where("created_at >= ?", *filters.StartTime)
	}
	if filters.EndTime != nil {
		query = query.Where("created_at <= ?", *filters.EndTime)
	}

	query = query.Order("updated_at DESC")
	if filters.Limit > 0 {
		query = query.Limit(filters.Limit)
	}
	if filters.Offset > 0 {
		query = query.Offset(filters.Offset)
	}

	var models []AgentConfigurationModel
	if err := query.Find(&models).Error; err != nil {
		return nil, fmt.Errorf("failed to query agent configurations: %w", err)
	}

	configurations := make([]*types.AgentConfiguration, 0, len(models))
	for i := range models {
		cfg, err := agentConfigurationFromModel(&models[i])
		if err != nil {
			return nil, err
		}
		configurations = append(configurations, cfg)
	}

	return configurations, nil
}

// UpdateAgentConfiguration updates an existing agent configuration record
func (ls *LocalStorage) UpdateAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error {
	// Fast-fail if context is already cancelled
	if err := ctx.Err(); err != nil {
		return err
	}

	configJSON, err := json.Marshal(config.Configuration)
	if err != nil {
		return fmt.Errorf("failed to marshal configuration: %w", err)
	}

	encryptedFieldsJSON, err := json.Marshal(config.EncryptedFields)
	if err != nil {
		return fmt.Errorf("failed to marshal encrypted fields: %w", err)
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	result := gormDB.Model(&AgentConfigurationModel{}).
		Where("agent_id = ? AND package_id = ?", config.AgentID, config.PackageID).
		Updates(map[string]interface{}{
			"configuration":    configJSON,
			"encrypted_fields": encryptedFieldsJSON,
			"status":           config.Status,
			"version":          gorm.Expr("version + 1"),
			"updated_at":       config.UpdatedAt,
			"updated_by":       config.UpdatedBy,
		})

	if result.Error != nil {
		return fmt.Errorf("failed to update agent configuration: %w", result.Error)
	}

	if result.RowsAffected == 0 {
		return fmt.Errorf("configuration for agent '%s' and package '%s' not found", config.AgentID, config.PackageID)
	}

	return nil
}

// DeleteAgentConfiguration deletes an agent configuration record
func (ls *LocalStorage) DeleteAgentConfiguration(ctx context.Context, agentID, packageID string) error {
	// Fast-fail if context is already cancelled
	if err := ctx.Err(); err != nil {
		return err
	}

	gormDB, err := ls.gormWithContext(ctx)
	if err != nil {
		return fmt.Errorf("failed to prepare gorm transaction: %w", err)
	}

	result := gormDB.Where("agent_id = ? AND package_id = ?", agentID, packageID).
		Delete(&AgentConfigurationModel{})

	if result.Error != nil {
		return fmt.Errorf("failed to delete agent configuration: %w", result.Error)
	}

	if result.RowsAffected == 0 {
		return fmt.Errorf("configuration for agent '%s' and package '%s' not found", agentID, packageID)
	}

	return nil
}

// ValidateAgentConfiguration validates a configuration against the package schema
func (ls *LocalStorage) ValidateAgentConfiguration(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during validate agent configuration: %w", err)
	}

	// Get the package to access its schema
	pkg, err := ls.GetAgentPackage(ctx, packageID)
	if err != nil {
		return &types.ConfigurationValidationResult{
			Valid:  false,
			Errors: []string{fmt.Sprintf("Package not found: %s", packageID)},
		}, nil
	}

	// Parse the configuration schema
	var schema map[string]interface{}
	if len(pkg.ConfigurationSchema) > 0 {
		if err := json.Unmarshal(pkg.ConfigurationSchema, &schema); err != nil {
			return &types.ConfigurationValidationResult{
				Valid:  false,
				Errors: []string{fmt.Sprintf("Invalid package schema: %v", err)},
			}, nil
		}
	}

	// TODO: Implement comprehensive validation logic
	// For now, return a basic validation result
	return &types.ConfigurationValidationResult{
		Valid:  true,
		Errors: []string{},
	}, nil
}

// Agent Package Management Methods

// StoreAgentPackage stores an agent package record in SQLite
func (ls *LocalStorage) StoreAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store agent package: %w", err)
	}

	query := `
		INSERT INTO agent_packages (
			id, name, version, description, author, repository,
			install_path, configuration_schema, status, configuration_status,
			installed_at, updated_at, metadata
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(id) DO UPDATE SET
			name = excluded.name,
			version = excluded.version,
			description = excluded.description,
			author = excluded.author,
			repository = excluded.repository,
			install_path = excluded.install_path,
			configuration_schema = excluded.configuration_schema,
			status = excluded.status,
			configuration_status = excluded.configuration_status,
			updated_at = excluded.updated_at,
			metadata = excluded.metadata;`

	metadataJSON, err := json.Marshal(pkg.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal package metadata: %w", err)
	}

	_, err = ls.db.ExecContext(ctx, query,
		pkg.ID, pkg.Name, pkg.Version, pkg.Description, pkg.Author,
		pkg.Repository, pkg.InstallPath, pkg.ConfigurationSchema,
		pkg.Status, pkg.ConfigurationStatus, pkg.InstalledAt,
		pkg.UpdatedAt, metadataJSON,
	)

	if err != nil {
		return fmt.Errorf("failed to store agent package: %w", err)
	}

	return nil
}

// GetAgentPackage retrieves an agent package record from SQLite
func (ls *LocalStorage) GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get agent package: %w", err)
	}

	query := `
		SELECT
			id, name, version, description, author, repository,
			install_path, configuration_schema, status, configuration_status,
			installed_at, updated_at, metadata
		FROM agent_packages WHERE id = ?`

	row := ls.db.QueryRowContext(ctx, query, packageID)

	pkg := &types.AgentPackage{}
	var metadataJSON []byte

	err := row.Scan(
		&pkg.ID, &pkg.Name, &pkg.Version, &pkg.Description, &pkg.Author,
		&pkg.Repository, &pkg.InstallPath, &pkg.ConfigurationSchema,
		&pkg.Status, &pkg.ConfigurationStatus, &pkg.InstalledAt,
		&pkg.UpdatedAt, &metadataJSON,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("package with ID '%s' not found", packageID)
		}
		return nil, fmt.Errorf("failed to get agent package: %w", err)
	}

	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &pkg.Metadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal package metadata: %w", err)
		}
	}

	return pkg, nil
}

// QueryAgentPackages retrieves agent package records from SQLite based on filters
func (ls *LocalStorage) QueryAgentPackages(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during query agent packages: %w", err)
	}

	query := `
		SELECT
			id, name, version, description, author, repository,
			install_path, configuration_schema, status, configuration_status,
			installed_at, updated_at, metadata
		FROM agent_packages`

	var conditions []string
	var args []interface{}

	// Add filters
	if filters.Status != nil {
		conditions = append(conditions, "status = ?")
		args = append(args, *filters.Status)
	}
	if filters.ConfigurationStatus != nil {
		conditions = append(conditions, "configuration_status = ?")
		args = append(args, *filters.ConfigurationStatus)
	}
	if filters.Name != nil {
		conditions = append(conditions, "name LIKE ?")
		args = append(args, "%"+*filters.Name+"%")
	}
	if filters.Author != nil {
		conditions = append(conditions, "author = ?")
		args = append(args, *filters.Author)
	}

	// Add WHERE clause if there are conditions
	if len(conditions) > 0 {
		query += " WHERE " + strings.Join(conditions, " AND ")
	}

	// Add ordering and pagination
	query += " ORDER BY updated_at DESC"
	if filters.Limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", filters.Limit)
	}
	if filters.Offset > 0 {
		query += fmt.Sprintf(" OFFSET %d", filters.Offset)
	}

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query agent packages: %w", err)
	}
	defer rows.Close()

	packages := []*types.AgentPackage{}
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during package iteration: %w", err)
		}

		pkg := &types.AgentPackage{}
		var metadataJSON []byte

		err := rows.Scan(
			&pkg.ID, &pkg.Name, &pkg.Version, &pkg.Description, &pkg.Author,
			&pkg.Repository, &pkg.InstallPath, &pkg.ConfigurationSchema,
			&pkg.Status, &pkg.ConfigurationStatus, &pkg.InstalledAt,
			&pkg.UpdatedAt, &metadataJSON,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan agent package row: %w", err)
		}

		if len(metadataJSON) > 0 {
			if err := json.Unmarshal(metadataJSON, &pkg.Metadata); err != nil {
				return nil, fmt.Errorf("failed to unmarshal package metadata: %w", err)
			}
		}

		packages = append(packages, pkg)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error after querying agent packages: %w", err)
	}

	return packages, nil
}

// UpdateAgentPackage updates an existing agent package record
func (ls *LocalStorage) UpdateAgentPackage(ctx context.Context, pkg *types.AgentPackage) error {
	// Fast-fail if context is already cancelled
	if err := ctx.Err(); err != nil {
		return err
	}

	query := `
		UPDATE agent_packages
		SET name = ?, version = ?, description = ?, author = ?, repository = ?,
			install_path = ?, configuration_schema = ?, status = ?,
			configuration_status = ?, updated_at = ?, metadata = ?
		WHERE id = ?;`

	metadataJSON, err := json.Marshal(pkg.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal package metadata: %w", err)
	}

	result, err := ls.db.ExecContext(ctx, query,
		pkg.Name, pkg.Version, pkg.Description, pkg.Author, pkg.Repository,
		pkg.InstallPath, pkg.ConfigurationSchema, pkg.Status,
		pkg.ConfigurationStatus, pkg.UpdatedAt, metadataJSON, pkg.ID,
	)

	if err != nil {
		return fmt.Errorf("failed to update agent package: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected for package update: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("package with ID '%s' not found", pkg.ID)
	}

	return nil
}

// DeleteAgentPackage deletes an agent package record
func (ls *LocalStorage) DeleteAgentPackage(ctx context.Context, packageID string) error {
	// Fast-fail if context is already cancelled
	if err := ctx.Err(); err != nil {
		return err
	}

	query := `DELETE FROM agent_packages WHERE id = ?;`

	result, err := ls.db.ExecContext(ctx, query, packageID)
	if err != nil {
		return fmt.Errorf("failed to delete agent package: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected for package deletion: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("package with ID '%s' not found", packageID)
	}

	return nil
}

// GetReasonerPerformanceMetrics retrieves performance metrics for a specific reasoner
// This is a read-only operation that leverages SQLite WAL mode for concurrent access
func (ls *LocalStorage) GetReasonerPerformanceMetrics(ctx context.Context, reasonerID string) (*types.ReasonerPerformanceMetrics, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get reasoner performance metrics: %w", err)
	}

	// Parse reasoner ID (format: "node_id.reasoner_id")
	parts := strings.SplitN(reasonerID, ".", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid reasoner_id format, expected 'node_id.reasoner_id'")
	}

	nodeID := parts[0]
	localReasonerID := parts[1]

	// Execute read-only query directly - no write mutex needed due to SQLite WAL mode
	// WAL mode allows concurrent readers without blocking writers
	return ls.executeReasonerMetricsQueryDirect(ctx, nodeID, localReasonerID)
}

// executeReasonerMetricsQuery performs the reasoner metrics query within a transaction
//
//nolint:unused // retained for upcoming analytics endpoints
func (ls *LocalStorage) executeReasonerMetricsQuery(tx DBTX, nodeID, localReasonerID string) (*types.ReasonerPerformanceMetrics, error) {
	// Query for metrics from workflow_executions table using separate node_id and reasoner_id
	metricsQuery := `
		SELECT
			COUNT(*) as total_executions,
			COALESCE(AVG(duration_ms), 0) as avg_duration,
			COALESCE(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END), 0) as successful_executions,
			COALESCE(SUM(CASE WHEN started_at >= datetime('now', '-24 hours') THEN 1 ELSE 0 END), 0) as executions_last_24h
		FROM workflow_executions
		WHERE agent_node_id = ? AND reasoner_id = ?`

	row := tx.QueryRow(metricsQuery, nodeID, localReasonerID)

	var totalExecutions, successfulExecutions, executionsLast24h int
	var avgDuration float64

	err := row.Scan(&totalExecutions, &avgDuration, &successfulExecutions, &executionsLast24h)
	if err != nil {
		return nil, fmt.Errorf("failed to query reasoner metrics: %w", err)
	}

	// Calculate success rate
	successRate := 0.0
	if totalExecutions > 0 {
		successRate = float64(successfulExecutions) / float64(totalExecutions)
	}

	// Get recent executions (last 5) - optimized query
	recentQuery := `
		SELECT execution_id, status, duration_ms, started_at
		FROM workflow_executions
		WHERE agent_node_id = ? AND reasoner_id = ?
		ORDER BY started_at DESC
		LIMIT 5`

	rows, err := tx.Query(recentQuery, nodeID, localReasonerID)
	if err != nil {
		return nil, fmt.Errorf("failed to query recent executions: %w", err)
	}
	defer rows.Close()

	var recentExecutions []types.RecentExecutionItem
	for rows.Next() {
		var item types.RecentExecutionItem
		var durationMs sql.NullInt64

		err := rows.Scan(&item.ExecutionID, &item.Status, &durationMs, &item.Timestamp)
		if err != nil {
			return nil, fmt.Errorf("failed to scan recent execution: %w", err)
		}

		if durationMs.Valid {
			item.DurationMs = durationMs.Int64
		}

		recentExecutions = append(recentExecutions, item)
	}

	avgResponseTimeMs := int(avgDuration)

	metrics := &types.ReasonerPerformanceMetrics{
		AvgResponseTimeMs: avgResponseTimeMs,
		SuccessRate:       successRate,
		TotalExecutions:   totalExecutions,
		ExecutionsLast24h: executionsLast24h,
		RecentExecutions:  recentExecutions,
	}

	return metrics, nil
}

// executeReasonerMetricsQueryDirect performs reasoner metrics query without transaction wrapper
// This is used when we detect we're already in a transaction context
func (ls *LocalStorage) executeReasonerMetricsQueryDirect(ctx context.Context, nodeID, localReasonerID string) (*types.ReasonerPerformanceMetrics, error) {
	// Query for metrics from workflow_executions table using separate node_id and reasoner_id
	metricsQuery := `
		SELECT
			COUNT(*) as total_executions,
			COALESCE(AVG(duration_ms), 0) as avg_duration,
			COALESCE(SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END), 0) as successful_executions,
			COALESCE(SUM(CASE WHEN started_at >= datetime('now', '-24 hours') THEN 1 ELSE 0 END), 0) as executions_last_24h
		FROM workflow_executions
		WHERE agent_node_id = ? AND reasoner_id = ?`

	row := ls.db.QueryRowContext(ctx, metricsQuery, nodeID, localReasonerID)

	var totalExecutions, successfulExecutions, executionsLast24h int
	var avgDuration float64

	err := row.Scan(&totalExecutions, &avgDuration, &successfulExecutions, &executionsLast24h)
	if err != nil {
		return nil, fmt.Errorf("failed to query reasoner metrics: %w", err)
	}

	// Calculate success rate
	successRate := 0.0
	if totalExecutions > 0 {
		successRate = float64(successfulExecutions) / float64(totalExecutions)
	}

	// Get recent executions (last 5) - optimized query
	recentQuery := `
		SELECT execution_id, status, duration_ms, started_at
		FROM workflow_executions
		WHERE agent_node_id = ? AND reasoner_id = ?
		ORDER BY started_at DESC
		LIMIT 5`

	rows, err := ls.db.QueryContext(ctx, recentQuery, nodeID, localReasonerID)
	if err != nil {
		return nil, fmt.Errorf("failed to query recent executions: %w", err)
	}
	defer rows.Close()

	var recentExecutions []types.RecentExecutionItem
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during recent executions iteration: %w", err)
		}

		var item types.RecentExecutionItem
		var durationMs sql.NullInt64

		err := rows.Scan(&item.ExecutionID, &item.Status, &durationMs, &item.Timestamp)
		if err != nil {
			return nil, fmt.Errorf("failed to scan recent execution: %w", err)
		}

		if durationMs.Valid {
			item.DurationMs = durationMs.Int64
		}

		recentExecutions = append(recentExecutions, item)
	}

	avgResponseTimeMs := int(avgDuration)

	metrics := &types.ReasonerPerformanceMetrics{
		AvgResponseTimeMs: avgResponseTimeMs,
		SuccessRate:       successRate,
		TotalExecutions:   totalExecutions,
		ExecutionsLast24h: executionsLast24h,
		RecentExecutions:  recentExecutions,
	}

	return metrics, nil
}

// GetReasonerExecutionHistory retrieves paginated execution history for a specific reasoner
// This is a read-only operation that leverages SQLite WAL mode for concurrent access
func (ls *LocalStorage) GetReasonerExecutionHistory(ctx context.Context, reasonerID string, page, limit int) (*types.ReasonerExecutionHistory, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get reasoner execution history: %w", err)
	}

	// Parse reasoner ID (format: "node_id.reasoner_id")
	parts := strings.SplitN(reasonerID, ".", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid reasoner_id format, expected 'node_id.reasoner_id'")
	}

	nodeID := parts[0]
	localReasonerID := parts[1]

	// Calculate offset
	offset := (page - 1) * limit

	// Execute read-only query directly - no write mutex needed due to SQLite WAL mode
	// WAL mode allows concurrent readers without blocking writers
	return ls.executeReasonerHistoryQueryDirect(ctx, nodeID, localReasonerID, page, limit, offset)
}

// executeReasonerHistoryQuery performs the reasoner history query within a transaction
//
//nolint:unused // retained for upcoming analytics endpoints
func (ls *LocalStorage) executeReasonerHistoryQuery(tx DBTX, nodeID, localReasonerID string, page, limit, offset int) (*types.ReasonerExecutionHistory, error) {
	// Use a single optimized query with window functions to get both count and data efficiently
	// This reduces lock time and improves performance
	combinedQuery := `
		WITH execution_data AS (
			SELECT
				execution_id, agent_node_id, reasoner_id, status, status_reason,
				input_data, output_data, error_message, duration_ms, retry_count,
				session_id, actor_id, started_at, completed_at,
				COUNT(*) OVER() as total_count,
				ROW_NUMBER() OVER(ORDER BY started_at DESC) as row_num
			FROM workflow_executions
			WHERE agent_node_id = ? AND reasoner_id = ?
		)
		SELECT execution_id, agent_node_id, reasoner_id, status, status_reason,
		       input_data, output_data, error_message, duration_ms, retry_count,
		       session_id, actor_id, started_at, completed_at, total_count
		FROM execution_data
		WHERE row_num > ? AND row_num <= ?
		ORDER BY started_at DESC`

	rows, err := tx.Query(combinedQuery, nodeID, localReasonerID, offset, offset+limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query execution history: %w", err)
	}
	defer rows.Close()

	var executions []types.ReasonerExecutionRecord
	var total int

	for rows.Next() {
		var record types.ReasonerExecutionRecord
		var inputData, outputData sql.NullString
		var errorMessage sql.NullString
		var durationMs sql.NullInt64
		var statusReason sql.NullString
		var sessionID sql.NullString
		var actorID sql.NullString
		var completedAt sql.NullTime

		err := rows.Scan(
			&record.ExecutionID, &record.AgentNodeID, &record.ReasonerID,
			&record.Status, &statusReason, &inputData, &outputData,
			&errorMessage, &durationMs, &record.RetryCount, &sessionID,
			&actorID, &record.StartedAt, &completedAt, &total,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan execution record: %w", err)
		}

		record.Timestamp = record.StartedAt
		if statusReason.Valid {
			record.StatusReason = &statusReason.String
			record.ErrorCategory = &statusReason.String
		}
		if completedAt.Valid {
			record.CompletedAt = &completedAt.Time
		}
		if sessionID.Valid {
			record.SessionID = &sessionID.String
		}
		if actorID.Valid {
			record.ActorID = &actorID.String
		}

		if inputData.Valid && inputData.String != "" {
			payload := types.DecodeStoredExecutionPayload(json.RawMessage(inputData.String))
			record.Input = payload.Input
			record.Context = payload.Context
			if record.Input == nil {
				record.Input = map[string]interface{}{"raw": inputData.String}
			}
		}

		// Parse output data
		if outputData.Valid && outputData.String != "" {
			if err := json.Unmarshal([]byte(outputData.String), &record.Output); err != nil {
				record.Output = map[string]interface{}{"raw": outputData.String}
			}
		}

		// Set error message
		if errorMessage.Valid {
			record.Error = errorMessage.String
		}

		// Set duration
		if durationMs.Valid {
			record.DurationMs = durationMs.Int64
		}

		executions = append(executions, record)
	}

	// When no executions are found, total remains 0 (correct behavior)
	// The window function COUNT(*) OVER() handles empty result sets efficiently

	hasMore := (page * limit) < total

	history := &types.ReasonerExecutionHistory{
		Executions: executions,
		Total:      total,
		Page:       page,
		Limit:      limit,
		HasMore:    hasMore,
	}

	return history, nil
}

// executeReasonerHistoryQueryDirect performs reasoner history query without transaction wrapper
// This is used when we detect we're already in a transaction context
func (ls *LocalStorage) executeReasonerHistoryQueryDirect(ctx context.Context, nodeID, localReasonerID string, page, limit, offset int) (*types.ReasonerExecutionHistory, error) {
	// Use a single optimized query with window functions to get both count and data efficiently
	// This reduces lock time and improves performance
	combinedQuery := `
		WITH execution_data AS (
			SELECT
				execution_id, agent_node_id, reasoner_id, status, status_reason,
				input_data, output_data, error_message, duration_ms, retry_count,
				session_id, actor_id, started_at, completed_at,
				COUNT(*) OVER() as total_count,
				ROW_NUMBER() OVER(ORDER BY started_at DESC) as row_num
			FROM workflow_executions
			WHERE agent_node_id = ? AND reasoner_id = ?
		)
		SELECT execution_id, agent_node_id, reasoner_id, status, status_reason,
		       input_data, output_data, error_message, duration_ms, retry_count,
		       session_id, actor_id, started_at, completed_at, total_count
		FROM execution_data
		WHERE row_num > ? AND row_num <= ?
		ORDER BY started_at DESC`

	rows, err := ls.db.QueryContext(ctx, combinedQuery, nodeID, localReasonerID, offset, offset+limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query execution history: %w", err)
	}
	defer rows.Close()

	var executions []types.ReasonerExecutionRecord
	var total int

	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during execution history iteration: %w", err)
		}

		var record types.ReasonerExecutionRecord
		var inputData, outputData sql.NullString
		var errorMessage sql.NullString
		var durationMs sql.NullInt64
		var statusReason sql.NullString
		var sessionID sql.NullString
		var actorID sql.NullString
		var completedAt sql.NullTime

		err := rows.Scan(
			&record.ExecutionID, &record.AgentNodeID, &record.ReasonerID,
			&record.Status, &statusReason, &inputData, &outputData,
			&errorMessage, &durationMs, &record.RetryCount, &sessionID,
			&actorID, &record.StartedAt, &completedAt, &total,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan execution record: %w", err)
		}

		record.Timestamp = record.StartedAt
		if statusReason.Valid {
			record.StatusReason = &statusReason.String
			record.ErrorCategory = &statusReason.String
		}
		if completedAt.Valid {
			record.CompletedAt = &completedAt.Time
		}
		if sessionID.Valid {
			record.SessionID = &sessionID.String
		}
		if actorID.Valid {
			record.ActorID = &actorID.String
		}

		if inputData.Valid && inputData.String != "" {
			payload := types.DecodeStoredExecutionPayload(json.RawMessage(inputData.String))
			record.Input = payload.Input
			record.Context = payload.Context
			if record.Input == nil {
				record.Input = map[string]interface{}{"raw": inputData.String}
			}
		}

		// Parse output data
		if outputData.Valid && outputData.String != "" {
			if err := json.Unmarshal([]byte(outputData.String), &record.Output); err != nil {
				record.Output = map[string]interface{}{"raw": outputData.String}
			}
		}

		// Set error message
		if errorMessage.Valid {
			record.Error = errorMessage.String
		}

		// Set duration
		if durationMs.Valid {
			record.DurationMs = durationMs.Int64
		}

		executions = append(executions, record)
	}

	// When no executions are found, total remains 0 (correct behavior)
	// The window function COUNT(*) OVER() handles empty result sets efficiently

	hasMore := (page * limit) < total

	history := &types.ReasonerExecutionHistory{
		Executions: executions,
		Total:      total,
		Page:       page,
		Limit:      limit,
		HasMore:    hasMore,
	}

	return history, nil
}

// GetExecutionEventBus returns the execution event bus for real-time updates
func (ls *LocalStorage) GetExecutionEventBus() *events.ExecutionEventBus {
	return ls.eventBus
}

// GetWorkflowExecutionEventBus returns the bus for workflow execution events.
func (ls *LocalStorage) GetWorkflowExecutionEventBus() *events.EventBus[*types.WorkflowExecutionEvent] {
	return ls.workflowExecutionEventBus
}

// GetExecutionLogEventBus returns the bus for structured execution logs.
func (ls *LocalStorage) GetExecutionLogEventBus() *events.EventBus[*types.ExecutionLogEntry] {
	return ls.executionLogEventBus
}

// AgentField Server DID operations
func (ls *LocalStorage) StoreAgentFieldServerDID(ctx context.Context, agentfieldServerID, rootDID string, masterSeed []byte, createdAt, lastKeyRotation time.Time) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store af server DID: %w", err)
	}

	// Validate input parameters
	if agentfieldServerID == "" {
		return &ValidationError{
			Field:   "agentfield_server_id",
			Value:   agentfieldServerID,
			Reason:  "af server ID cannot be empty",
			Context: "StoreAgentFieldServerDID",
		}
	}
	if rootDID == "" {
		return &ValidationError{
			Field:   "root_did",
			Value:   rootDID,
			Reason:  "root DID cannot be empty",
			Context: "StoreAgentFieldServerDID",
		}
	}
	if len(masterSeed) == 0 {
		return &ValidationError{
			Field:   "master_seed",
			Value:   "<encrypted>",
			Reason:  "master seed cannot be empty",
			Context: "StoreAgentFieldServerDID",
		}
	}

	// Use transaction for data consistency
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() {
		if err != nil {
			rollbackTx(tx, "StoreAgentFieldServerDID")
		}
	}()

	// Execute with retry logic
	err = ls.retryOnConstraintFailure(ctx, func() error {
		query := `
                        INSERT OR REPLACE INTO did_registry (agentfield_server_id, root_did, master_seed_encrypted, created_at, last_key_rotation, total_dids)
                        VALUES (?, ?, ?, ?, ?, 0)
                `
		if ls.mode == "postgres" {
			query = `
                                INSERT INTO did_registry (agentfield_server_id, root_did, master_seed_encrypted, created_at, last_key_rotation, total_dids)
                                VALUES (?, ?, ?, ?, ?, 0)
                                ON CONFLICT (agentfield_server_id) DO UPDATE SET
                                        root_did = EXCLUDED.root_did,
                                        master_seed_encrypted = EXCLUDED.master_seed_encrypted,
                                        created_at = EXCLUDED.created_at,
                                        last_key_rotation = EXCLUDED.last_key_rotation,
                                        total_dids = did_registry.total_dids
                        `
		}
		_, execErr := tx.ExecContext(ctx, query, agentfieldServerID, rootDID, masterSeed, createdAt, lastKeyRotation)
		if execErr != nil {
			return fmt.Errorf("failed to store af server DID: %w", execErr)
		}
		return nil
	}, 3) // Retry up to 3 times for transient errors

	if err != nil {
		return err
	}

	// Commit transaction
	if err = tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	log.Printf("Successfully stored af server DID: agentfield_server_id=%s, root_did=%s", agentfieldServerID, rootDID)
	return nil
}

// StoreAgentDIDWithComponents stores an agent DID along with its component DIDs in a single transaction
func (ls *LocalStorage) StoreAgentDIDWithComponents(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int, components []ComponentDIDRequest) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store agent DID with components: %w", err)
	}

	// Pre-storage validation
	if err := ls.validateAgentFieldServerExists(ctx, agentfieldServerDID); err != nil {
		return fmt.Errorf("pre-storage validation failed: %w", err)
	}

	// Use transaction for data consistency across all operations
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() {
		if err != nil {
			rollbackTx(tx, "StoreAgentDIDWithComponents")
		}
	}()

	// Store agent DID first
	err = ls.retryOnConstraintFailure(ctx, func() error {
		query := `
			INSERT INTO agent_dids (
				agent_node_id, did, agentfield_server_id, public_key_jwk, derivation_path, registered_at, status
			) VALUES (?, ?, ?, ?, ?, ?, ?)`

		derivationPath := fmt.Sprintf("m/44'/0'/0'/%d", derivationIndex)
		_, execErr := tx.ExecContext(ctx, query, agentID, agentDID, agentfieldServerDID, publicKeyJWK, derivationPath, time.Now(), "active")
		if execErr != nil {
			if strings.Contains(execErr.Error(), "UNIQUE constraint failed") || strings.Contains(execErr.Error(), "agent_dids") {
				return &DuplicateDIDError{
					DID:  fmt.Sprintf("agent:%s@%s", agentID, agentfieldServerDID),
					Type: "agent",
				}
			}
			if strings.Contains(execErr.Error(), "FOREIGN KEY constraint failed") {
				return &ForeignKeyConstraintError{
					Table:           "agent_dids",
					Column:          "agentfield_server_id",
					ReferencedTable: "did_registry",
					ReferencedValue: agentfieldServerDID,
					Operation:       "INSERT",
				}
			}
			return fmt.Errorf("failed to store agent DID: %w", execErr)
		}
		return nil
	}, 3)

	if err != nil {
		var dupErr *DuplicateDIDError
		if errors.As(err, &dupErr) {
			return dupErr
		}
		return fmt.Errorf("failed to store agent DID: %w", err)
	}

	// Store component DIDs
	for i, component := range components {
		err = ls.retryOnConstraintFailure(ctx, func() error {
			query := `
				INSERT INTO component_dids (
					did, agent_did, component_type, function_name, public_key_jwk, derivation_path
				) VALUES (?, ?, ?, ?, ?, ?)`

			derivationPath := fmt.Sprintf("m/44'/0'/0'/%d", component.DerivationIndex)
			_, execErr := tx.ExecContext(ctx, query, component.ComponentDID, agentDID, component.ComponentType, component.ComponentName, component.PublicKeyJWK, derivationPath)
			if execErr != nil {
				if strings.Contains(execErr.Error(), "UNIQUE constraint failed") || strings.Contains(execErr.Error(), "component_dids") {
					return &DuplicateDIDError{
						DID:  fmt.Sprintf("component:%s/%s@%s", component.ComponentType, component.ComponentName, agentDID),
						Type: "component",
					}
				}
				if strings.Contains(execErr.Error(), "FOREIGN KEY constraint failed") {
					return &ForeignKeyConstraintError{
						Table:           "component_dids",
						Column:          "agent_did",
						ReferencedTable: "agent_dids",
						ReferencedValue: agentDID,
						Operation:       "INSERT",
					}
				}
				return fmt.Errorf("failed to store component DID %d: %w", i, execErr)
			}
			return nil
		}, 3)

		if err != nil {
			var dupErr *DuplicateDIDError
			if errors.As(err, &dupErr) {
				return dupErr
			}
			return fmt.Errorf("failed to store component DID %d (%s): %w", i, component.ComponentName, err)
		}
	}

	// Commit transaction
	if err = tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	log.Printf("Successfully stored agent DID with %d components: agent_id=%s, did=%s", len(components), agentID, agentDID)
	return nil
}

func (ls *LocalStorage) GetAgentFieldServerDID(ctx context.Context, agentfieldServerID string) (*types.AgentFieldServerDIDInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get af server DID: %w", err)
	}

	query := `
		SELECT agentfield_server_id, root_did, master_seed_encrypted, created_at, last_key_rotation
		FROM did_registry WHERE agentfield_server_id = ?
	`
	row := ls.db.QueryRowContext(ctx, query, agentfieldServerID)
	info := &types.AgentFieldServerDIDInfo{}

	err := row.Scan(&info.AgentFieldServerID, &info.RootDID, &info.MasterSeed, &info.CreatedAt, &info.LastKeyRotation)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil // Return nil, nil for "not found"
		}
		return nil, fmt.Errorf("failed to get af server DID: %w", err)
	}
	return info, nil
}

func (ls *LocalStorage) ListAgentFieldServerDIDs(ctx context.Context) ([]*types.AgentFieldServerDIDInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list af server DIDs: %w", err)
	}

	query := `
		SELECT agentfield_server_id, root_did, master_seed_encrypted, created_at, last_key_rotation
		FROM did_registry ORDER BY created_at DESC
	`
	rows, err := ls.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list af server DIDs: %w", err)
	}
	defer rows.Close()

	var infos []*types.AgentFieldServerDIDInfo
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during af server DID list iteration: %w", err)
		}

		info := &types.AgentFieldServerDIDInfo{}
		err := rows.Scan(&info.AgentFieldServerID, &info.RootDID, &info.MasterSeed, &info.CreatedAt, &info.LastKeyRotation)
		if err != nil {
			return nil, fmt.Errorf("failed to scan af server DID: %w", err)
		}
		infos = append(infos, info)
	}
	return infos, nil
}

// DID Registry operations
func (ls *LocalStorage) StoreDID(ctx context.Context, did string, didDocument, publicKey, privateKeyRef, derivationPath string) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store DID: %w", err)
	}

	// INSERT-only query - no ON CONFLICT clause for security
	query := `
		INSERT INTO did_registry (
			did, did_document, public_key, private_key_ref, derivation_path,
			created_at, updated_at, status
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`

	now := time.Now()
	_, err := ls.db.ExecContext(ctx, query, did, didDocument, publicKey, privateKeyRef, derivationPath, now, now, "active")
	if err != nil {
		// Check if this is a unique constraint violation (duplicate DID)
		if strings.Contains(err.Error(), "UNIQUE constraint failed") || strings.Contains(err.Error(), "did_registry.did") {
			log.Printf("Duplicate DID registry entry detected: %s", did)
			return &DuplicateDIDError{
				DID:  did,
				Type: "registry",
			}
		}
		return fmt.Errorf("failed to store DID: %w", err)
	}

	log.Printf("Successfully stored DID registry entry: %s", did)
	return nil
}

func (ls *LocalStorage) GetDID(ctx context.Context, did string) (*types.DIDRegistryEntry, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get DID: %w", err)
	}

	query := `
		SELECT did, did_document, public_key, private_key_ref, derivation_path,
			   created_at, updated_at, status
		FROM did_registry WHERE did = ?`

	row := ls.db.QueryRowContext(ctx, query, did)
	entry := &types.DIDRegistryEntry{}

	err := row.Scan(&entry.DID, &entry.DIDDocument, &entry.PublicKey, &entry.PrivateKeyRef,
		&entry.DerivationPath, &entry.CreatedAt, &entry.UpdatedAt, &entry.Status)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("DID %s not found", did)
		}
		return nil, fmt.Errorf("failed to get DID: %w", err)
	}
	return entry, nil
}

func (ls *LocalStorage) ListDIDs(ctx context.Context) ([]*types.DIDRegistryEntry, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list DIDs: %w", err)
	}

	query := `
		SELECT did, did_document, public_key, private_key_ref, derivation_path,
			   created_at, updated_at, status
		FROM did_registry ORDER BY created_at DESC`

	rows, err := ls.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list DIDs: %w", err)
	}
	defer rows.Close()

	var entries []*types.DIDRegistryEntry
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during DID list iteration: %w", err)
		}

		entry := &types.DIDRegistryEntry{}
		err := rows.Scan(&entry.DID, &entry.DIDDocument, &entry.PublicKey, &entry.PrivateKeyRef,
			&entry.DerivationPath, &entry.CreatedAt, &entry.UpdatedAt, &entry.Status)
		if err != nil {
			return nil, fmt.Errorf("failed to scan DID entry: %w", err)
		}
		entries = append(entries, entry)
	}
	return entries, nil
}

// validateAgentFieldServerExists checks if a af server registry exists
func (ls *LocalStorage) validateAgentFieldServerExists(ctx context.Context, agentfieldServerID string) error {
	if agentfieldServerID == "" {
		return &ValidationError{
			Field:   "agentfield_server_id",
			Value:   agentfieldServerID,
			Reason:  "af server ID cannot be empty",
			Context: "pre-storage validation",
		}
	}

	query := `SELECT 1 FROM did_registry WHERE agentfield_server_id = ? LIMIT 1`
	var exists int
	err := ls.db.QueryRowContext(ctx, query, agentfieldServerID).Scan(&exists)
	if err != nil {
		if err == sql.ErrNoRows {
			return &ForeignKeyConstraintError{
				Table:           "agent_dids",
				Column:          "agentfield_server_id",
				ReferencedTable: "did_registry",
				ReferencedValue: agentfieldServerID,
				Operation:       "INSERT",
			}
		}
		return fmt.Errorf("failed to validate af server existence: %w", err)
	}
	return nil
}

// validateAgentDIDExists checks if an agent DID exists
func (ls *LocalStorage) validateAgentDIDExists(ctx context.Context, agentDID string) error {
	if agentDID == "" {
		return &ValidationError{
			Field:   "agent_did",
			Value:   agentDID,
			Reason:  "agent DID cannot be empty",
			Context: "pre-storage validation",
		}
	}

	query := `SELECT 1 FROM agent_dids WHERE did = ? LIMIT 1`
	var exists int
	err := ls.db.QueryRowContext(ctx, query, agentDID).Scan(&exists)
	if err != nil {
		if err == sql.ErrNoRows {
			return &ForeignKeyConstraintError{
				Table:           "component_dids",
				Column:          "agent_did",
				ReferencedTable: "agent_dids",
				ReferencedValue: agentDID,
				Operation:       "INSERT",
			}
		}
		return fmt.Errorf("failed to validate agent DID existence: %w", err)
	}
	return nil
}

// retryOnConstraintFailure executes a function with retry logic for transient constraint issues
func (ls *LocalStorage) retryOnConstraintFailure(ctx context.Context, operation func() error, maxRetries int) error {
	var lastErr error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		if err := ctx.Err(); err != nil {
			return fmt.Errorf("context cancelled during retry attempt %d: %w", attempt, err)
		}

		lastErr = operation()
		if lastErr == nil {
			return nil
		}

		// Don't retry on validation errors or permanent constraint violations
		if _, isValidationErr := lastErr.(*ValidationError); isValidationErr {
			return lastErr
		}
		if _, isFKErr := lastErr.(*ForeignKeyConstraintError); isFKErr {
			return lastErr
		}
		if _, isDuplicateErr := lastErr.(*DuplicateDIDError); isDuplicateErr {
			return lastErr
		}

		// Only retry on database-level transient errors
		if strings.Contains(lastErr.Error(), "database is locked") ||
			strings.Contains(lastErr.Error(), "SQLITE_BUSY") ||
			strings.Contains(lastErr.Error(), "database is temporarily unavailable") {
			if attempt < maxRetries {
				// Exponential backoff: 10ms, 20ms, 40ms
				backoff := time.Duration(10*(1<<attempt)) * time.Millisecond
				time.Sleep(backoff)
				continue
			}
		}

		// For other errors, don't retry
		return lastErr
	}
	return lastErr
}

// Agent DID operations
func (ls *LocalStorage) StoreAgentDID(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store agent DID: %w", err)
	}

	// Pre-storage validation
	if err := ls.validateAgentFieldServerExists(ctx, agentfieldServerDID); err != nil {
		return fmt.Errorf("pre-storage validation failed: %w", err)
	}

	// Validate input parameters
	if agentID == "" {
		return &ValidationError{
			Field:   "agent_node_id",
			Value:   agentID,
			Reason:  "agent ID cannot be empty",
			Context: "StoreAgentDID",
		}
	}
	if agentDID == "" {
		return &ValidationError{
			Field:   "did",
			Value:   agentDID,
			Reason:  "agent DID cannot be empty",
			Context: "StoreAgentDID",
		}
	}
	if publicKeyJWK == "" {
		return &ValidationError{
			Field:   "public_key_jwk",
			Value:   publicKeyJWK,
			Reason:  "public key JWK cannot be empty",
			Context: "StoreAgentDID",
		}
	}

	// Use transaction for data consistency
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() {
		if err != nil {
			rollbackTx(tx, "StoreAgentDID")
		}
	}()

	// Execute with retry logic
	err = ls.retryOnConstraintFailure(ctx, func() error {
		// INSERT-only query - no ON CONFLICT clause for security
		query := `
			INSERT INTO agent_dids (
				agent_node_id, did, agentfield_server_id, public_key_jwk, derivation_path, registered_at, status
			) VALUES (?, ?, ?, ?, ?, ?, ?)`

		derivationPath := fmt.Sprintf("m/44'/0'/0'/%d", derivationIndex)
		_, execErr := tx.ExecContext(ctx, query, agentID, agentDID, agentfieldServerDID, publicKeyJWK, derivationPath, time.Now(), "active")
		if execErr != nil {
			// Check if this is a unique constraint violation (duplicate agent DID)
			if strings.Contains(execErr.Error(), "UNIQUE constraint failed") || strings.Contains(execErr.Error(), "agent_dids") {
				log.Printf("Duplicate agent DID entry detected: agent_id=%s, agentfield_server_id=%s", agentID, agentfieldServerDID)
				return &DuplicateDIDError{
					DID:  fmt.Sprintf("agent:%s@%s", agentID, agentfieldServerDID),
					Type: "agent",
				}
			}
			// Check for foreign key constraint violations
			if strings.Contains(execErr.Error(), "FOREIGN KEY constraint failed") {
				return &ForeignKeyConstraintError{
					Table:           "agent_dids",
					Column:          "agentfield_server_id",
					ReferencedTable: "did_registry",
					ReferencedValue: agentfieldServerDID,
					Operation:       "INSERT",
				}
			}
			return fmt.Errorf("failed to store agent DID: %w", execErr)
		}
		return nil
	}, 3) // Retry up to 3 times for transient errors

	if err != nil {
		return err
	}

	// Commit transaction
	if err = tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	log.Printf("Successfully stored agent DID entry: agent_id=%s, did=%s", agentID, agentDID)
	return nil
}

func (ls *LocalStorage) GetAgentDID(ctx context.Context, agentID string) (*types.AgentDIDInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get agent DID: %w", err)
	}

	query := `
		SELECT agent_node_id, did, agentfield_server_id, public_key_jwk, derivation_path,
		       reasoners, skills, status, registered_at
		FROM agent_dids WHERE agent_node_id = ?`

	row := ls.db.QueryRowContext(ctx, query, agentID)
	info := &types.AgentDIDInfo{}

	var reasonersJSON, skillsJSON, publicKeyJWK string
	err := row.Scan(&info.AgentNodeID, &info.DID, &info.AgentFieldServerID, &publicKeyJWK,
		&info.DerivationPath, &reasonersJSON, &skillsJSON, &info.Status, &info.RegisteredAt)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("agent DID for %s not found", agentID)
		}
		return nil, fmt.Errorf("failed to get agent DID: %w", err)
	}
	info.PublicKeyJWK = json.RawMessage(publicKeyJWK)

	// Parse JSON fields
	if reasonersJSON != "" {
		if err := json.Unmarshal([]byte(reasonersJSON), &info.Reasoners); err != nil {
			return nil, fmt.Errorf("failed to parse reasoners JSON: %w", err)
		}
	} else {
		info.Reasoners = make(map[string]types.ReasonerDIDInfo)
	}

	if skillsJSON != "" {
		if err := json.Unmarshal([]byte(skillsJSON), &info.Skills); err != nil {
			return nil, fmt.Errorf("failed to parse skills JSON: %w", err)
		}
	} else {
		info.Skills = make(map[string]types.SkillDIDInfo)
	}

	return info, nil
}

func (ls *LocalStorage) ListAgentDIDs(ctx context.Context) ([]*types.AgentDIDInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list agent DIDs: %w", err)
	}

	query := `
		SELECT agent_node_id, did, agentfield_server_id, public_key_jwk, derivation_path,
		       reasoners, skills, status, registered_at
		FROM agent_dids ORDER BY registered_at DESC`

	rows, err := ls.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list agent DIDs: %w", err)
	}
	defer rows.Close()

	var infos []*types.AgentDIDInfo
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during agent DID list iteration: %w", err)
		}

		info := &types.AgentDIDInfo{}
		var reasonersJSON, skillsJSON, publicKeyJWK string
		err := rows.Scan(&info.AgentNodeID, &info.DID, &info.AgentFieldServerID, &publicKeyJWK,
			&info.DerivationPath, &reasonersJSON, &skillsJSON, &info.Status, &info.RegisteredAt)
		if err != nil {
			return nil, fmt.Errorf("failed to scan agent DID: %w", err)
		}
		info.PublicKeyJWK = json.RawMessage(publicKeyJWK)

		// Parse JSON fields
		if reasonersJSON != "" {
			if err := json.Unmarshal([]byte(reasonersJSON), &info.Reasoners); err != nil {
				return nil, fmt.Errorf("failed to parse reasoners JSON: %w", err)
			}
		} else {
			info.Reasoners = make(map[string]types.ReasonerDIDInfo)
		}

		if skillsJSON != "" {
			if err := json.Unmarshal([]byte(skillsJSON), &info.Skills); err != nil {
				return nil, fmt.Errorf("failed to parse skills JSON: %w", err)
			}
		} else {
			info.Skills = make(map[string]types.SkillDIDInfo)
		}

		infos = append(infos, info)
	}
	return infos, nil
}

// Component DID operations
func (ls *LocalStorage) StoreComponentDID(ctx context.Context, componentID, componentDID, agentDID, componentType, componentName string, derivationIndex int) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store component DID: %w", err)
	}

	// Pre-storage validation
	if err := ls.validateAgentDIDExists(ctx, agentDID); err != nil {
		return fmt.Errorf("pre-storage validation failed: %w", err)
	}

	// Validate input parameters
	if componentDID == "" {
		return &ValidationError{
			Field:   "component_did",
			Value:   componentDID,
			Reason:  "component DID cannot be empty",
			Context: "StoreComponentDID",
		}
	}
	if componentType == "" {
		return &ValidationError{
			Field:   "component_type",
			Value:   componentType,
			Reason:  "component type cannot be empty",
			Context: "StoreComponentDID",
		}
	}
	if componentName == "" {
		return &ValidationError{
			Field:   "component_name",
			Value:   componentName,
			Reason:  "component name cannot be empty",
			Context: "StoreComponentDID",
		}
	}
	// Validate component type
	validTypes := map[string]bool{"reasoner": true, "skill": true}
	if !validTypes[componentType] {
		return &ValidationError{
			Field:   "component_type",
			Value:   componentType,
			Reason:  "component type must be 'reasoner' or 'skill'",
			Context: "StoreComponentDID",
		}
	}

	// Use transaction for data consistency
	tx, err := ls.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer func() {
		if err != nil {
			rollbackTx(tx, "StoreComponentDID")
		}
	}()

	// Execute with retry logic
	err = ls.retryOnConstraintFailure(ctx, func() error {
		// INSERT-only query - no ON CONFLICT clause for security
		query := `
			INSERT INTO component_dids (
				did, agent_did, component_type, function_name, public_key_jwk, derivation_path
			) VALUES (?, ?, ?, ?, ?, ?)`

		derivationPath := fmt.Sprintf("m/44'/0'/0'/%d", derivationIndex)
		// For now, use empty public key - this should be passed as a parameter in the future
		publicKeyJWK := ""
		_, execErr := tx.ExecContext(ctx, query, componentDID, agentDID, componentType, componentName, publicKeyJWK, derivationPath)
		if execErr != nil {
			// Check if this is a unique constraint violation (duplicate component DID)
			if strings.Contains(execErr.Error(), "UNIQUE constraint failed") || strings.Contains(execErr.Error(), "component_dids") {
				log.Printf("Duplicate component DID entry detected: agent_did=%s, function_name=%s, component_type=%s", agentDID, componentName, componentType)
				return &DuplicateDIDError{
					DID:  fmt.Sprintf("component:%s/%s@%s", componentType, componentName, agentDID),
					Type: "component",
				}
			}
			// Check for foreign key constraint violations
			if strings.Contains(execErr.Error(), "FOREIGN KEY constraint failed") {
				return &ForeignKeyConstraintError{
					Table:           "component_dids",
					Column:          "agent_did",
					ReferencedTable: "agent_dids",
					ReferencedValue: agentDID,
					Operation:       "INSERT",
				}
			}
			return fmt.Errorf("failed to store component DID: %w", execErr)
		}
		return nil
	}, 3) // Retry up to 3 times for transient errors

	if err != nil {
		return err
	}

	// Commit transaction
	if err = tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	log.Printf("Successfully stored component DID entry: component_did=%s, agent_did=%s, type=%s", componentDID, agentDID, componentType)
	return nil
}

func (ls *LocalStorage) GetComponentDID(ctx context.Context, componentID string) (*types.ComponentDIDInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get component DID: %w", err)
	}

	// Use function_name as the componentID since there's no separate component_id column
	query := `
		SELECT function_name, did, agent_did, component_type, function_name,
			   derivation_path, created_at
		FROM component_dids WHERE function_name = ?`

	row := ls.db.QueryRowContext(ctx, query, componentID)
	info := &types.ComponentDIDInfo{}

	var derivationPath string
	var createdAt sql.NullTime

	err := row.Scan(&info.ComponentID, &info.ComponentDID, &info.AgentDID,
		&info.ComponentType, &info.ComponentName, &derivationPath, &createdAt)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("component DID for %s not found", componentID)
		}
		return nil, fmt.Errorf("failed to get component DID: %w", err)
	}

	if createdAt.Valid {
		info.CreatedAt = createdAt.Time
	}

	// Parse derivation index from derivation path (e.g., "m/44'/0'/0'/123" -> 123)
	if derivationPath != "" {
		parts := strings.Split(derivationPath, "/")
		if len(parts) > 0 {
			lastPart := parts[len(parts)-1]
			if derivationIndex, parseErr := strconv.Atoi(strings.Trim(lastPart, "'")); parseErr == nil {
				info.DerivationIndex = derivationIndex
			}
		}
	}

	return info, nil
}

func (ls *LocalStorage) ListComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list component DIDs: %w", err)
	}

	var query string
	var rows *sql.Rows
	var err error

	if agentDID == "" {
		// Get all components when agentDID is empty
		query = `
			SELECT function_name, did, agent_did, component_type, function_name,
				   derivation_path, created_at
			FROM component_dids ORDER BY created_at DESC`
		rows, err = ls.db.QueryContext(ctx, query)
	} else {
		// Get components for specific agent
		query = `
			SELECT function_name, did, agent_did, component_type, function_name,
				   derivation_path, created_at
			FROM component_dids WHERE agent_did = ? ORDER BY created_at DESC`
		rows, err = ls.db.QueryContext(ctx, query, agentDID)
	}
	if err != nil {
		return nil, fmt.Errorf("failed to list component DIDs: %w", err)
	}
	defer rows.Close()

	var infos []*types.ComponentDIDInfo
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during component DID list iteration: %w", err)
		}

		info := &types.ComponentDIDInfo{}
		var derivationPath string
		var createdAt sql.NullTime

		err := rows.Scan(&info.ComponentID, &info.ComponentDID, &info.AgentDID,
			&info.ComponentType, &info.ComponentName, &derivationPath, &createdAt)
		if err != nil {
			return nil, fmt.Errorf("failed to scan component DID: %w", err)
		}

		if createdAt.Valid {
			info.CreatedAt = createdAt.Time
		}

		// Parse derivation index from derivation path
		if derivationPath != "" {
			parts := strings.Split(derivationPath, "/")
			if len(parts) > 0 {
				lastPart := parts[len(parts)-1]
				if derivationIndex, parseErr := strconv.Atoi(strings.Trim(lastPart, "'")); parseErr == nil {
					info.DerivationIndex = derivationIndex
				}
			}
		}

		infos = append(infos, info)
	}
	return infos, nil
}

// Execution VC operations
func (ls *LocalStorage) StoreExecutionVC(ctx context.Context, vcID, executionID, workflowID, sessionID, issuerDID, targetDID, callerDID, inputHash, outputHash, status string, vcDocument []byte, signature string, storageURI string, documentSizeBytes int64) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store execution VC: %w", err)
	}

	query := `
		INSERT INTO execution_vcs (
			vc_id, execution_id, workflow_id, session_id, issuer_did, target_did,
			caller_did, vc_document, signature, storage_uri, document_size_bytes,
			input_hash, output_hash, status, created_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(vc_id) DO UPDATE SET
			status = excluded.status,
			vc_document = excluded.vc_document,
			signature = excluded.signature,
			storage_uri = excluded.storage_uri,
			document_size_bytes = excluded.document_size_bytes;`

	_, err := ls.db.ExecContext(ctx, query, vcID, executionID, workflowID, sessionID, issuerDID, targetDID,
		callerDID, vcDocument, signature, storageURI, documentSizeBytes, inputHash, outputHash, status, time.Now())
	if err != nil {
		return fmt.Errorf("failed to store execution VC: %w", err)
	}
	return nil
}

func (ls *LocalStorage) GetExecutionVC(ctx context.Context, vcID string) (*types.ExecutionVCInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get execution VC: %w", err)
	}

	query := `
		SELECT vc_id, execution_id, workflow_id, session_id, issuer_did, target_did,
			   caller_did, input_hash, output_hash, status, created_at, storage_uri, document_size_bytes
		FROM execution_vcs WHERE vc_id = ?`

	row := ls.db.QueryRowContext(ctx, query, vcID)
	info := &types.ExecutionVCInfo{}

	err := row.Scan(&info.VCID, &info.ExecutionID, &info.WorkflowID, &info.SessionID,
		&info.IssuerDID, &info.TargetDID, &info.CallerDID, &info.InputHash,
		&info.OutputHash, &info.Status, &info.CreatedAt, &info.StorageURI, &info.DocumentSize)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("execution VC %s not found", vcID)
		}
		return nil, fmt.Errorf("failed to get execution VC: %w", err)
	}
	return info, nil
}

func buildExecutionVCFilterClauses(filters types.VCFilters) (string, []interface{}) {
	var (
		conditions []string
		args       []interface{}
	)

	if filters.ExecutionID != nil {
		conditions = append(conditions, "evc.execution_id = ?")
		args = append(args, *filters.ExecutionID)
	}
	if filters.WorkflowID != nil {
		conditions = append(conditions, "evc.workflow_id = ?")
		args = append(args, *filters.WorkflowID)
	}
	if filters.SessionID != nil {
		conditions = append(conditions, "evc.session_id = ?")
		args = append(args, *filters.SessionID)
	}
	if filters.IssuerDID != nil {
		conditions = append(conditions, "evc.issuer_did = ?")
		args = append(args, *filters.IssuerDID)
	}
	if filters.TargetDID != nil {
		conditions = append(conditions, "evc.target_did = ?")
		args = append(args, *filters.TargetDID)
	}
	if filters.CallerDID != nil {
		conditions = append(conditions, "evc.caller_did = ?")
		args = append(args, *filters.CallerDID)
	}
	if filters.AgentNodeID != nil {
		conditions = append(conditions, "COALESCE(we.agent_node_id, '') = ?")
		args = append(args, *filters.AgentNodeID)
	}
	if filters.Status != nil {
		conditions = append(conditions, "evc.status = ?")
		args = append(args, *filters.Status)
	}
	if filters.CreatedAfter != nil {
		conditions = append(conditions, "evc.created_at >= ?")
		args = append(args, filters.CreatedAfter.UTC())
	}
	if filters.CreatedBefore != nil {
		conditions = append(conditions, "evc.created_at <= ?")
		args = append(args, filters.CreatedBefore.UTC())
	}

	if filters.Search != nil {
		if trimmed := strings.TrimSpace(*filters.Search); trimmed != "" {
			search := "%" + strings.ToLower(trimmed) + "%"
			conditions = append(conditions, "("+
				"LOWER(evc.execution_id) LIKE ? OR "+
				"LOWER(evc.workflow_id) LIKE ? OR "+
				"LOWER(evc.issuer_did) LIKE ? OR "+
				"LOWER(evc.target_did) LIKE ? OR "+
				"LOWER(evc.caller_did) LIKE ? OR "+
				"LOWER(evc.session_id) LIKE ? OR "+
				"LOWER(COALESCE(we.agent_node_id, '')) LIKE ? OR "+
				"LOWER(COALESCE(we.workflow_name, '')) LIKE ?"+
				")")
			for i := 0; i < 8; i++ {
				args = append(args, search)
			}
		}
	}

	return strings.Join(conditions, " AND "), args
}

func (ls *LocalStorage) ListExecutionVCs(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list execution VCs: %w", err)
	}

	query := `
		SELECT evc.vc_id, evc.execution_id, evc.workflow_id, evc.session_id,
		       evc.issuer_did, evc.target_did, evc.caller_did, evc.input_hash,
		       evc.output_hash, evc.status, evc.created_at, evc.storage_uri,
		       evc.document_size_bytes, we.agent_node_id, we.workflow_name
		FROM execution_vcs evc
		LEFT JOIN workflow_executions we ON we.execution_id = evc.execution_id`

	whereClause, args := buildExecutionVCFilterClauses(filters)
	if whereClause != "" {
		query += " WHERE " + whereClause
	}

	query += " ORDER BY evc.created_at DESC"

	if filters.Limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", filters.Limit)
	}

	if filters.Offset > 0 {
		query += fmt.Sprintf(" OFFSET %d", filters.Offset)
	}

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list execution VCs: %w", err)
	}
	defer rows.Close()

	var infos []*types.ExecutionVCInfo
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during execution VC list iteration: %w", err)
		}

		info := &types.ExecutionVCInfo{}
		err := rows.Scan(&info.VCID, &info.ExecutionID, &info.WorkflowID, &info.SessionID,
			&info.IssuerDID, &info.TargetDID, &info.CallerDID, &info.InputHash,
			&info.OutputHash, &info.Status, &info.CreatedAt, &info.StorageURI,
			&info.DocumentSize, &info.AgentNodeID, &info.WorkflowName)
		if err != nil {
			return nil, fmt.Errorf("failed to scan execution VC: %w", err)
		}
		infos = append(infos, info)
	}
	return infos, nil
}

func (ls *LocalStorage) ListWorkflowVCStatusSummaries(ctx context.Context, workflowIDs []string) ([]*types.WorkflowVCStatusAggregation, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during workflow VC status summary query: %w", err)
	}

	if len(workflowIDs) == 0 {
		return []*types.WorkflowVCStatusAggregation{}, nil
	}

	placeholders := make([]string, len(workflowIDs))
	for i := range workflowIDs {
		placeholders[i] = "?"
	}

	query := fmt.Sprintf(`
		SELECT workflow_id,
		       COUNT(*) AS vc_count,
		       SUM(CASE WHEN status = ? THEN 1 ELSE 0 END) AS verified_count,
		       SUM(CASE WHEN status = ? OR status = ? THEN 1 ELSE 0 END) AS failed_count,
		       MAX(created_at) AS last_created_at
		FROM execution_vcs
		WHERE workflow_id IN (%s)
		GROUP BY workflow_id
	`, strings.Join(placeholders, ","))

	args := []interface{}{
		string(types.ExecutionStatusSucceeded),
		string(types.ExecutionStatusFailed),
		string(types.ExecutionStatusTimeout),
	}
	for _, id := range workflowIDs {
		args = append(args, id)
	}

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query workflow VC status summaries: %w", err)
	}
	defer rows.Close()

	var summaries []*types.WorkflowVCStatusAggregation
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during workflow VC status iteration: %w", err)
		}

		var lastCreated sql.NullTime
		summary := &types.WorkflowVCStatusAggregation{}
		if err := rows.Scan(
			&summary.WorkflowID,
			&summary.VCCount,
			&summary.VerifiedCount,
			&summary.FailedCount,
			&lastCreated,
		); err != nil {
			return nil, fmt.Errorf("failed to scan workflow VC status summary: %w", err)
		}

		if lastCreated.Valid {
			summary.LastCreatedAt = &lastCreated.Time
		}

		summaries = append(summaries, summary)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("workflow VC status summary rows error: %w", err)
	}

	return summaries, nil
}

func (ls *LocalStorage) CountExecutionVCs(ctx context.Context, filters types.VCFilters) (int, error) {
	if err := ctx.Err(); err != nil {
		return 0, fmt.Errorf("context cancelled during count execution VCs: %w", err)
	}

	query := `
		SELECT COUNT(*)
		FROM execution_vcs evc
		LEFT JOIN workflow_executions we ON we.execution_id = evc.execution_id`

	whereClause, args := buildExecutionVCFilterClauses(filters)
	if whereClause != "" {
		query += " WHERE " + whereClause
	}

	var total int
	if err := ls.db.QueryRowContext(ctx, query, args...).Scan(&total); err != nil {
		return 0, fmt.Errorf("failed to count execution VCs: %w", err)
	}
	return total, nil
}

// Workflow VC operations
func (ls *LocalStorage) StoreWorkflowVC(ctx context.Context, workflowVCID, workflowID, sessionID string, componentVCIDs []string, status string, startTime, endTime *time.Time, totalSteps, completedSteps int, storageURI string, documentSizeBytes int64) error {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store workflow VC: %w", err)
	}

	componentVCIDsJSON, err := json.Marshal(componentVCIDs)
	if err != nil {
		return fmt.Errorf("failed to marshal component VC IDs: %w", err)
	}

	query := `
		INSERT INTO workflow_vcs (
			workflow_vc_id, workflow_id, session_id, component_vc_ids, status,
			start_time, end_time, total_steps, completed_steps, storage_uri, document_size_bytes
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(workflow_vc_id) DO UPDATE SET
			component_vc_ids = excluded.component_vc_ids,
			status = excluded.status,
			end_time = excluded.end_time,
			completed_steps = excluded.completed_steps,
			storage_uri = excluded.storage_uri,
			document_size_bytes = excluded.document_size_bytes;`

	_, err = ls.db.ExecContext(ctx, query, workflowVCID, workflowID, sessionID, componentVCIDsJSON, status,
		startTime, endTime, totalSteps, completedSteps, storageURI, documentSizeBytes)
	if err != nil {
		return fmt.Errorf("failed to store workflow VC: %w", err)
	}
	return nil
}

func (ls *LocalStorage) GetWorkflowVC(ctx context.Context, workflowVCID string) (*types.WorkflowVCInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get workflow VC: %w", err)
	}

	query := `
		SELECT workflow_vc_id, workflow_id, session_id, component_vc_ids, status,
			   start_time, end_time, total_steps, completed_steps, storage_uri, document_size_bytes
		FROM workflow_vcs WHERE workflow_vc_id = ?`

	row := ls.db.QueryRowContext(ctx, query, workflowVCID)
	info := &types.WorkflowVCInfo{}
	var componentVCIDsJSON []byte

	err := row.Scan(&info.WorkflowVCID, &info.WorkflowID, &info.SessionID, &componentVCIDsJSON,
		&info.Status, &info.StartTime, &info.EndTime, &info.TotalSteps, &info.CompletedSteps, &info.StorageURI, &info.DocumentSize)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("workflow VC %s not found", workflowVCID)
		}
		return nil, fmt.Errorf("failed to get workflow VC: %w", err)
	}

	if len(componentVCIDsJSON) > 0 {
		if err := json.Unmarshal(componentVCIDsJSON, &info.ComponentVCIDs); err != nil {
			return nil, fmt.Errorf("failed to unmarshal component VC IDs: %w", err)
		}
	}

	return info, nil
}

func (ls *LocalStorage) ListWorkflowVCs(ctx context.Context, workflowID string) ([]*types.WorkflowVCInfo, error) {
	// Check context cancellation early
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list workflow VCs: %w", err)
	}

	var query string
	var args []interface{}

	if workflowID == "" {
		// Get all workflow VCs
		query = `
			SELECT workflow_vc_id, workflow_id, session_id, component_vc_ids, status,
				   start_time, end_time, total_steps, completed_steps, storage_uri, document_size_bytes
			FROM workflow_vcs ORDER BY start_time DESC`
	} else {
		// Get workflow VCs for specific workflow
		query = `
			SELECT workflow_vc_id, workflow_id, session_id, component_vc_ids, status,
				   start_time, end_time, total_steps, completed_steps, storage_uri, document_size_bytes
			FROM workflow_vcs WHERE workflow_id = ? ORDER BY start_time DESC`
		args = append(args, workflowID)
	}

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to list workflow VCs: %w", err)
	}
	defer rows.Close()

	var infos []*types.WorkflowVCInfo
	for rows.Next() {
		// Check context cancellation during iteration
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during workflow VC list iteration: %w", err)
		}

		info := &types.WorkflowVCInfo{}
		var componentVCIDsJSON []byte

		err := rows.Scan(&info.WorkflowVCID, &info.WorkflowID, &info.SessionID, &componentVCIDsJSON,
			&info.Status, &info.StartTime, &info.EndTime, &info.TotalSteps, &info.CompletedSteps, &info.StorageURI, &info.DocumentSize)
		if err != nil {
			return nil, fmt.Errorf("failed to scan workflow VC: %w", err)
		}

		if len(componentVCIDsJSON) > 0 {
			if err := json.Unmarshal(componentVCIDsJSON, &info.ComponentVCIDs); err != nil {
				return nil, fmt.Errorf("failed to unmarshal component VC IDs: %w", err)
			}
		}

		infos = append(infos, info)
	}
	return infos, nil
}

// GetFullExecutionVC retrieves the full execution VC including the VC document and signature
func (ls *LocalStorage) GetFullExecutionVC(vcID string) (json.RawMessage, string, error) {
	query := `
		SELECT vc_document, signature
		FROM execution_vcs WHERE vc_id = ?`

	row := ls.db.QueryRow(query, vcID)

	var vcDocument string
	var signature string

	err := row.Scan(&vcDocument, &signature)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, "", fmt.Errorf("execution VC %s not found", vcID)
		}
		return nil, "", fmt.Errorf("failed to get full execution VC: %w", err)
	}

	return json.RawMessage(vcDocument), signature, nil
}

// TransactionalStorage methods (not fully implemented for local storage yet)
func (ls *LocalStorage) BeginTransaction() (Transaction, error) {
	return nil, fmt.Errorf("transactions not fully implemented for LocalStorage")
}

// StoreWorkflowExecutionEvent inserts an immutable execution event into SQLite.
func (ls *LocalStorage) StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error {
	if event == nil {
		return fmt.Errorf("workflow execution event is nil")
	}

	// Use retry logic for database lock errors
	return ls.retryDatabaseOperation(ctx, event.ExecutionID, func() error {
		tx, err := ls.db.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("failed to begin transaction: %w", err)
		}
		defer rollbackTx(tx, "StoreWorkflowExecutionEvent:"+event.ExecutionID)

		if err := ls.storeWorkflowExecutionEventTx(ctx, tx, event); err != nil {
			return err
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("failed to commit event transaction: %w", err)
		}

		eventCopy := *event
		ls.workflowExecutionEventBus.Publish(&eventCopy)

		return nil
	})
}

// storeWorkflowExecutionEventTx inserts an execution event within an existing transaction.
// This allows atomic operations where the event storage and execution update happen together.
func (ls *LocalStorage) storeWorkflowExecutionEventTx(ctx context.Context, tx DBTX, event *types.WorkflowExecutionEvent) error {
	payload := string(event.Payload)
	if len(event.Payload) == 0 {
		payload = "{}"
	}

	query := `
		INSERT INTO workflow_execution_events (
			execution_id, workflow_id, run_id, parent_execution_id, sequence, previous_sequence,
			event_type, status, status_reason, payload, emitted_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	result, err := tx.ExecContext(ctx, query,
		event.ExecutionID,
		event.WorkflowID,
		event.RunID,
		event.ParentExecutionID,
		event.Sequence,
		event.PreviousSequence,
		event.EventType,
		event.Status,
		event.StatusReason,
		payload,
		event.EmittedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to insert workflow execution event: %w", err)
	}

	if id, err := result.LastInsertId(); err == nil {
		event.EventID = id
	}
	if event.RecordedAt.IsZero() {
		event.RecordedAt = time.Now().UTC()
	}

	return nil
}

// ListWorkflowExecutionEvents retrieves execution events ordered by sequence.
func (ls *LocalStorage) ListWorkflowExecutionEvents(ctx context.Context, executionID string, afterSeq *int64, limit int) ([]*types.WorkflowExecutionEvent, error) {
	query := `
		SELECT event_id, execution_id, workflow_id, run_id, parent_execution_id, sequence, previous_sequence,
		       event_type, status, status_reason, payload, emitted_at, recorded_at
		FROM workflow_execution_events
		WHERE execution_id = ?`
	args := []interface{}{executionID}

	if afterSeq != nil {
		query += " AND sequence > ?"
		args = append(args, *afterSeq)
	}

	query += " ORDER BY sequence ASC"
	if limit > 0 {
		query += fmt.Sprintf(" LIMIT %d", limit)
	}

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query workflow execution events: %w", err)
	}
	defer rows.Close()

	var events []*types.WorkflowExecutionEvent
	for rows.Next() {
		evt := &types.WorkflowExecutionEvent{}
		var runID sql.NullString
		var parentID sql.NullString
		var status sql.NullString
		var statusReason sql.NullString
		var payload sql.NullString

		if err := rows.Scan(
			&evt.EventID,
			&evt.ExecutionID,
			&evt.WorkflowID,
			&runID,
			&parentID,
			&evt.Sequence,
			&evt.PreviousSequence,
			&evt.EventType,
			&status,
			&statusReason,
			&payload,
			&evt.EmittedAt,
			&evt.RecordedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan workflow execution event: %w", err)
		}

		if runID.Valid {
			evt.RunID = &runID.String
		}
		if parentID.Valid {
			evt.ParentExecutionID = &parentID.String
		}
		if status.Valid {
			value := status.String
			evt.Status = &value
		}
		if statusReason.Valid {
			value := statusReason.String
			evt.StatusReason = &value
		}
		if payload.Valid {
			evt.Payload = json.RawMessage(payload.String)
		} else {
			evt.Payload = json.RawMessage("{}")
		}

		events = append(events, evt)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating workflow execution events: %w", err)
	}

	return events, nil
}

// StoreExecutionLogEntry inserts a structured execution log entry and publishes it to subscribers.
func (ls *LocalStorage) StoreExecutionLogEntry(ctx context.Context, entry *types.ExecutionLogEntry) error {
	if entry == nil {
		return fmt.Errorf("execution log entry is nil")
	}
	if strings.TrimSpace(entry.ExecutionID) == "" {
		return fmt.Errorf("execution_id is required")
	}
	if strings.TrimSpace(entry.WorkflowID) == "" {
		entry.WorkflowID = entry.ExecutionID
	}
	if strings.TrimSpace(entry.Level) == "" {
		entry.Level = "info"
	}
	if strings.TrimSpace(entry.Source) == "" {
		entry.Source = "sdk.logger"
	}
	if entry.EmittedAt.IsZero() {
		entry.EmittedAt = time.Now().UTC()
	}
	ls.mu.Lock()
	defer ls.mu.Unlock()

	return ls.retryDatabaseOperation(ctx, entry.ExecutionID, func() error {
		tx, err := ls.db.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("failed to begin execution log transaction: %w", err)
		}
		defer rollbackTx(tx, "StoreExecutionLogEntry:"+entry.ExecutionID)

		if err := ls.storeExecutionLogEntryTx(ctx, tx, entry); err != nil {
			return err
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("failed to commit execution log transaction: %w", err)
		}

		entryCopy := *entry
		ls.executionLogEventBus.Publish(&entryCopy)
		return nil
	})
}

// StoreExecutionLogEntries atomically stores a batch of structured execution logs for one execution.
func (ls *LocalStorage) StoreExecutionLogEntries(ctx context.Context, executionID string, entries []*types.ExecutionLogEntry) error {
	if len(entries) == 0 {
		return nil
	}

	ls.mu.Lock()
	defer ls.mu.Unlock()

	return ls.retryDatabaseOperation(ctx, executionID, func() error {
		tx, err := ls.db.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("failed to begin execution log batch transaction: %w", err)
		}
		defer rollbackTx(tx, "StoreExecutionLogEntries:"+executionID)

		for _, entry := range entries {
			if entry == nil {
				continue
			}
			if err := ls.storeExecutionLogEntryTx(ctx, tx, entry); err != nil {
				return err
			}
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("failed to commit execution log batch transaction: %w", err)
		}

		for _, entry := range entries {
			if entry == nil {
				continue
			}
			entryCopy := *entry
			ls.executionLogEventBus.Publish(&entryCopy)
		}
		return nil
	})
}

func (ls *LocalStorage) storeExecutionLogEntryTx(ctx context.Context, tx DBTX, entry *types.ExecutionLogEntry) error {
	var nextSeq int64 = 1
	if err := tx.QueryRowContext(ctx,
		`SELECT COALESCE(MAX(sequence), 0) + 1 FROM execution_logs WHERE execution_id = ?`,
		entry.ExecutionID,
	).Scan(&nextSeq); err != nil {
		return fmt.Errorf("failed to compute execution log sequence: %w", err)
	}
	entry.Sequence = nextSeq

	attributes := "{}"
	if len(entry.Attributes) > 0 {
		attributes = string(entry.Attributes)
	}

	query := `
		INSERT INTO execution_logs (
			execution_id, workflow_id, run_id, root_workflow_id, parent_execution_id, sequence,
			agent_node_id, reasoner_id, level, source, event_type, message, attributes,
			system_generated, sdk_language, attempt, span_id, step_id, error_category, emitted_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	result, err := tx.ExecContext(ctx, query,
		entry.ExecutionID,
		entry.WorkflowID,
		entry.RunID,
		entry.RootWorkflowID,
		entry.ParentExecutionID,
		entry.Sequence,
		entry.AgentNodeID,
		entry.ReasonerID,
		entry.Level,
		entry.Source,
		entry.EventType,
		entry.Message,
		attributes,
		entry.SystemGenerated,
		entry.SDKLanguage,
		entry.Attempt,
		entry.SpanID,
		entry.StepID,
		entry.ErrorCategory,
		entry.EmittedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to insert execution log entry: %w", err)
	}
	if id, err := result.LastInsertId(); err == nil {
		entry.EventID = id
	}
	if entry.RecordedAt.IsZero() {
		entry.RecordedAt = time.Now().UTC()
	}
	return nil
}

// ListExecutionLogEntries retrieves structured execution logs ordered by sequence.
func (ls *LocalStorage) ListExecutionLogEntries(ctx context.Context, executionID string, afterSeq *int64, limit int, levels []string, nodeIDs []string, sources []string, query string) ([]*types.ExecutionLogEntry, error) {
	baseQuery := `
		SELECT event_id, execution_id, workflow_id, run_id, root_workflow_id, parent_execution_id, sequence,
		       agent_node_id, reasoner_id, level, source, event_type, message, attributes, system_generated,
		       sdk_language, attempt, span_id, step_id, error_category, emitted_at, recorded_at
		FROM execution_logs
		WHERE execution_id = ?`
	args := []interface{}{executionID}

	appendIn := func(column string, values []string) {
		if len(values) == 0 {
			return
		}
		holders := make([]string, 0, len(values))
		for _, value := range values {
			if strings.TrimSpace(value) == "" {
				continue
			}
			holders = append(holders, "?")
			args = append(args, value)
		}
		if len(holders) > 0 {
			baseQuery += " AND " + column + " IN (" + strings.Join(holders, ",") + ")"
		}
	}

	if afterSeq != nil {
		baseQuery += " AND sequence > ?"
		args = append(args, *afterSeq)
	}
	appendIn("level", levels)
	appendIn("agent_node_id", nodeIDs)
	appendIn("source", sources)
	if trimmed := strings.TrimSpace(query); trimmed != "" {
		baseQuery += " AND (message LIKE ? OR attributes LIKE ?)"
		like := "%" + trimmed + "%"
		args = append(args, like, like)
	}

	descendingTail := afterSeq == nil && limit > 0
	if descendingTail {
		baseQuery += " ORDER BY sequence DESC"
		baseQuery += fmt.Sprintf(" LIMIT %d", limit)
	} else {
		baseQuery += " ORDER BY sequence ASC"
		if limit > 0 {
			baseQuery += fmt.Sprintf(" LIMIT %d", limit)
		}
	}

	rows, err := ls.db.QueryContext(ctx, baseQuery, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query execution logs: %w", err)
	}
	defer rows.Close()

	var entries []*types.ExecutionLogEntry
	for rows.Next() {
		entry := &types.ExecutionLogEntry{}
		var runID, rootWorkflowID, parentExecutionID, reasonerID, eventType, sdkLanguage, spanID, stepID, errorCategory sql.NullString
		var attributes sql.NullString
		var attempt sql.NullInt64
		var emittedAt sql.NullTime
		var recordedAt sql.NullTime
		if err := rows.Scan(
			&entry.EventID,
			&entry.ExecutionID,
			&entry.WorkflowID,
			&runID,
			&rootWorkflowID,
			&parentExecutionID,
			&entry.Sequence,
			&entry.AgentNodeID,
			&reasonerID,
			&entry.Level,
			&entry.Source,
			&eventType,
			&entry.Message,
			&attributes,
			&entry.SystemGenerated,
			&sdkLanguage,
			&attempt,
			&spanID,
			&stepID,
			&errorCategory,
			&emittedAt,
			&recordedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan execution log entry: %w", err)
		}

		if runID.Valid {
			entry.RunID = &runID.String
		}
		if rootWorkflowID.Valid {
			entry.RootWorkflowID = &rootWorkflowID.String
		}
		if parentExecutionID.Valid {
			entry.ParentExecutionID = &parentExecutionID.String
		}
		if reasonerID.Valid {
			entry.ReasonerID = &reasonerID.String
		}
		if eventType.Valid {
			entry.EventType = &eventType.String
		}
		if sdkLanguage.Valid {
			entry.SDKLanguage = &sdkLanguage.String
		}
		if spanID.Valid {
			entry.SpanID = &spanID.String
		}
		if stepID.Valid {
			entry.StepID = &stepID.String
		}
		if errorCategory.Valid {
			entry.ErrorCategory = &errorCategory.String
		}
		if attempt.Valid {
			value := int(attempt.Int64)
			entry.Attempt = &value
		}
		if emittedAt.Valid {
			entry.EmittedAt = emittedAt.Time
		}
		if recordedAt.Valid {
			entry.RecordedAt = recordedAt.Time
		} else {
			entry.RecordedAt = entry.EmittedAt
		}
		if attributes.Valid {
			entry.Attributes = json.RawMessage(attributes.String)
		} else {
			entry.Attributes = json.RawMessage("{}")
		}

		entries = append(entries, entry)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating execution logs: %w", err)
	}
	if descendingTail {
		for left, right := 0, len(entries)-1; left < right; left, right = left+1, right-1 {
			entries[left], entries[right] = entries[right], entries[left]
		}
	}
	return entries, nil
}

// PruneExecutionLogEntries trims old or excessive execution logs for a single execution.
func (ls *LocalStorage) PruneExecutionLogEntries(ctx context.Context, executionID string, maxEntries int, olderThan time.Time) error {
	if strings.TrimSpace(executionID) == "" {
		return nil
	}
	if !olderThan.IsZero() {
		if _, err := ls.db.ExecContext(ctx,
			`DELETE FROM execution_logs WHERE execution_id = ? AND emitted_at < ?`,
			executionID, olderThan,
		); err != nil {
			return fmt.Errorf("failed to prune execution logs by age: %w", err)
		}
	}
	if maxEntries > 0 {
		if _, err := ls.db.ExecContext(ctx, `
			DELETE FROM execution_logs
			WHERE execution_id = ?
			  AND event_id NOT IN (
			    SELECT event_id FROM execution_logs
			    WHERE execution_id = ?
			    ORDER BY sequence DESC
			    LIMIT ?
			  )`,
			executionID, executionID, maxEntries,
		); err != nil {
			return fmt.Errorf("failed to prune execution logs by count: %w", err)
		}
	}
	return nil
}

// StoreExecutionWebhookEvent records webhook delivery attempts for SQLite deployments.
func (ls *LocalStorage) StoreExecutionWebhookEvent(ctx context.Context, event *types.ExecutionWebhookEvent) error {
	if event == nil {
		return fmt.Errorf("execution webhook event is nil")
	}

	payload := interface{}(nil)
	if len(event.Payload) > 0 {
		payload = string(event.Payload)
	}

	query := `
		INSERT INTO execution_webhook_events (
			execution_id, event_type, status, http_status, payload, response_body, error_message, created_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)`

	_, err := ls.db.ExecContext(ctx, query,
		event.ExecutionID,
		event.EventType,
		event.Status,
		event.HTTPStatus,
		payload,
		event.ResponseBody,
		event.ErrorMessage,
	)
	return err
}

// ListExecutionWebhookEvents returns webhook attempts ordered by creation time.
func (ls *LocalStorage) ListExecutionWebhookEvents(ctx context.Context, executionID string) ([]*types.ExecutionWebhookEvent, error) {
	query := `
		SELECT id, execution_id, event_type, status, http_status, payload, response_body, error_message, created_at
		FROM execution_webhook_events
		WHERE execution_id = ?
		ORDER BY created_at ASC, id ASC`

	rows, err := ls.db.QueryContext(ctx, query, executionID)
	if err != nil {
		return nil, fmt.Errorf("failed to query execution webhook events: %w", err)
	}
	defer rows.Close()

	var events []*types.ExecutionWebhookEvent
	for rows.Next() {
		evt := &types.ExecutionWebhookEvent{}
		var payload sql.NullString
		var response sql.NullString
		var errMsg sql.NullString
		var status sql.NullInt64

		if err := rows.Scan(
			&evt.ID,
			&evt.ExecutionID,
			&evt.EventType,
			&evt.Status,
			&status,
			&payload,
			&response,
			&errMsg,
			&evt.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan execution webhook event: %w", err)
		}

		if status.Valid {
			s := int(status.Int64)
			evt.HTTPStatus = &s
		}
		if payload.Valid {
			evt.Payload = json.RawMessage(payload.String)
		} else {
			evt.Payload = json.RawMessage("{}")
		}
		if response.Valid {
			value := response.String
			evt.ResponseBody = &value
		}
		if errMsg.Valid {
			value := errMsg.String
			evt.ErrorMessage = &value
		}

		events = append(events, evt)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating execution webhook events: %w", err)
	}

	return events, nil
}

// ListExecutionWebhookEventsBatch fetches webhook events for multiple executions in a single query.
func (ls *LocalStorage) ListExecutionWebhookEventsBatch(ctx context.Context, executionIDs []string) (map[string][]*types.ExecutionWebhookEvent, error) {
	results := make(map[string][]*types.ExecutionWebhookEvent)
	if len(executionIDs) == 0 {
		return results, nil
	}

	unique := make([]string, 0, len(executionIDs))
	seen := make(map[string]struct{}, len(executionIDs))
	for _, id := range executionIDs {
		trimmed := strings.TrimSpace(id)
		if trimmed == "" {
			continue
		}
		if _, ok := seen[trimmed]; ok {
			continue
		}
		seen[trimmed] = struct{}{}
		unique = append(unique, trimmed)
	}
	if len(unique) == 0 {
		return results, nil
	}

	placeholders := make([]string, len(unique))
	args := make([]interface{}, len(unique))
	for i, id := range unique {
		placeholders[i] = "?"
		args[i] = id
	}

	query := fmt.Sprintf(`
		SELECT execution_id, id, event_type, status, http_status, payload, response_body, error_message, created_at
		FROM execution_webhook_events
		WHERE execution_id IN (%s)
		ORDER BY execution_id ASC, created_at ASC, id ASC`, strings.Join(placeholders, ","))

	rows, err := ls.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query batch webhook events: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		evt := &types.ExecutionWebhookEvent{}
		var payload sql.NullString
		var response sql.NullString
		var errMsg sql.NullString
		var status sql.NullInt64
		if err := rows.Scan(
			&evt.ExecutionID,
			&evt.ID,
			&evt.EventType,
			&evt.Status,
			&status,
			&payload,
			&response,
			&errMsg,
			&evt.CreatedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan batch webhook event: %w", err)
		}

		if status.Valid {
			s := int(status.Int64)
			evt.HTTPStatus = &s
		}
		if payload.Valid {
			evt.Payload = json.RawMessage(payload.String)
		} else {
			evt.Payload = json.RawMessage("{}")
		}
		if response.Valid {
			value := response.String
			evt.ResponseBody = &value
		}
		if errMsg.Valid {
			value := errMsg.String
			evt.ErrorMessage = &value
		}

		results[evt.ExecutionID] = append(results[evt.ExecutionID], evt)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating batch webhook events: %w", err)
	}

	return results, nil
}

// =============================================================================
// DID Document Operations (did:web Resolution)
// =============================================================================

// StoreDIDDocument stores a DID document record.
func (ls *LocalStorage) StoreDIDDocument(ctx context.Context, record *types.DIDDocumentRecord) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store DID document: %w", err)
	}

	query := `
		INSERT INTO did_documents (
			did, agent_id, did_document, public_key_jwk, revoked_at, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(did) DO UPDATE SET
			agent_id = excluded.agent_id,
			did_document = excluded.did_document,
			public_key_jwk = excluded.public_key_jwk,
			updated_at = excluded.updated_at`

	_, err := ls.db.ExecContext(ctx, query,
		record.DID, record.AgentID, record.DIDDocument, record.PublicKeyJWK,
		record.RevokedAt, record.CreatedAt, record.UpdatedAt,
	)
	if err != nil {
		return fmt.Errorf("failed to store DID document: %w", err)
	}

	return nil
}

// GetDIDDocument retrieves a DID document by its DID.
func (ls *LocalStorage) GetDIDDocument(ctx context.Context, did string) (*types.DIDDocumentRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get DID document: %w", err)
	}

	query := `
		SELECT did, agent_id, did_document, public_key_jwk, revoked_at, created_at, updated_at
		FROM did_documents WHERE did = ?`

	row := ls.db.QueryRowContext(ctx, query, did)

	record := &types.DIDDocumentRecord{}
	var revokedAt sql.NullTime

	err := row.Scan(
		&record.DID, &record.AgentID, &record.DIDDocument, &record.PublicKeyJWK,
		&revokedAt, &record.CreatedAt, &record.UpdatedAt,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("DID document not found: %s", did)
		}
		return nil, fmt.Errorf("failed to get DID document: %w", err)
	}

	if revokedAt.Valid {
		record.RevokedAt = &revokedAt.Time
	}

	return record, nil
}

// GetDIDDocumentByAgentID retrieves a DID document by agent ID.
func (ls *LocalStorage) GetDIDDocumentByAgentID(ctx context.Context, agentID string) (*types.DIDDocumentRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get DID document by agent ID: %w", err)
	}

	query := `
		SELECT did, agent_id, did_document, public_key_jwk, revoked_at, created_at, updated_at
		FROM did_documents WHERE agent_id = ? AND revoked_at IS NULL
		ORDER BY created_at DESC LIMIT 1`

	row := ls.db.QueryRowContext(ctx, query, agentID)

	record := &types.DIDDocumentRecord{}
	var revokedAt sql.NullTime

	err := row.Scan(
		&record.DID, &record.AgentID, &record.DIDDocument, &record.PublicKeyJWK,
		&revokedAt, &record.CreatedAt, &record.UpdatedAt,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("DID document not found for agent: %s", agentID)
		}
		return nil, fmt.Errorf("failed to get DID document by agent ID: %w", err)
	}

	if revokedAt.Valid {
		record.RevokedAt = &revokedAt.Time
	}

	return record, nil
}

// RevokeDIDDocument revokes a DID document by setting its revoked_at timestamp.
func (ls *LocalStorage) RevokeDIDDocument(ctx context.Context, did string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during revoke DID document: %w", err)
	}

	query := `UPDATE did_documents SET revoked_at = ?, updated_at = ? WHERE did = ?`

	now := time.Now()
	result, err := ls.db.ExecContext(ctx, query, now, now, did)
	if err != nil {
		return fmt.Errorf("failed to revoke DID document: %w", err)
	}

	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rows == 0 {
		return fmt.Errorf("DID document not found: %s", did)
	}

	return nil
}

// ListDIDDocuments lists all DID documents.
func (ls *LocalStorage) ListDIDDocuments(ctx context.Context) ([]*types.DIDDocumentRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list DID documents: %w", err)
	}

	query := `
		SELECT did, agent_id, did_document, public_key_jwk, revoked_at, created_at, updated_at
		FROM did_documents ORDER BY created_at DESC`

	rows, err := ls.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list DID documents: %w", err)
	}
	defer rows.Close()

	var records []*types.DIDDocumentRecord
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during scan: %w", err)
		}

		record := &types.DIDDocumentRecord{}
		var revokedAt sql.NullTime

		err := rows.Scan(
			&record.DID, &record.AgentID, &record.DIDDocument, &record.PublicKeyJWK,
			&revokedAt, &record.CreatedAt, &record.UpdatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan DID document: %w", err)
		}

		if revokedAt.Valid {
			record.RevokedAt = &revokedAt.Time
		}

		records = append(records, record)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating DID documents: %w", err)
	}

	return records, nil
}

// ListAgentsByLifecycleStatus lists agents filtered by lifecycle status.
func (ls *LocalStorage) ListAgentsByLifecycleStatus(ctx context.Context, status types.AgentLifecycleStatus) ([]*types.AgentNode, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list agents by lifecycle status: %w", err)
	}

	query := `
		SELECT
			id, version, group_id, team_id, base_url, traffic_weight, deployment_type, invocation_url, reasoners, skills,
			communication_config, health_status, lifecycle_status, last_heartbeat,
			registered_at, features, metadata, proposed_tags, approved_tags
		FROM agent_nodes WHERE lifecycle_status = ? ORDER BY registered_at DESC`

	rows, err := ls.db.QueryContext(ctx, query, string(status))
	if err != nil {
		return nil, fmt.Errorf("failed to list agents by lifecycle status: %w", err)
	}
	defer rows.Close()

	return ls.scanAgentNodes(ctx, rows)
}

// reconstructAgentLevelTags ensures agent-level ProposedTags and ApprovedTags
// are populated. If the dedicated DB columns were empty (e.g., on older records),
// it reconstructs them from per-reasoner/per-skill fields as a fallback.
func reconstructAgentLevelTags(agent *types.AgentNode) {
	// Only reconstruct if DB columns were empty
	if len(agent.ApprovedTags) == 0 {
		seen := make(map[string]struct{})
		for _, r := range agent.Reasoners {
			for _, t := range r.ApprovedTags {
				if _, exists := seen[t]; !exists {
					seen[t] = struct{}{}
					agent.ApprovedTags = append(agent.ApprovedTags, t)
				}
			}
		}
		for _, sk := range agent.Skills {
			for _, t := range sk.ApprovedTags {
				if _, exists := seen[t]; !exists {
					seen[t] = struct{}{}
					agent.ApprovedTags = append(agent.ApprovedTags, t)
				}
			}
		}
	}

	if len(agent.ProposedTags) == 0 {
		proposedSeen := make(map[string]struct{})
		for _, r := range agent.Reasoners {
			source := r.ProposedTags
			if len(source) == 0 {
				source = r.Tags
			}
			for _, t := range source {
				if _, exists := proposedSeen[t]; !exists {
					proposedSeen[t] = struct{}{}
					agent.ProposedTags = append(agent.ProposedTags, t)
				}
			}
		}
		for _, sk := range agent.Skills {
			source := sk.ProposedTags
			if len(source) == 0 {
				source = sk.Tags
			}
			for _, t := range source {
				if _, exists := proposedSeen[t]; !exists {
					proposedSeen[t] = struct{}{}
					agent.ProposedTags = append(agent.ProposedTags, t)
				}
			}
		}
	}
}

// ============================================================================
// Access Policy Storage
// ============================================================================

// GetAccessPolicies retrieves all enabled access policies, sorted by priority descending.
func (ls *LocalStorage) GetAccessPolicies(ctx context.Context) ([]*types.AccessPolicy, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get access policies: %w", err)
	}

	query := `
		SELECT id, name, caller_tags, target_tags, allow_functions, deny_functions,
		       constraints, action, priority, enabled, description, created_at, updated_at
		FROM access_policies WHERE enabled = true ORDER BY priority DESC, created_at DESC`

	rows, err := ls.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to get access policies: %w", err)
	}
	defer rows.Close()

	var policies []*types.AccessPolicy
	for rows.Next() {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("context cancelled during scan: %w", err)
		}

		policy, err := scanAccessPolicy(rows)
		if err != nil {
			return nil, err
		}
		policies = append(policies, policy)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating access policies: %w", err)
	}

	return policies, nil
}

// GetAccessPolicyByID retrieves a single access policy by its ID.
func (ls *LocalStorage) GetAccessPolicyByID(ctx context.Context, id int64) (*types.AccessPolicy, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get access policy: %w", err)
	}

	query := `
		SELECT id, name, caller_tags, target_tags, allow_functions, deny_functions,
		       constraints, action, priority, enabled, description, created_at, updated_at
		FROM access_policies WHERE id = ?`

	row := ls.db.QueryRowContext(ctx, query, id)

	policy := &types.AccessPolicy{}
	var callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON string
	var description sql.NullString

	err := row.Scan(
		&policy.ID, &policy.Name, &callerTagsJSON, &targetTagsJSON,
		&allowFuncsJSON, &denyFuncsJSON, &constraintsJSON,
		&policy.Action, &policy.Priority, &policy.Enabled, &description,
		&policy.CreatedAt, &policy.UpdatedAt,
	)
	if err != nil {
		return nil, fmt.Errorf("access policy with ID %d not found: %w", id, err)
	}

	if description.Valid {
		policy.Description = &description.String
	}
	if err := unmarshalAccessPolicyJSON(policy, callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON); err != nil {
		return nil, fmt.Errorf("failed to unmarshal access policy %d: %w", id, err)
	}

	return policy, nil
}

// CreateAccessPolicy creates a new access policy.
func (ls *LocalStorage) CreateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error {
	if ls.mode == "postgres" {
		return ls.createAccessPolicyPostgres(ctx, policy)
	}

	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during create access policy: %w", err)
	}

	callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON, err := marshalAccessPolicyJSON(policy)
	if err != nil {
		return fmt.Errorf("failed to marshal access policy fields: %w", err)
	}

	query := `
		INSERT INTO access_policies (
			name, caller_tags, target_tags, allow_functions, deny_functions,
			constraints, action, priority, enabled, description, created_at, updated_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`

	result, err := ls.db.ExecContext(ctx, query,
		policy.Name, callerTagsJSON, targetTagsJSON,
		allowFuncsJSON, denyFuncsJSON, constraintsJSON,
		policy.Action, policy.Priority, policy.Enabled, policy.Description,
		policy.CreatedAt, policy.UpdatedAt,
	)
	if err != nil {
		if strings.Contains(err.Error(), "UNIQUE constraint") {
			return fmt.Errorf("access policy with name %q already exists", policy.Name)
		}
		return fmt.Errorf("failed to create access policy: %w", err)
	}

	id, err := result.LastInsertId()
	if err == nil {
		policy.ID = id
	}

	return nil
}

// createAccessPolicyPostgres creates an access policy using PostgreSQL's RETURNING clause.
func (ls *LocalStorage) createAccessPolicyPostgres(ctx context.Context, policy *types.AccessPolicy) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during create access policy: %w", err)
	}

	callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON, err := marshalAccessPolicyJSON(policy)
	if err != nil {
		return fmt.Errorf("failed to marshal access policy fields: %w", err)
	}

	query := `
		INSERT INTO access_policies (
			name, caller_tags, target_tags, allow_functions, deny_functions,
			constraints, action, priority, enabled, description, created_at, updated_at
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
		RETURNING id`

	row := ls.db.DB.QueryRowContext(ctx, query,
		policy.Name, callerTagsJSON, targetTagsJSON,
		allowFuncsJSON, denyFuncsJSON, constraintsJSON,
		policy.Action, policy.Priority, policy.Enabled, policy.Description,
		policy.CreatedAt, policy.UpdatedAt,
	)

	if err := row.Scan(&policy.ID); err != nil {
		if strings.Contains(err.Error(), "duplicate key") {
			return fmt.Errorf("access policy with name %q already exists", policy.Name)
		}
		return fmt.Errorf("failed to create access policy: %w", err)
	}

	return nil
}

// UpdateAccessPolicy updates an existing access policy.
func (ls *LocalStorage) UpdateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during update access policy: %w", err)
	}

	callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON, err := marshalAccessPolicyJSON(policy)
	if err != nil {
		return fmt.Errorf("failed to marshal access policy fields: %w", err)
	}

	query := `
		UPDATE access_policies SET
			name = ?, caller_tags = ?, target_tags = ?, allow_functions = ?,
			deny_functions = ?, constraints = ?, action = ?, priority = ?,
			enabled = ?, description = ?, updated_at = ?
		WHERE id = ?`

	result, err := ls.db.ExecContext(ctx, query,
		policy.Name, callerTagsJSON, targetTagsJSON,
		allowFuncsJSON, denyFuncsJSON, constraintsJSON,
		policy.Action, policy.Priority, policy.Enabled, policy.Description,
		policy.UpdatedAt, policy.ID,
	)
	if err != nil {
		return fmt.Errorf("failed to update access policy: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("access policy with ID %d not found", policy.ID)
	}

	return nil
}

// DeleteAccessPolicy deletes an access policy by ID.
func (ls *LocalStorage) DeleteAccessPolicy(ctx context.Context, id int64) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during delete access policy: %w", err)
	}

	query := `DELETE FROM access_policies WHERE id = ?`

	result, err := ls.db.ExecContext(ctx, query, id)
	if err != nil {
		return fmt.Errorf("failed to delete access policy: %w", err)
	}

	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rows == 0 {
		return fmt.Errorf("access policy with ID %d not found", id)
	}

	return nil
}

// scanAccessPolicy scans a row into an AccessPolicy struct.
func scanAccessPolicy(rows *sql.Rows) (*types.AccessPolicy, error) {
	policy := &types.AccessPolicy{}
	var callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON string
	var description sql.NullString

	err := rows.Scan(
		&policy.ID, &policy.Name, &callerTagsJSON, &targetTagsJSON,
		&allowFuncsJSON, &denyFuncsJSON, &constraintsJSON,
		&policy.Action, &policy.Priority, &policy.Enabled, &description,
		&policy.CreatedAt, &policy.UpdatedAt,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to scan access policy: %w", err)
	}

	if description.Valid {
		policy.Description = &description.String
	}
	if err := unmarshalAccessPolicyJSON(policy, callerTagsJSON, targetTagsJSON, allowFuncsJSON, denyFuncsJSON, constraintsJSON); err != nil {
		return nil, fmt.Errorf("failed to unmarshal access policy %d: %w", policy.ID, err)
	}

	return policy, nil
}

// unmarshalAccessPolicyJSON populates the JSON fields of an AccessPolicy.
// Returns an error if any JSON field cannot be deserialized, preventing
// corrupted data from silently producing empty policy rules.
func unmarshalAccessPolicyJSON(policy *types.AccessPolicy, callerTags, targetTags, allowFuncs, denyFuncs, constraints string) error {
	if callerTags != "" {
		if err := json.Unmarshal([]byte(callerTags), &policy.CallerTags); err != nil {
			return fmt.Errorf("failed to unmarshal caller_tags: %w", err)
		}
	}
	if targetTags != "" {
		if err := json.Unmarshal([]byte(targetTags), &policy.TargetTags); err != nil {
			return fmt.Errorf("failed to unmarshal target_tags: %w", err)
		}
	}
	if allowFuncs != "" {
		if err := json.Unmarshal([]byte(allowFuncs), &policy.AllowFunctions); err != nil {
			return fmt.Errorf("failed to unmarshal allow_functions: %w", err)
		}
	}
	if denyFuncs != "" {
		if err := json.Unmarshal([]byte(denyFuncs), &policy.DenyFunctions); err != nil {
			return fmt.Errorf("failed to unmarshal deny_functions: %w", err)
		}
	}
	if constraints != "" {
		if err := json.Unmarshal([]byte(constraints), &policy.Constraints); err != nil {
			return fmt.Errorf("failed to unmarshal constraints: %w", err)
		}
	}
	return nil
}

// marshalAccessPolicyJSON serializes the JSON fields of an AccessPolicy for storage.
func marshalAccessPolicyJSON(policy *types.AccessPolicy) (callerTags, targetTags, allowFuncs, denyFuncs, constraints string, err error) {
	ct, err := json.Marshal(policy.CallerTags)
	if err != nil {
		return "", "", "", "", "", fmt.Errorf("caller_tags: %w", err)
	}
	tt, err := json.Marshal(policy.TargetTags)
	if err != nil {
		return "", "", "", "", "", fmt.Errorf("target_tags: %w", err)
	}
	af, err := json.Marshal(policy.AllowFunctions)
	if err != nil {
		return "", "", "", "", "", fmt.Errorf("allow_functions: %w", err)
	}
	df, err := json.Marshal(policy.DenyFunctions)
	if err != nil {
		return "", "", "", "", "", fmt.Errorf("deny_functions: %w", err)
	}
	cn, err := json.Marshal(policy.Constraints)
	if err != nil {
		return "", "", "", "", "", fmt.Errorf("constraints: %w", err)
	}
	return string(ct), string(tt), string(af), string(df), string(cn), nil
}

// ========== Agent Tag VC operations ==========

// StoreAgentTagVC stores or replaces an agent's tag VC.
func (ls *LocalStorage) StoreAgentTagVC(ctx context.Context, agentID, agentDID, vcID, vcDocument, signature string, issuedAt time.Time, expiresAt *time.Time) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during store agent tag VC: %w", err)
	}

	query := `
		INSERT INTO agent_tag_vcs (agent_id, agent_did, vc_id, vc_document, signature, issued_at, expires_at, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
		ON CONFLICT(agent_id) DO UPDATE SET
			agent_did = excluded.agent_did,
			vc_id = excluded.vc_id,
			vc_document = excluded.vc_document,
			signature = excluded.signature,
			issued_at = excluded.issued_at,
			expires_at = excluded.expires_at,
			revoked_at = NULL,
			updated_at = excluded.updated_at`

	now := time.Now()
	_, err := ls.db.ExecContext(ctx, query, agentID, agentDID, vcID, vcDocument, signature, issuedAt, expiresAt, now, now)
	if err != nil {
		return fmt.Errorf("failed to store agent tag VC: %w", err)
	}
	return nil
}

// GetAgentTagVC retrieves an agent's tag VC record.
func (ls *LocalStorage) GetAgentTagVC(ctx context.Context, agentID string) (*types.AgentTagVCRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during get agent tag VC: %w", err)
	}

	query := `
		SELECT id, agent_id, agent_did, vc_id, vc_document, signature, issued_at, expires_at, revoked_at
		FROM agent_tag_vcs WHERE agent_id = ?`

	row := ls.db.QueryRowContext(ctx, query, agentID)

	record := &types.AgentTagVCRecord{}
	var expiresAt, revokedAt sql.NullTime
	var signature sql.NullString

	err := row.Scan(
		&record.ID, &record.AgentID, &record.AgentDID, &record.VCID,
		&record.VCDocument, &signature, &record.IssuedAt, &expiresAt, &revokedAt,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("agent tag VC not found for agent %s", agentID)
		}
		return nil, fmt.Errorf("failed to get agent tag VC: %w", err)
	}

	if signature.Valid {
		record.Signature = signature.String
	}
	if expiresAt.Valid {
		record.ExpiresAt = &expiresAt.Time
	}
	if revokedAt.Valid {
		record.RevokedAt = &revokedAt.Time
	}

	return record, nil
}

// ListAgentTagVCs returns all non-revoked agent tag VCs.
func (ls *LocalStorage) ListAgentTagVCs(ctx context.Context) ([]*types.AgentTagVCRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("context cancelled during list agent tag VCs: %w", err)
	}

	query := `
		SELECT id, agent_id, agent_did, vc_id, vc_document, signature, issued_at, expires_at, revoked_at
		FROM agent_tag_vcs WHERE revoked_at IS NULL`

	rows, err := ls.db.QueryContext(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list agent tag VCs: %w", err)
	}
	defer rows.Close()

	var records []*types.AgentTagVCRecord
	for rows.Next() {
		record := &types.AgentTagVCRecord{}
		var expiresAt, revokedAt sql.NullTime
		var signature sql.NullString

		if err := rows.Scan(
			&record.ID, &record.AgentID, &record.AgentDID, &record.VCID,
			&record.VCDocument, &signature, &record.IssuedAt, &expiresAt, &revokedAt,
		); err != nil {
			return nil, fmt.Errorf("failed to scan agent tag VC: %w", err)
		}

		if signature.Valid {
			record.Signature = signature.String
		}
		if expiresAt.Valid {
			record.ExpiresAt = &expiresAt.Time
		}
		if revokedAt.Valid {
			record.RevokedAt = &revokedAt.Time
		}
		records = append(records, record)
	}

	return records, rows.Err()
}

// RevokeAgentTagVC marks an agent's tag VC as revoked.
func (ls *LocalStorage) RevokeAgentTagVC(ctx context.Context, agentID string) error {
	if err := ctx.Err(); err != nil {
		return fmt.Errorf("context cancelled during revoke agent tag VC: %w", err)
	}

	query := `UPDATE agent_tag_vcs SET revoked_at = ?, updated_at = ? WHERE agent_id = ? AND revoked_at IS NULL`

	now := time.Now()
	result, err := ls.db.ExecContext(ctx, query, now, now, agentID)
	if err != nil {
		return fmt.Errorf("failed to revoke agent tag VC: %w", err)
	}

	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rows == 0 {
		return fmt.Errorf("no active agent tag VC found for agent %s", agentID)
	}

	return nil
}
