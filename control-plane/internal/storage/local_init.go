package storage

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/internal/logger"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/boltdb/bolt"
	"github.com/jackc/pgx/v5/pgconn"
	_ "github.com/jackc/pgx/v5/stdlib"
	_ "github.com/mattn/go-sqlite3"
)

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
		logger.Logger.Warn().Msgf("Database path was relative (%s), converted to absolute path: %s", dbPath, absPath)
		dbPath = absPath
	}

	// Ensure the directory exists
	dbDir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dbDir, 0755); err != nil {
		return fmt.Errorf("failed to create database directory %s: %w", dbDir, err)
	}

	logger.Logger.Info().Msgf("Initializing SQLite database at: %s", dbPath)

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
		logger.Logger.Warn().Msgf("Invalid integer for %s=%s, using fallback %d", key, raw, fallback)
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
