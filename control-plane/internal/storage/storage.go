package storage

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

// RunSummaryAggregation holds aggregated statistics for a single workflow run
type RunSummaryAggregation struct {
	RunID            string
	TotalExecutions  int
	StatusCounts     map[string]int
	EarliestStarted  time.Time
	LatestStarted    time.Time
	RootExecutionID  *string
	RootAgentNodeID  *string
	RootReasonerID   *string
	SessionID        *string
	ActorID          *string
	MaxDepth         int
	ActiveExecutions int
}

// StorageProvider is the interface for the primary data storage backend.
type StorageProvider interface {
	// Lifecycle
	Initialize(ctx context.Context, config StorageConfig) error
	Close(ctx context.Context) error
	HealthCheck(ctx context.Context) error

	// Execution operations
	StoreExecution(ctx context.Context, execution *types.AgentExecution) error
	GetExecution(ctx context.Context, id int64) (*types.AgentExecution, error)
	QueryExecutions(ctx context.Context, filters types.ExecutionFilters) ([]*types.AgentExecution, error)

	// Workflow execution operations
	StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error
	GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error)
	QueryWorkflowExecutions(ctx context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error)
	UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error)) error
	CreateExecutionRecord(ctx context.Context, execution *types.Execution) error
	GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error)
	UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error)
	QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error)
	QueryRunSummaries(ctx context.Context, filter types.ExecutionFilter) ([]*RunSummaryAggregation, int, error)
	RegisterExecutionWebhook(ctx context.Context, webhook *types.ExecutionWebhook) error
	GetExecutionWebhook(ctx context.Context, executionID string) (*types.ExecutionWebhook, error)
	ListDueExecutionWebhooks(ctx context.Context, limit int) ([]*types.ExecutionWebhook, error)
	TryMarkExecutionWebhookInFlight(ctx context.Context, executionID string, now time.Time) (bool, error)
	UpdateExecutionWebhookState(ctx context.Context, executionID string, update types.ExecutionWebhookStateUpdate) error
	HasExecutionWebhook(ctx context.Context, executionID string) (bool, error)
	ListExecutionWebhooksRegistered(ctx context.Context, executionIDs []string) (map[string]bool, error)
	StoreExecutionWebhookEvent(ctx context.Context, event *types.ExecutionWebhookEvent) error
	ListExecutionWebhookEvents(ctx context.Context, executionID string) ([]*types.ExecutionWebhookEvent, error)
	ListExecutionWebhookEventsBatch(ctx context.Context, executionIDs []string) (map[string][]*types.ExecutionWebhookEvent, error)
	StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error
	ListWorkflowExecutionEvents(ctx context.Context, executionID string, afterSeq *int64, limit int) ([]*types.WorkflowExecutionEvent, error)

	// Execution cleanup operations
	CleanupOldExecutions(ctx context.Context, retentionPeriod time.Duration, batchSize int) (int, error)
	MarkStaleExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error)

	// Workflow cleanup operations - deletes all data related to a workflow ID
	CleanupWorkflow(ctx context.Context, workflowID string, dryRun bool) (*types.WorkflowCleanupResult, error)

	// DAG operations - optimized single-query DAG building
	QueryWorkflowDAG(ctx context.Context, rootWorkflowID string) ([]*types.WorkflowExecution, error)

	// Workflow operations
	CreateOrUpdateWorkflow(ctx context.Context, workflow *types.Workflow) error
	GetWorkflow(ctx context.Context, workflowID string) (*types.Workflow, error)
	QueryWorkflows(ctx context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error)

	// Session operations
	CreateOrUpdateSession(ctx context.Context, session *types.Session) error
	GetSession(ctx context.Context, sessionID string) (*types.Session, error)
	QuerySessions(ctx context.Context, filters types.SessionFilters) ([]*types.Session, error)

	// Memory operations
	SetMemory(ctx context.Context, memory *types.Memory) error
	GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error)
	DeleteMemory(ctx context.Context, scope, scopeID, key string) error
	ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error)
	SetVector(ctx context.Context, record *types.VectorRecord) error
	DeleteVector(ctx context.Context, scope, scopeID, key string) error
	DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error)
	SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error)

	// Event operations
	StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error
	GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error)

	// Distributed Lock operations
	AcquireLock(ctx context.Context, key string, timeout time.Duration) (*types.DistributedLock, error)
	ReleaseLock(ctx context.Context, lockID string) error
	RenewLock(ctx context.Context, lockID string) (*types.DistributedLock, error)
	GetLockStatus(ctx context.Context, key string) (*types.DistributedLock, error)

	// Agent registry
	RegisterAgent(ctx context.Context, agent *types.AgentNode) error
	GetAgent(ctx context.Context, id string) (*types.AgentNode, error)
	ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error)
	UpdateAgentHealth(ctx context.Context, id string, status types.HealthStatus) error
	UpdateAgentHealthAtomic(ctx context.Context, id string, status types.HealthStatus, expectedLastHeartbeat *time.Time) error
	UpdateAgentHeartbeat(ctx context.Context, id string, heartbeatTime time.Time) error
	UpdateAgentLifecycleStatus(ctx context.Context, id string, status types.AgentLifecycleStatus) error

	// Configuration
	SetConfig(ctx context.Context, key string, value interface{}) error
	GetConfig(ctx context.Context, key string) (interface{}, error)

	// Reasoner Performance and History
	GetReasonerPerformanceMetrics(ctx context.Context, reasonerID string) (*types.ReasonerPerformanceMetrics, error)
	GetReasonerExecutionHistory(ctx context.Context, reasonerID string, page, limit int) (*types.ReasonerExecutionHistory, error)

	// Agent Configuration Management
	StoreAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error
	GetAgentConfiguration(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error)
	QueryAgentConfigurations(ctx context.Context, filters types.ConfigurationFilters) ([]*types.AgentConfiguration, error)
	UpdateAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error
	DeleteAgentConfiguration(ctx context.Context, agentID, packageID string) error
	ValidateAgentConfiguration(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error)

	// Agent Package Management
	StoreAgentPackage(ctx context.Context, pkg *types.AgentPackage) error
	GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error)
	QueryAgentPackages(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error)
	UpdateAgentPackage(ctx context.Context, pkg *types.AgentPackage) error
	DeleteAgentPackage(ctx context.Context, packageID string) error

	// Real-time features (optional, may be handled by CacheProvider)
	SubscribeToMemoryChanges(ctx context.Context, scope, scopeID string) (<-chan types.MemoryChangeEvent, error)
	PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error

	// Execution event bus for real-time updates
	GetExecutionEventBus() *events.ExecutionEventBus
	GetWorkflowExecutionEventBus() *events.EventBus[*types.WorkflowExecutionEvent]

	// DID Registry operations
	StoreDID(ctx context.Context, did string, didDocument, publicKey, privateKeyRef, derivationPath string) error
	GetDID(ctx context.Context, did string) (*types.DIDRegistryEntry, error)
	ListDIDs(ctx context.Context) ([]*types.DIDRegistryEntry, error)

	// AgentField Server DID operations
	StoreAgentFieldServerDID(ctx context.Context, agentfieldServerID, rootDID string, masterSeed []byte, createdAt, lastKeyRotation time.Time) error
	GetAgentFieldServerDID(ctx context.Context, agentfieldServerID string) (*types.AgentFieldServerDIDInfo, error)
	ListAgentFieldServerDIDs(ctx context.Context) ([]*types.AgentFieldServerDIDInfo, error)

	// Agent DID operations
	StoreAgentDID(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int) error
	GetAgentDID(ctx context.Context, agentID string) (*types.AgentDIDInfo, error)
	ListAgentDIDs(ctx context.Context) ([]*types.AgentDIDInfo, error)

	// Component DID operations
	StoreComponentDID(ctx context.Context, componentID, componentDID, agentDID, componentType, componentName string, derivationIndex int) error
	GetComponentDID(ctx context.Context, componentID string) (*types.ComponentDIDInfo, error)
	ListComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error)

	// Multi-step DID operations with transaction safety
	StoreAgentDIDWithComponents(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int, components []ComponentDIDRequest) error

	// Execution VC operations
	StoreExecutionVC(ctx context.Context, vcID, executionID, workflowID, sessionID, issuerDID, targetDID, callerDID, inputHash, outputHash, status string, vcDocument []byte, signature string, storageURI string, documentSizeBytes int64) error
	GetExecutionVC(ctx context.Context, vcID string) (*types.ExecutionVCInfo, error)
	ListExecutionVCs(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error)
	ListWorkflowVCStatusSummaries(ctx context.Context, workflowIDs []string) ([]*types.WorkflowVCStatusAggregation, error)
	CountExecutionVCs(ctx context.Context, filters types.VCFilters) (int, error)

	// Workflow VC operations
	StoreWorkflowVC(ctx context.Context, workflowVCID, workflowID, sessionID string, componentVCIDs []string, status string, startTime, endTime *time.Time, totalSteps, completedSteps int, storageURI string, documentSizeBytes int64) error
	GetWorkflowVC(ctx context.Context, workflowVCID string) (*types.WorkflowVCInfo, error)
	ListWorkflowVCs(ctx context.Context, workflowID string) ([]*types.WorkflowVCInfo, error)
}

// ComponentDIDRequest represents a component DID to be stored
type ComponentDIDRequest struct {
	ComponentDID    string
	ComponentType   string
	ComponentName   string
	PublicKeyJWK    string
	DerivationIndex int
}

// CacheProvider is the interface for the high-performance caching layer.
type CacheProvider interface {
	Set(key string, value interface{}, ttl time.Duration) error
	Get(key string, dest interface{}) error
	Delete(key string) error
	Exists(key string) bool

	// Pub/Sub for real-time features
	Subscribe(channel string) (<-chan CacheMessage, error)
	Publish(channel string, message interface{}) error
}

// CacheMessage represents a message received from the cache's pub/sub.
type CacheMessage struct {
	Channel string
	Payload []byte
}

// StorageConfig holds the configuration for the storage provider.
type StorageConfig struct {
	Mode     string                `yaml:"mode" mapstructure:"mode"`
	Local    LocalStorageConfig    `yaml:"local" mapstructure:"local"`
	Postgres PostgresStorageConfig `yaml:"postgres" mapstructure:"postgres"`
	Vector   VectorStoreConfig     `yaml:"vector" mapstructure:"vector"`
}

// PostgresStorageConfig holds configuration for the PostgreSQL storage provider.
type PostgresStorageConfig struct {
	DSN             string        `yaml:"dsn" mapstructure:"dsn"`
	URL             string        `yaml:"url" mapstructure:"url"`
	Host            string        `yaml:"host" mapstructure:"host"`
	Port            int           `yaml:"port" mapstructure:"port"`
	Database        string        `yaml:"database" mapstructure:"database"`
	User            string        `yaml:"user" mapstructure:"user"`
	Password        string        `yaml:"password" mapstructure:"password"`
	SSLMode         string        `yaml:"sslmode" mapstructure:"sslmode"`
	AdminDatabase   string        `yaml:"admin_database" mapstructure:"admin_database"`
	ConnMaxLifetime time.Duration `yaml:"conn_max_lifetime" mapstructure:"conn_max_lifetime"`
	MaxOpenConns    int           `yaml:"max_open_conns" mapstructure:"max_open_conns"`
	MaxIdleConns    int           `yaml:"max_idle_conns" mapstructure:"max_idle_conns"`
}

// LocalStorageConfig holds configuration for the local storage provider.
type LocalStorageConfig struct {
	DatabasePath string `yaml:"database_path" mapstructure:"database_path"`
	KVStorePath  string `yaml:"kv_store_path" mapstructure:"kv_store_path"`
}

// VectorStoreConfig controls vector storage behavior.
type VectorStoreConfig struct {
	Enabled  *bool  `yaml:"enabled" mapstructure:"enabled"`
	Distance string `yaml:"distance" mapstructure:"distance"`
}

func (cfg VectorStoreConfig) isEnabled() bool {
	if cfg.Enabled == nil {
		return true
	}
	return *cfg.Enabled
}

func (cfg VectorStoreConfig) normalized() VectorStoreConfig {
	if cfg.Distance == "" {
		cfg.Distance = "cosine"
	}
	return cfg
}

// StorageFactory is responsible for creating the appropriate storage backend.
type StorageFactory struct{}

// CreateStorage creates a StorageProvider and CacheProvider based on the configuration.
func (sf *StorageFactory) CreateStorage(config StorageConfig) (StorageProvider, CacheProvider, error) {
	ctx := context.Background() // Use background context for initialization

	mode := config.Mode
	if mode == "" {
		mode = "local"
	}

	// Allow environment variable to override mode
	if envMode := os.Getenv("AGENTFIELD_STORAGE_MODE"); envMode != "" {
		mode = envMode
	}

	config.Vector = config.Vector.normalized()

	switch mode {
	case "local":
		localStorage := NewLocalStorage(config.Local)
		localStorage.vectorConfig = config.Vector
		// Pass the full StorageConfig to Initialize
		if err := localStorage.Initialize(ctx, StorageConfig{
			Mode:     mode,
			Local:    config.Local,
			Postgres: config.Postgres,
			Vector:   config.Vector,
		}); err != nil {
			return nil, nil, fmt.Errorf("failed to initialize local storage: %w", err)
		}
		return localStorage, localStorage, nil // Local storage acts as both

	case "postgres":
		pgStorage := NewPostgresStorage(config.Postgres)
		pgStorage.vectorConfig = config.Vector
		if err := pgStorage.Initialize(ctx, StorageConfig{
			Mode:     mode,
			Local:    config.Local,
			Postgres: config.Postgres,
			Vector:   config.Vector,
		}); err != nil {
			return nil, nil, fmt.Errorf("failed to initialize postgres storage: %w", err)
		}
		return pgStorage, pgStorage, nil

	default:
		return nil, nil, fmt.Errorf("unsupported storage mode: %s (supported modes: local, postgres)", mode)
	}
}
