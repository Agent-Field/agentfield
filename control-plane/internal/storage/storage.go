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
	RootStatus       *string
	RootAgentNodeID  *string
	RootReasonerID   *string
	SessionID        *string
	ActorID          *string
	MaxDepth         int
	ActiveExecutions int
}

// ConfigEntry represents a database-stored configuration file.
type ConfigEntry struct {
	Key       string    `json:"key"`
	Value     string    `json:"value"`
	Version   int       `json:"version"`
	CreatedBy string    `json:"created_by,omitempty"`
	UpdatedBy string    `json:"updated_by,omitempty"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// StorageProvider is the interface for the primary data storage backend.
type StorageProvider interface {
	// Lifecycle
	// Initialize prepares the storage backend using the provided configuration.
	// The ctx controls setup lifetime, and config supplies backend settings.
	// Returns an error if initialization fails.
	Initialize(ctx context.Context, config StorageConfig) error
	// Close releases storage resources associated with the provider.
	// The ctx bounds shutdown and cleanup work for the backend.
	// Returns an error if resources cannot be closed cleanly.
	Close(ctx context.Context) error
	// HealthCheck verifies that the storage backend is reachable and operational.
	// The ctx controls how long the health probe may run.
	// Returns an error when the backend is unhealthy or unreachable.
	HealthCheck(ctx context.Context) error

	// Execution operations
	// StoreExecution persists an agent execution record.
	// The ctx scopes the write, and execution contains the execution data to store.
	// Returns an error if the execution cannot be saved.
	StoreExecution(ctx context.Context, execution *types.AgentExecution) error
	// GetExecution fetches an agent execution by its numeric identifier.
	// The ctx scopes the read, and id selects the execution to load.
	// Returns the execution record or an error if it cannot be found or read.
	GetExecution(ctx context.Context, id int64) (*types.AgentExecution, error)
	// QueryExecutions lists agent executions matching the provided filters.
	// The ctx scopes the query, and filters define the selection criteria.
	// Returns matching executions or an error if the query fails.
	QueryExecutions(ctx context.Context, filters types.ExecutionFilters) ([]*types.AgentExecution, error)

	// Workflow execution operations
	// StoreWorkflowExecution persists a workflow execution record.
	// The ctx scopes the write, and execution contains the workflow execution data.
	// Returns an error if the workflow execution cannot be saved.
	StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error
	// GetWorkflowExecution retrieves a workflow execution by its identifier.
	// The ctx scopes the read, and executionID identifies the workflow execution.
	// Returns the workflow execution or an error if it is missing or unreadable.
	GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error)
	// QueryWorkflowExecutions lists workflow executions that match the given filters.
	// The ctx scopes the query, and filters specify which executions to include.
	// Returns matching workflow executions or an error if the query fails.
	QueryWorkflowExecutions(ctx context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error)
	// UpdateWorkflowExecution applies an update function to a workflow execution record.
	// The ctx scopes the operation, executionID selects the record, and updateFunc mutates it.
	// Returns an error if the execution cannot be loaded, updated, or saved.
	UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error)) error
	// CreateExecutionRecord creates a primary execution record.
	// The ctx scopes the write, and execution contains the record to insert.
	// Returns an error if the record cannot be created.
	CreateExecutionRecord(ctx context.Context, execution *types.Execution) error
	// GetExecutionRecord retrieves a primary execution record by identifier.
	// The ctx scopes the read, and executionID selects the record to fetch.
	// Returns the execution record or an error if it is missing or unreadable.
	GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error)
	// UpdateExecutionRecord updates a primary execution record through a callback.
	// The ctx scopes the operation, executionID selects the record, and update transforms it.
	// Returns the updated execution or an error if the update cannot be completed.
	UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error)
	// QueryExecutionRecords lists execution records that satisfy the provided filter.
	// The ctx scopes the query, and filter describes the records to include.
	// Returns matching execution records or an error if the query fails.
	QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error)
	// QueryRunSummaries aggregates execution records into run-level summaries.
	// The ctx scopes the query, and filter limits which executions are aggregated.
	// Returns summaries, a total count, and an error if aggregation fails.
	QueryRunSummaries(ctx context.Context, filter types.ExecutionFilter) ([]*RunSummaryAggregation, int, error)
	// RegisterExecutionWebhook stores webhook metadata for an execution.
	// The ctx scopes the write, and webhook contains the registration details.
	// Returns an error if the webhook cannot be registered.
	RegisterExecutionWebhook(ctx context.Context, webhook *types.ExecutionWebhook) error
	// GetExecutionWebhook retrieves webhook metadata for an execution.
	// The ctx scopes the read, and executionID identifies the webhook to load.
	// Returns the webhook record or an error if it is missing or unreadable.
	GetExecutionWebhook(ctx context.Context, executionID string) (*types.ExecutionWebhook, error)
	// ListDueExecutionWebhooks returns pending execution webhooks ready for delivery.
	// The ctx scopes the query, and limit caps how many due webhooks to return.
	// Returns due webhooks or an error if the lookup fails.
	ListDueExecutionWebhooks(ctx context.Context, limit int) ([]*types.ExecutionWebhook, error)
	// TryMarkExecutionWebhookInFlight marks a due webhook as being processed.
	// The ctx scopes the update, executionID selects the webhook, and now timestamps the attempt.
	// Returns true if the webhook was claimed, or false with an error on failure.
	TryMarkExecutionWebhookInFlight(ctx context.Context, executionID string, now time.Time) (bool, error)
	// UpdateExecutionWebhookState updates stored state for an execution webhook.
	// The ctx scopes the write, executionID selects the webhook, and update carries new state.
	// Returns an error if the webhook state cannot be updated.
	UpdateExecutionWebhookState(ctx context.Context, executionID string, update types.ExecutionWebhookStateUpdate) error
	// HasExecutionWebhook reports whether an execution has a registered webhook.
	// The ctx scopes the lookup, and executionID identifies the execution to check.
	// Returns a boolean result or an error if the check fails.
	HasExecutionWebhook(ctx context.Context, executionID string) (bool, error)
	// ListExecutionWebhooksRegistered reports webhook registration status for executions.
	// The ctx scopes the lookup, and executionIDs lists the executions to inspect.
	// Returns a map keyed by execution ID or an error if the query fails.
	ListExecutionWebhooksRegistered(ctx context.Context, executionIDs []string) (map[string]bool, error)
	// StoreExecutionWebhookEvent persists a webhook delivery event for an execution.
	// The ctx scopes the write, and event contains the webhook event payload.
	// Returns an error if the event cannot be stored.
	StoreExecutionWebhookEvent(ctx context.Context, event *types.ExecutionWebhookEvent) error
	// ListExecutionWebhookEvents retrieves webhook events for one execution.
	// The ctx scopes the query, and executionID identifies which events to return.
	// Returns the event list or an error if the lookup fails.
	ListExecutionWebhookEvents(ctx context.Context, executionID string) ([]*types.ExecutionWebhookEvent, error)
	// ListExecutionWebhookEventsBatch retrieves webhook events for multiple executions.
	// The ctx scopes the query, and executionIDs identifies the executions to load.
	// Returns events keyed by execution ID or an error if the lookup fails.
	ListExecutionWebhookEventsBatch(ctx context.Context, executionIDs []string) (map[string][]*types.ExecutionWebhookEvent, error)
	// StoreWorkflowExecutionEvent persists an event associated with a workflow execution.
	// The ctx scopes the write, and event contains the workflow execution event data.
	// Returns an error if the event cannot be stored.
	StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error
	// ListWorkflowExecutionEvents lists events for a workflow execution in sequence order.
	// The ctx scopes the query, executionID selects the execution, and afterSeq and limit page results.
	// Returns matching workflow events or an error if the query fails.
	ListWorkflowExecutionEvents(ctx context.Context, executionID string, afterSeq *int64, limit int) ([]*types.WorkflowExecutionEvent, error)
	StoreExecutionLogEntry(ctx context.Context, entry *types.ExecutionLogEntry) error
	ListExecutionLogEntries(ctx context.Context, executionID string, afterSeq *int64, limit int, levels []string, nodeIDs []string, sources []string, query string) ([]*types.ExecutionLogEntry, error)
	PruneExecutionLogEntries(ctx context.Context, executionID string, maxEntries int, olderThan time.Time) error

	// Execution cleanup operations
	// CleanupOldExecutions deletes execution data older than the retention period.
	// The ctx scopes the cleanup, retentionPeriod sets the age threshold, and batchSize limits each pass.
	// Returns the number of cleaned executions or an error if cleanup fails.
	CleanupOldExecutions(ctx context.Context, retentionPeriod time.Duration, batchSize int) (int, error)
	// MarkStaleExecutions marks long-running executions as stale.
	// The ctx scopes the update, staleAfter sets the cutoff age, and limit bounds affected rows.
	// Returns the number marked stale or an error if the update fails.
	MarkStaleExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error)
	// MarkStaleWorkflowExecutions marks long-running workflow executions as stale.
	// The ctx scopes the update, staleAfter sets the cutoff age, and limit bounds affected rows.
	// Returns the number marked stale or an error if the update fails.
	MarkStaleWorkflowExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error)
	// RetryStaleWorkflowExecutions resets stale workflow executions with retry_count < maxRetries
	// back to "pending" status with incremented retry_count. Returns IDs of retried executions.
	RetryStaleWorkflowExecutions(ctx context.Context, staleAfter time.Duration, maxRetries int, limit int) ([]string, error)

	// Workflow cleanup operations - deletes all data related to a workflow ID
	// CleanupWorkflow removes all stored data associated with a workflow.
	// The ctx scopes the operation, workflowID selects the workflow, and dryRun skips destructive changes.
	// Returns a cleanup summary or an error if the operation fails.
	CleanupWorkflow(ctx context.Context, workflowID string, dryRun bool) (*types.WorkflowCleanupResult, error)

	// DAG operations - optimized single-query DAG building
	// QueryWorkflowDAG loads workflow executions needed to assemble a workflow DAG.
	// The ctx scopes the query, and rootWorkflowID identifies the DAG root workflow.
	// Returns workflow execution nodes or an error if the query fails.
	QueryWorkflowDAG(ctx context.Context, rootWorkflowID string) ([]*types.WorkflowExecution, error)

	// Workflow operations
	// CreateOrUpdateWorkflow stores a workflow, creating it or replacing existing state.
	// The ctx scopes the write, and workflow contains the workflow definition to persist.
	// Returns an error if the workflow cannot be saved.
	CreateOrUpdateWorkflow(ctx context.Context, workflow *types.Workflow) error
	// GetWorkflow retrieves a workflow by its identifier.
	// The ctx scopes the read, and workflowID selects the workflow to load.
	// Returns the workflow or an error if it is missing or unreadable.
	GetWorkflow(ctx context.Context, workflowID string) (*types.Workflow, error)
	// QueryWorkflows lists workflows that satisfy the provided filters.
	// The ctx scopes the query, and filters define which workflows to include.
	// Returns matching workflows or an error if the query fails.
	QueryWorkflows(ctx context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error)

	// Session operations
	// CreateOrUpdateSession stores a session, creating it or replacing existing state.
	// The ctx scopes the write, and session contains the session data to persist.
	// Returns an error if the session cannot be saved.
	CreateOrUpdateSession(ctx context.Context, session *types.Session) error
	// GetSession retrieves a session by its identifier.
	// The ctx scopes the read, and sessionID selects the session to load.
	// Returns the session or an error if it is missing or unreadable.
	GetSession(ctx context.Context, sessionID string) (*types.Session, error)
	// QuerySessions lists sessions matching the provided filters.
	// The ctx scopes the query, and filters define which sessions to return.
	// Returns matching sessions or an error if the query fails.
	QuerySessions(ctx context.Context, filters types.SessionFilters) ([]*types.Session, error)

	// Memory operations
	// SetMemory stores or replaces a memory record.
	// The ctx scopes the write, and memory contains the scoped memory data to persist.
	// Returns an error if the memory record cannot be saved.
	SetMemory(ctx context.Context, memory *types.Memory) error
	// GetMemory retrieves a memory record by scope, scope ID, and key.
	// The ctx scopes the read, and scope, scopeID, and key identify the memory entry.
	// Returns the memory record or an error if it is missing or unreadable.
	GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error)
	// DeleteMemory removes a memory record by scope, scope ID, and key.
	// The ctx scopes the delete, and scope, scopeID, and key identify the record.
	// Returns an error if the memory record cannot be deleted.
	DeleteMemory(ctx context.Context, scope, scopeID, key string) error
	// ListMemory lists memory records within a scope and scope identifier.
	// The ctx scopes the query, and scope and scopeID identify the memory namespace.
	// Returns matching memory records or an error if the query fails.
	ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error)
	// SetVector stores or replaces a vector record.
	// The ctx scopes the write, and record contains the vector data to persist.
	// Returns an error if the vector record cannot be saved.
	SetVector(ctx context.Context, record *types.VectorRecord) error
	// GetVector retrieves a vector record by scope, scope ID, and key.
	// The ctx scopes the read, and scope, scopeID, and key identify the vector entry.
	// Returns the vector record or an error if it is missing or unreadable.
	GetVector(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error)
	// DeleteVector removes a vector record by scope, scope ID, and key.
	// The ctx scopes the delete, and scope, scopeID, and key identify the vector entry.
	// Returns an error if the vector record cannot be deleted.
	DeleteVector(ctx context.Context, scope, scopeID, key string) error
	// DeleteVectorsByPrefix removes vector records that share a key prefix.
	// The ctx scopes the delete, and scope, scopeID, and prefix define the target set.
	// Returns the number deleted or an error if the operation fails.
	DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error)
	// SimilaritySearch finds the closest vector matches for an embedding query.
	// The ctx scopes the search, scope and scopeID choose the namespace, and queryEmbedding, topK, and filters shape results.
	// Returns ranked matches or an error if the search fails.
	SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error)

	// Event operations
	// StoreEvent persists a memory change event.
	// The ctx scopes the write, and event contains the change event data to record.
	// Returns an error if the event cannot be stored.
	StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error
	// GetEventHistory retrieves memory change events matching a filter.
	// The ctx scopes the query, and filter defines which events to include.
	// Returns matching events or an error if the query fails.
	GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error)

	// Distributed Lock operations
	// AcquireLock attempts to create a distributed lock for a key.
	// The ctx scopes the request, key identifies the lock, and timeout sets its lease duration.
	// Returns the acquired lock or an error if the lock cannot be obtained.
	AcquireLock(ctx context.Context, key string, timeout time.Duration) (*types.DistributedLock, error)
	// ReleaseLock releases a distributed lock by lock identifier.
	// The ctx scopes the request, and lockID identifies the held lock to release.
	// Returns an error if the lock cannot be released.
	ReleaseLock(ctx context.Context, lockID string) error
	// RenewLock extends the lease for an existing distributed lock.
	// The ctx scopes the request, and lockID identifies the lock to renew.
	// Returns the renewed lock state or an error if renewal fails.
	RenewLock(ctx context.Context, lockID string) (*types.DistributedLock, error)
	// GetLockStatus retrieves the current state of a distributed lock key.
	// The ctx scopes the lookup, and key identifies the lock to inspect.
	// Returns the lock status or an error if it cannot be read.
	GetLockStatus(ctx context.Context, key string) (*types.DistributedLock, error)

	// Agent registry
	// RegisterAgent stores agent metadata in the registry.
	// The ctx scopes the write, and agent contains the agent node information to persist.
	// Returns an error if the agent cannot be registered.
	RegisterAgent(ctx context.Context, agent *types.AgentNode) error
	// GetAgent retrieves the current agent record by identifier.
	// The ctx scopes the read, and id identifies the agent to load.
	// Returns the agent record or an error if it is missing or unreadable.
	GetAgent(ctx context.Context, id string) (*types.AgentNode, error)
	// GetAgentVersion retrieves a specific version of an agent record.
	// The ctx scopes the read, and id and version identify the versioned agent entry.
	// Returns the agent version or an error if it is missing or unreadable.
	GetAgentVersion(ctx context.Context, id string, version string) (*types.AgentNode, error)
	// DeleteAgentVersion removes a specific agent version from the registry.
	// The ctx scopes the delete, and id and version identify the versioned agent entry.
	// Returns an error if the agent version cannot be deleted.
	DeleteAgentVersion(ctx context.Context, id string, version string) error
	// ListAgentVersions lists all stored versions for an agent.
	// The ctx scopes the query, and id identifies the agent whose versions to return.
	// Returns version records or an error if the query fails.
	ListAgentVersions(ctx context.Context, id string) ([]*types.AgentNode, error)
	// ListAgents lists registered agents matching the provided filters.
	// The ctx scopes the query, and filters define which agents to include.
	// Returns matching agents or an error if the query fails.
	ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error)
	// ListAgentsByGroup lists agents assigned to a group.
	// The ctx scopes the query, and groupID identifies the group to inspect.
	// Returns matching agents or an error if the query fails.
	ListAgentsByGroup(ctx context.Context, groupID string) ([]*types.AgentNode, error)
	// ListAgentGroups lists agent group summaries for a team.
	// The ctx scopes the query, and teamID identifies the team to inspect.
	// Returns group summaries or an error if the query fails.
	ListAgentGroups(ctx context.Context, teamID string) ([]types.AgentGroupSummary, error)
	// UpdateAgentHealth records a new health status for an agent.
	// The ctx scopes the update, and id and status identify the agent and new health state.
	// Returns an error if the health status cannot be updated.
	UpdateAgentHealth(ctx context.Context, id string, status types.HealthStatus) error
	// UpdateAgentHealthAtomic updates agent health when the last heartbeat matches expectations.
	// The ctx scopes the update, and id, status, and expectedLastHeartbeat control the conditional write.
	// Returns an error if the conditional health update fails.
	UpdateAgentHealthAtomic(ctx context.Context, id string, status types.HealthStatus, expectedLastHeartbeat *time.Time) error
	// UpdateAgentHeartbeat records a heartbeat timestamp for an agent version.
	// The ctx scopes the update, and id, version, and heartbeatTime identify the heartbeat to save.
	// Returns an error if the heartbeat cannot be updated.
	UpdateAgentHeartbeat(ctx context.Context, id string, version string, heartbeatTime time.Time) error
	// UpdateAgentLifecycleStatus updates the lifecycle status of an agent.
	// The ctx scopes the update, and id and status identify the agent and new lifecycle state.
	// Returns an error if the lifecycle status cannot be updated.
	UpdateAgentLifecycleStatus(ctx context.Context, id string, status types.AgentLifecycleStatus) error
	// UpdateAgentVersion marks which version is active for an agent.
	// The ctx scopes the update, and id and version identify the agent and selected version.
	// Returns an error if the active version cannot be updated.
	UpdateAgentVersion(ctx context.Context, id string, version string) error
	// UpdateAgentTrafficWeight changes the traffic weight assigned to an agent version.
	// The ctx scopes the update, and id, version, and weight identify the target routing change.
	// Returns an error if the traffic weight cannot be updated.
	UpdateAgentTrafficWeight(ctx context.Context, id string, version string, weight int) error

	// Configuration Storage (database-backed config files)
	// SetConfig stores a configuration value under a key.
	// The ctx scopes the write, and key, value, and updatedBy describe the config change.
	// Returns an error if the configuration cannot be saved.
	SetConfig(ctx context.Context, key string, value string, updatedBy string) error
	// GetConfig retrieves a stored configuration entry by key.
	// The ctx scopes the read, and key identifies the configuration entry to load.
	// Returns the configuration entry or an error if it is missing or unreadable.
	GetConfig(ctx context.Context, key string) (*ConfigEntry, error)
	// ListConfigs returns all stored configuration entries.
	// The ctx scopes the query and does not require additional parameters.
	// Returns configuration entries or an error if the query fails.
	ListConfigs(ctx context.Context) ([]*ConfigEntry, error)
	// DeleteConfig removes a stored configuration entry by key.
	// The ctx scopes the delete, and key identifies the configuration entry to remove.
	// Returns an error if the configuration cannot be deleted.
	DeleteConfig(ctx context.Context, key string) error

	// Reasoner Performance and History
	// GetReasonerPerformanceMetrics retrieves aggregated performance metrics for a reasoner.
	// The ctx scopes the read, and reasonerID identifies the reasoner to inspect.
	// Returns performance metrics or an error if they cannot be loaded.
	GetReasonerPerformanceMetrics(ctx context.Context, reasonerID string) (*types.ReasonerPerformanceMetrics, error)
	// GetReasonerExecutionHistory retrieves paginated execution history for a reasoner.
	// The ctx scopes the query, and reasonerID, page, and limit define which history page to load.
	// Returns execution history or an error if the query fails.
	GetReasonerExecutionHistory(ctx context.Context, reasonerID string, page, limit int) (*types.ReasonerExecutionHistory, error)

	// Agent Configuration Management
	// StoreAgentConfiguration persists an agent configuration record.
	// The ctx scopes the write, and config contains the configuration to store.
	// Returns an error if the configuration cannot be saved.
	StoreAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error
	// GetAgentConfiguration retrieves configuration for an agent and package pair.
	// The ctx scopes the read, and agentID and packageID identify the configuration to load.
	// Returns the agent configuration or an error if it is missing or unreadable.
	GetAgentConfiguration(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error)
	// QueryAgentConfigurations lists agent configurations that match provided filters.
	// The ctx scopes the query, and filters define which configurations to include.
	// Returns matching configurations or an error if the query fails.
	QueryAgentConfigurations(ctx context.Context, filters types.ConfigurationFilters) ([]*types.AgentConfiguration, error)
	// UpdateAgentConfiguration replaces stored state for an agent configuration.
	// The ctx scopes the write, and config contains the updated configuration values.
	// Returns an error if the configuration cannot be updated.
	UpdateAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error
	// DeleteAgentConfiguration removes configuration for an agent and package pair.
	// The ctx scopes the delete, and agentID and packageID identify the configuration to remove.
	// Returns an error if the configuration cannot be deleted.
	DeleteAgentConfiguration(ctx context.Context, agentID, packageID string) error
	// ValidateAgentConfiguration validates configuration data for an agent package.
	// The ctx scopes the check, and agentID, packageID, and config identify the configuration to validate.
	// Returns validation results or an error if validation cannot be performed.
	ValidateAgentConfiguration(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error)

	// Agent Package Management
	// StoreAgentPackage persists an agent package record.
	// The ctx scopes the write, and pkg contains the package metadata to store.
	// Returns an error if the package cannot be saved.
	StoreAgentPackage(ctx context.Context, pkg *types.AgentPackage) error
	// GetAgentPackage retrieves an agent package by identifier.
	// The ctx scopes the read, and packageID identifies the package to load.
	// Returns the package record or an error if it is missing or unreadable.
	GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error)
	// QueryAgentPackages lists agent packages matching the provided filters.
	// The ctx scopes the query, and filters define which packages to include.
	// Returns matching packages or an error if the query fails.
	QueryAgentPackages(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error)
	// UpdateAgentPackage replaces stored state for an agent package.
	// The ctx scopes the write, and pkg contains the updated package metadata.
	// Returns an error if the package cannot be updated.
	UpdateAgentPackage(ctx context.Context, pkg *types.AgentPackage) error
	// DeleteAgentPackage removes an agent package by identifier.
	// The ctx scopes the delete, and packageID identifies the package to remove.
	// Returns an error if the package cannot be deleted.
	DeleteAgentPackage(ctx context.Context, packageID string) error

	// Real-time features (optional, may be handled by CacheProvider)
	// SubscribeToMemoryChanges opens a stream of memory change events for a scope.
	// The ctx scopes the subscription, and scope and scopeID identify the memory namespace to watch.
	// Returns an event channel or an error if subscription setup fails.
	SubscribeToMemoryChanges(ctx context.Context, scope, scopeID string) (<-chan types.MemoryChangeEvent, error)
	// PublishMemoryChange broadcasts a memory change event to subscribers.
	// The ctx scopes the publish, and event contains the change notification to send.
	// Returns an error if the event cannot be published.
	PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error

	// Execution event bus for real-time updates
	// GetExecutionEventBus returns the in-process event bus for execution updates.
	// This method takes no parameters and exposes the shared execution event bus.
	// Returns the execution event bus pointer.
	GetExecutionEventBus() *events.ExecutionEventBus
	// GetWorkflowExecutionEventBus returns the in-process event bus for workflow execution updates.
	// This method takes no parameters and exposes the shared workflow execution event bus.
	// Returns the workflow execution event bus pointer.
	GetWorkflowExecutionEventBus() *events.EventBus[*types.WorkflowExecutionEvent]
	GetExecutionLogEventBus() *events.EventBus[*types.ExecutionLogEntry]

	// DID Registry operations
	// StoreDID persists DID registry data for a decentralized identifier.
	// The ctx scopes the write, and the DID, document, key, and derivation parameters describe the record to store.
	// Returns an error if the DID record cannot be saved.
	StoreDID(ctx context.Context, did string, didDocument, publicKey, privateKeyRef, derivationPath string) error
	// GetDID retrieves a DID registry entry by decentralized identifier.
	// The ctx scopes the read, and did identifies the registry entry to load.
	// Returns the DID registry entry or an error if it is missing or unreadable.
	GetDID(ctx context.Context, did string) (*types.DIDRegistryEntry, error)
	// ListDIDs lists all stored DID registry entries.
	// The ctx scopes the query and does not require additional parameters.
	// Returns DID registry entries or an error if the query fails.
	ListDIDs(ctx context.Context) ([]*types.DIDRegistryEntry, error)

	// AgentField Server DID operations
	// StoreAgentFieldServerDID stores DID material for an AgentField server.
	// The ctx scopes the write, and the server ID, DID, seed, and rotation timestamps describe the stored record.
	// Returns an error if the server DID cannot be saved.
	StoreAgentFieldServerDID(ctx context.Context, agentfieldServerID, rootDID string, masterSeed []byte, createdAt, lastKeyRotation time.Time) error
	// GetAgentFieldServerDID retrieves DID information for an AgentField server.
	// The ctx scopes the read, and agentfieldServerID identifies the server record to load.
	// Returns the server DID info or an error if it is missing or unreadable.
	GetAgentFieldServerDID(ctx context.Context, agentfieldServerID string) (*types.AgentFieldServerDIDInfo, error)
	// ListAgentFieldServerDIDs lists stored AgentField server DID records.
	// The ctx scopes the query and does not require additional parameters.
	// Returns server DID records or an error if the query fails.
	ListAgentFieldServerDIDs(ctx context.Context) ([]*types.AgentFieldServerDIDInfo, error)

	// Agent DID operations
	// StoreAgentDID stores DID information for an agent.
	// The ctx scopes the write, and the agent, server, key, and derivation parameters describe the DID record.
	// Returns an error if the agent DID cannot be saved.
	StoreAgentDID(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int) error
	// GetAgentDID retrieves DID information for an agent.
	// The ctx scopes the read, and agentID identifies the agent DID record to load.
	// Returns the agent DID info or an error if it is missing or unreadable.
	GetAgentDID(ctx context.Context, agentID string) (*types.AgentDIDInfo, error)
	// ListAgentDIDs lists stored DID records for agents.
	// The ctx scopes the query and does not require additional parameters.
	// Returns agent DID records or an error if the query fails.
	ListAgentDIDs(ctx context.Context) ([]*types.AgentDIDInfo, error)

	// Component DID operations
	// StoreComponentDID stores DID information for a component.
	// The ctx scopes the write, and the component, agent, naming, and derivation parameters describe the DID record.
	// Returns an error if the component DID cannot be saved.
	StoreComponentDID(ctx context.Context, componentID, componentDID, agentDID, componentType, componentName string, derivationIndex int) error
	// GetComponentDID retrieves DID information for a component.
	// The ctx scopes the read, and componentID identifies the component DID record to load.
	// Returns the component DID info or an error if it is missing or unreadable.
	GetComponentDID(ctx context.Context, componentID string) (*types.ComponentDIDInfo, error)
	// ListComponentDIDs lists component DID records associated with an agent DID.
	// The ctx scopes the query, and agentDID identifies the component DID owner to inspect.
	// Returns component DID records or an error if the query fails.
	ListComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error)

	// Multi-step DID operations with transaction safety
	// StoreAgentDIDWithComponents stores an agent DID and its component DIDs atomically.
	// The ctx scopes the transaction, and the agent, server, key, derivation, and components parameters define the stored records.
	// Returns an error if any part of the transaction fails.
	StoreAgentDIDWithComponents(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int, components []ComponentDIDRequest) error

	// Execution VC operations
	// StoreExecutionVC stores verifiable credential data for an execution.
	// The ctx scopes the write, and the identifiers, hashes, document, signature, and storage fields describe the VC record.
	// Returns an error if the execution VC cannot be saved.
	StoreExecutionVC(ctx context.Context, vcID, executionID, workflowID, sessionID, issuerDID, targetDID, callerDID, inputHash, outputHash, status string, vcDocument []byte, signature string, storageURI string, documentSizeBytes int64) error
	// GetExecutionVC retrieves execution verifiable credential data by VC identifier.
	// The ctx scopes the read, and vcID identifies the execution VC record to load.
	// Returns the execution VC info or an error if it is missing or unreadable.
	GetExecutionVC(ctx context.Context, vcID string) (*types.ExecutionVCInfo, error)
	// ListExecutionVCs lists execution verifiable credentials matching the provided filters.
	// The ctx scopes the query, and filters define which execution VC records to include.
	// Returns matching execution VC records or an error if the query fails.
	ListExecutionVCs(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error)
	// ListWorkflowVCStatusSummaries aggregates workflow VC status for workflow identifiers.
	// The ctx scopes the query, and workflowIDs identifies the workflows to summarize.
	// Returns workflow VC status summaries or an error if aggregation fails.
	ListWorkflowVCStatusSummaries(ctx context.Context, workflowIDs []string) ([]*types.WorkflowVCStatusAggregation, error)
	// CountExecutionVCs counts execution VC records that match the provided filters.
	// The ctx scopes the query, and filters define which execution VC records to count.
	// Returns the count or an error if the query fails.
	CountExecutionVCs(ctx context.Context, filters types.VCFilters) (int, error)

	// Workflow VC operations
	// StoreWorkflowVC stores verifiable credential data for a workflow.
	// The ctx scopes the write, and the workflow, status, timing, component, and storage parameters describe the VC record.
	// Returns an error if the workflow VC cannot be saved.
	StoreWorkflowVC(ctx context.Context, workflowVCID, workflowID, sessionID string, componentVCIDs []string, status string, startTime, endTime *time.Time, totalSteps, completedSteps int, storageURI string, documentSizeBytes int64) error
	// GetWorkflowVC retrieves workflow verifiable credential data by VC identifier.
	// The ctx scopes the read, and workflowVCID identifies the workflow VC record to load.
	// Returns the workflow VC info or an error if it is missing or unreadable.
	GetWorkflowVC(ctx context.Context, workflowVCID string) (*types.WorkflowVCInfo, error)
	// ListWorkflowVCs lists workflow verifiable credentials for a workflow.
	// The ctx scopes the query, and workflowID identifies the workflow whose VC records to return.
	// Returns matching workflow VC records or an error if the query fails.
	ListWorkflowVCs(ctx context.Context, workflowID string) ([]*types.WorkflowVCInfo, error)

	// Observability Webhook configuration (singleton pattern)
	// GetObservabilityWebhook retrieves the singleton observability webhook configuration.
	// The ctx scopes the read and does not require additional parameters.
	// Returns the webhook configuration or an error if it is missing or unreadable.
	GetObservabilityWebhook(ctx context.Context) (*types.ObservabilityWebhookConfig, error)
	// SetObservabilityWebhook stores the singleton observability webhook configuration.
	// The ctx scopes the write, and config contains the webhook configuration to persist.
	// Returns an error if the configuration cannot be saved.
	SetObservabilityWebhook(ctx context.Context, config *types.ObservabilityWebhookConfig) error
	// DeleteObservabilityWebhook removes the singleton observability webhook configuration.
	// The ctx scopes the delete and does not require additional parameters.
	// Returns an error if the configuration cannot be deleted.
	DeleteObservabilityWebhook(ctx context.Context) error

	// Observability Dead Letter Queue
	// AddToDeadLetterQueue stores a failed observability event for later inspection.
	// The ctx scopes the write, and event, errorMessage, and retryCount describe the failed delivery.
	// Returns an error if the dead-letter entry cannot be saved.
	AddToDeadLetterQueue(ctx context.Context, event *types.ObservabilityEvent, errorMessage string, retryCount int) error
	// GetDeadLetterQueueCount returns the number of stored dead-letter entries.
	// The ctx scopes the read and does not require additional parameters.
	// Returns the entry count or an error if it cannot be determined.
	GetDeadLetterQueueCount(ctx context.Context) (int64, error)
	// GetDeadLetterQueue retrieves paginated dead-letter entries.
	// The ctx scopes the query, and limit and offset define which entries to return.
	// Returns dead-letter entries or an error if the query fails.
	GetDeadLetterQueue(ctx context.Context, limit, offset int) ([]types.ObservabilityDeadLetterEntry, error)
	// DeleteFromDeadLetterQueue removes selected dead-letter entries by identifier.
	// The ctx scopes the delete, and ids lists the entries to remove.
	// Returns an error if the entries cannot be deleted.
	DeleteFromDeadLetterQueue(ctx context.Context, ids []int64) error
	// ClearDeadLetterQueue removes all stored dead-letter entries.
	// The ctx scopes the delete and does not require additional parameters.
	// Returns an error if the queue cannot be cleared.
	ClearDeadLetterQueue(ctx context.Context) error

	// Access policy operations (tag-based authorization)
	// GetAccessPolicies lists all stored access policies.
	// The ctx scopes the query and does not require additional parameters.
	// Returns access policies or an error if the query fails.
	GetAccessPolicies(ctx context.Context) ([]*types.AccessPolicy, error)
	// GetAccessPolicyByID retrieves an access policy by identifier.
	// The ctx scopes the read, and id identifies the access policy to load.
	// Returns the access policy or an error if it is missing or unreadable.
	GetAccessPolicyByID(ctx context.Context, id int64) (*types.AccessPolicy, error)
	// CreateAccessPolicy stores a new access policy.
	// The ctx scopes the write, and policy contains the access policy data to persist.
	// Returns an error if the policy cannot be created.
	CreateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error
	// UpdateAccessPolicy replaces stored state for an access policy.
	// The ctx scopes the write, and policy contains the updated policy data.
	// Returns an error if the policy cannot be updated.
	UpdateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error
	// DeleteAccessPolicy removes an access policy by identifier.
	// The ctx scopes the delete, and id identifies the access policy to remove.
	// Returns an error if the policy cannot be deleted.
	DeleteAccessPolicy(ctx context.Context, id int64) error

	// Agent Tag VC operations (tag-based PermissionVC)
	// StoreAgentTagVC stores a tag-based verifiable credential for an agent.
	// The ctx scopes the write, and the agent, credential, signature, and timing parameters describe the VC record.
	// Returns an error if the agent tag VC cannot be saved.
	StoreAgentTagVC(ctx context.Context, agentID, agentDID, vcID, vcDocument, signature string, issuedAt time.Time, expiresAt *time.Time) error
	// GetAgentTagVC retrieves the tag-based verifiable credential for an agent.
	// The ctx scopes the read, and agentID identifies the agent VC record to load.
	// Returns the agent tag VC record or an error if it is missing or unreadable.
	GetAgentTagVC(ctx context.Context, agentID string) (*types.AgentTagVCRecord, error)
	// ListAgentTagVCs lists all stored agent tag verifiable credentials.
	// The ctx scopes the query and does not require additional parameters.
	// Returns agent tag VC records or an error if the query fails.
	ListAgentTagVCs(ctx context.Context) ([]*types.AgentTagVCRecord, error)
	// RevokeAgentTagVC revokes the tag-based verifiable credential for an agent.
	// The ctx scopes the update, and agentID identifies the agent VC record to revoke.
	// Returns an error if the credential cannot be revoked.
	RevokeAgentTagVC(ctx context.Context, agentID string) error

	// DID Document operations (did:web resolution)
	// StoreDIDDocument persists a DID document record.
	// The ctx scopes the write, and record contains the DID document data to store.
	// Returns an error if the DID document cannot be saved.
	StoreDIDDocument(ctx context.Context, record *types.DIDDocumentRecord) error
	// GetDIDDocument retrieves a DID document by DID string.
	// The ctx scopes the read, and did identifies the DID document to load.
	// Returns the DID document record or an error if it is missing or unreadable.
	GetDIDDocument(ctx context.Context, did string) (*types.DIDDocumentRecord, error)
	// GetDIDDocumentByAgentID retrieves a DID document associated with an agent.
	// The ctx scopes the read, and agentID identifies the agent-linked DID document to load.
	// Returns the DID document record or an error if it is missing or unreadable.
	GetDIDDocumentByAgentID(ctx context.Context, agentID string) (*types.DIDDocumentRecord, error)
	// RevokeDIDDocument revokes a DID document by DID string.
	// The ctx scopes the update, and did identifies the DID document to revoke.
	// Returns an error if the DID document cannot be revoked.
	RevokeDIDDocument(ctx context.Context, did string) error
	// ListDIDDocuments lists all stored DID document records.
	// The ctx scopes the query and does not require additional parameters.
	// Returns DID document records or an error if the query fails.
	ListDIDDocuments(ctx context.Context) ([]*types.DIDDocumentRecord, error)

	// Agent lifecycle queries (tag approval workflow)
	// ListAgentsByLifecycleStatus lists agents currently in a lifecycle status.
	// The ctx scopes the query, and status identifies which lifecycle state to filter by.
	// Returns matching agents or an error if the query fails.
	ListAgentsByLifecycleStatus(ctx context.Context, status types.AgentLifecycleStatus) ([]*types.AgentNode, error)
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
