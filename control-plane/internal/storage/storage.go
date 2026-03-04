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
// It defines all operations for managing executions, workflows, agents, memory,
// configurations, and various other data entities in the system.
type StorageProvider interface {
	// Lifecycle

	// Initialize prepares the storage backend for use with the given configuration.
	// It should establish connections, create necessary tables/schema, and validate configuration.
	// Returns an error if initialization fails.
	Initialize(ctx context.Context, config StorageConfig) error

	// Close gracefully shuts down the storage backend, releasing any resources
	// such as database connections, file handles, or background goroutines.
	Close(ctx context.Context) error

	// HealthCheck verifies that the storage backend is accessible and functioning correctly.
	// Returns nil if healthy, or an error if there's a problem (e.g., connection failure).
	HealthCheck(ctx context.Context) error

	// Execution operations

	// StoreExecution persists a new agent execution record.
	// Returns an error if the operation fails (e.g., database constraint violation).
	StoreExecution(ctx context.Context, execution *types.AgentExecution) error

	// GetExecution retrieves a single agent execution by its numeric ID.
	// Returns the execution if found, or an error if not found or on database failure.
	GetExecution(ctx context.Context, id int64) (*types.AgentExecution, error)

	// QueryExecutions returns a list of executions matching the provided filters.
	// The filters can specify criteria such as status, agent ID, time range, etc.
	QueryExecutions(ctx context.Context, filters types.ExecutionFilters) ([]*types.AgentExecution, error)

	// Workflow execution operations

	// StoreWorkflowExecution persists a workflow execution record.
	// If a record with the same executionID exists, it will be updated.
	StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error

	// GetWorkflowExecution retrieves a workflow execution by its unique executionID (UUID).
	// Returns the execution if found, or an error if not found or on database failure.
	GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error)

	// QueryWorkflowExecutions returns workflow executions matching the provided filters.
	// Filters can include status, workflow ID, session ID, time ranges, etc.
	QueryWorkflowExecutions(ctx context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error)

	// UpdateWorkflowExecution atomically updates a workflow execution using the provided update function.
	// The function receives the current execution and returns the updated version.
	// Returns the updated execution, or an error if the execution is not found or the update fails.
	UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(execution *types.WorkflowExecution) (*types.WorkflowExecution, error)) error

	// CreateExecutionRecord creates a new execution record for tracking agent reasoner executions.
	// Returns an error if the executionID already exists or on database failure.
	CreateExecutionRecord(ctx context.Context, execution *types.Execution) error

	// GetExecutionRecord retrieves an execution record by its unique executionID (UUID).
	// Returns the record if found, or an error if not found or on database failure.
	GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error)

	// UpdateExecutionRecord atomically updates an execution record using the provided update function.
	// Similar to UpdateWorkflowExecution but for reasoner-level execution tracking.
	// Returns the updated record, or an error if not found or on update failure.
	UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error)

	// QueryExecutionRecords returns execution records matching the provided filter criteria.
	QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error)

	// QueryRunSummaries returns aggregated statistics for workflow runs matching the filter.
	// The returned int is the total count of matching runs (for pagination).
	// The aggregation includes status counts, execution timing, and depth information.
	QueryRunSummaries(ctx context.Context, filter types.ExecutionFilter) ([]*RunSummaryAggregation, int, error)

	// RegisterExecutionWebhook registers a webhook to be called when an execution reaches a terminal state.
	RegisterExecutionWebhook(ctx context.Context, webhook *types.ExecutionWebhook) error

	// GetExecutionWebhook retrieves the webhook registered for a specific execution.
	GetExecutionWebhook(ctx context.Context, executionID string) (*types.ExecutionWebhook, error)

	// ListDueExecutionWebhooks returns webhooks that are due for delivery (pending and ready).
	// The limit parameter controls the maximum number of webhooks to return.
	ListDueExecutionWebhooks(ctx context.Context, limit int) ([]*types.ExecutionWebhook, error)

	// TryMarkExecutionWebhookInFlight atomically marks a webhook as in-flight if it's still pending.
	// Returns true if successfully marked, false if it was already processed or doesn't exist.
	TryMarkExecutionWebhookInFlight(ctx context.Context, executionID string, now time.Time) (bool, error)

	// UpdateExecutionWebhookState updates the state of an execution webhook (e.g., success, failed, retry).
	UpdateExecutionWebhookState(ctx context.Context, executionID string, update types.ExecutionWebhookStateUpdate) error

	// HasExecutionWebhook checks whether a webhook exists for the given executionID.
	HasExecutionWebhook(ctx context.Context, executionID string) (bool, error)

	// ListExecutionWebhooksRegistered returns a map of executionID to whether it has a registered webhook.
	ListExecutionWebhooksRegistered(ctx context.Context, executionIDs []string) (map[string]bool, error)

	// StoreExecutionWebhookEvent stores an event related to webhook delivery (attempt, success, failure).
	StoreExecutionWebhookEvent(ctx context.Context, event *types.ExecutionWebhookEvent) error

	// ListExecutionWebhookEvents returns all webhook events for a specific execution.
	ListExecutionWebhookEvents(ctx context.Context, executionID string) ([]*types.ExecutionWebhookEvent, error)

	// ListExecutionWebhookEventsBatch returns webhook events for multiple executions in a single query.
	// Returns a map keyed by executionID.
	ListExecutionWebhookEventsBatch(ctx context.Context, executionIDs []string) (map[string][]*types.ExecutionWebhookEvent, error)

	// StoreWorkflowExecutionEvent stores an event related to workflow execution (started, completed, failed).
	StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error

	// ListWorkflowExecutionEvents returns workflow execution events for a specific execution.
	// The afterSeq parameter can be used to retrieve events after a specific sequence number.
	// The limit parameter controls the maximum number of events to return.
	ListWorkflowExecutionEvents(ctx context.Context, executionID string, afterSeq *int64, limit int) ([]*types.WorkflowExecutionEvent, error)

	// Execution cleanup operations

	// CleanupOldExecutions removes execution records older than the retention period.
	// The batchSize controls how many records are deleted per transaction.
	// Returns the number of records deleted, or an error if the operation fails.
	CleanupOldExecutions(ctx context.Context, retentionPeriod time.Duration, batchSize int) (int, error)

	// MarkStaleExecutions marks executions that have been running longer than staleAfter as failed.
	// The limit controls how many executions are processed in a single call.
	// Returns the number of executions marked, or an error if the operation fails.
	MarkStaleExecutions(ctx context.Context, staleAfter time.Duration, limit int) (int, error)

	// CleanupWorkflow deletes all data related to a workflow ID.
	// If dryRun is true, returns the would-be-deleted items without actually deleting.
	CleanupWorkflow(ctx context.Context, workflowID string, dryRun bool) (*types.WorkflowCleanupResult, error)

	// QueryWorkflowDAG retrieves all workflow executions belonging to a root workflow.
	// This enables efficient DAG reconstruction in a single query.
	QueryWorkflowDAG(ctx context.Context, rootWorkflowID string) ([]*types.WorkflowExecution, error)

	// Workflow operations

	// CreateOrUpdateWorkflow creates a new workflow or updates an existing one.
	// If the workflowID already exists, it updates the existing record.
	CreateOrUpdateWorkflow(ctx context.Context, workflow *types.Workflow) error

	// GetWorkflow retrieves a workflow by its unique workflowID.
	// Returns the workflow if found, or an error if not found.
	GetWorkflow(ctx context.Context, workflowID string) (*types.Workflow, error)

	// QueryWorkflows returns workflows matching the provided filter criteria.
	QueryWorkflows(ctx context.Context, filters types.WorkflowFilters) ([]*types.Workflow, error)

	// Session operations

	// CreateOrUpdateSession creates a new session or updates an existing one.
	CreateOrUpdateSession(ctx context.Context, session *types.Session) error

	// GetSession retrieves a session by its unique sessionID.
	// Returns the session if found, or an error if not found.
	GetSession(ctx context.Context, sessionID string) (*types.Session, error)

	// QuerySessions returns sessions matching the provided filter criteria.
	QuerySessions(ctx context.Context, filters types.SessionFilters) ([]*types.Session, error)

	// Memory operations

	// SetMemory stores a memory value for a given scope, scopeID, and key.
	// Scope can be "global", "workflow", "session", or "actor".
	// If the key already exists, it updates the existing value.
	SetMemory(ctx context.Context, memory *types.Memory) error

	// GetMemory retrieves a memory value by scope, scopeID, and key.
	// Returns the memory if found, or an error if not found.
	GetMemory(ctx context.Context, scope, scopeID, key string) (*types.Memory, error)

	// DeleteMemory removes a memory value by scope, scopeID, and key.
	// Returns an error if the memory doesn't exist or deletion fails.
	DeleteMemory(ctx context.Context, scope, scopeID, key string) error

	// ListMemory returns all memory values for a given scope and scopeID.
	// Returns an empty slice if no memories exist (never nil).
	ListMemory(ctx context.Context, scope, scopeID string) ([]*types.Memory, error)

	// SetVector stores a vector embedding for semantic search capabilities.
	// The vector is associated with a scope, scopeID, key, and optional metadata.
	SetVector(ctx context.Context, record *types.VectorRecord) error

	// GetVector retrieves a vector record by scope, scopeID, and key.
	GetVector(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error)

	// DeleteVector removes a vector record by scope, scopeID, and key.
	DeleteVector(ctx context.Context, scope, scopeID, key string) error

	// DeleteVectorsByPrefix removes all vectors whose keys start with the given prefix
	// within a specific scope. Returns the count of deleted vectors.
	DeleteVectorsByPrefix(ctx context.Context, scope, scopeID, prefix string) (int, error)

	// SimilaritySearch performs a vector similarity search.
	// It finds the topK most similar vectors based on the query embedding.
	// Optional filters can be applied to narrow results.
	SimilaritySearch(ctx context.Context, scope, scopeID string, queryEmbedding []float32, topK int, filters map[string]interface{}) ([]*types.VectorSearchResult, error)

	// Event operations

	// StoreEvent persists a memory change event for audit trail and event sourcing.
	StoreEvent(ctx context.Context, event *types.MemoryChangeEvent) error

	// GetEventHistory retrieves memory change events matching the provided filter.
	// Filters can specify scope, scopeID, event types, time ranges, etc.
	GetEventHistory(ctx context.Context, filter types.EventFilter) ([]*types.MemoryChangeEvent, error)

	// Distributed Lock operations

	// AcquireLock attempts to acquire a distributed lock with the given key and timeout.
	// Returns the lock token if successful, or an error if the lock is already held or times out.
	AcquireLock(ctx context.Context, key string, timeout time.Duration) (*types.DistributedLock, error)

	// ReleaseLock releases a previously acquired lock using its lockID.
	// Returns an error if the lock doesn't exist or isn't held by the caller.
	ReleaseLock(ctx context.Context, lockID string) error

	// RenewLock extends the expiration time of an existing lock.
	// Returns the updated lock token, or an error if the lock no longer exists.
	RenewLock(ctx context.Context, lockID string) (*types.DistributedLock, error)

	// GetLockStatus retrieves the current status of a lock by its key.
	// Returns nil if no lock exists with that key.
	GetLockStatus(ctx context.Context, key string) (*types.DistributedLock, error)

	// Agent registry

	// RegisterAgent registers a new agent node in the system.
	// Returns an error if the agent ID already exists or validation fails.
	RegisterAgent(ctx context.Context, agent *types.AgentNode) error

	// GetAgent retrieves an agent by its unique ID.
	// Returns the agent if found, or an error if not found.
	GetAgent(ctx context.Context, id string) (*types.AgentNode, error)

	// GetAgentVersion retrieves a specific version of an agent.
	// Returns the agent version if found, or an error if not found.
	GetAgentVersion(ctx context.Context, id string, version string) (*types.AgentNode, error)

	// DeleteAgentVersion removes a specific version of an agent.
	// Returns an error if the version doesn't exist.
	DeleteAgentVersion(ctx context.Context, id string, version string) error

	// ListAgentVersions returns all versions of a specific agent.
	// Returns an empty slice if no versions exist.
	ListAgentVersions(ctx context.Context, id string) ([]*types.AgentNode, error)

	// ListAgents returns agents matching the provided filter criteria.
	// Filters can include health status, lifecycle status, tags, etc.
	ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error)

	// ListAgentsByGroup returns all agents belonging to a specific group.
	ListAgentsByGroup(ctx context.Context, groupID string) ([]*types.AgentNode, error)

	// ListAgentGroups returns summaries of all agent groups for a team.
	ListAgentGroups(ctx context.Context, teamID string) ([]types.AgentGroupSummary, error)

	// UpdateAgentHealth updates the health status of an agent.
	// Status values are typically "healthy", "unhealthy", or "unknown".
	UpdateAgentHealth(ctx context.Context, id string, status types.HealthStatus) error

	// UpdateAgentHealthAtomic atomically updates agent health with a condition on the last heartbeat.
	// This prevents race conditions when multiple processes update health concurrently.
	// Returns an error if the expected heartbeat doesn't match or the agent doesn't exist.
	UpdateAgentHealthAtomic(ctx context.Context, id string, status types.HealthStatus, expectedLastHeartbeat *time.Time) error

	// UpdateAgentHeartbeat updates the last heartbeat timestamp and version for an agent.
	// This is used for health monitoring and to detect stale agents.
	UpdateAgentHeartbeat(ctx context.Context, id string, version string, heartbeatTime time.Time) error

	// UpdateAgentLifecycleStatus updates the lifecycle status of an agent.
	// Statuses include "pending_approval", "approved", "rejected", "running", etc.
	UpdateAgentLifecycleStatus(ctx context.Context, id string, status types.AgentLifecycleStatus) error

	// UpdateAgentVersion updates the active version of an agent.
	UpdateAgentVersion(ctx context.Context, id string, version string) error

	// UpdateAgentTrafficWeight updates the traffic weight for a specific agent version.
	// This is used for gradual rollouts and A/B testing.
	UpdateAgentTrafficWeight(ctx context.Context, id string, version string, weight int) error

	// Configuration

	// SetConfig stores a configuration key-value pair.
	// These are typically system-wide settings.
	SetConfig(ctx context.Context, key string, value interface{}) error

	// GetConfig retrieves a configuration value by key.
	// Returns nil if the key doesn't exist.
	GetConfig(ctx context.Context, key string) (interface{}, error)

	// Reasoner Performance and History

	// GetReasonerPerformanceMetrics retrieves performance metrics for a specific reasoner.
	// Includes statistics like average execution time, success rate, etc.
	GetReasonerPerformanceMetrics(ctx context.Context, reasonerID string) (*types.ReasonerPerformanceMetrics, error)

	// GetReasonerExecutionHistory retrieves paginated execution history for a reasoner.
	// Page numbers start at 1. Limit controls results per page.
	GetReasonerExecutionHistory(ctx context.Context, reasonerID string, page, limit int) (*types.ReasonerExecutionHistory, error)

	// Agent Configuration Management

	// StoreAgentConfiguration stores configuration for a specific agent/package combination.
	StoreAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error

	// GetAgentConfiguration retrieves agent configuration by agentID and packageID.
	GetAgentConfiguration(ctx context.Context, agentID, packageID string) (*types.AgentConfiguration, error)

	// QueryAgentConfigurations returns agent configurations matching the provided filters.
	QueryAgentConfigurations(ctx context.Context, filters types.ConfigurationFilters) ([]*types.AgentConfiguration, error)

	// UpdateAgentConfiguration updates an existing agent configuration.
	UpdateAgentConfiguration(ctx context.Context, config *types.AgentConfiguration) error

	// DeleteAgentConfiguration removes an agent configuration.
	DeleteAgentConfiguration(ctx context.Context, agentID, packageID string) error

	// ValidateAgentConfiguration validates an agent's configuration against its schema.
	// Returns validation result with any errors or warnings.
	ValidateAgentConfiguration(ctx context.Context, agentID, packageID string, config map[string]interface{}) (*types.ConfigurationValidationResult, error)

	// Agent Package Management

	// StoreAgentPackage stores an agent package (versioned collection of skills/reasoners).
	StoreAgentPackage(ctx context.Context, pkg *types.AgentPackage) error

	// GetAgentPackage retrieves an agent package by its packageID.
	GetAgentPackage(ctx context.Context, packageID string) (*types.AgentPackage, error)

	// QueryAgentPackages returns agent packages matching the provided filters.
	QueryAgentPackages(ctx context.Context, filters types.PackageFilters) ([]*types.AgentPackage, error)

	// UpdateAgentPackage updates an existing agent package.
	UpdateAgentPackage(ctx context.Context, pkg *types.AgentPackage) error

	// DeleteAgentPackage removes an agent package.
	DeleteAgentPackage(ctx context.Context, packageID string) error

	// Real-time features (optional, may be handled by CacheProvider)

	// SubscribeToMemoryChanges returns a channel that emits memory change events.
	// The caller should read from the channel until it's closed or the context is cancelled.
	SubscribeToMemoryChanges(ctx context.Context, scope, scopeID string) (<-chan types.MemoryChangeEvent, error)

	// PublishMemoryChange publishes a memory change event to subscribers.
	PublishMemoryChange(ctx context.Context, event types.MemoryChangeEvent) error

	// GetExecutionEventBus returns the event bus for execution-related events.
	GetExecutionEventBus() *events.ExecutionEventBus

	// GetWorkflowExecutionEventBus returns the event bus for workflow execution events.
	GetWorkflowExecutionEventBus() *events.EventBus[*types.WorkflowExecutionEvent]

	// DID Registry operations

	// StoreDID stores a DID (Decentralized Identifier) with its document and keys.
	StoreDID(ctx context.Context, did string, didDocument, publicKey, privateKeyRef, derivationPath string) error

	// GetDID retrieves a DID registry entry by DID string.
	GetDID(ctx context.Context, did string) (*types.DIDRegistryEntry, error)

	// ListDIDs returns all registered DIDs.
	ListDIDs(ctx context.Context) ([]*types.DIDRegistryEntry, error)

	// AgentField Server DID operations

	// StoreAgentFieldServerDID stores the root DID for an AgentField server.
	StoreAgentFieldServerDID(ctx context.Context, agentfieldServerID, rootDID string, masterSeed []byte, createdAt, lastKeyRotation time.Time) error

	// GetAgentFieldServerDID retrieves the DID info for an AgentField server.
	GetAgentFieldServerDID(ctx context.Context, agentfieldServerID string) (*types.AgentFieldServerDIDInfo, error)

	// ListAgentFieldServerDIDs returns all AgentField server DIDs.
	ListAgentFieldServerDIDs(ctx context.Context) ([]*types.AgentFieldServerDIDInfo, error)

	// Agent DID operations

	// StoreAgentDID stores a DID associated with an agent.
	StoreAgentDID(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int) error

	// GetAgentDID retrieves DID information for an agent.
	GetAgentDID(ctx context.Context, agentID string) (*types.AgentDIDInfo, error)

	// ListAgentDIDs returns all agent DIDs.
	ListAgentDIDs(ctx context.Context) ([]*types.AgentDIDInfo, error)

	// Component DID operations

	// StoreComponentDID stores a DID for a component (skill/reasoner) of an agent.
	StoreComponentDID(ctx context.Context, componentID, componentDID, agentDID, componentType, componentName string, derivationIndex int) error

	// GetComponentDID retrieves DID information for a component.
	GetComponentDID(ctx context.Context, componentID string) (*types.ComponentDIDInfo, error)

	// ListComponentDIDs returns all DIDs for components of an agent.
	ListComponentDIDs(ctx context.Context, agentDID string) ([]*types.ComponentDIDInfo, error)

	// StoreAgentDIDWithComponents stores an agent DID along with all its component DIDs atomically.
	// This ensures all or nothing are stored, maintaining referential integrity.
	StoreAgentDIDWithComponents(ctx context.Context, agentID, agentDID, agentfieldServerDID, publicKeyJWK string, derivationIndex int, components []ComponentDIDRequest) error

	// Execution VC operations

	// StoreExecutionVC stores a Verifiable Credential for an execution.
	StoreExecutionVC(ctx context.Context, vcID, executionID, workflowID, sessionID, issuerDID, targetDID, callerDID, inputHash, outputHash, status string, vcDocument []byte, signature string, storageURI string, documentSizeBytes int64) error

	// GetExecutionVC retrieves a VC by its ID.
	GetExecutionVC(ctx context.Context, vcID string) (*types.ExecutionVCInfo, error)

	// ListExecutionVCs returns execution VCs matching the provided filters.
	ListExecutionVCs(ctx context.Context, filters types.VCFilters) ([]*types.ExecutionVCInfo, error)

	// ListWorkflowVCStatusSummaries returns VC status summaries for multiple workflows.
	ListWorkflowVCStatusSummaries(ctx context.Context, workflowIDs []string) ([]*types.WorkflowVCStatusAggregation, error)

	// CountExecutionVCs returns the count of execution VCs matching filters.
	CountExecutionVCs(ctx context.Context, filters types.VCFilters) (int, error)

	// Workflow VC operations

	// StoreWorkflowVC stores a Verifiable Credential for a workflow execution.
	StoreWorkflowVC(ctx context.Context, workflowVCID, workflowID, sessionID string, componentVCIDs []string, status string, startTime, endTime *time.Time, totalSteps, completedSteps int, storageURI string, documentSizeBytes int64) error

	// GetWorkflowVC retrieves a workflow VC by its ID.
	GetWorkflowVC(ctx context.Context, workflowVCID string) (*types.WorkflowVCInfo, error)

	// ListWorkflowVCs returns all VCs for a specific workflow.
	ListWorkflowVCs(ctx context.Context, workflowID string) ([]*types.WorkflowVCInfo, error)

	// Observability Webhook configuration (singleton pattern)

	// GetObservabilityWebhook retrieves the configured observability webhook (if any).
	// Returns nil if no webhook is configured.
	GetObservabilityWebhook(ctx context.Context) (*types.ObservabilityWebhookConfig, error)

	// SetObservabilityWebhook sets or updates the observability webhook configuration.
	// There can only be one webhook, so this replaces any existing configuration.
	SetObservabilityWebhook(ctx context.Context, config *types.ObservabilityWebhookConfig) error

	// DeleteObservabilityWebhook removes the observability webhook configuration.
	DeleteObservabilityWebhook(ctx context.Context) error

	// Observability Dead Letter Queue

	// AddToDeadLetterQueue adds a failed event to the dead letter queue for later inspection/retry.
	AddToDeadLetterQueue(ctx context.Context, event *types.ObservabilityEvent, errorMessage string, retryCount int) error

	// GetDeadLetterQueueCount returns the total number of items in the dead letter queue.
	GetDeadLetterQueueCount(ctx context.Context) (int64, error)

	// GetDeadLetterQueue retrieves paginated items from the dead letter queue.
	GetDeadLetterQueue(ctx context.Context, limit, offset int) ([]types.ObservabilityDeadLetterEntry, error)

	// DeleteFromDeadLetterQueue removes specific items from the dead letter queue by IDs.
	DeleteFromDeadLetterQueue(ctx context.Context, ids []int64) error

	// ClearDeadLetterQueue removes all items from the dead letter queue.
	ClearDeadLetterQueue(ctx context.Context) error

	// Access policy operations (tag-based authorization)

	// GetAccessPolicies returns all access policies.
	GetAccessPolicies(ctx context.Context) ([]*types.AccessPolicy, error)

	// GetAccessPolicyByID retrieves a specific access policy by its ID.
	GetAccessPolicyByID(ctx context.Context, id int64) (*types.AccessPolicy, error)

	// CreateAccessPolicy creates a new access policy.
	CreateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error

	// UpdateAccessPolicy updates an existing access policy.
	UpdateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error

	// DeleteAccessPolicy removes an access policy by its ID.
	DeleteAccessPolicy(ctx context.Context, id int64) error

	// Agent Tag VC operations (tag-based PermissionVC)

	// StoreAgentTagVC stores a Verifiable Credential for agent permissions/tags.
	StoreAgentTagVC(ctx context.Context, agentID, agentDID, vcID, vcDocument, signature string, issuedAt time.Time, expiresAt *time.Time) error

	// GetAgentTagVC retrieves the tag VC for a specific agent.
	GetAgentTagVC(ctx context.Context, agentID string) (*types.AgentTagVCRecord, error)

	// ListAgentTagVCs returns all agent tag VCs.
	ListAgentTagVCs(ctx context.Context) ([]*types.AgentTagVCRecord, error)

	// RevokeAgentTagVC revokes the tag VC for a specific agent.
	RevokeAgentTagVC(ctx context.Context, agentID string) error

	// DID Document operations (did:web resolution)

	// StoreDIDDocument stores a DID document for did:web resolution.
	StoreDIDDocument(ctx context.Context, record *types.DIDDocumentRecord) error

	// GetDIDDocument retrieves a DID document by DID string.
	GetDIDDocument(ctx context.Context, did string) (*types.DIDDocumentRecord, error)

	// GetDIDDocumentByAgentID retrieves a DID document by associated agent ID.
	GetDIDDocumentByAgentID(ctx context.Context, agentID string) (*types.DIDDocumentRecord, error)

	// RevokeDIDDocument marks a DID document as revoked.
	RevokeDIDDocument(ctx context.Context, did string) error

	// ListDIDDocuments returns all DID documents.
	ListDIDDocuments(ctx context.Context) ([]*types.DIDDocumentRecord, error)

	// ListAgentsByLifecycleStatus returns all agents with a specific lifecycle status.
	// Common statuses include "pending_approval", "approved", "rejected".
	ListAgentsByLifecycleStatus(ctx context.Context, status types.AgentLifecycleStatus) ([]*types.AgentNode, error)
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
	GetVector(ctx context.Context, scope, scopeID, key string) (*types.VectorRecord, error)
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
	GetAgentVersion(ctx context.Context, id string, version string) (*types.AgentNode, error)
	DeleteAgentVersion(ctx context.Context, id string, version string) error
	ListAgentVersions(ctx context.Context, id string) ([]*types.AgentNode, error)
	ListAgents(ctx context.Context, filters types.AgentFilters) ([]*types.AgentNode, error)
	ListAgentsByGroup(ctx context.Context, groupID string) ([]*types.AgentNode, error)
	ListAgentGroups(ctx context.Context, teamID string) ([]types.AgentGroupSummary, error)
	UpdateAgentHealth(ctx context.Context, id string, status types.HealthStatus) error
	UpdateAgentHealthAtomic(ctx context.Context, id string, status types.HealthStatus, expectedLastHeartbeat *time.Time) error
	UpdateAgentHeartbeat(ctx context.Context, id string, version string, heartbeatTime time.Time) error
	UpdateAgentLifecycleStatus(ctx context.Context, id string, status types.AgentLifecycleStatus) error
	UpdateAgentVersion(ctx context.Context, id string, version string) error
	UpdateAgentTrafficWeight(ctx context.Context, id string, version string, weight int) error

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

	// Observability Webhook configuration (singleton pattern)
	GetObservabilityWebhook(ctx context.Context) (*types.ObservabilityWebhookConfig, error)
	SetObservabilityWebhook(ctx context.Context, config *types.ObservabilityWebhookConfig) error
	DeleteObservabilityWebhook(ctx context.Context) error

	// Observability Dead Letter Queue
	AddToDeadLetterQueue(ctx context.Context, event *types.ObservabilityEvent, errorMessage string, retryCount int) error
	GetDeadLetterQueueCount(ctx context.Context) (int64, error)
	GetDeadLetterQueue(ctx context.Context, limit, offset int) ([]types.ObservabilityDeadLetterEntry, error)
	DeleteFromDeadLetterQueue(ctx context.Context, ids []int64) error
	ClearDeadLetterQueue(ctx context.Context) error

	// Access policy operations (tag-based authorization)
	GetAccessPolicies(ctx context.Context) ([]*types.AccessPolicy, error)
	GetAccessPolicyByID(ctx context.Context, id int64) (*types.AccessPolicy, error)
	CreateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error
	UpdateAccessPolicy(ctx context.Context, policy *types.AccessPolicy) error
	DeleteAccessPolicy(ctx context.Context, id int64) error

	// Agent Tag VC operations (tag-based PermissionVC)
	StoreAgentTagVC(ctx context.Context, agentID, agentDID, vcID, vcDocument, signature string, issuedAt time.Time, expiresAt *time.Time) error
	GetAgentTagVC(ctx context.Context, agentID string) (*types.AgentTagVCRecord, error)
	ListAgentTagVCs(ctx context.Context) ([]*types.AgentTagVCRecord, error)
	RevokeAgentTagVC(ctx context.Context, agentID string) error

	// DID Document operations (did:web resolution)
	StoreDIDDocument(ctx context.Context, record *types.DIDDocumentRecord) error
	GetDIDDocument(ctx context.Context, did string) (*types.DIDDocumentRecord, error)
	GetDIDDocumentByAgentID(ctx context.Context, agentID string) (*types.DIDDocumentRecord, error)
	RevokeDIDDocument(ctx context.Context, did string) error
	ListDIDDocuments(ctx context.Context) ([]*types.DIDDocumentRecord, error)

	// Agent lifecycle queries (tag approval workflow)
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
