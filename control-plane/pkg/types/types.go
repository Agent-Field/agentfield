package types

import (
	"encoding/json"
	"time"
)

// AgentExecution represents a single execution of a reasoner or skill.
type AgentExecution struct {
	ID          int64   `json:"id" db:"id"`
	WorkflowID  string  `json:"workflow_id" db:"workflow_id"`
	SessionID   *string `json:"session_id,omitempty" db:"session_id"`
	AgentNodeID string  `json:"agent_node_id" db:"agent_node_id"`
	ReasonerID  string  `json:"reasoner_id" db:"reasoner_id"`

	InputData  json.RawMessage `json:"input_data" db:"input_data"`
	OutputData json.RawMessage `json:"output_data" db:"output_data"`
	InputSize  int             `json:"input_size" db:"input_size"`
	OutputSize int             `json:"output_size" db:"output_size"`

	DurationMS   int     `json:"duration_ms" db:"duration_ms"`
	Status       string  `json:"status" db:"status"`
	ErrorMessage *string `json:"error_message,omitempty" db:"error_message"`

	UserID *string `json:"user_id,omitempty" db:"user_id"`
	NodeID *string `json:"node_id,omitempty" db:"node_id"`

	Metadata ExecutionMetadata `json:"metadata" db:"metadata"`

	CreatedAt time.Time `json:"created_at" db:"created_at"`
}

// ExecutionMetadata holds extensible metadata for an agent execution.
type ExecutionMetadata struct {
	Cost        *CostMetadata          `json:"cost,omitempty"`
	ABTest      *ABTestMetadata        `json:"ab_test,omitempty"`
	Model       *ModelMetadata         `json:"model,omitempty"`
	Compliance  *ComplianceMetadata    `json:"compliance,omitempty"`
	Performance *PerformanceMetadata   `json:"performance,omitempty"`
	Custom      map[string]interface{} `json:"custom,omitempty"`
}

// CostMetadata holds cost-related metadata.
type CostMetadata struct {
	USD        *float64 `json:"usd,omitempty"`
	Currency   string   `json:"currency,omitempty"`
	Provider   string   `json:"provider,omitempty"`
	TokensUsed *int     `json:"tokens_used,omitempty"`
}

// ABTestMetadata holds A/B testing metadata.
type ABTestMetadata struct {
	TestID       string `json:"test_id"`
	Variant      string `json:"variant"`
	ControlGroup bool   `json:"control_group"`
}

// ModelMetadata holds model-related metadata.
type ModelMetadata struct {
	Name        string   `json:"name"`
	Version     string   `json:"version,omitempty"`
	Provider    string   `json:"provider"`
	Temperature *float64 `json:"temperature,omitempty"`
	MaxTokens   *int     `json:"max_tokens,omitempty"`
}

// ComplianceMetadata holds compliance and audit metadata.
type ComplianceMetadata struct {
	AuditRequired bool   `json:"audit_required"`
	DataClass     string `json:"data_class,omitempty"`
	RetentionDays *int   `json:"retention_days,omitempty"`
	PIIDetected   bool   `json:"pii_detected"`
}

// PerformanceMetadata holds performance-related metadata.
type PerformanceMetadata struct {
	QueueTimeMS   *int `json:"queue_time_ms,omitempty"`
	NetworkTimeMS *int `json:"network_time_ms,omitempty"`
	CacheHit      bool `json:"cache_hit"`
	RetryCount    int  `json:"retry_count"`
}

// Memory represents a piece of memory stored in the system.
type Memory struct {
	Scope       string          `json:"scope" db:"scope"`
	ScopeID     string          `json:"scope_id" db:"scope_id"`
	Key         string          `json:"key" db:"key"`
	Data        json.RawMessage `json:"data" db:"data"`
	AccessLevel string          `json:"access_level" db:"access_level"`

	TTL       *time.Duration `json:"ttl,omitempty" db:"ttl"`
	CreatedAt time.Time      `json:"created_at" db:"created_at"`
	UpdatedAt time.Time      `json:"updated_at" db:"updated_at"`

	Metadata MemoryMetadata `json:"metadata" db:"metadata"`
}

// MemoryMetadata holds extensible metadata for memory.
type MemoryMetadata struct {
	Encryption    *EncryptionMetadata    `json:"encryption,omitempty"`
	Replication   *ReplicationMetadata   `json:"replication,omitempty"`
	Analytics     *AnalyticsMetadata     `json:"analytics,omitempty"`
	AccessControl *AccessControlMetadata `json:"access_control,omitempty"`
	Custom        map[string]interface{} `json:"custom,omitempty"`
}

// EncryptionMetadata holds encryption-related metadata.
type EncryptionMetadata struct {
	Encrypted bool   `json:"encrypted"`
	KeyID     string `json:"key_id,omitempty"`
	Algorithm string `json:"algorithm,omitempty"`
}

// ReplicationMetadata holds replication-related metadata.
type ReplicationMetadata struct {
	Replicated bool     `json:"replicated"`
	Regions    []string `json:"regions,omitempty"`
	SyncStatus string   `json:"sync_status,omitempty"`
}

// AnalyticsMetadata holds analytics-related metadata for memory.
type AnalyticsMetadata struct {
	AccessCount  int       `json:"access_count"`
	LastAccessed time.Time `json:"last_accessed,omitempty"`
}

// AccessControlMetadata holds access control metadata for memory.
type AccessControlMetadata struct {
	RequiredRoles  []string `json:"required_roles,omitempty"`
	TeamRestricted bool     `json:"team_restricted"`
	AuditAccess    bool     `json:"audit_access"`
}

// AgentNode represents a registered agent service.
type AgentNode struct {
	ID      string `json:"id" db:"id"`
	TeamID  string `json:"team_id" db:"team_id"`
	BaseURL string `json:"base_url" db:"base_url"`
	Version string `json:"version" db:"version"`

	// Serverless support
	DeploymentType string  `json:"deployment_type" db:"deployment_type"`         // "long_running" or "serverless"
	InvocationURL  *string `json:"invocation_url,omitempty" db:"invocation_url"` // For serverless agents

	CallbackDiscovery *CallbackDiscoveryInfo `json:"callback_discovery,omitempty" db:"-"`

	Reasoners           []ReasonerDefinition `json:"reasoners" db:"reasoners"`
	Skills              []SkillDefinition    `json:"skills" db:"skills"`
	CommunicationConfig CommunicationConfig  `json:"communication_config" db:"communication_config"`

	HealthStatus    HealthStatus         `json:"health_status" db:"health_status"`
	LifecycleStatus AgentLifecycleStatus `json:"lifecycle_status" db:"lifecycle_status"`
	LastHeartbeat   time.Time            `json:"last_heartbeat" db:"last_heartbeat"`
	RegisteredAt    time.Time            `json:"registered_at" db:"registered_at"`

	Features AgentFeatures `json:"features" db:"features"`
	Metadata AgentMetadata `json:"metadata" db:"metadata"`
}

// CallbackDiscoveryInfo captures how the AgentField server resolved an agent callback URL.
type CallbackDiscoveryInfo struct {
	Mode        string               `json:"mode,omitempty"`
	Preferred   string               `json:"preferred,omitempty"`
	Resolved    string               `json:"resolved,omitempty"`
	Candidates  []string             `json:"candidates,omitempty"`
	Tests       []CallbackTestResult `json:"tests,omitempty"`
	SubmittedAt string               `json:"submitted_at,omitempty"`
}

// CallbackTestResult describes the outcome of probing a callback candidate URL.
type CallbackTestResult struct {
	URL       string `json:"url"`
	Success   bool   `json:"success"`
	Status    int    `json:"status,omitempty"`
	Endpoint  string `json:"endpoint,omitempty"`
	Error     string `json:"error,omitempty"`
	LatencyMS int64  `json:"latency_ms,omitempty"`
}

// ReasonerDefinition defines a reasoner provided by an agent node.
type ReasonerDefinition struct {
	ID           string          `json:"id"`
	InputSchema  json.RawMessage `json:"input_schema"`
	OutputSchema json.RawMessage `json:"output_schema"`
	MemoryConfig MemoryConfig    `json:"memory_config"`
}

// SkillDefinition defines a skill provided by an agent node.
type SkillDefinition struct {
	ID          string          `json:"id"`
	InputSchema json.RawMessage `json:"input_schema"`
	Tags        []string        `json:"tags"`
}

// MemoryConfig defines memory configuration for a reasoner.
type MemoryConfig struct {
	AutoInject      []string `json:"auto_inject"`
	MemoryRetention string   `json:"memory_retention"`
	CacheResults    bool     `json:"cache_results"`
}

// CommunicationConfig defines communication protocols supported by an agent node.
type CommunicationConfig struct {
	Protocols         []string `json:"protocols"`
	WebSocketEndpoint string   `json:"websocket_endpoint"`
	HeartbeatInterval string   `json:"heartbeat_interval"`
}

// HealthStatus represents the health status of an agent node.
type HealthStatus string

const (
	HealthStatusActive   HealthStatus = "active"
	HealthStatusInactive HealthStatus = "inactive"
	HealthStatusUnknown  HealthStatus = "unknown"
)

// AgentLifecycleStatus represents the lifecycle status of an agent node.
type AgentLifecycleStatus string

const (
	AgentStatusStarting AgentLifecycleStatus = "starting" // Initializing (covers registering + initializing)
	AgentStatusReady    AgentLifecycleStatus = "ready"    // Fully operational
	AgentStatusDegraded AgentLifecycleStatus = "degraded" // Partial functionality
	AgentStatusOffline  AgentLifecycleStatus = "offline"  // Not responding
)

// AgentStatus represents the unified status model for agent nodes.
// This simplifies the current complex status system by providing a single source of truth.
type AgentStatus struct {
	// Core status fields
	State       AgentState `json:"state"`        // Primary state: active, inactive, starting, stopping
	HealthScore int        `json:"health_score"` // Health score from 0-100
	LastSeen    time.Time  `json:"last_seen"`    // Last heartbeat timestamp

	// Lifecycle information
	LifecycleStatus AgentLifecycleStatus `json:"lifecycle_status"` // Backward compatibility
	HealthStatus    HealthStatus         `json:"health_status"`    // Backward compatibility

	// MCP status (optional)
	MCPStatus *MCPStatusInfo `json:"mcp_status,omitempty"` // MCP server status if available

	// Transition tracking
	StateTransition *StateTransition `json:"state_transition,omitempty"` // Current transition if any

	// Metadata
	LastUpdated  time.Time    `json:"last_updated"`            // When this status was last updated
	LastVerified *time.Time   `json:"last_verified,omitempty"` // When live health check was last performed
	Source       StatusSource `json:"source"`                  // Source of this status update
}

// AgentState represents the primary state of an agent (simplified from complex status types)
type AgentState string

const (
	AgentStateActive   AgentState = "active"   // Agent is running and healthy
	AgentStateInactive AgentState = "inactive" // Agent is not responding or offline
	AgentStateStarting AgentState = "starting" // Agent is initializing
	AgentStateStopping AgentState = "stopping" // Agent is shutting down
)

// MCPStatusInfo represents MCP server status information
type MCPStatusInfo struct {
	TotalServers   int       `json:"total_servers"`
	RunningServers int       `json:"running_servers"`
	TotalTools     int       `json:"total_tools"`
	OverallHealth  float64   `json:"overall_health"`
	ServiceStatus  string    `json:"service_status"` // "ready", "degraded", "unavailable"
	LastChecked    time.Time `json:"last_checked"`
}

// StateTransition represents an ongoing state transition
type StateTransition struct {
	From      AgentState `json:"from"`
	To        AgentState `json:"to"`
	StartedAt time.Time  `json:"started_at"`
	Reason    string     `json:"reason,omitempty"`
}

// StatusSource indicates where a status update originated
type StatusSource string

const (
	StatusSourceHeartbeat   StatusSource = "heartbeat"    // From agent heartbeat
	StatusSourceHealthCheck StatusSource = "health_check" // From health monitor
	StatusSourceManual      StatusSource = "manual"       // Manual update
	StatusSourceReconcile   StatusSource = "reconcile"    // From reconciliation service
	StatusSourcePresence    StatusSource = "presence"     // From presence lease expirations
)

// AgentStatusUpdate represents a status update request
type AgentStatusUpdate struct {
	State           *AgentState           `json:"state,omitempty"`
	HealthScore     *int                  `json:"health_score,omitempty"`
	LifecycleStatus *AgentLifecycleStatus `json:"lifecycle_status,omitempty"`
	MCPStatus       *MCPStatusInfo        `json:"mcp_status,omitempty"`
	Source          StatusSource          `json:"source"`
	Reason          string                `json:"reason,omitempty"`
}

// Helper methods for AgentStatus

// IsHealthy returns true if the agent is in a healthy state
func (as *AgentStatus) IsHealthy() bool {
	return as.State == AgentStateActive && as.HealthScore >= 70
}

// IsTransitioning returns true if the agent is currently transitioning between states
func (as *AgentStatus) IsTransitioning() bool {
	return as.StateTransition != nil
}

// GetEffectiveState returns the current effective state, considering transitions
func (as *AgentStatus) GetEffectiveState() AgentState {
	if as.IsTransitioning() {
		return as.StateTransition.To
	}
	return as.State
}

// ToLegacyHealthStatus converts the unified status to legacy HealthStatus for backward compatibility
func (as *AgentStatus) ToLegacyHealthStatus() HealthStatus {
	switch as.State {
	case AgentStateActive:
		return HealthStatusActive
	case AgentStateInactive, AgentStateStopping:
		return HealthStatusInactive
	default:
		return HealthStatusUnknown
	}
}

// ToLegacyLifecycleStatus converts the unified status to legacy AgentLifecycleStatus for backward compatibility
func (as *AgentStatus) ToLegacyLifecycleStatus() AgentLifecycleStatus {
	// If we have explicit lifecycle status, use it
	if as.LifecycleStatus != "" {
		return as.LifecycleStatus
	}

	// Otherwise, derive from state
	switch as.State {
	case AgentStateActive:
		if as.HealthScore >= 90 {
			return AgentStatusReady
		} else if as.HealthScore >= 50 {
			return AgentStatusDegraded
		}
		return AgentStatusReady
	case AgentStateStarting:
		return AgentStatusStarting
	case AgentStateInactive, AgentStateStopping:
		return AgentStatusOffline
	default:
		return AgentStatusOffline
	}
}

// NewAgentStatus creates a new AgentStatus with default values
func NewAgentStatus(state AgentState, source StatusSource) *AgentStatus {
	now := time.Now()
	return &AgentStatus{
		State:       state,
		HealthScore: 100, // Default to healthy
		LastSeen:    now,
		LastUpdated: now,
		Source:      source,
		// Set backward compatibility fields
		HealthStatus:    HealthStatusUnknown,
		LifecycleStatus: AgentStatusStarting,
	}
}

// FromLegacyStatus creates a unified AgentStatus from legacy status fields
func FromLegacyStatus(healthStatus HealthStatus, lifecycleStatus AgentLifecycleStatus, lastHeartbeat time.Time) *AgentStatus {
	now := time.Now()

	// Determine primary state from legacy statuses
	var state AgentState
	var healthScore int

	switch healthStatus {
	case HealthStatusActive:
		state = AgentStateActive
		healthScore = 85 // Good health
	case HealthStatusInactive:
		state = AgentStateInactive
		healthScore = 0 // No health
	default:
		// Derive from lifecycle status
		switch lifecycleStatus {
		case AgentStatusReady:
			state = AgentStateActive
			healthScore = 90
		case AgentStatusStarting:
			state = AgentStateStarting
			healthScore = 50
		case AgentStatusDegraded:
			state = AgentStateActive
			healthScore = 60
		default:
			state = AgentStateInactive
			healthScore = 0
		}
	}

	return &AgentStatus{
		State:           state,
		HealthScore:     healthScore,
		LastSeen:        lastHeartbeat,
		LifecycleStatus: lifecycleStatus,
		HealthStatus:    healthStatus,
		LastUpdated:     now,
		Source:          StatusSourceReconcile,
	}
}

// UpdateFromHeartbeat updates the status based on heartbeat data
func (as *AgentStatus) UpdateFromHeartbeat(lifecycleStatus *AgentLifecycleStatus, mcpStatus *MCPStatusInfo) {
	now := time.Now()
	as.LastSeen = now
	as.LastUpdated = now
	as.Source = StatusSourceHeartbeat

	// Update lifecycle status if provided
	if lifecycleStatus != nil {
		as.LifecycleStatus = *lifecycleStatus

		// Update primary state based on lifecycle status
		switch *lifecycleStatus {
		case AgentStatusReady:
			as.State = AgentStateActive
			if as.HealthScore < 70 {
				as.HealthScore = 85 // Boost health score for ready agents
			}
		case AgentStatusStarting:
			as.State = AgentStateStarting
			as.HealthScore = 50
		case AgentStatusDegraded:
			as.State = AgentStateActive
			as.HealthScore = 60
		case AgentStatusOffline:
			as.State = AgentStateInactive
			as.HealthScore = 0
		}
	}

	// Update MCP status if provided
	if mcpStatus != nil {
		as.MCPStatus = mcpStatus

		// Adjust health score based on MCP health
		if mcpStatus.OverallHealth > 0 {
			mcpHealthContribution := int(mcpStatus.OverallHealth * 20) // Up to 20 points from MCP
			as.HealthScore = min(100, as.HealthScore+mcpHealthContribution)
		}
	}

	// Update backward compatibility fields
	as.HealthStatus = as.ToLegacyHealthStatus()
}

// StartTransition begins a state transition
func (as *AgentStatus) StartTransition(to AgentState, reason string) {
	as.StateTransition = &StateTransition{
		From:      as.State,
		To:        to,
		StartedAt: time.Now(),
		Reason:    reason,
	}
	as.LastUpdated = time.Now()
}

// CompleteTransition completes the current state transition
func (as *AgentStatus) CompleteTransition() {
	if as.StateTransition != nil {
		as.State = as.StateTransition.To
		as.StateTransition = nil
		as.LastUpdated = time.Now()

		// Update backward compatibility fields
		as.HealthStatus = as.ToLegacyHealthStatus()
		as.LifecycleStatus = as.ToLegacyLifecycleStatus()
	}
}

// min helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// AgentFeatures holds feature flags for an agent node.
type AgentFeatures struct {
	ABTesting       bool            `json:"ab_testing"`
	AdvancedMetrics bool            `json:"advanced_metrics"`
	Compliance      bool            `json:"compliance"`
	AuditLogging    bool            `json:"audit_logging"`
	RoleBasedAccess bool            `json:"role_based_access"`
	Experimental    map[string]bool `json:"experimental,omitempty"`
}

// AgentMetadata holds extensible metadata for an agent node.
type AgentMetadata struct {
	Deployment  *DeploymentMetadata       `json:"deployment,omitempty"`
	Performance *AgentPerformanceMetadata `json:"performance,omitempty"`
	Custom      map[string]interface{}    `json:"custom,omitempty"`
}

// DeploymentMetadata holds deployment-related metadata for an agent node.
type DeploymentMetadata struct {
	Environment string            `json:"environment"`
	Platform    string            `json:"platform"`
	Region      string            `json:"region,omitempty"`
	Tags        map[string]string `json:"tags,omitempty"`
}

// AgentPerformanceMetadata holds performance-related metadata for an agent node.
type AgentPerformanceMetadata struct {
	LatencyMS    int `json:"latency_ms"`
	ThroughputPS int `json:"throughput_ps"`
}

// ExecutionFilters holds filters for querying agent executions.
type ExecutionFilters struct {
	WorkflowID  *string    `json:"workflow_id,omitempty"`
	SessionID   *string    `json:"session_id,omitempty"`
	AgentNodeID *string    `json:"agent_node_id,omitempty"`
	ReasonerID  *string    `json:"reasoner_id,omitempty"`
	Status      *string    `json:"status,omitempty"`
	UserID      *string    `json:"user_id,omitempty"`
	TeamID      *string    `json:"team_id,omitempty"`
	StartTime   *time.Time `json:"start_time,omitempty"`
	EndTime     *time.Time `json:"end_time,omitempty"`
	Limit       int        `json:"limit,omitempty"`
	Offset      int        `json:"offset,omitempty"`
}

// AgentFilters holds filters for querying agent nodes.
type AgentFilters struct {
	TeamID       *string       `json:"team_id,omitempty"`
	HealthStatus *HealthStatus `json:"health_status,omitempty"`
	Features     []string      `json:"features,omitempty"`
}

// EventFilter holds filters for querying memory events.
type EventFilter struct {
	Scope    *string    `json:"scope,omitempty"`
	ScopeID  *string    `json:"scope_id,omitempty"`
	Patterns []string   `json:"patterns,omitempty"`
	Since    *time.Time `json:"since,omitempty"`
	Limit    int        `json:"limit,omitempty"`
}

// MemoryChangeEvent represents a detailed event when memory changes.
type MemoryChangeEvent struct {
	ID           string          `json:"id"`
	Type         string          `json:"type"`
	Timestamp    time.Time       `json:"timestamp"`
	Scope        string          `json:"scope"`
	ScopeID      string          `json:"scope_id"`
	Key          string          `json:"key"`
	Action       string          `json:"action"` // "set" or "delete"
	Data         json.RawMessage `json:"data,omitempty"`
	PreviousData json.RawMessage `json:"previous_data,omitempty"`
	Metadata     EventMetadata   `json:"metadata"`
}

// EventMetadata holds context for a memory change event.
type EventMetadata struct {
	AgentID    string `json:"agent_id"`
	WorkflowID string `json:"workflow_id"`
}

// DistributedLock represents a lock in the distributed system.
type DistributedLock struct {
	LockID    string    `json:"lock_id"`
	Key       string    `json:"lock_key"`
	Holder    string    `json:"holder"`
	ExpiresAt time.Time `json:"expires_at"`
	CreatedAt time.Time `json:"created_at"`
}

// LockEvent represents an event related to a distributed lock.
type LockEvent struct {
	ID        string    `json:"id"`
	Type      string    `json:"type"`
	Timestamp time.Time `json:"timestamp"`
	Action    string    `json:"action"` // "acquired", "released", "expired"
	LockKey   string    `json:"lock_key"`
	LockID    string    `json:"lock_id"`
	Holder    string    `json:"holder"`
	ExpiresAt time.Time `json:"expires_at"`
}

// WorkflowExecution represents the new comprehensive execution tracking
type WorkflowExecution struct {
	ID int64 `json:"id" db:"id"`

	// Core IDs
	WorkflowID          string  `json:"workflow_id" db:"workflow_id"`
	ExecutionID         string  `json:"execution_id" db:"execution_id"`
	AgentFieldRequestID string  `json:"agentfield_request_id" db:"agentfield_request_id"`
	RunID               *string `json:"run_id,omitempty" db:"run_id"`
	SessionID           *string `json:"session_id,omitempty" db:"session_id"`
	ActorID             *string `json:"actor_id,omitempty" db:"actor_id"`
	AgentNodeID         string  `json:"agent_node_id" db:"agent_node_id"`

	// DAG Relationship Fields
	ParentWorkflowID  *string `json:"parent_workflow_id,omitempty" db:"parent_workflow_id"`
	ParentExecutionID *string `json:"parent_execution_id,omitempty" db:"parent_execution_id"`
	RootWorkflowID    *string `json:"root_workflow_id,omitempty" db:"root_workflow_id"`
	WorkflowDepth     int     `json:"workflow_depth" db:"workflow_depth"`

	// Request details
	ReasonerID string          `json:"reasoner_id" db:"reasoner_id"`
	InputData  json.RawMessage `json:"input_data" db:"input_data"`
	OutputData json.RawMessage `json:"output_data" db:"output_data"`
	InputSize  int             `json:"input_size" db:"input_size"`
	OutputSize int             `json:"output_size" db:"output_size"`

	// Workflow metadata
	WorkflowName *string  `json:"workflow_name,omitempty" db:"workflow_name"`
	WorkflowTags []string `json:"workflow_tags" db:"workflow_tags"`

	// Execution details
	Status      string     `json:"status" db:"status"`
	StartedAt   time.Time  `json:"started_at" db:"started_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty" db:"completed_at"`
	DurationMS  *int64     `json:"duration_ms,omitempty" db:"duration_ms"`

	// State coordination for distributed workflows
	StateVersion          int64      `json:"state_version" db:"state_version"`
	LastEventSequence     int64      `json:"last_event_sequence" db:"last_event_sequence"`
	ActiveChildren        int        `json:"active_children" db:"active_children"`
	PendingChildren       int        `json:"pending_children" db:"pending_children"`
	PendingTerminalStatus *string    `json:"pending_terminal_status,omitempty" db:"pending_terminal_status"`
	StatusReason          *string    `json:"status_reason,omitempty" db:"status_reason"`
	LeaseOwner            *string    `json:"lease_owner,omitempty" db:"lease_owner"`
	LeaseExpiresAt        *time.Time `json:"lease_expires_at,omitempty" db:"lease_expires_at"`

	// Error handling
	ErrorMessage *string `json:"error_message,omitempty" db:"error_message"`
	RetryCount   int     `json:"retry_count" db:"retry_count"`

	// Webhook observability (non-persisted)
	WebhookRegistered bool                     `json:"webhook_registered,omitempty" db:"-"`
	WebhookEvents     []*ExecutionWebhookEvent `json:"webhook_events,omitempty" db:"-"`

	// Notes for app.note() feature
	Notes []ExecutionNote `json:"notes" db:"notes"`

	// Timestamps
	CreatedAt time.Time `json:"created_at" db:"created_at"`
	UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}

// WorkflowExecutionEvent captures immutable lifecycle transitions for an execution.
type WorkflowExecutionEvent struct {
	EventID           int64           `json:"event_id" db:"event_id"`
	ExecutionID       string          `json:"execution_id" db:"execution_id"`
	WorkflowID        string          `json:"workflow_id" db:"workflow_id"`
	RunID             *string         `json:"run_id,omitempty" db:"run_id"`
	ParentExecutionID *string         `json:"parent_execution_id,omitempty" db:"parent_execution_id"`
	Sequence          int64           `json:"sequence" db:"sequence"`
	PreviousSequence  int64           `json:"previous_sequence" db:"previous_sequence"`
	EventType         string          `json:"event_type" db:"event_type"`
	Status            *string         `json:"status,omitempty" db:"status"`
	StatusReason      *string         `json:"status_reason,omitempty" db:"status_reason"`
	Payload           json.RawMessage `json:"payload" db:"payload"`
	EmittedAt         time.Time       `json:"emitted_at" db:"emitted_at"`
	RecordedAt        time.Time       `json:"recorded_at" db:"recorded_at"`
}

// WorkflowRunEvent mirrors execution events at the workflow-run level.
type WorkflowRunEvent struct {
	EventID          int64           `json:"event_id" db:"event_id"`
	RunID            string          `json:"run_id" db:"run_id"`
	Sequence         int64           `json:"sequence" db:"sequence"`
	PreviousSequence int64           `json:"previous_sequence" db:"previous_sequence"`
	EventType        string          `json:"event_type" db:"event_type"`
	Status           *string         `json:"status,omitempty" db:"status"`
	StatusReason     *string         `json:"status_reason,omitempty" db:"status_reason"`
	Payload          json.RawMessage `json:"payload" db:"payload"`
	EmittedAt        time.Time       `json:"emitted_at" db:"emitted_at"`
	RecordedAt       time.Time       `json:"recorded_at" db:"recorded_at"`
}

// ExecutionWebhookEvent records outbound webhook delivery attempts.
type ExecutionWebhookEvent struct {
	ID           int64           `json:"id" db:"id"`
	ExecutionID  string          `json:"execution_id" db:"execution_id"`
	EventType    string          `json:"event_type" db:"event_type"`
	Status       string          `json:"status" db:"status"`
	HTTPStatus   *int            `json:"http_status,omitempty" db:"http_status"`
	Payload      json.RawMessage `json:"payload,omitempty" db:"payload"`
	ResponseBody *string         `json:"response_body,omitempty" db:"response_body"`
	ErrorMessage *string         `json:"error_message,omitempty" db:"error_message"`
	CreatedAt    time.Time       `json:"created_at" db:"created_at"`
}

// WorkflowRun tracks the lifecycle of an orchestrated workflow execution tree.
type WorkflowRun struct {
	RunID             string          `json:"run_id" db:"run_id"`
	RootWorkflowID    string          `json:"root_workflow_id" db:"root_workflow_id"`
	RootExecutionID   *string         `json:"root_execution_id,omitempty" db:"root_execution_id"`
	Status            string          `json:"status" db:"status"`
	TotalSteps        int             `json:"total_steps" db:"total_steps"`
	CompletedSteps    int             `json:"completed_steps" db:"completed_steps"`
	FailedSteps       int             `json:"failed_steps" db:"failed_steps"`
	StateVersion      int64           `json:"state_version" db:"state_version"`
	LastEventSequence int64           `json:"last_event_sequence" db:"last_event_sequence"`
	Metadata          json.RawMessage `json:"metadata" db:"metadata"`
	CreatedAt         time.Time       `json:"created_at" db:"created_at"`
	UpdatedAt         time.Time       `json:"updated_at" db:"updated_at"`
	CompletedAt       *time.Time      `json:"completed_at,omitempty" db:"completed_at"`
}

// WorkflowRunUpdate defines atomic modifications applied to a workflow run.
type WorkflowRunUpdate struct {
	Status              *string         `json:"status,omitempty"`
	Metadata            json.RawMessage `json:"metadata,omitempty"`
	CompletedStepsDelta int             `json:"completed_steps_delta,omitempty"`
	FailedStepsDelta    int             `json:"failed_steps_delta,omitempty"`
	CompletedAt         *time.Time      `json:"completed_at,omitempty"`
}

// WorkflowStep represents a single unit of work inside a workflow run.
type WorkflowStep struct {
	StepID       string          `json:"step_id" db:"step_id"`
	RunID        string          `json:"run_id" db:"run_id"`
	ParentStepID *string         `json:"parent_step_id,omitempty" db:"parent_step_id"`
	ExecutionID  *string         `json:"execution_id,omitempty" db:"execution_id"`
	AgentNodeID  *string         `json:"agent_node_id,omitempty" db:"agent_node_id"`
	Target       *string         `json:"target,omitempty" db:"target"`
	Status       string          `json:"status" db:"status"`
	Attempt      int             `json:"attempt" db:"attempt"`
	Priority     int             `json:"priority" db:"priority"`
	NotBefore    time.Time       `json:"not_before" db:"not_before"`
	InputURI     *string         `json:"input_uri,omitempty" db:"input_uri"`
	ResultURI    *string         `json:"result_uri,omitempty" db:"result_uri"`
	ErrorMessage *string         `json:"error_message,omitempty" db:"error_message"`
	Metadata     json.RawMessage `json:"metadata" db:"metadata"`
	StartedAt    *time.Time      `json:"started_at,omitempty" db:"started_at"`
	CompletedAt  *time.Time      `json:"completed_at,omitempty" db:"completed_at"`
	LeasedAt     *time.Time      `json:"leased_at,omitempty" db:"leased_at"`
	LeaseTimeout *time.Time      `json:"lease_timeout,omitempty" db:"lease_timeout"`
	CreatedAt    time.Time       `json:"created_at" db:"created_at"`
	UpdatedAt    time.Time       `json:"updated_at" db:"updated_at"`
}

// WorkflowStepFilters limit step queries.
type WorkflowStepFilters struct {
	RunID  *string `json:"run_id,omitempty"`
	Status *string `json:"status,omitempty"`
}

// WorkflowStepLeaseOptions configure step leasing semantics.
type WorkflowStepLeaseOptions struct {
	RunID         *string       `json:"run_id,omitempty"`
	Statuses      []string      `json:"statuses,omitempty"`
	Now           time.Time     `json:"now"`
	LeaseDuration time.Duration `json:"lease_duration"`
	MaxAttempts   int           `json:"max_attempts"`
}

// ExecutionNote represents a single note entry for workflow executions
type ExecutionNote struct {
	Message   string    `json:"message"`
	Tags      []string  `json:"tags"`
	Timestamp time.Time `json:"timestamp"`
}

// Workflow represents aggregated workflow information
type Workflow struct {
	WorkflowID   string   `json:"workflow_id" db:"workflow_id"`
	WorkflowName *string  `json:"workflow_name,omitempty" db:"workflow_name"`
	WorkflowTags []string `json:"workflow_tags" db:"workflow_tags"`
	SessionID    *string  `json:"session_id,omitempty" db:"session_id"`
	ActorID      *string  `json:"actor_id,omitempty" db:"actor_id"`

	// DAG Relationship Fields
	ParentWorkflowID *string `json:"parent_workflow_id,omitempty" db:"parent_workflow_id"`
	RootWorkflowID   *string `json:"root_workflow_id,omitempty" db:"root_workflow_id"`
	WorkflowDepth    int     `json:"workflow_depth" db:"workflow_depth"`

	// Aggregated metrics
	TotalExecutions      int   `json:"total_executions" db:"total_executions"`
	SuccessfulExecutions int   `json:"successful_executions" db:"successful_executions"`
	FailedExecutions     int   `json:"failed_executions" db:"failed_executions"`
	TotalDurationMS      int64 `json:"total_duration_ms" db:"total_duration_ms"`

	// Status
	Status string `json:"status" db:"status"`

	// Timestamps
	StartedAt   time.Time  `json:"started_at" db:"started_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty" db:"completed_at"`
	CreatedAt   time.Time  `json:"created_at" db:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at" db:"updated_at"`
}

// Session represents session-level aggregation
type Session struct {
	SessionID   string  `json:"session_id" db:"session_id"`
	ActorID     *string `json:"actor_id,omitempty" db:"actor_id"`
	SessionName *string `json:"session_name,omitempty" db:"session_name"`

	// DAG Relationship Fields
	ParentSessionID *string `json:"parent_session_id,omitempty" db:"parent_session_id"`
	RootSessionID   *string `json:"root_session_id,omitempty" db:"root_session_id"`

	// Aggregated metrics
	TotalWorkflows  int   `json:"total_workflows" db:"total_workflows"`
	TotalExecutions int   `json:"total_executions" db:"total_executions"`
	TotalDurationMS int64 `json:"total_duration_ms" db:"total_duration_ms"`

	// Timestamps
	StartedAt      time.Time `json:"started_at" db:"started_at"`
	LastActivityAt time.Time `json:"last_activity_at" db:"last_activity_at"`
	CreatedAt      time.Time `json:"created_at" db:"created_at"`
	UpdatedAt      time.Time `json:"updated_at" db:"updated_at"`
}

// WorkflowExecutionFilters holds filters for querying workflow executions
type WorkflowExecutionFilters struct {
	WorkflowID        *string    `json:"workflow_id,omitempty"`
	ParentExecutionID *string    `json:"parent_execution_id,omitempty"`
	SessionID         *string    `json:"session_id,omitempty"`
	ActorID           *string    `json:"actor_id,omitempty"`
	AgentNodeID       *string    `json:"agent_node_id,omitempty"`
	Status            *string    `json:"status,omitempty"`
	StartTime         *time.Time `json:"start_time,omitempty"`
	EndTime           *time.Time `json:"end_time,omitempty"`
	Search            *string    `json:"search,omitempty"`
	SortBy            *string    `json:"sort_by,omitempty"`
	SortOrder         *string    `json:"sort_order,omitempty"`
	Limit             int        `json:"limit,omitempty"`
	Offset            int        `json:"offset,omitempty"`
}

// WorkflowRunFilters holds filters for querying workflow runs
type WorkflowRunFilters struct {
	RunID      *string    `json:"run_id,omitempty"`
	RunIDs     []string   `json:"run_ids,omitempty"`
	WorkflowID *string    `json:"workflow_id,omitempty"`
	SessionID  *string    `json:"session_id,omitempty"`
	ActorID    *string    `json:"actor_id,omitempty"`
	Statuses   []string   `json:"statuses,omitempty"`
	Search     *string    `json:"search,omitempty"`
	Since      *time.Time `json:"since,omitempty"`
	Until      *time.Time `json:"until,omitempty"`
	SortBy     *string    `json:"sort_by,omitempty"`
	SortOrder  *string    `json:"sort_order,omitempty"`
	Limit      int        `json:"limit,omitempty"`
	Offset     int        `json:"offset,omitempty"`
}

// WorkflowFilters holds filters for querying workflows
type WorkflowFilters struct {
	SessionID   *string    `json:"session_id,omitempty"`
	ActorID     *string    `json:"actor_id,omitempty"`
	AgentNodeID *string    `json:"agent_node_id,omitempty"`
	Status      *string    `json:"status,omitempty"`
	StartTime   *time.Time `json:"start_time,omitempty"`
	EndTime     *time.Time `json:"end_time,omitempty"`
	Limit       int        `json:"limit,omitempty"`
	Offset      int        `json:"offset,omitempty"`
	SortBy      *string    `json:"sort_by,omitempty"`
	SortOrder   *string    `json:"sort_order,omitempty"`
}

// SessionFilters holds filters for querying sessions
type SessionFilters struct {
	ActorID   *string    `json:"actor_id,omitempty"`
	StartTime *time.Time `json:"start_time,omitempty"`
	EndTime   *time.Time `json:"end_time,omitempty"`
	Limit     int        `json:"limit,omitempty"`
	Offset    int        `json:"offset,omitempty"`
}

// ReasonerPerformanceMetrics represents performance data for a reasoner
type ReasonerPerformanceMetrics struct {
	AvgResponseTimeMs int                   `json:"avg_response_time_ms"`
	SuccessRate       float64               `json:"success_rate"`
	TotalExecutions   int                   `json:"total_executions"`
	ExecutionsLast24h int                   `json:"executions_last_24h"`
	RecentExecutions  []RecentExecutionItem `json:"recent_executions"`
}

// RecentExecutionItem represents a recent execution for metrics
type RecentExecutionItem struct {
	ExecutionID string    `json:"execution_id"`
	Status      string    `json:"status"`
	DurationMs  int64     `json:"duration_ms"`
	Timestamp   time.Time `json:"timestamp"`
}

// ReasonerExecutionHistory represents paginated execution history
type ReasonerExecutionHistory struct {
	Executions []ReasonerExecutionRecord `json:"executions"`
	Total      int                       `json:"total"`
	Page       int                       `json:"page"`
	Limit      int                       `json:"limit"`
	HasMore    bool                      `json:"has_more"`
}

// ReasonerExecutionRecord represents a single execution record for reasoner history
type ReasonerExecutionRecord struct {
	ExecutionID string                 `json:"execution_id"`
	Status      string                 `json:"status"`
	Input       map[string]interface{} `json:"input"`
	Output      map[string]interface{} `json:"output,omitempty"`
	Error       string                 `json:"error,omitempty"`
	DurationMs  int64                  `json:"duration_ms"`
	Timestamp   time.Time              `json:"timestamp"`
}

// WorkflowSummaryData represents pre-aggregated workflow summary data from database
type WorkflowSummaryData struct {
	WorkflowID      string    `json:"workflow_id" db:"workflow_id"`
	TotalExecutions int       `json:"total_executions" db:"total_executions"`
	LatestActivity  time.Time `json:"latest_activity" db:"latest_activity"`
	StartedAt       time.Time `json:"started_at" db:"started_at"`
	RootReasoner    *string   `json:"root_reasoner" db:"root_reasoner"`
	AgentNodeID     *string   `json:"agent_node_id" db:"agent_node_id"`
	WorkflowStatus  *string   `json:"workflow_status" db:"workflow_status"`
	TotalDurationMS *int64    `json:"total_duration_ms" db:"total_duration_ms"`
	MaxDepth        int       `json:"max_depth" db:"max_depth"`
	WorkflowName    *string   `json:"workflow_name" db:"workflow_name"`
	SessionID       *string   `json:"session_id" db:"session_id"`
	ActorID         *string   `json:"actor_id" db:"actor_id"`
	CurrentTask     *string   `json:"current_task" db:"current_task"`
}

// WorkflowCleanupResult represents the result of a workflow cleanup operation
type WorkflowCleanupResult struct {
	WorkflowID      string         `json:"workflow_id"`
	DryRun          bool           `json:"dry_run"`
	DeletedRecords  map[string]int `json:"deleted_records"`
	FreedSpaceBytes int64          `json:"freed_space_bytes"`
	DurationMS      int64          `json:"duration_ms"`
	Success         bool           `json:"success"`
	ErrorMessage    *string        `json:"error_message,omitempty"`
}
