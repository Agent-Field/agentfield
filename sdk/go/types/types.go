package types

import (
	"encoding/json"
	"time"
)

// ReasonerDefinition mirrors the AgentField server registration contract.
type ReasonerDefinition struct {
	ID           string          `json:"id"`
	InputSchema  json.RawMessage `json:"input_schema"`
	OutputSchema json.RawMessage `json:"output_schema"`
	Tags         []string        `json:"tags,omitempty"`
	ProposedTags []string        `json:"proposed_tags,omitempty"`
}

// SkillDefinition is included for completeness.
type SkillDefinition struct {
	ID           string          `json:"id"`
	InputSchema  json.RawMessage `json:"input_schema"`
	Tags         []string        `json:"tags,omitempty"`
	ProposedTags []string        `json:"proposed_tags,omitempty"`
}

// CommunicationConfig declares supported protocols for the agent.
type CommunicationConfig struct {
	Protocols         []string `json:"protocols"`
	WebSocketEndpoint string   `json:"websocket_endpoint,omitempty"`
	HeartbeatInterval string   `json:"heartbeat_interval,omitempty"`
}

// NodeRegistrationRequest is the legacy-compatible registration payload.
type NodeRegistrationRequest struct {
	ID                   string               `json:"id"`
	TeamID               string               `json:"team_id"`
	BaseURL              string               `json:"base_url"`
	Version              string               `json:"version"`
	Reasoners            []ReasonerDefinition `json:"reasoners"`
	Skills               []SkillDefinition    `json:"skills"`
	CommunicationConfig  CommunicationConfig  `json:"communication_config"`
	HealthStatus         string               `json:"health_status"`
	LastHeartbeat        time.Time            `json:"last_heartbeat"`
	RegisteredAt         time.Time            `json:"registered_at"`
	Metadata             map[string]any       `json:"metadata,omitempty"`
	Features             map[string]any       `json:"features,omitempty"`
	DeploymentType       string               `json:"deployment_type,omitempty"`
	InvocationURL        *string              `json:"invocation_url,omitempty"`
	CallbackDiscovery    map[string]any       `json:"callback_discovery,omitempty"`
	CommunicationChannel map[string]any       `json:"communication_channel,omitempty"`
}

// NodeRegistrationResponse captures the subset we rely on.
type NodeRegistrationResponse struct {
	ID                string    `json:"id"`
	ResolvedBaseURL   string    `json:"resolved_base_url"`
	CallbackDiscovery any       `json:"callback_discovery,omitempty"`
	Message           string    `json:"message,omitempty"`
	Success           bool      `json:"success"`
	RegisteredAt      time.Time `json:"-"`
	Status            string    `json:"status,omitempty"`
	ProposedTags      []string  `json:"proposed_tags,omitempty"`
	PendingTags       []string  `json:"pending_tags,omitempty"`
	AutoApprovedTags  []string  `json:"auto_approved_tags,omitempty"`
}

// NodeStatusUpdate is used for lease renewals.
type NodeStatusUpdate struct {
	Phase       string `json:"phase"`
	Version     string `json:"version,omitempty"`
	HealthScore *int   `json:"health_score,omitempty"`
}

// Canonical execution status values used by the control plane.
const (
	ExecutionStatusPending   = "pending"
	ExecutionStatusQueued    = "queued"
	ExecutionStatusWaiting   = "waiting"
	ExecutionStatusRunning   = "running"
	ExecutionStatusSucceeded = "succeeded"
	ExecutionStatusFailed    = "failed"
	ExecutionStatusCancelled = "cancelled"
	ExecutionStatusTimeout   = "timeout"
)

// LeaseResponse informs the agent how long the lease lasts.
type LeaseResponse struct {
	LeaseSeconds     int    `json:"lease_seconds"`
	NextLeaseRenewal string `json:"next_lease_renewal"`
	Message          string `json:"message,omitempty"`
}

// ActionAckRequest accompanies push-based workloads.
type ActionAckRequest struct {
	ActionID   string          `json:"action_id"`
	Status     string          `json:"status"`
	DurationMS *int            `json:"duration_ms,omitempty"`
	ResultRef  string          `json:"result_ref,omitempty"`
	Result     json.RawMessage `json:"result,omitempty"`
	Error      string          `json:"error_message,omitempty"`
	Notes      []string        `json:"notes,omitempty"`
}

// ShutdownRequest notifies the control plane that the node is draining.
type ShutdownRequest struct {
	Reason          string `json:"reason,omitempty"`
	Version         string `json:"version,omitempty"`
	ExpectedRestart string `json:"expected_restart,omitempty"`
}

// WorkflowExecutionEvent mirrors the control plane's event ingestion payload.
// It allows agents to emit parent/child execution details without routing work
// through the control plane.
type WorkflowExecutionEvent struct {
	ExecutionID       string                 `json:"execution_id"`
	WorkflowID        string                 `json:"workflow_id,omitempty"`
	RunID             string                 `json:"run_id,omitempty"`
	ReasonerID        string                 `json:"reasoner_id,omitempty"`
	Type              string                 `json:"type,omitempty"`
	AgentNodeID       string                 `json:"agent_node_id,omitempty"`
	Status            string                 `json:"status"`
	StatusReason      *string                `json:"status_reason,omitempty"`
	ParentExecutionID *string                `json:"parent_execution_id,omitempty"`
	ParentWorkflowID  *string                `json:"parent_workflow_id,omitempty"`
	InputData         map[string]interface{} `json:"input_data,omitempty"`
	Result            interface{}            `json:"result,omitempty"`
	Error             string                 `json:"error,omitempty"`
	DurationMS        *int64                 `json:"duration_ms,omitempty"`
}
