package types

import (
	"encoding/json"
	"time"
)

// ReasonerDefinition mirrors the Haxen server registration contract.
type ReasonerDefinition struct {
	ID           string          `json:"id"`
	InputSchema  json.RawMessage `json:"input_schema"`
	OutputSchema json.RawMessage `json:"output_schema"`
}

// SkillDefinition is included for completeness.
type SkillDefinition struct {
	ID          string          `json:"id"`
	InputSchema json.RawMessage `json:"input_schema"`
	Tags        []string        `json:"tags,omitempty"`
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
}

// NodeStatusUpdate is used for lease renewals.
type NodeStatusUpdate struct {
	Phase       string `json:"phase"`
	HealthScore *int   `json:"health_score,omitempty"`
}

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
	ExpectedRestart string `json:"expected_restart,omitempty"`
}
