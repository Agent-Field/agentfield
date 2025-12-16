package types

import (
	"encoding/json"
	"time"
)

// Execution captures a single agent invocation. A workflow run is represented by a
// shared RunID across multiple executions. ParentExecutionID creates the DAG edges.
type Execution struct {
	// Primary identifiers
	ExecutionID       string  `json:"execution_id" db:"execution_id"`
	RunID             string  `json:"run_id" db:"run_id"`
	ParentExecutionID *string `json:"parent_execution_id,omitempty" db:"parent_execution_id"`

	// Agent metadata
	AgentNodeID string `json:"agent_node_id" db:"agent_node_id"`
	ReasonerID  string `json:"reasoner_id" db:"reasoner_id"`
	NodeID      string `json:"node_id" db:"node_id"`

	// Payloads
	InputPayload  json.RawMessage `json:"input" db:"input_payload"`
	ResultPayload json.RawMessage `json:"result,omitempty" db:"result_payload"`
	ErrorMessage  *string         `json:"error,omitempty" db:"error_message"`
	InputURI      *string         `json:"input_uri,omitempty" db:"input_uri"`
	ResultURI     *string         `json:"result_uri,omitempty" db:"result_uri"`

	// Lifecycle
	Status      string     `json:"status" db:"status"`
	StartedAt   time.Time  `json:"started_at" db:"started_at"`
	CompletedAt *time.Time `json:"completed_at,omitempty" db:"completed_at"`
	DurationMS  *int64     `json:"duration_ms,omitempty" db:"duration_ms"`

	// Optional metadata
	SessionID *string `json:"session_id,omitempty" db:"session_id"`
	ActorID   *string `json:"actor_id,omitempty" db:"actor_id"`

	// Notes for debugging and tracking
	Notes []ExecutionNote `json:"notes,omitempty" db:"notes"`

	// Webhook state (computed, not stored in executions table)
	WebhookRegistered bool                     `json:"webhook_registered,omitempty" db:"-"`
	WebhookEvents     []*ExecutionWebhookEvent `json:"webhook_events,omitempty" db:"-"`

	CreatedAt time.Time `json:"created_at" db:"created_at"`
	UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}

// ExecutionFilter describes supported filters when querying executions.
type ExecutionFilter struct {
	RunID             *string
	ExecutionID       *string
	ParentExecutionID *string
	AgentNodeID       *string
	ReasonerID        *string
	Status            *string
	SessionID         *string
	ActorID           *string
	Limit             int
	Offset            int
	StartTime         *time.Time
	EndTime           *time.Time
	SortBy            string
	SortDescending    bool
}

// ExecutionDAGEdge captures a parent→child relationship inside a run. The UI uses
// this information to render workflow graphs.
type ExecutionDAGEdge struct {
	Parent string `json:"parent"`
	Child  string `json:"child"`
}

// GroupExecutionsByRun buckets executions by RunID. Input order is preserved within
// each run slice to match natural timeline ordering.
func GroupExecutionsByRun(executions []*Execution) map[string][]*Execution {
	grouped := make(map[string][]*Execution, len(executions))
	for _, exec := range executions {
		if exec == nil {
			continue
		}
		runID := exec.RunID
		grouped[runID] = append(grouped[runID], exec)
	}
	return grouped
}

// BuildExecutionGraph produces node and edge collections suitable for DAG rendering.
func BuildExecutionGraph(executions []*Execution) (map[string]*Execution, []ExecutionDAGEdge, []string) {
	nodes := make(map[string]*Execution, len(executions))
	var edges []ExecutionDAGEdge
	rootSet := make(map[string]struct{}, len(executions))

	for _, exec := range executions {
		if exec == nil {
			continue
		}

		nodes[exec.ExecutionID] = exec

		if exec.ParentExecutionID != nil && *exec.ParentExecutionID != "" {
			edges = append(edges, ExecutionDAGEdge{
				Parent: *exec.ParentExecutionID,
				Child:  exec.ExecutionID,
			})
		} else {
			rootSet[exec.ExecutionID] = struct{}{}
		}
	}

	var roots []string
	for root := range rootSet {
		roots = append(roots, root)
	}

	return nodes, edges, roots
}

// LegacyExecutionEndpoints enumerates HTTP handlers that depend on execution data.
// Keeping the list close to the data model helps the migration away from the legacy
// workflow tables by giving us a definitive checklist.
var LegacyExecutionEndpoints = struct {
	ExecuteSync       string
	ExecuteAsync      string
	ExecuteStatus     string
	ExecuteBatchState string
	RunSnapshot       string
	ExecutionSnapshot string
	RunEvents         string
	ExecutionEvents   string
	RunEventStream    string
	ExecutionStream   string
	RunCleanup        string

	UIWorkflowSummary        string
	UIWorkflowSummaryFast    string
	UIWorkflowDag            string
	UIWorkflowNotes          string
	UIAgentExecutionList     string
	UIAgentExecutionDetail   string
	UIAgentExecutionTimeline string

	UIWorkflowRunList   string
	UIWorkflowRunDetail string

	UISessionRuns string
}{
	ExecuteSync:       "/api/v1/agents/:agent/execute/:target",
	ExecuteAsync:      "/api/v1/agents/:agent/execute/async/:target",
	ExecuteStatus:     "/api/v1/agents/:agent/executions/:execution_id",
	ExecuteBatchState: "/api/v1/agents/:agent/executions/batch-status",
	RunSnapshot:       "/api/v1/agents/:agent/workflow/runs/:run_id/snapshot",
	ExecutionSnapshot: "/api/v1/agents/:agent/workflow/executions/:execution_id/snapshot",
	RunEvents:         "/api/v1/agents/:agent/workflow/runs/:run_id/events",
	ExecutionEvents:   "/api/v1/agents/:agent/workflow/executions/:execution_id/events",
	RunEventStream:    "/api/v1/agents/:agent/workflow/runs/:run_id/events/stream",
	ExecutionStream:   "/api/v1/agents/:agent/workflow/executions/:execution_id/events/stream",
	RunCleanup:        "/api/v1/agents/:agent/workflows/:workflow_id/cleanup",

	UIWorkflowSummary:        "/api/ui/v1/workflows/summary",
	UIWorkflowSummaryFast:    "/api/ui/v1/workflows/summary/optimized",
	UIWorkflowDag:            "/api/ui/v1/workflows/:workflowId/dag",
	UIWorkflowNotes:          "/api/ui/v1/workflows/:workflowId/notes/events",
	UIAgentExecutionList:     "/api/ui/v1/agents/:agentId/executions",
	UIAgentExecutionDetail:   "/api/ui/v1/agents/:agentId/executions/:executionId",
	UIAgentExecutionTimeline: "/api/ui/v1/agents/:agentId/executions/:executionId/timeline",

	UIWorkflowRunList:   "/api/ui/v2/workflow-runs",
	UIWorkflowRunDetail: "/api/ui/v2/workflow-runs/:run_id",

	UISessionRuns: "/api/ui/v1/sessions/:session_id/workflows",
}
