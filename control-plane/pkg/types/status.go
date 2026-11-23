package types

import "strings"

// ExecutionStatus represents the canonical set of execution statuses supported by the system.
type ExecutionStatus = string

const (
	ExecutionStatusUnknown   ExecutionStatus = "unknown"
	ExecutionStatusPending   ExecutionStatus = "pending"
	ExecutionStatusQueued    ExecutionStatus = "queued"
	ExecutionStatusWaiting   ExecutionStatus = "waiting"
	ExecutionStatusRunning   ExecutionStatus = "running"
	ExecutionStatusPaused    ExecutionStatus = "paused"
	ExecutionStatusSucceeded ExecutionStatus = "succeeded"
	ExecutionStatusFailed    ExecutionStatus = "failed"
	ExecutionStatusCancelled ExecutionStatus = "cancelled"
	ExecutionStatusTimeout   ExecutionStatus = "timeout"
)

// ExecutionReasonAwaitingAgentRestart is written to an execution's
// status_reason while the control plane is deliberately holding it through a
// restart of its agent node — the dispatch found the port closed and is
// waiting for the process to come back rather than failing the run.
//
// Such an execution is NOT an orphan: the control plane still owns it and is
// about to re-dispatch it. MarkAgentExecutionsOrphaned therefore skips rows
// carrying this reason, so the re-registration that ends the restart does not
// fail the very execution that was waiting for it.
const ExecutionReasonAwaitingAgentRestart = "awaiting_agent_restart"

// ExecutionReasonCancelled is the status_reason of an execution cancelled
// without a reason of its own (a person's Stop). Cancelling always writes a
// reason: leaving the field alone kept whatever the run said last, and a run
// stopped after its plan was approved reported "approval_granted" as its
// root error category (WorkAF live run, 2026-11-07).
const ExecutionReasonCancelled = "cancelled"

// THE OFFLINE HOLD. A node that registers with
// metadata.custom.holds_executions_offline = true promises that its work
// survives the node going away for a while (a laptop asleep overnight, a
// process quit and started again later): when it comes back it lists every
// execution it still holds as inflight on its first lease. For such a node the
// stale sweep does not time out an execution merely because nothing renewed it
// while the node was offline; it marks the execution held instead.
const (
	// ExecutionReasonNodeOffline is the status_reason of a running execution
	// the control plane is holding for an opted-in node whose lease lapsed.
	// The row stays running; the stale sweep and the stale-retry sweep skip
	// it, and updated_at records when the hold began.
	ExecutionReasonNodeOffline = "node_offline"
	// ExecutionReasonNodeLostExecution fails a held execution the node did
	// not list as inflight when it came back: the node returned without it.
	ExecutionReasonNodeLostExecution = "node_lost_execution"
	// ExecutionReasonNodeRemoved fails a held execution whose node is no
	// longer registered.
	ExecutionReasonNodeRemoved = "node_removed"
	// ExecutionReasonNodeOfflineTooLong fails a held execution whose node has
	// stayed away past the offline hold ceiling.
	ExecutionReasonNodeOfflineTooLong = "node_offline_too_long"
)

// AgentMetadataHoldsExecutionsOffline is the registration metadata key
// (metadata.custom) by which a node opts in to the offline hold.
const AgentMetadataHoldsExecutionsOffline = "holds_executions_offline"

// HoldsExecutionsOffline reports whether the node opted in to the offline
// hold. Absent, false, or anything other than true / "true" is no.
func (a *AgentNode) HoldsExecutionsOffline() bool {
	if a == nil || a.Metadata.Custom == nil {
		return false
	}
	switch v := a.Metadata.Custom[AgentMetadataHoldsExecutionsOffline].(type) {
	case bool:
		return v
	case string:
		return v == "true"
	}
	return false
}

const (
	// ExecutionReasonAgentShutdownCancelled identifies an execution cancelled by
	// an SDK while its agent process was draining for shutdown.
	ExecutionReasonAgentShutdownCancelled = "agent_shutdown_cancelled"
	// AgentShutdownCancellationError is emitted identically by every SDK when
	// graceful shutdown cancels an in-flight reasoner.
	AgentShutdownCancellationError = "cancelled during graceful shutdown"
)

var canonicalExecutionStatuses = map[ExecutionStatus]struct{}{
	ExecutionStatusUnknown:   {},
	ExecutionStatusPending:   {},
	ExecutionStatusQueued:    {},
	ExecutionStatusWaiting:   {},
	ExecutionStatusRunning:   {},
	ExecutionStatusPaused:    {},
	ExecutionStatusSucceeded: {},
	ExecutionStatusFailed:    {},
	ExecutionStatusCancelled: {},
	ExecutionStatusTimeout:   {},
}

var executionStatusAliases = map[string]ExecutionStatus{
	"success":           ExecutionStatusSucceeded,
	"successful":        ExecutionStatusSucceeded,
	"completed":         ExecutionStatusSucceeded,
	"complete":          ExecutionStatusSucceeded,
	"done":              ExecutionStatusSucceeded,
	"ok":                ExecutionStatusSucceeded,
	"error":             ExecutionStatusFailed,
	"failure":           ExecutionStatusFailed,
	"errored":           ExecutionStatusFailed,
	"canceled":          ExecutionStatusCancelled,
	"cancel":            ExecutionStatusCancelled,
	"pause":             ExecutionStatusPaused,
	"hold":              ExecutionStatusPaused,
	"on_hold":           ExecutionStatusPaused,
	"suspended":         ExecutionStatusPaused,
	"timed_out":         ExecutionStatusTimeout,
	"wait":              ExecutionStatusQueued,
	"awaiting_approval": ExecutionStatusWaiting,
	"awaiting_human":    ExecutionStatusWaiting,
	"approval_pending":  ExecutionStatusWaiting,
	"in_progress":       ExecutionStatusRunning,
	"processing":        ExecutionStatusRunning,
}

// NormalizeExecutionStatus maps arbitrary status strings onto the canonical execution statuses used by the AgentField platform.
// Unknown or unsupported statuses resolve to ExecutionStatusUnknown.
func NormalizeExecutionStatus(status string) string {
	normalized := ExecutionStatus(strings.ToLower(strings.TrimSpace(status)))
	if normalized == "" {
		return string(ExecutionStatusUnknown)
	}
	if _, ok := canonicalExecutionStatuses[normalized]; ok {
		return string(normalized)
	}
	if mapped, ok := executionStatusAliases[string(normalized)]; ok {
		return string(mapped)
	}
	return string(ExecutionStatusUnknown)
}

// IsTerminalExecutionStatus reports whether the provided status string represents a terminal execution state.
func IsTerminalExecutionStatus(status string) bool {
	switch NormalizeExecutionStatus(status) {
	case string(ExecutionStatusSucceeded), string(ExecutionStatusFailed), string(ExecutionStatusCancelled), string(ExecutionStatusTimeout):
		return true
	default:
		return false
	}
}
