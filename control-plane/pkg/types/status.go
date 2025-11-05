package types

import "strings"

// ExecutionStatus represents the canonical set of execution statuses supported by the system.
type ExecutionStatus = string

const (
	ExecutionStatusUnknown   ExecutionStatus = "unknown"
	ExecutionStatusPending   ExecutionStatus = "pending"
	ExecutionStatusQueued    ExecutionStatus = "queued"
	ExecutionStatusRunning   ExecutionStatus = "running"
	ExecutionStatusSucceeded ExecutionStatus = "succeeded"
	ExecutionStatusFailed    ExecutionStatus = "failed"
	ExecutionStatusCancelled ExecutionStatus = "cancelled"
	ExecutionStatusTimeout   ExecutionStatus = "timeout"
)

var canonicalExecutionStatuses = map[ExecutionStatus]struct{}{
	ExecutionStatusUnknown:   {},
	ExecutionStatusPending:   {},
	ExecutionStatusQueued:    {},
	ExecutionStatusRunning:   {},
	ExecutionStatusSucceeded: {},
	ExecutionStatusFailed:    {},
	ExecutionStatusCancelled: {},
	ExecutionStatusTimeout:   {},
}

var executionStatusAliases = map[string]ExecutionStatus{
	"success":     ExecutionStatusSucceeded,
	"successful":  ExecutionStatusSucceeded,
	"completed":   ExecutionStatusSucceeded,
	"complete":    ExecutionStatusSucceeded,
	"done":        ExecutionStatusSucceeded,
	"ok":          ExecutionStatusSucceeded,
	"error":       ExecutionStatusFailed,
	"failure":     ExecutionStatusFailed,
	"errored":     ExecutionStatusFailed,
	"canceled":    ExecutionStatusCancelled,
	"cancel":      ExecutionStatusCancelled,
	"timed_out":   ExecutionStatusTimeout,
	"wait":        ExecutionStatusQueued,
	"waiting":     ExecutionStatusQueued,
	"in_progress": ExecutionStatusRunning,
	"processing":  ExecutionStatusRunning,
}

// NormalizeExecutionStatus maps arbitrary status strings onto the canonical execution statuses used by the Haxen platform.
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
