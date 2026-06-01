package types

import "strings"

// TerminalStatuses contains execution statuses that represent completed work.
var TerminalStatuses = map[string]bool{
	ExecutionStatusSucceeded: true,
	ExecutionStatusFailed:    true,
	ExecutionStatusCancelled: true,
	ExecutionStatusTimeout:   true,
}

// ActiveStatuses contains execution statuses that are still in progress and
// should be polled.
var ActiveStatuses = map[string]bool{
	ExecutionStatusPending: true,
	ExecutionStatusQueued:  true,
	ExecutionStatusWaiting: true,
	ExecutionStatusRunning: true,
}

// statusAliases maps common alternative names to canonical status values.
var statusAliases = map[string]string{
	"success":            ExecutionStatusSucceeded,
	"successful":         ExecutionStatusSucceeded,
	"completed":          ExecutionStatusSucceeded,
	"complete":           ExecutionStatusSucceeded,
	"done":               ExecutionStatusSucceeded,
	"ok":                 ExecutionStatusSucceeded,
	"error":              ExecutionStatusFailed,
	"failure":            ExecutionStatusFailed,
	"errored":            ExecutionStatusFailed,
	"canceled":           ExecutionStatusCancelled,
	"cancel":             ExecutionStatusCancelled,
	"timed_out":          ExecutionStatusTimeout,
	"wait":               ExecutionStatusQueued,
	"awaiting_approval":  ExecutionStatusWaiting,
	"awaiting_human":     ExecutionStatusWaiting,
	"approval_pending":   ExecutionStatusWaiting,
	"in_progress":        ExecutionStatusRunning,
	"processing":         ExecutionStatusRunning,
}

// canonicalStatuses is the set of all known canonical status strings.
var canonicalStatuses = map[string]bool{
	ExecutionStatusPending:   true,
	ExecutionStatusQueued:    true,
	ExecutionStatusWaiting:   true,
	ExecutionStatusRunning:   true,
	ExecutionStatusSucceeded: true,
	ExecutionStatusFailed:    true,
	ExecutionStatusCancelled: true,
	ExecutionStatusTimeout:   true,
	"unknown":               true,
}

// NormalizeStatus maps an arbitrary status string to its canonical form.
// Returns "unknown" for unrecognized values.
func NormalizeStatus(status string) string {
	s := strings.TrimSpace(strings.ToLower(status))
	if s == "" {
		return "unknown"
	}
	if canonicalStatuses[s] {
		return s
	}
	if alias, ok := statusAliases[s]; ok {
		return alias
	}
	return "unknown"
}

// IsTerminalStatus returns true if the given status represents a completed
// execution that will not transition further.
func IsTerminalStatus(status string) bool {
	return TerminalStatuses[NormalizeStatus(status)]
}

// IsActiveStatus returns true if the given status represents an execution
// that is still in progress and should continue to be polled.
func IsActiveStatus(status string) bool {
	return ActiveStatuses[NormalizeStatus(status)]
}
