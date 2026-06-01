package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNormalizeStatus(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		// Canonical statuses pass through unchanged
		{"pending canonical", "pending", "pending"},
		{"queued canonical", "queued", "queued"},
		{"waiting canonical", "waiting", "waiting"},
		{"running canonical", "running", "running"},
		{"succeeded canonical", "succeeded", "succeeded"},
		{"failed canonical", "failed", "failed"},
		{"cancelled canonical", "cancelled", "cancelled"},
		{"timeout canonical", "timeout", "timeout"},
		{"unknown canonical", "unknown", "unknown"},

		// Alias mappings
		{"success alias", "success", "succeeded"},
		{"error alias", "error", "failed"},
		{"completed alias", "completed", "succeeded"},
		{"canceled alias", "canceled", "cancelled"},
		{"timed_out alias", "timed_out", "timeout"},
		{"in_progress alias", "in_progress", "running"},
		{"processing alias", "processing", "running"},
		{"awaiting_approval alias", "awaiting_approval", "waiting"},

		// Additional aliases from the source
		{"successful alias", "successful", "succeeded"},
		{"complete alias", "complete", "succeeded"},
		{"done alias", "done", "succeeded"},
		{"ok alias", "ok", "succeeded"},
		{"failure alias", "failure", "failed"},
		{"errored alias", "errored", "failed"},
		{"cancel alias", "cancel", "cancelled"},
		{"wait alias", "wait", "queued"},
		{"awaiting_human alias", "awaiting_human", "waiting"},
		{"approval_pending alias", "approval_pending", "waiting"},

		// Empty string returns "unknown"
		{"empty string", "", "unknown"},

		// Unknown string returns "unknown"
		{"unknown string", "foobar", "unknown"},
		{"unknown random", "xyz123", "unknown"},

		// Case insensitive
		{"SUCCESS uppercase", "SUCCESS", "succeeded"},
		{"Failed mixed case", "Failed", "failed"},
		{"RUNNING uppercase", "RUNNING", "running"},
		{"Pending mixed case", "Pending", "pending"},
		{"CANCELLED uppercase", "CANCELLED", "cancelled"},

		// Whitespace trimmed
		{"running with spaces", " running ", "running"},
		{"succeeded with leading space", " succeeded", "succeeded"},
		{"failed with trailing space", "failed ", "failed"},
		{"spaces around success alias", " success ", "succeeded"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := NormalizeStatus(tt.input)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsTerminalStatus(t *testing.T) {
	tests := []struct {
		name     string
		status   string
		expected bool
	}{
		// Terminal statuses
		{"succeeded is terminal", "succeeded", true},
		{"failed is terminal", "failed", true},
		{"cancelled is terminal", "cancelled", true},
		{"timeout is terminal", "timeout", true},

		// Aliases that map to terminal statuses
		{"success is terminal via alias", "success", true},
		{"error is terminal via alias", "error", true},
		{"completed is terminal via alias", "completed", true},
		{"canceled is terminal via alias", "canceled", true},
		{"timed_out is terminal via alias", "timed_out", true},

		// Active statuses are NOT terminal
		{"pending is not terminal", "pending", false},
		{"queued is not terminal", "queued", false},
		{"waiting is not terminal", "waiting", false},
		{"running is not terminal", "running", false},

		// Unknown status is not terminal
		{"unknown is not terminal", "unknown", false},
		{"empty is not terminal", "", false},
		{"gibberish is not terminal", "gibberish", false},

		// Case insensitive
		{"SUCCEEDED uppercase is terminal", "SUCCEEDED", true},
		{"Failed mixed case is terminal", "Failed", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsTerminalStatus(tt.status)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestIsActiveStatus(t *testing.T) {
	tests := []struct {
		name     string
		status   string
		expected bool
	}{
		// Active statuses
		{"pending is active", "pending", true},
		{"queued is active", "queued", true},
		{"waiting is active", "waiting", true},
		{"running is active", "running", true},

		// Aliases that map to active statuses
		{"in_progress is active via alias", "in_progress", true},
		{"processing is active via alias", "processing", true},
		{"awaiting_approval is active via alias", "awaiting_approval", true},
		{"wait is active via alias", "wait", true},

		// Terminal statuses are NOT active
		{"succeeded is not active", "succeeded", false},
		{"failed is not active", "failed", false},
		{"cancelled is not active", "cancelled", false},
		{"timeout is not active", "timeout", false},

		// Unknown status is not active
		{"unknown is not active", "unknown", false},
		{"empty is not active", "", false},
		{"gibberish is not active", "gibberish", false},

		// Case insensitive
		{"RUNNING uppercase is active", "RUNNING", true},
		{"Pending mixed case is active", "Pending", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsActiveStatus(tt.status)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func TestTerminalAndActiveAreMutuallyExclusive(t *testing.T) {
	statuses := []string{
		"pending", "queued", "waiting", "running",
		"succeeded", "failed", "cancelled", "timeout",
		"unknown", "", "garbage",
	}

	for _, s := range statuses {
		t.Run(s, func(t *testing.T) {
			terminal := IsTerminalStatus(s)
			active := IsActiveStatus(s)
			// A status cannot be both terminal and active simultaneously
			assert.False(t, terminal && active, "status %q should not be both terminal and active", s)
		})
	}
}
