package storage

import (
	"testing"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// allCanonicalStatuses lists every canonical execution status defined by the
// platform. Keeping this in sync with types.go is intentional — a divergence
// here would itself be caught by the reachability test.
var allCanonicalStatuses = []string{
	types.ExecutionStatusUnknown,
	types.ExecutionStatusPending,
	types.ExecutionStatusQueued,
	types.ExecutionStatusWaiting,
	types.ExecutionStatusRunning,
	types.ExecutionStatusPaused,
	types.ExecutionStatusSucceeded,
	types.ExecutionStatusFailed,
	types.ExecutionStatusCancelled,
	types.ExecutionStatusTimeout,
}

var terminalStatuses = []string{
	types.ExecutionStatusSucceeded,
	types.ExecutionStatusFailed,
	types.ExecutionStatusCancelled,
}

// semiTerminalStatuses lists states that are terminal-like but allow specific
// recovery transitions. For example, timeout allows transitions to running
// (stale execution reaper may timeout executions that are still active, e.g.
// waiting for HITL approval) and cancelled.
var semiTerminalStatuses = []string{
	types.ExecutionStatusTimeout,
}

// ---------------------------------------------------------------------------
// Terminal irreversibility
// ---------------------------------------------------------------------------

// TestInvariant_ExecutionState_TerminalStatesAreIrreversible verifies that no
// fully terminal state can transition to any non-identical state.
func TestInvariant_ExecutionState_TerminalStatesAreIrreversible(t *testing.T) {
	for _, terminal := range terminalStatuses {
		for _, target := range allCanonicalStatuses {
			if target == terminal {
				continue // same-state is allowed
			}
			err := validateExecutionStateTransition(terminal, target)
			assert.Error(t, err,
				"terminal state %q must not transition to %q", terminal, target)
		}
	}
}

// TestInvariant_ExecutionState_SemiTerminalAllowedTransitions verifies that
// semi-terminal states (e.g. timeout) forbid all transitions except the
// explicitly allowed recovery paths.
func TestInvariant_ExecutionState_SemiTerminalAllowedTransitions(t *testing.T) {
	// timeout allows transitions to running and cancelled only.
	allowedFromTimeout := map[string]bool{
		types.ExecutionStatusRunning:   true,
		types.ExecutionStatusCancelled: true,
		types.ExecutionStatusTimeout:   true, // self-transition
	}

	for _, target := range allCanonicalStatuses {
		err := validateExecutionStateTransition(types.ExecutionStatusTimeout, target)
		if allowedFromTimeout[target] {
			assert.NoError(t, err,
				"semi-terminal state %q must allow transition to %q", types.ExecutionStatusTimeout, target)
		} else {
			assert.Error(t, err,
				"semi-terminal state %q must not transition to %q", types.ExecutionStatusTimeout, target)
		}
	}
}

// TestInvariant_ExecutionState_TerminalToSelfIsAllowed verifies that a
// terminal state may "transition" to itself without error (idempotent write).
func TestInvariant_ExecutionState_TerminalToSelfIsAllowed(t *testing.T) {
	for _, terminal := range terminalStatuses {
		err := validateExecutionStateTransition(terminal, terminal)
		assert.NoError(t, err,
			"terminal state %q must be allowed to transition to itself", terminal)
	}
}

// ---------------------------------------------------------------------------
// Reachability — all states reachable from "unknown"
// ---------------------------------------------------------------------------

// TestInvariant_ExecutionState_AllStatesReachableFromUnknown performs a BFS
// from ExecutionStatusUnknown and verifies that every canonical state is
// reachable through valid transitions.
func TestInvariant_ExecutionState_AllStatesReachableFromUnknown(t *testing.T) {
	visited := map[string]bool{types.ExecutionStatusUnknown: true}
	queue := []string{types.ExecutionStatusUnknown}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		for _, candidate := range allCanonicalStatuses {
			if visited[candidate] {
				continue
			}
			if err := validateExecutionStateTransition(current, candidate); err == nil {
				visited[candidate] = true
				queue = append(queue, candidate)
			}
		}
	}

	for _, status := range allCanonicalStatuses {
		assert.True(t, visited[status],
			"status %q must be reachable from %q via valid transitions", status, types.ExecutionStatusUnknown)
	}
}

// ---------------------------------------------------------------------------
// Determinism
// ---------------------------------------------------------------------------

// TestInvariant_ExecutionState_TransitionsAreDeterministic verifies that for
// every (from, to) pair, validateExecutionStateTransition always returns the
// same error/nil result across repeated calls.
func TestInvariant_ExecutionState_TransitionsAreDeterministic(t *testing.T) {
	const iterations = 5

	for _, from := range allCanonicalStatuses {
		for _, to := range allCanonicalStatuses {
			var firstResult error
			for i := 0; i < iterations; i++ {
				err := validateExecutionStateTransition(from, to)
				if i == 0 {
					firstResult = err
					continue
				}
				if firstResult == nil {
					assert.NoError(t, err,
						"transition %q→%q returned non-nil on iteration %d after nil on first", from, to, i)
				} else {
					assert.Error(t, err,
						"transition %q→%q returned nil on iteration %d after error on first", from, to, i)
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Alias normalization equivalence
// ---------------------------------------------------------------------------

// TestInvariant_ExecutionState_AliasesNormalizeBeforeValidation verifies that
// alias forms (e.g. "success") behave identically to their canonical forms
// (e.g. "succeeded") when used as the source or target of a transition.
func TestInvariant_ExecutionState_AliasesNormalizeBeforeValidation(t *testing.T) {
	cases := []struct {
		alias     string
		canonical string
	}{
		{"success", types.ExecutionStatusSucceeded},
		{"completed", types.ExecutionStatusSucceeded},
		{"error", types.ExecutionStatusFailed},
		{"canceled", types.ExecutionStatusCancelled},
		{"in_progress", types.ExecutionStatusRunning},
	}

	for _, tc := range cases {
		for _, target := range allCanonicalStatuses {
			errAlias := validateExecutionStateTransition(tc.alias, target)
			errCanon := validateExecutionStateTransition(tc.canonical, target)

			// Both must agree: either both nil or both non-nil.
			if errCanon == nil {
				assert.NoError(t, errAlias,
					"alias %q→%q must be allowed if canonical %q→%q is allowed",
					tc.alias, target, tc.canonical, target)
			} else {
				assert.Error(t, errAlias,
					"alias %q→%q must be forbidden if canonical %q→%q is forbidden",
					tc.alias, target, tc.canonical, target)
			}
		}
	}
}

// ---------------------------------------------------------------------------
// Known valid transitions (regression guard)
// ---------------------------------------------------------------------------

// TestInvariant_ExecutionState_KnownValidTransitions guards specific
// transitions that the state machine must always allow.
func TestInvariant_ExecutionState_KnownValidTransitions(t *testing.T) {
	valid := [][2]string{
		{types.ExecutionStatusUnknown, types.ExecutionStatusPending},
		{types.ExecutionStatusPending, types.ExecutionStatusQueued},
		{types.ExecutionStatusPending, types.ExecutionStatusRunning},
		{types.ExecutionStatusPending, types.ExecutionStatusCancelled},
		{types.ExecutionStatusQueued, types.ExecutionStatusRunning},
		{types.ExecutionStatusRunning, types.ExecutionStatusSucceeded},
		{types.ExecutionStatusRunning, types.ExecutionStatusFailed},
		{types.ExecutionStatusRunning, types.ExecutionStatusCancelled},
		{types.ExecutionStatusRunning, types.ExecutionStatusTimeout},
		{types.ExecutionStatusRunning, types.ExecutionStatusPaused},
		{types.ExecutionStatusRunning, types.ExecutionStatusWaiting},
		{types.ExecutionStatusPaused, types.ExecutionStatusRunning},
		{types.ExecutionStatusPaused, types.ExecutionStatusCancelled},
		{types.ExecutionStatusWaiting, types.ExecutionStatusRunning},
		{types.ExecutionStatusWaiting, types.ExecutionStatusCancelled},
		{types.ExecutionStatusWaiting, types.ExecutionStatusFailed},
		{types.ExecutionStatusTimeout, types.ExecutionStatusRunning},
		{types.ExecutionStatusTimeout, types.ExecutionStatusCancelled},
	}

	for _, pair := range valid {
		from, to := pair[0], pair[1]
		err := validateExecutionStateTransition(from, to)
		require.NoError(t, err, "transition %q→%q must be valid", from, to)
	}
}

// TestInvariant_ExecutionState_KnownInvalidTransitions guards specific
// transitions that the state machine must always forbid.
func TestInvariant_ExecutionState_KnownInvalidTransitions(t *testing.T) {
	invalid := [][2]string{
		{types.ExecutionStatusSucceeded, types.ExecutionStatusRunning},
		{types.ExecutionStatusSucceeded, types.ExecutionStatusPending},
		{types.ExecutionStatusFailed, types.ExecutionStatusRunning},
		{types.ExecutionStatusFailed, types.ExecutionStatusSucceeded},
		{types.ExecutionStatusCancelled, types.ExecutionStatusPending},
		{types.ExecutionStatusTimeout, types.ExecutionStatusPending},
		{types.ExecutionStatusTimeout, types.ExecutionStatusSucceeded},
		{types.ExecutionStatusPending, types.ExecutionStatusSucceeded}, // must go through running
		{types.ExecutionStatusQueued, types.ExecutionStatusSucceeded},
	}

	for _, pair := range invalid {
		from, to := pair[0], pair[1]
		err := validateExecutionStateTransition(from, to)
		require.Error(t, err, "transition %q→%q must be invalid", from, to)
	}
}
