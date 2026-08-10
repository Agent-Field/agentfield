package storage

import (
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

// newRunningExecution inserts a running execution, optionally parented, whose
// updated_at is backdated by the given age. A zero age leaves updated_at at
// "now", i.e. actively progressing.
func newRunningExecution(t *testing.T, ls *LocalStorage, id, parent string, age time.Duration) {
	t.Helper()
	ctx := t.Context()
	now := time.Now().UTC()

	exec := &types.Execution{
		ExecutionID: id,
		RunID:       "run-parented",
		AgentNodeID: "agent-1",
		ReasonerID:  "reasoner-1",
		NodeID:      "node-1",
		Status:      "running",
		StartedAt:   now.Add(-2 * time.Hour),
	}
	if parent != "" {
		exec.ParentExecutionID = strPtr(parent)
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, exec))
	if age > 0 {
		backdateExecutionUpdatedAt(t, ls, "executions", id, now.Add(-age))
	}
}

func executionStatus(t *testing.T, ls *LocalStorage, id string) string {
	t.Helper()
	rec, err := ls.GetExecutionRecord(t.Context(), id)
	require.NoError(t, err)
	return rec.Status
}

// TestMarkStaleExecutions_KeepsParentBlockedOnActiveChild is the regression
// this guards: a parent's own updated_at stops moving while it waits on a
// child, so an agent doing many minutes of work inside one request used to get
// its whole ancestor chain reaped mid-flight — reported to the caller as
// "execution timed out (no activity)" while the work was still running.
func TestMarkStaleExecutions_KeepsParentBlockedOnActiveChild(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)

	// build (idle 1h, waiting) -> adapter (idle 1h, waiting) -> engine (working now)
	newRunningExecution(t, ls, "exec-build", "", time.Hour)
	newRunningExecution(t, ls, "exec-adapter", "exec-build", time.Hour)
	newRunningExecution(t, ls, "exec-engine", "exec-adapter", 0)

	reaped, err := ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 0, reaped, "nothing may be reaped while a descendant is still working")

	for _, id := range []string{"exec-build", "exec-adapter", "exec-engine"} {
		require.Equal(t, "running", executionStatus(t, ls, id), "%s must survive", id)
	}
}

// TestMarkStaleExecutions_UnwindsChainBottomUp: when work genuinely stops, the
// chain still drains — one level per sweep, leaf first — so skipping parents
// cannot strand executions in `running` forever.
func TestMarkStaleExecutions_UnwindsChainBottomUp(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)

	newRunningExecution(t, ls, "exec-build", "", time.Hour)
	newRunningExecution(t, ls, "exec-adapter", "exec-build", time.Hour)
	newRunningExecution(t, ls, "exec-engine", "exec-adapter", time.Hour)

	// Sweep 1: only the leaf is childless.
	reaped, err := ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped)
	require.Equal(t, "timeout", executionStatus(t, ls, "exec-engine"))
	require.Equal(t, "running", executionStatus(t, ls, "exec-adapter"))
	require.Equal(t, "running", executionStatus(t, ls, "exec-build"))

	// Sweep 2: the adapter is now childless.
	reaped, err = ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped)
	require.Equal(t, "timeout", executionStatus(t, ls, "exec-adapter"))
	require.Equal(t, "running", executionStatus(t, ls, "exec-build"))

	// Sweep 3: the root drains last.
	reaped, err = ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped)
	require.Equal(t, "timeout", executionStatus(t, ls, "exec-build"))
}

// TestMarkStaleExecutions_TerminalChildDoesNotShieldParent: only a
// *non-terminal* child protects a parent. A finished child must not keep a
// genuinely stuck parent alive.
func TestMarkStaleExecutions_TerminalChildDoesNotShieldParent(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	newRunningExecution(t, ls, "exec-build", "", time.Hour)
	done := &types.Execution{
		ExecutionID:       "exec-done",
		RunID:             "run-parented",
		AgentNodeID:       "agent-1",
		ReasonerID:        "reasoner-1",
		NodeID:            "node-1",
		Status:            "succeeded",
		StartedAt:         now.Add(-2 * time.Hour),
		ParentExecutionID: strPtr("exec-build"),
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, done))

	reaped, err := ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped)
	require.Equal(t, "timeout", executionStatus(t, ls, "exec-build"))
	require.Equal(t, "succeeded", executionStatus(t, ls, "exec-done"), "terminal child untouched")
}

// TestMarkStaleExecutions_UnrelatedStuckExecutionStillReaped: the new skip is
// scoped to actual parents — an unrelated stuck execution is still collected in
// the same sweep as a protected chain.
func TestMarkStaleExecutions_UnrelatedStuckExecutionStillReaped(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)

	newRunningExecution(t, ls, "exec-build", "", time.Hour)
	newRunningExecution(t, ls, "exec-engine", "exec-build", 0) // working
	newRunningExecution(t, ls, "exec-orphan", "", time.Hour)   // genuinely stuck

	reaped, err := ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped)
	require.Equal(t, "timeout", executionStatus(t, ls, "exec-orphan"))
	require.Equal(t, "running", executionStatus(t, ls, "exec-build"))
}
