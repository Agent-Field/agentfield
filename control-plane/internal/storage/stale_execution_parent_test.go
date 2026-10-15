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

// TestMarkStaleExecutions_LongFinishedChildDoesNotShieldParent: a child that
// reached a terminal state *before* the cutoff no longer protects a genuinely
// stuck parent, so orphan cleanup keeps working (issue #1059 acceptance).
func TestMarkStaleExecutions_LongFinishedChildDoesNotShieldParent(t *testing.T) {
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
	// The child finished an hour ago — well before the 30-minute cutoff — so it
	// must not shield the stuck parent. CreateExecutionRecord stamps updated_at
	// at "now", so backdate it to make the completion genuinely old.
	backdateExecutionUpdatedAt(t, ls, "executions", "exec-done", now.Add(-time.Hour))

	reaped, err := ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped)
	require.Equal(t, "timeout", executionStatus(t, ls, "exec-build"))
	require.Equal(t, "succeeded", executionStatus(t, ls, "exec-done"), "terminal child untouched")
}

// TestMarkStaleExecutions_RecentlyFinishedChildShieldsParent guards issue #1059:
// a child that reached a terminal state *after* the cutoff protects its parent
// during the brief window between the child reporting success and the parent
// posting its own result. Without this the parent is timed out mid-flight and
// its real success callback is rejected with HTTP 409.
func TestMarkStaleExecutions_RecentlyFinishedChildShieldsParent(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	// Parent is stale by its own clock (idle 1h while it waited on the child).
	newRunningExecution(t, ls, "exec-build", "", time.Hour)
	// Child just succeeded (updated_at ~= now, well after the 30-minute cutoff).
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
	require.Equal(t, 0, reaped, "parent must survive while its child only just finished")
	require.Equal(t, "running", executionStatus(t, ls, "exec-build"))
}

// TestMarkStaleExecutions_TimeoutChildDoesNotShieldParent: a child the reaper
// itself timed out must not shield its parent, even though its timeout is
// recent — otherwise a reaped leaf would keep its stuck ancestor alive and the
// bottom-up unwind (one level per sweep) would stall.
func TestMarkStaleExecutions_TimeoutChildDoesNotShieldParent(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	newRunningExecution(t, ls, "exec-build", "", time.Hour)
	timedOut := &types.Execution{
		ExecutionID:       "exec-timeout",
		RunID:             "run-parented",
		AgentNodeID:       "agent-1",
		ReasonerID:        "reasoner-1",
		NodeID:            "node-1",
		Status:            "timeout",
		StartedAt:         now.Add(-2 * time.Hour),
		ParentExecutionID: strPtr("exec-build"),
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, timedOut))

	reaped, err := ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped)
	require.Equal(t, "timeout", executionStatus(t, ls, "exec-build"))
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

// workflowStatus returns the current status of a workflow_executions row.
func workflowStatus(t *testing.T, ls *LocalStorage, id string) string {
	t.Helper()
	wf, err := ls.GetWorkflowExecution(t.Context(), id)
	require.NoError(t, err)
	return wf.Status
}

// TestMarkStaleWorkflowExecutions_RecentlyFinishedChildShieldsParent is the
// workflow-table half of the issue #1059 fix: a parent workflow that is stale
// by its own clock must not be reaped while a child workflow only just reached
// a terminal state, so the parent's own success callback is not rejected 409.
func TestMarkStaleWorkflowExecutions_RecentlyFinishedChildShieldsParent(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()
	staleAt := now.Add(-2 * time.Hour)

	const parentID = "wf-parent-recent-child"
	require.NoError(t, ls.StoreWorkflowExecution(ctx, retryTestWorkflow(parentID, staleAt)))

	child := retryTestWorkflow("wf-child-recent", now) // updated_at ~= now
	child.Status = "succeeded"
	child.ParentExecutionID = strPtr(parentID)
	require.NoError(t, ls.StoreWorkflowExecution(ctx, child))

	reaped, err := ls.MarkStaleWorkflowExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 0, reaped, "parent workflow must survive while its child only just finished")
	require.Equal(t, "running", workflowStatus(t, ls, parentID))
}

// TestMarkStaleWorkflowExecutions_LongFinishedChildDoesNotShieldParent: a child
// workflow that finished well before the cutoff no longer shields a genuinely
// stuck parent, preserving orphan cleanup (issue #1059 acceptance).
func TestMarkStaleWorkflowExecutions_LongFinishedChildDoesNotShieldParent(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()
	staleAt := now.Add(-2 * time.Hour)

	const parentID = "wf-parent-old-child"
	require.NoError(t, ls.StoreWorkflowExecution(ctx, retryTestWorkflow(parentID, staleAt)))

	child := retryTestWorkflow("wf-child-old", staleAt) // finished long ago
	child.Status = "succeeded"
	child.ParentExecutionID = strPtr(parentID)
	require.NoError(t, ls.StoreWorkflowExecution(ctx, child))

	reaped, err := ls.MarkStaleWorkflowExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped)
	require.Equal(t, "timeout", workflowStatus(t, ls, parentID))
	require.Equal(t, "succeeded", workflowStatus(t, ls, "wf-child-old"), "terminal child untouched")
}

// TestMarkStaleExecutions_LateChildCallbackStillShieldsParent covers the review
// finding on #1063: completed_at is the agent's clock (stored verbatim), so a
// skewed agent clock or a callback that took a while to land can make
// completed_at older than the cutoff at the instant the control plane writes
// the terminal row. The shield reads the later of completed_at and updated_at
// (the control plane's own write clock), so the parent is still protected.
func TestMarkStaleExecutions_LateChildCallbackStillShieldsParent(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	newRunningExecution(t, ls, "exec-parent", "", time.Hour)
	done := &types.Execution{
		ExecutionID:       "exec-child",
		RunID:             "run-parented",
		AgentNodeID:       "agent-1",
		ReasonerID:        "reasoner-1",
		NodeID:            "node-1",
		Status:            "succeeded",
		StartedAt:         now.Add(-2 * time.Hour),
		ParentExecutionID: strPtr("exec-parent"),
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, done))

	// Child reports it finished 45 min ago (skewed agent clock, or a callback
	// that was stuck); the control plane only wrote the terminal row just now.
	_, err := ls.requireSQLDB().Exec(
		"UPDATE executions SET completed_at = ?, updated_at = ? WHERE execution_id = ?",
		now.Add(-45*time.Minute), now, "exec-child")
	require.NoError(t, err)

	reaped, err := ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 0, reaped, "a late-landing child callback (recent updated_at) must still shield the parent")
	require.Equal(t, "running", executionStatus(t, ls, "exec-parent"))
}
