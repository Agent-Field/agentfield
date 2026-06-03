package storage

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

// strPtr is shared with stale_execution_reaper_test.go (same package). If the
// helper is missing from this package later, this file will fail to compile —
// that is the intended signal.

// seedRunningWorkflowExecution inserts a workflow_executions row in `running`
// state owned by the given agent, plus a paired executions-table row so the
// orphan reaper has both sides to reap.
func seedRunningWorkflowExecution(
	t *testing.T,
	ls *LocalStorage,
	executionID, agentNodeID string,
	startedAt time.Time,
) {
	t.Helper()

	wf := &types.WorkflowExecution{
		WorkflowID:          "wf-" + executionID,
		ExecutionID:         executionID,
		AgentFieldRequestID: "req-" + executionID,
		AgentNodeID:         agentNodeID,
		ReasonerID:          "test.reasoner",
		Status:              "running",
		StartedAt:           startedAt,
		CreatedAt:           startedAt,
		UpdatedAt:           startedAt,
		WorkflowTags:        []string{},
		InputData:           json.RawMessage("{}"),
		OutputData:          json.RawMessage("{}"),
	}
	require.NoError(t, ls.StoreWorkflowExecution(t.Context(), wf))

	exec := &types.Execution{
		ExecutionID: executionID,
		RunID:       "run-" + executionID,
		AgentNodeID: agentNodeID,
		ReasonerID:  "test.reasoner",
		NodeID:      agentNodeID,
		Status:      "running",
		StartedAt:   startedAt,
	}
	require.NoError(t, ls.CreateExecutionRecord(t.Context(), exec))
}

// TestMarkAgentExecutionsOrphaned_ReapsByAgent confirms the core invariant:
// every non-terminal execution owned by the given agent_node_id is failed,
// and rows belonging to OTHER agents are untouched. This is the load-bearing
// behavior — without it, a single redeploy of one agent would never clean up
// orphaned cross-agent calls and could potentially fail unrelated work.
func TestMarkAgentExecutionsOrphaned_ReapsByAgent(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	// Two running executions owned by github-buddy.
	seedRunningWorkflowExecution(t, ls, "exec-gh-1", "github-buddy", now.Add(-30*time.Minute))
	seedRunningWorkflowExecution(t, ls, "exec-gh-2", "github-buddy", now.Add(-2*time.Minute))

	// One running execution owned by an UNRELATED agent — must NOT be reaped.
	seedRunningWorkflowExecution(t, ls, "exec-pr-af-1", "pr-af", now.Add(-30*time.Minute))

	// One COMPLETED execution owned by github-buddy — must be left alone
	// (terminal status is final; reaper never resurrects).
	completed := &types.WorkflowExecution{
		WorkflowID:          "wf-exec-gh-done",
		ExecutionID:         "exec-gh-done",
		AgentFieldRequestID: "req-exec-gh-done",
		AgentNodeID:         "github-buddy",
		ReasonerID:          "test.reasoner",
		Status:              "succeeded",
		StartedAt:           now.Add(-1 * time.Hour),
		CreatedAt:           now.Add(-1 * time.Hour),
		UpdatedAt:           now.Add(-30 * time.Minute),
		WorkflowTags:        []string{},
		InputData:           json.RawMessage("{}"),
		OutputData:          json.RawMessage("{}"),
	}
	completedAt := now.Add(-30 * time.Minute)
	completed.CompletedAt = &completedAt
	require.NoError(t, ls.StoreWorkflowExecution(ctx, completed))

	reaped, err := ls.MarkAgentExecutionsOrphaned(ctx, "github-buddy",
		"agent_restart_orphaned: github-buddy re-registered with new instance")
	require.NoError(t, err)
	require.Equal(t, 2, reaped, "should reap exactly the two running github-buddy executions")

	// Both github-buddy running execs are now failed with reason set.
	for _, id := range []string{"exec-gh-1", "exec-gh-2"} {
		got, err := ls.GetWorkflowExecution(ctx, id)
		require.NoError(t, err, "execution %s should exist", id)
		require.Equal(t, "failed", got.Status, "execution %s should be failed", id)
		require.NotNil(t, got.CompletedAt, "execution %s should have completed_at", id)
		require.NotNil(t, got.StatusReason, "execution %s should have status_reason set", id)
		require.True(t,
			strings.Contains(*got.StatusReason, "agent_restart_orphaned"),
			"execution %s status_reason should mention agent_restart_orphaned, got %q",
			id, *got.StatusReason,
		)
		require.NotNil(t, got.ErrorMessage, "execution %s should have error_message set", id)
	}

	// pr-af exec is still running (different agent — must not be touched).
	prAf, err := ls.GetWorkflowExecution(ctx, "exec-pr-af-1")
	require.NoError(t, err)
	require.Equal(t, "running", prAf.Status,
		"pr-af execution must NOT be reaped when github-buddy restarts")

	// Completed exec is still succeeded.
	doneRec, err := ls.GetWorkflowExecution(ctx, "exec-gh-done")
	require.NoError(t, err)
	require.Equal(t, "succeeded", doneRec.Status,
		"already-succeeded execution must NOT flip to failed")
}

// TestMarkAgentExecutionsOrphaned_ReapsAllNonTerminalStatuses ensures we don't
// silently leave `pending`, `queued`, or `waiting` executions behind. They're
// equally orphaned by a process restart.
func TestMarkAgentExecutionsOrphaned_ReapsAllNonTerminalStatuses(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	for _, status := range []string{"running", "pending", "queued", "waiting"} {
		wf := &types.WorkflowExecution{
			WorkflowID:          "wf-" + status,
			ExecutionID:         "exec-" + status,
			AgentFieldRequestID: "req-" + status,
			AgentNodeID:         "github-buddy",
			ReasonerID:          "test.reasoner",
			Status:              status,
			StartedAt:           now.Add(-5 * time.Minute),
			CreatedAt:           now.Add(-5 * time.Minute),
			UpdatedAt:           now.Add(-5 * time.Minute),
			WorkflowTags:        []string{},
			InputData:           json.RawMessage("{}"),
			OutputData:          json.RawMessage("{}"),
		}
		require.NoError(t, ls.StoreWorkflowExecution(ctx, wf))
	}

	reaped, err := ls.MarkAgentExecutionsOrphaned(ctx, "github-buddy", "test reap")
	require.NoError(t, err)
	require.Equal(t, 4, reaped, "all four non-terminal statuses must be reaped")

	for _, status := range []string{"running", "pending", "queued", "waiting"} {
		got, err := ls.GetWorkflowExecution(ctx, "exec-"+status)
		require.NoError(t, err)
		require.Equal(t, "failed", got.Status,
			"status=%s execution should be reaped to failed", status)
	}
}

// TestMarkAgentExecutionsOrphaned_DoesNotResurrectTerminalStatuses pins that
// even with the reaper triggered, succeeded/failed/cancelled/timeout rows are
// never touched. A subtle bug here would let a redeploy "un-finish" an
// already-completed execution, scrambling the audit trail.
func TestMarkAgentExecutionsOrphaned_DoesNotResurrectTerminalStatuses(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	for _, status := range []string{"succeeded", "failed", "cancelled", "timeout"} {
		startedAt := now.Add(-30 * time.Minute)
		completedAt := now.Add(-15 * time.Minute)
		wf := &types.WorkflowExecution{
			WorkflowID:          "wf-" + status,
			ExecutionID:         "exec-" + status,
			AgentFieldRequestID: "req-" + status,
			AgentNodeID:         "github-buddy",
			ReasonerID:          "test.reasoner",
			Status:              status,
			StartedAt:           startedAt,
			CompletedAt:         &completedAt,
			CreatedAt:           startedAt,
			UpdatedAt:           completedAt,
			WorkflowTags:        []string{},
			InputData:           json.RawMessage("{}"),
			OutputData:          json.RawMessage("{}"),
		}
		require.NoError(t, ls.StoreWorkflowExecution(ctx, wf))
	}

	reaped, err := ls.MarkAgentExecutionsOrphaned(ctx, "github-buddy", "test reap")
	require.NoError(t, err)
	require.Equal(t, 0, reaped, "no rows should be reaped when none are non-terminal")

	for _, status := range []string{"succeeded", "failed", "cancelled", "timeout"} {
		got, err := ls.GetWorkflowExecution(ctx, "exec-"+status)
		require.NoError(t, err)
		require.Equal(t, status, got.Status,
			"terminal status %s must not be modified by orphan reap", status)
	}
}

// TestMarkAgentExecutionsOrphaned_NoOpOnEmptyAgent guards against accidentally
// reaping every execution in the database when the caller passes an empty
// agent_node_id (e.g. via uninitialized variable). This must error, not run.
func TestMarkAgentExecutionsOrphaned_NoOpOnEmptyAgent(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()
	seedRunningWorkflowExecution(t, ls, "exec-1", "github-buddy", now)

	_, err := ls.MarkAgentExecutionsOrphaned(ctx, "", "test reap")
	require.Error(t, err, "empty agent_node_id must be rejected to prevent global reap")

	// Verify the running exec was untouched.
	got, err := ls.GetWorkflowExecution(ctx, "exec-1")
	require.NoError(t, err)
	require.Equal(t, "running", got.Status)
}

// TestMarkAgentExecutionsOrphaned_DefaultReasonWhenEmpty ensures that an empty
// reason still produces a meaningful audit string rather than a NULL/empty
// status_reason. Helps when an operator inspects the row after the fact.
func TestMarkAgentExecutionsOrphaned_DefaultReasonWhenEmpty(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()
	seedRunningWorkflowExecution(t, ls, "exec-1", "github-buddy", now)

	reaped, err := ls.MarkAgentExecutionsOrphaned(ctx, "github-buddy", "")
	require.NoError(t, err)
	require.Equal(t, 1, reaped)

	got, err := ls.GetWorkflowExecution(ctx, "exec-1")
	require.NoError(t, err)
	require.NotNil(t, got.StatusReason)
	require.NotEmpty(t, *got.StatusReason, "empty reason must be replaced with a default audit string")
}
