package storage

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

// setupTestLocalStorage creates a real LocalStorage backed by an ephemeral SQLite database.
func setupTestLocalStorage(t *testing.T) (*LocalStorage, context.Context) {
	t.Helper()

	ctx := context.Background()
	tempDir := t.TempDir()
	cfg := StorageConfig{
		Mode: "local",
		Local: LocalStorageConfig{
			DatabasePath: filepath.Join(tempDir, "agentfield.db"),
			KVStorePath:  filepath.Join(tempDir, "agentfield.bolt"),
		},
	}

	ls := NewLocalStorage(LocalStorageConfig{})
	if err := ls.Initialize(ctx, cfg); err != nil {
		if strings.Contains(strings.ToLower(err.Error()), "fts5") {
			t.Skip("sqlite3 compiled without FTS5; skipping test")
		}
		t.Fatalf("initialize local storage: %v", err)
	}

	t.Cleanup(func() {
		_ = ls.Close(ctx)
	})

	return ls, ctx
}

// backdateExecutionUpdatedAt manually sets updated_at to a past time to simulate a stuck execution.
func backdateExecutionUpdatedAt(t *testing.T, ls *LocalStorage, table, executionID string, updatedAt time.Time) {
	t.Helper()
	db := ls.requireSQLDB()
	_, err := db.Exec(
		"UPDATE "+table+" SET updated_at = ? WHERE execution_id = ?",
		updatedAt.UTC(), executionID,
	)
	require.NoError(t, err)
}

func TestMarkStaleExecutions_ReapsStuckExecutions(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	// Insert a "stuck" execution: running, but updated_at is 1 hour ago
	stuck := &types.Execution{
		ExecutionID: "exec-stuck-1",
		RunID:       "run-1",
		AgentNodeID: "agent-1",
		ReasonerID:  "reasoner-1",
		NodeID:      "node-1",
		Status:      "running",
		StartedAt:   now.Add(-2 * time.Hour),
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, stuck))
	backdateExecutionUpdatedAt(t, ls, "executions", "exec-stuck-1", now.Add(-1*time.Hour))

	// Insert an active execution: running, updated_at is recent
	active := &types.Execution{
		ExecutionID: "exec-active-1",
		RunID:       "run-1",
		AgentNodeID: "agent-1",
		ReasonerID:  "reasoner-1",
		NodeID:      "node-1",
		Status:      "running",
		StartedAt:   now.Add(-2 * time.Hour), // started long ago, but still active
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, active))
	// Don't backdate — updated_at is now, so this one is still "active"

	// Insert a completed execution (should not be touched)
	completed := &types.Execution{
		ExecutionID: "exec-completed-1",
		RunID:       "run-1",
		AgentNodeID: "agent-1",
		ReasonerID:  "reasoner-1",
		NodeID:      "node-1",
		Status:      "succeeded",
		StartedAt:   now.Add(-3 * time.Hour),
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, completed))

	// Reap with 30-minute threshold
	reaped, err := ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped, "should reap exactly the stuck execution")

	// Verify the stuck one is now timed out
	stuckRecord, err := ls.GetExecutionRecord(ctx, "exec-stuck-1")
	require.NoError(t, err)
	require.Equal(t, "timeout", stuckRecord.Status)
	require.NotNil(t, stuckRecord.CompletedAt)
	require.NotNil(t, stuckRecord.DurationMS)
	require.Contains(t, *stuckRecord.ErrorMessage, "no activity")

	// Verify the active one is untouched
	activeRecord, err := ls.GetExecutionRecord(ctx, "exec-active-1")
	require.NoError(t, err)
	require.Equal(t, "running", activeRecord.Status)

	// Verify the completed one is untouched
	completedRecord, err := ls.GetExecutionRecord(ctx, "exec-completed-1")
	require.NoError(t, err)
	require.Equal(t, "succeeded", completedRecord.Status)
}

func TestMarkStaleExecutions_DoesNotReapActiveExecutions(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	// Insert an execution that started a long time ago but was recently updated
	longRunning := &types.Execution{
		ExecutionID: "exec-long-running",
		RunID:       "run-2",
		AgentNodeID: "agent-1",
		ReasonerID:  "reasoner-1",
		NodeID:      "node-1",
		Status:      "running",
		StartedAt:   now.Add(-5 * time.Hour), // started 5 hours ago
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, longRunning))
	// updated_at is "now" from the insert — should NOT be reaped

	reaped, err := ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 0, reaped, "long-running execution with recent activity should not be reaped")

	record, err := ls.GetExecutionRecord(ctx, "exec-long-running")
	require.NoError(t, err)
	require.Equal(t, "running", record.Status)
}

func TestMarkStaleWorkflowExecutions_ReapsOrphanedChildren(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	// Simulate the real scenario: parent failed, child is stuck running
	parentExec := &types.WorkflowExecution{
		WorkflowID:          "wf-orphan-test",
		ExecutionID:         "exec-parent",
		AgentFieldRequestID: "req-parent",
		AgentNodeID:         "agent-1",
		ReasonerID:          "orchestrator.review",
		Status:              "failed",
		StartedAt:           now.Add(-2 * time.Hour),
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-1 * time.Hour),
		WorkflowTags:        []string{},
		InputData:           json.RawMessage("{}"),
		OutputData:          json.RawMessage("{}"),
	}
	completedAt := now.Add(-1 * time.Hour)
	parentExec.CompletedAt = &completedAt
	require.NoError(t, ls.StoreWorkflowExecution(ctx, parentExec))

	// Orphaned child: still "running" but parent already failed
	orphanChild := &types.WorkflowExecution{
		WorkflowID:          "wf-orphan-test",
		ExecutionID:         "exec-orphan-child",
		AgentFieldRequestID: "req-child",
		ParentExecutionID:   strPtr("exec-parent"),
		AgentNodeID:         "agent-1",
		ReasonerID:          "reasoner.intake",
		Status:              "running",
		StartedAt:           now.Add(-2 * time.Hour),
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-2 * time.Hour),
		WorkflowTags:        []string{},
		InputData:           json.RawMessage("{}"),
		OutputData:          json.RawMessage("{}"),
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, orphanChild))
	// Backdate updated_at to ensure it looks stuck
	backdateExecutionUpdatedAt(t, ls, "workflow_executions", "exec-orphan-child", now.Add(-1*time.Hour))

	// Active child in a different workflow: running and recently updated
	activeChild := &types.WorkflowExecution{
		WorkflowID:          "wf-active-test",
		ExecutionID:         "exec-active-child",
		AgentFieldRequestID: "req-active-child",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reasoner.intake",
		Status:              "running",
		StartedAt:           now.Add(-3 * time.Hour),
		CreatedAt:           now.Add(-3 * time.Hour),
		UpdatedAt:           now, // recently active
		WorkflowTags:        []string{},
		InputData:           json.RawMessage("{}"),
		OutputData:          json.RawMessage("{}"),
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, activeChild))

	// Reap with 30-minute threshold
	reaped, err := ls.MarkStaleWorkflowExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped, "should reap exactly the orphaned child")

	// Verify the orphaned child is now timed out
	orphan, err := ls.GetWorkflowExecution(ctx, "exec-orphan-child")
	require.NoError(t, err)
	require.Equal(t, "timeout", orphan.Status)
	require.NotNil(t, orphan.CompletedAt)
	require.Contains(t, *orphan.ErrorMessage, "no activity")

	// Verify the active child is untouched
	active, err := ls.GetWorkflowExecution(ctx, "exec-active-child")
	require.NoError(t, err)
	require.Equal(t, "running", active.Status)

	// Verify the already-failed parent is untouched
	parent, err := ls.GetWorkflowExecution(ctx, "exec-parent")
	require.NoError(t, err)
	require.Equal(t, "failed", parent.Status)
}

func TestMarkStaleWorkflowExecutions_ReapsWaitingState(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	// Execution stuck in "waiting" state (e.g., waiting for approval that never came)
	waiting := &types.WorkflowExecution{
		WorkflowID:          "wf-waiting",
		ExecutionID:         "exec-waiting",
		AgentFieldRequestID: "req-waiting",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reasoner.approval",
		Status:              "waiting",
		StartedAt:           now.Add(-3 * time.Hour),
		CreatedAt:           now.Add(-3 * time.Hour),
		UpdatedAt:           now.Add(-3 * time.Hour),
		WorkflowTags:        []string{},
		InputData:           json.RawMessage("{}"),
		OutputData:          json.RawMessage("{}"),
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, waiting))
	backdateExecutionUpdatedAt(t, ls, "workflow_executions", "exec-waiting", now.Add(-1*time.Hour))

	reaped, err := ls.MarkStaleWorkflowExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped)

	record, err := ls.GetWorkflowExecution(ctx, "exec-waiting")
	require.NoError(t, err)
	require.Equal(t, "timeout", record.Status)
}

func TestMarkStaleWorkflowExecutions_MultipleStuckExecutions(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	// Create 5 stuck executions (simulating a real batch scenario)
	for i := 0; i < 5; i++ {
		exec := &types.WorkflowExecution{
			WorkflowID:          "wf-batch",
			ExecutionID:         "exec-batch-" + string(rune('a'+i)),
			AgentFieldRequestID: "req-batch-" + string(rune('a'+i)),
			AgentNodeID:         "agent-1",
			ReasonerID:          "reasoner.review",
			Status:              "running",
			StartedAt:           now.Add(-time.Duration(i+1) * time.Hour),
			CreatedAt:           now.Add(-time.Duration(i+1) * time.Hour),
			UpdatedAt:           now.Add(-time.Duration(i+1) * time.Hour),
			WorkflowTags:        []string{},
			InputData:           json.RawMessage("{}"),
			OutputData:          json.RawMessage("{}"),
		}
		require.NoError(t, ls.StoreWorkflowExecution(ctx, exec))
		backdateExecutionUpdatedAt(t, ls, "workflow_executions", exec.ExecutionID, now.Add(-time.Duration(i+1)*time.Hour))
	}

	// Reap with limit=3 to test batching
	reaped, err := ls.MarkStaleWorkflowExecutions(ctx, 30*time.Minute, 3)
	require.NoError(t, err)
	require.Equal(t, 3, reaped, "should respect the limit parameter")

	// Reap the remaining
	reaped2, err := ls.MarkStaleWorkflowExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 2, reaped2, "should reap the remaining stuck executions")

	// Third call should find nothing
	reaped3, err := ls.MarkStaleWorkflowExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 0, reaped3, "all stuck executions should already be reaped")
}

func TestMarkStaleExecutions_ReapsWhenUpdatedAtIsNULL(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	// Insert a stuck execution, then NULL out updated_at to simulate a legacy row
	// where updated_at was never populated.
	stuck := &types.Execution{
		ExecutionID: "exec-null-updated",
		RunID:       "run-null",
		AgentNodeID: "agent-1",
		ReasonerID:  "reasoner-1",
		NodeID:      "node-1",
		Status:      "running",
		StartedAt:   now.Add(-2 * time.Hour),
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, stuck))

	// NULL out updated_at and backdate created_at to simulate a legacy row
	// COALESCE(updated_at, created_at, started_at) should fall back to created_at
	db := ls.requireSQLDB()
	_, err := db.Exec(
		"UPDATE executions SET updated_at = NULL, created_at = ? WHERE execution_id = ?",
		now.Add(-2*time.Hour), "exec-null-updated",
	)
	require.NoError(t, err)

	reaped, err := ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped, "should reap execution with NULL updated_at via COALESCE fallback")

	record, err := ls.GetExecutionRecord(ctx, "exec-null-updated")
	require.NoError(t, err)
	require.Equal(t, "timeout", record.Status)
	require.Contains(t, *record.ErrorMessage, "no activity")
}

func TestMarkStaleWorkflowExecutions_ReapsWhenUpdatedAtIsNULL(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	// Insert a stuck workflow execution, then NULL out updated_at
	stuck := &types.WorkflowExecution{
		WorkflowID:          "wf-null-updated",
		ExecutionID:         "wfexec-null-updated",
		AgentFieldRequestID: "req-null",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reasoner-1",
		Status:              "running",
		StartedAt:           now.Add(-2 * time.Hour),
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-2 * time.Hour),
		WorkflowTags:        []string{},
		InputData:           json.RawMessage("{}"),
		OutputData:          json.RawMessage("{}"),
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, stuck))

	// NULL out updated_at — COALESCE should fall back to created_at or started_at
	db := ls.requireSQLDB()
	_, err := db.Exec("UPDATE workflow_executions SET updated_at = NULL WHERE execution_id = ?", "wfexec-null-updated")
	require.NoError(t, err)

	reaped, err := ls.MarkStaleWorkflowExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped, "should reap workflow execution with NULL updated_at via COALESCE fallback")

	record, err := ls.GetWorkflowExecution(ctx, "wfexec-null-updated")
	require.NoError(t, err)
	require.Equal(t, "timeout", record.Status)
	require.Contains(t, *record.ErrorMessage, "no activity")
}

func TestStaleReaper_EndToEnd_WithCleanupService(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()

	// Set up a realistic scenario: a workflow with parent + multiple children,
	// where the parent failed but children are stuck.
	runID := "run-e2e-reaper"

	// Parent execution (already failed)
	parentExec := &types.Execution{
		ExecutionID: "exec-e2e-parent",
		RunID:       runID,
		AgentNodeID: "pr-af",
		ReasonerID:  "orchestrator.review",
		NodeID:      "pr-af-node",
		Status:      "failed",
		StartedAt:   now.Add(-2 * time.Hour),
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, parentExec))

	// Child 1: stuck in running (no activity for 1 hour)
	child1 := &types.Execution{
		ExecutionID:       "exec-e2e-child-intake",
		RunID:             runID,
		ParentExecutionID: strPtr("exec-e2e-parent"),
		AgentNodeID:       "pr-af",
		ReasonerID:        "reasoner.intake",
		NodeID:            "pr-af-node",
		Status:            "running",
		StartedAt:         now.Add(-2 * time.Hour),
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, child1))
	backdateExecutionUpdatedAt(t, ls, "executions", "exec-e2e-child-intake", now.Add(-1*time.Hour))

	// Child 2: stuck in queued (no activity for 1 hour)
	child2 := &types.Execution{
		ExecutionID:       "exec-e2e-child-anatomy",
		RunID:             runID,
		ParentExecutionID: strPtr("exec-e2e-parent"),
		AgentNodeID:       "pr-af",
		ReasonerID:        "reasoner.anatomy",
		NodeID:            "pr-af-node",
		Status:            "queued",
		StartedAt:         now.Add(-2 * time.Hour),
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, child2))
	backdateExecutionUpdatedAt(t, ls, "executions", "exec-e2e-child-anatomy", now.Add(-1*time.Hour))

	// Same scenario in workflow_executions table
	parentWf := &types.WorkflowExecution{
		WorkflowID:          "wf-e2e",
		ExecutionID:         "wfexec-e2e-parent",
		AgentFieldRequestID: "req-e2e-parent",
		AgentNodeID:         "pr-af",
		ReasonerID:          "orchestrator.review",
		Status:              "failed",
		StartedAt:           now.Add(-2 * time.Hour),
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-1 * time.Hour),
		WorkflowTags:        []string{},
		InputData:           json.RawMessage("{}"),
		OutputData:          json.RawMessage("{}"),
	}
	parentWfCompletedAt := now.Add(-1 * time.Hour)
	parentWf.CompletedAt = &parentWfCompletedAt
	require.NoError(t, ls.StoreWorkflowExecution(ctx, parentWf))

	orphanWf := &types.WorkflowExecution{
		WorkflowID:          "wf-e2e",
		ExecutionID:         "wfexec-e2e-orphan",
		AgentFieldRequestID: "req-e2e-orphan",
		ParentExecutionID:   strPtr("wfexec-e2e-parent"),
		AgentNodeID:         "pr-af",
		ReasonerID:          "reasoner.intake",
		Status:              "running",
		StartedAt:           now.Add(-2 * time.Hour),
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-2 * time.Hour),
		WorkflowTags:        []string{},
		InputData:           json.RawMessage("{}"),
		OutputData:          json.RawMessage("{}"),
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, orphanWf))
	backdateExecutionUpdatedAt(t, ls, "workflow_executions", "wfexec-e2e-orphan", now.Add(-1*time.Hour))

	// Run both reaper methods (this is what the cleanup service does)
	staleTimeout := 30 * time.Minute
	batchSize := 100

	execReaped, err := ls.MarkStaleExecutions(ctx, staleTimeout, batchSize)
	require.NoError(t, err)
	require.Equal(t, 2, execReaped, "should reap 2 stuck child executions")

	wfReaped, err := ls.MarkStaleWorkflowExecutions(ctx, staleTimeout, batchSize)
	require.NoError(t, err)
	require.Equal(t, 1, wfReaped, "should reap 1 stuck workflow execution")

	// Verify all stuck executions are now timeout
	for _, id := range []string{"exec-e2e-child-intake", "exec-e2e-child-anatomy"} {
		rec, err := ls.GetExecutionRecord(ctx, id)
		require.NoError(t, err)
		require.Equal(t, "timeout", rec.Status, "execution %s should be timed out", id)
		require.NotNil(t, rec.CompletedAt, "execution %s should have completed_at set", id)
		require.NotNil(t, rec.DurationMS, "execution %s should have duration_ms set", id)
		require.True(t, *rec.DurationMS > 0, "execution %s duration should be positive", id)
	}

	orphanRecord, err := ls.GetWorkflowExecution(ctx, "wfexec-e2e-orphan")
	require.NoError(t, err)
	require.Equal(t, "timeout", orphanRecord.Status)
	require.NotNil(t, orphanRecord.CompletedAt)

	// Verify parent statuses were NOT changed
	parentRecord, err := ls.GetExecutionRecord(ctx, "exec-e2e-parent")
	require.NoError(t, err)
	require.Equal(t, "failed", parentRecord.Status, "parent should remain failed")

	parentWfRecord, err := ls.GetWorkflowExecution(ctx, "wfexec-e2e-parent")
	require.NoError(t, err)
	require.Equal(t, "failed", parentWfRecord.Status, "parent workflow should remain failed")

	// Running again should find nothing
	execReaped2, err := ls.MarkStaleExecutions(ctx, staleTimeout, batchSize)
	require.NoError(t, err)
	require.Equal(t, 0, execReaped2, "second pass should find no stale executions")

	wfReaped2, err := ls.MarkStaleWorkflowExecutions(ctx, staleTimeout, batchSize)
	require.NoError(t, err)
	require.Equal(t, 0, wfReaped2, "second pass should find no stale workflow executions")
}

func strPtr(s string) *string {
	return &s
}
