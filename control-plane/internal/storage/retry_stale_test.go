package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Interleaving inventory for RetryStaleWorkflowExecutions.
//
// A candidate is a workflow row W and its paired execution row E. The sweep
// selects candidates and then, inside one transaction, updates W and then E
// with each candidate wrapped in a savepoint. The windows are:
//
//  1. Activity lands on E before selection. E is no longer stale, the selection
//     predicate drops the candidate, and neither row moves. Covered by
//     TestRetryStaleWorkflowExecutions_ExecutionActivityProtectsWorkflow and
//     TestRetryStaleWorkflowExecutions_FreshNonUTCTimestampNotRetried.
//  2. Activity lands on W before selection. W is no longer stale, the selection
//     predicate drops the candidate. Covered by TestRetryStaleWorkflowExecutions
//     (fresh workflow fixture).
//  3. Activity lands on W or E after selection but before the transaction's
//     first statement. The workflow UPDATE repeats the selection predicates, so
//     it matches zero rows and the candidate is skipped without a savepoint.
//     Covered by ..._ActivityAfterSelectionSkipsUpdate (E) and
//     ..._WorkflowActivityAfterSelectionSkipsUpdate (W).
//  4. Activity lands on E after the workflow UPDATE and before the execution
//     UPDATE. The execution UPDATE matches zero rows, the recheck sees an
//     active execution that is no longer stale, and the whole candidate is
//     rolled back to its savepoint: W untouched, retry_count unchanged, id
//     absent from the returned set, and E keeps the fresh heartbeat. Covered by
//     ..._HeartbeatBetweenStatementsSparesExecution on SQLite and live
//     PostgreSQL, and at batch level by ..._BatchReportsOnlyMovedCandidates.
//  5. E becomes terminal or waiting between selection and the execution UPDATE.
//     No running/pending/queued paired record is left, so the pre-existing
//     workflow-only recovery path applies: only W moves and E is never
//     modified. Terminal is covered by ..._TerminalPairedExecutionStillRetried;
//     the waiting variant reaches the same branch because the execution UPDATE
//     deliberately excludes 'waiting' (see commit c73dc4d1). Treating a
//     terminal execution as "no live work" is the documented contract that the
//     fix preserves.
//  6. A heartbeat on E lands after the execution UPDATE but before commit. The
//     update holds E's row lock on PostgreSQL, so the heartbeat serialises
//     behind the commit and applies to the pending row from outside the call;
//     the call cannot lose a heartbeat it has not yet observed.
//  7. A statement fails or the context is cancelled mid-batch. defer
//     rollbackTx discards every candidate, so no half-moved pair can survive;
//     the returned error is authoritative over the returned ids. The invariant
//     is "nothing commits", so it needs no interleaving proof beyond the
//     deferred rollback.
//
// MarkStaleWorkflowExecutions has a different shape: once its workflow guard
// has passed, its workflow UPDATE and its executions sync UPDATE both run, and
// the sync UPDATE carries no staleness predicate. A heartbeat that lands
// between the two statements is therefore overwritten by the sync, leaving
// both rows terminal together. The pair never splits (the consistency property
// this inventory tracks); the lost heartbeat is a pre-existing false-positive
// window in the reaper's sync mirror, unchanged by the retry fix and out of
// scope here.
//
// Helpers shared by the regressions below.

// retryTestWorkflow builds a stale-eligible workflow row whose timestamps are
// exactly the supplied instant so tests can assert that a skipped candidate
// keeps its committed updated_at.
func retryTestWorkflow(id string, updatedAt time.Time) *types.WorkflowExecution {
	return &types.WorkflowExecution{
		WorkflowID:          "wf-" + id,
		ExecutionID:         id,
		AgentFieldRequestID: "req-" + id,
		AgentNodeID:         "agent-retry",
		ReasonerID:          "reasoner-retry",
		Status:              "running",
		StartedAt:           updatedAt,
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          0,
		CreatedAt:           updatedAt,
		UpdatedAt:           updatedAt,
	}
}

// retryTestExecution builds the paired legacy row. CreateExecutionRecord
// stamps created_at/updated_at to now, so callers backdate updated_at when the
// fixture must look stale.
func retryTestExecution(id string, startedAt time.Time) *types.Execution {
	return &types.Execution{
		ExecutionID:  id,
		RunID:        "run-" + id,
		AgentNodeID:  "agent-retry",
		ReasonerID:   "reasoner-retry",
		NodeID:       "agent-retry",
		Status:       "running",
		StartedAt:    startedAt,
		CreatedAt:    startedAt,
		UpdatedAt:    startedAt,
		InputPayload: json.RawMessage(`{}`),
	}
}

func setupRetryTestStorage(t *testing.T) (*LocalStorage, context.Context) {
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
	t.Cleanup(func() { _ = ls.Close(ctx) })
	return ls, ctx
}

func TestRetryStaleWorkflowExecutions(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()

	// Create a stale workflow execution with retry_count=0
	staleExec := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-test",
		ExecutionID:         "exec-retry-1",
		AgentFieldRequestID: "req-1",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reason-1",
		Status:              "running",
		StartedAt:           now.Add(-2 * time.Hour),
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          0,
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-2 * time.Hour),
	}
	err := ls.StoreWorkflowExecution(ctx, staleExec)
	require.NoError(t, err)
	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID:  "exec-retry-1",
		RunID:        "run-retry-1",
		AgentNodeID:  "agent-1",
		ReasonerID:   "reason-1",
		NodeID:       "agent-1",
		Status:       "running",
		StartedAt:    now.Add(-2 * time.Hour),
		CreatedAt:    now.Add(-2 * time.Hour),
		UpdatedAt:    now.Add(-2 * time.Hour),
		InputPayload: json.RawMessage(`{}`),
	}))
	// CreateExecutionRecord initializes activity timestamps to the current time;
	// backdate the paired row so both clocks are stale for this retry case.
	backdateExecutionUpdatedAt(t, ls, "executions", "exec-retry-1", now.Add(-2*time.Hour))

	// Create a stale execution that already exhausted retries
	exhaustedExec := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-test",
		ExecutionID:         "exec-exhausted",
		AgentFieldRequestID: "req-2",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reason-1",
		Status:              "running",
		StartedAt:           now.Add(-2 * time.Hour),
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          3,
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-2 * time.Hour),
	}
	err = ls.StoreWorkflowExecution(ctx, exhaustedExec)
	require.NoError(t, err)

	// Create a fresh (non-stale) execution
	freshExec := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-test",
		ExecutionID:         "exec-fresh",
		AgentFieldRequestID: "req-3",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reason-1",
		Status:              "running",
		StartedAt:           now,
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          0,
		CreatedAt:           now,
		UpdatedAt:           now,
	}
	err = ls.StoreWorkflowExecution(ctx, freshExec)
	require.NoError(t, err)

	// Retry with maxRetries=3 and staleAfter=1 hour
	retriedIDs, err := ls.RetryStaleWorkflowExecutions(ctx, 1*time.Hour, 3, 100)
	require.NoError(t, err)

	// Only exec-retry-1 should be retried (stale + under max retries)
	assert.Equal(t, 1, len(retriedIDs))
	assert.Equal(t, "exec-retry-1", retriedIDs[0])

	// Verify the execution was reset to pending with incremented retry_count
	retried, err := ls.GetWorkflowExecution(ctx, "exec-retry-1")
	require.NoError(t, err)
	assert.Equal(t, "pending", retried.Status)
	assert.Equal(t, 1, retried.RetryCount)
	assert.Nil(t, retried.CompletedAt)

	executionRecord, err := ls.GetExecutionRecord(ctx, "exec-retry-1")
	require.NoError(t, err)
	assert.Equal(t, "pending", executionRecord.Status)
	assert.Nil(t, executionRecord.CompletedAt)

	// Verify exhausted execution was NOT retried
	exhausted, err := ls.GetWorkflowExecution(ctx, "exec-exhausted")
	require.NoError(t, err)
	assert.Equal(t, "running", exhausted.Status)
	assert.Equal(t, 3, exhausted.RetryCount)

	// Verify fresh execution was NOT retried
	fresh, err := ls.GetWorkflowExecution(ctx, "exec-fresh")
	require.NoError(t, err)
	assert.Equal(t, "running", fresh.Status)
	assert.Equal(t, 0, fresh.RetryCount)
}

func TestRetryStaleWorkflowExecutions_ExecutionActivityProtectsWorkflow(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()

	workflow := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-heartbeat",
		ExecutionID:         "exec-retry-heartbeat",
		AgentFieldRequestID: "req-retry-heartbeat",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reason-1",
		Status:              "running",
		StartedAt:           now.Add(-2 * time.Hour),
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-1 * time.Hour),
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          0,
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, workflow))

	// The execution is fresh even though its paired workflow row is stale.
	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID: "exec-retry-heartbeat",
		RunID:       "run-retry-heartbeat",
		AgentNodeID: "agent-1",
		ReasonerID:  "reason-1",
		NodeID:      "agent-1",
		Status:      "running",
		StartedAt:   now.Add(-2 * time.Hour),
	}))

	retried, err := ls.RetryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100)
	require.NoError(t, err)
	require.Empty(t, retried, "fresh paired execution activity must protect a stale workflow")

	workflowRecord, err := ls.GetWorkflowExecution(ctx, workflow.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "running", workflowRecord.Status)
	require.Equal(t, 0, workflowRecord.RetryCount)
}

func TestRetryStaleWorkflowExecutions_ActivityAfterSelectionSkipsUpdate(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()

	workflow := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-selection-race",
		ExecutionID:         "exec-retry-selection-race",
		AgentFieldRequestID: "req-retry-selection-race",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reason-1",
		Status:              "running",
		StartedAt:           now.Add(-2 * time.Hour),
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-1 * time.Hour),
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          0,
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, workflow))

	execution := &types.Execution{
		ExecutionID: "exec-retry-selection-race",
		RunID:       "run-retry-selection-race",
		AgentNodeID: "agent-1",
		ReasonerID:  "reason-1",
		NodeID:      "agent-1",
		Status:      "running",
		StartedAt:   now.Add(-2 * time.Hour),
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, execution))
	backdateExecutionUpdatedAt(t, ls, "executions", execution.ExecutionID, now.Add(-1*time.Hour))

	var heartbeatErr error
	retried, err := ls.retryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100, func() {
		_, heartbeatErr = ls.UpdateExecutionRecord(ctx, execution.ExecutionID, func(current *types.Execution) (*types.Execution, error) {
			current.Notes = append(current.Notes, types.ExecutionNote{
				Message:   "heartbeat",
				Timestamp: now,
			})
			return current, nil
		})
	})
	require.NoError(t, heartbeatErr)
	require.NoError(t, err)
	require.Empty(t, retried, "activity after selection must prevent the retry update")

	workflowRecord, err := ls.GetWorkflowExecution(ctx, workflow.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "running", workflowRecord.Status)
	require.Equal(t, 0, workflowRecord.RetryCount)

	executionRecord, err := ls.GetExecutionRecord(ctx, execution.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "running", executionRecord.Status)
	require.Len(t, executionRecord.Notes, 1)
}

// TestRetryStaleWorkflowExecutions_HeartbeatBetweenStatementsSparesExecution
// covers the window between the workflow UPDATE and the paired execution
// UPDATE. An AFTER UPDATE trigger on workflow_executions simulates a heartbeat
// that commits after the workflow guard has been evaluated: the execution's
// activity clock moves while the retry is between its two statements. The
// execution UPDATE must repeat the staleness predicate, and when it loses to
// the heartbeat the candidate must be all-or-nothing: the workflow half is
// rolled back untouched, retry_count stays put, the id is absent from the
// returned set, and the execution half keeps its live state.
func TestRetryStaleWorkflowExecutions_HeartbeatBetweenStatementsSparesExecution(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()
	staleAt := now.Add(-2 * time.Hour)

	workflow := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-between-statements",
		ExecutionID:         "exec-retry-between-statements",
		AgentFieldRequestID: "req-retry-between-statements",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reason-1",
		Status:              "running",
		StartedAt:           staleAt,
		CreatedAt:           staleAt,
		UpdatedAt:           staleAt,
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          0,
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, workflow))

	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID:  workflow.ExecutionID,
		RunID:        "run-retry-between-statements",
		AgentNodeID:  "agent-1",
		ReasonerID:   "reason-1",
		NodeID:       "agent-1",
		Status:       "running",
		StartedAt:    staleAt,
		InputPayload: json.RawMessage(`{}`),
	}))
	backdateExecutionUpdatedAt(t, ls, "executions", workflow.ExecutionID, staleAt)

	// The trigger stamps a heartbeat on the paired execution as soon as the
	// retry's workflow UPDATE has matched, i.e. between the retry's two
	// statements.
	db := ls.requireSQLDB()
	_, err := db.Exec(`
		CREATE TRIGGER retry_heartbeat_between_statements
		AFTER UPDATE ON workflow_executions
		FOR EACH ROW
		BEGIN
			UPDATE executions
			SET updated_at = CURRENT_TIMESTAMP,
			    status_reason = 'heartbeat-between-statements'
			WHERE execution_id = NEW.execution_id;
		END`)
	require.NoError(t, err)
	t.Cleanup(func() {
		_, _ = db.Exec("DROP TRIGGER IF EXISTS retry_heartbeat_between_statements")
	})

	retried, err := ls.RetryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100)
	require.NoError(t, err)
	require.Empty(t, retried,
		"a candidate whose execution update loses to a heartbeat must not be reported as retried")

	executionRecord, err := ls.GetExecutionRecord(ctx, workflow.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "running", executionRecord.Status,
		"a heartbeat between the retry statements must not be dragged back to pending")
	require.Nil(t, executionRecord.CompletedAt,
		"the execution half must keep its live state")
	require.Nil(t, executionRecord.ErrorMessage,
		"the execution half must not keep the retry error message")
	require.True(t, executionRecord.UpdatedAt.Equal(staleAt),
		"the execution half must be rolled back with its candidate")
	// The trigger's heartbeat is written inside the candidate's transaction, so
	// the rollback returns the row to its committed state. The live PostgreSQL
	// test covers a heartbeat that commits independently and must survive.

	workflowRecord, err := ls.GetWorkflowExecution(ctx, workflow.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "running", workflowRecord.Status,
		"the workflow half of a candidate that cannot move as a pair must be rolled back")
	require.Equal(t, 0, workflowRecord.RetryCount,
		"retry_count must not be incremented when the pair cannot move together")
	require.Nil(t, workflowRecord.CompletedAt,
		"the rolled-back workflow must not keep a completion time")
	require.Nil(t, workflowRecord.ErrorMessage,
		"the rolled-back workflow must not keep the retry error message")
	require.True(t, workflowRecord.UpdatedAt.Equal(workflow.UpdatedAt),
		"the rolled-back workflow must keep its original updated_at")
}

// TestRetryStaleWorkflowExecutions_WorkflowWithoutPairedExecutionStillRetried
// pins the pre-existing workflow-only path: a stale workflow with no paired
// execution row is still reset, because there is no second record that could
// fall out of sync with it.
func TestRetryStaleWorkflowExecutions_WorkflowWithoutPairedExecutionStillRetried(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()

	workflow := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-unpaired",
		ExecutionID:         "exec-retry-unpaired",
		AgentFieldRequestID: "req-retry-unpaired",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reason-1",
		Status:              "running",
		StartedAt:           now.Add(-2 * time.Hour),
		CreatedAt:           now.Add(-2 * time.Hour),
		UpdatedAt:           now.Add(-2 * time.Hour),
		InputData:           json.RawMessage(`{}`),
		OutputData:          json.RawMessage(`{}`),
		RetryCount:          0,
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, workflow))

	retried, err := ls.RetryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100)
	require.NoError(t, err)
	require.Equal(t, []string{workflow.ExecutionID}, retried)

	workflowRecord, err := ls.GetWorkflowExecution(ctx, workflow.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "pending", workflowRecord.Status)
	require.Equal(t, 1, workflowRecord.RetryCount)
}

// TestRetryStaleWorkflowExecutions_WorkflowActivityAfterSelectionSkipsUpdate
// covers the workflow half of the selection-to-update window: activity that
// moves the workflow clock after candidate selection must fail the repeated
// workflow predicate and leave both records alone.
func TestRetryStaleWorkflowExecutions_WorkflowActivityAfterSelectionSkipsUpdate(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()

	workflow := retryTestWorkflow("exec-retry-workflow-selection-race", now.Add(-2*time.Hour))
	require.NoError(t, ls.StoreWorkflowExecution(ctx, workflow))

	execution := retryTestExecution(workflow.ExecutionID, now.Add(-2*time.Hour))
	require.NoError(t, ls.CreateExecutionRecord(ctx, execution))
	backdateExecutionUpdatedAt(t, ls, "executions", execution.ExecutionID, now.Add(-2*time.Hour))

	var heartbeatErr error
	retried, err := ls.retryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100, func() {
		// The workflow row gains activity after selection; the conditional
		// workflow update repeats the staleness predicate and must drop the
		// candidate without touching the savepoint.
		heartbeatErr = ls.UpdateWorkflowExecution(ctx, workflow.ExecutionID,
			func(current *types.WorkflowExecution) (*types.WorkflowExecution, error) {
				return current, nil
			})
	})
	require.NoError(t, heartbeatErr)
	require.NoError(t, err)
	require.Empty(t, retried, "workflow activity after selection must prevent the retry")

	workflowRecord, err := ls.GetWorkflowExecution(ctx, workflow.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "running", workflowRecord.Status)
	require.Equal(t, 0, workflowRecord.RetryCount)
	require.True(t, workflowRecord.UpdatedAt.After(now.Add(-time.Minute)),
		"the heartbeat timestamp must be the surviving one")

	executionRecord, err := ls.GetExecutionRecord(ctx, execution.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "running", executionRecord.Status)
}

// TestRetryStaleWorkflowExecutions_TerminalPairedExecutionStillRetried pins the
// pre-existing workflow-only recovery path: once the paired execution is
// terminal there is no live work left to protect, so the stale workflow is
// still reset. The retry never modifies the terminal execution row, so only
// the workflow half of the pair moves by design; this is the same contract
// that keeps an unpaired workflow retriable.
func TestRetryStaleWorkflowExecutions_TerminalPairedExecutionStillRetried(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()

	workflow := retryTestWorkflow("exec-retry-terminal-paired", now.Add(-2*time.Hour))
	require.NoError(t, ls.StoreWorkflowExecution(ctx, workflow))

	execution := retryTestExecution(workflow.ExecutionID, now.Add(-2*time.Hour))
	execution.Status = string(types.ExecutionStatusSucceeded)
	require.NoError(t, ls.CreateExecutionRecord(ctx, execution))
	executionUpdatedAt := now.Add(-2 * time.Hour)
	backdateExecutionUpdatedAt(t, ls, "executions", execution.ExecutionID, executionUpdatedAt)

	retried, err := ls.RetryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100)
	require.NoError(t, err)
	require.Equal(t, []string{workflow.ExecutionID}, retried,
		"a stale workflow with no live paired record keeps the recovery retry")

	workflowRecord, err := ls.GetWorkflowExecution(ctx, workflow.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, "pending", workflowRecord.Status)
	require.Equal(t, 1, workflowRecord.RetryCount)
	require.Nil(t, workflowRecord.CompletedAt)

	executionRecord, err := ls.GetExecutionRecord(ctx, execution.ExecutionID)
	require.NoError(t, err)
	require.Equal(t, string(types.ExecutionStatusSucceeded), executionRecord.Status,
		"the terminal execution half must not be rewritten")
	require.True(t, executionRecord.UpdatedAt.Equal(executionUpdatedAt),
		"the terminal execution half must keep its committed timestamp")
}

// TestStaleParentWithLiveChildIsSparedByReaperAndRetry is the regression for
// the reaper's parent/child guard leaking into the retry sweep: a parent whose
// own workflow and execution clocks are stale while it waits on a live child
// must be spared by MarkStaleWorkflowExecutions and, equally, by
// RetryStaleWorkflowExecutions. Without the child guard in the retry, the
// parent is reset to pending and its paired execution is dragged back to
// pending while the child is still working.
func TestStaleParentWithLiveChildIsSparedByReaperAndRetry(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()
	staleAt := now.Add(-2 * time.Hour)

	// Parent: both clocks are two hours stale, so absent the child guard it is
	// a candidate for both sweeps.
	const parentID = "exec-parent-live-child"
	require.NoError(t, ls.StoreWorkflowExecution(ctx, retryTestWorkflow(parentID, staleAt)))
	require.NoError(t, ls.CreateExecutionRecord(ctx, retryTestExecution(parentID, staleAt)))
	backdateExecutionUpdatedAt(t, ls, "executions", parentID, staleAt)

	// Child: created now, still running, and parented to the parent. Its
	// activity is what must shield the parent.
	const childID = "exec-child-live"
	childWorkflow := retryTestWorkflow(childID, now)
	childWorkflow.ParentExecutionID = strPtr(parentID)
	require.NoError(t, ls.StoreWorkflowExecution(ctx, childWorkflow))

	childExecution := retryTestExecution(childID, now)
	childExecution.ParentExecutionID = strPtr(parentID)
	require.NoError(t, ls.CreateExecutionRecord(ctx, childExecution))

	parentBefore, err := ls.GetWorkflowExecution(ctx, parentID)
	require.NoError(t, err)
	childBefore, err := ls.GetWorkflowExecution(ctx, childID)
	require.NoError(t, err)
	childExecutionBefore, err := ls.GetExecutionRecord(ctx, childID)
	require.NoError(t, err)

	reaped, err := ls.MarkStaleWorkflowExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 0, reaped, "a live child must shield its parent from the reaper")

	retried, err := ls.RetryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100)
	require.NoError(t, err)
	require.Empty(t, retried, "a live child must shield its parent from the retry sweep")

	parent, err := ls.GetWorkflowExecution(ctx, parentID)
	require.NoError(t, err)
	require.Equal(t, "running", parent.Status)
	require.Equal(t, 0, parent.RetryCount)
	require.Nil(t, parent.ErrorMessage)
	require.Nil(t, parent.CompletedAt)
	require.True(t, parent.UpdatedAt.Equal(parentBefore.UpdatedAt),
		"the parent must keep its committed updated_at")

	parentExecution, err := ls.GetExecutionRecord(ctx, parentID)
	require.NoError(t, err)
	require.Equal(t, "running", parentExecution.Status)

	child, err := ls.GetWorkflowExecution(ctx, childID)
	require.NoError(t, err)
	require.Equal(t, "running", child.Status)
	require.Equal(t, 0, child.RetryCount)
	require.Nil(t, child.CompletedAt)
	require.True(t, child.UpdatedAt.Equal(childBefore.UpdatedAt), "the child must be untouched")

	childExecutionAfter, err := ls.GetExecutionRecord(ctx, childID)
	require.NoError(t, err)
	require.Equal(t, "running", childExecutionAfter.Status)
	require.True(t, childExecutionAfter.UpdatedAt.Equal(childExecutionBefore.UpdatedAt),
		"the child execution must be untouched")
}

// TestRetryStaleWorkflowExecutions_BatchReportsOnlyMovedCandidates covers the
// batch accounting contract: when one sweep selects several stale candidates
// and only some of them lose their paired execution to a heartbeat between the
// retry's two statements, the reported count and id list must name exactly the
// candidates whose records moved as a pair. The losers must be all-or-nothing
// (workflow untouched, retry_count unchanged, id absent, execution state
// intact) and the winner must move both halves.
func TestRetryStaleWorkflowExecutions_BatchReportsOnlyMovedCandidates(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()

	const (
		losesOldest = "exec-batch-loses-oldest"
		moves       = "exec-batch-moves"
		losesNewest = "exec-batch-loses-newest"
	)

	// Distinct workflow clocks pin the selection order: oldest first, so the
	// single sweep visits every candidate.
	candidates := []struct {
		id          string
		workflowAge time.Duration
	}{
		{id: losesOldest, workflowAge: 3 * time.Hour},
		{id: moves, workflowAge: 2 * time.Hour},
		{id: losesNewest, workflowAge: 1 * time.Hour},
	}
	workflowUpdatedAt := make(map[string]time.Time, len(candidates))
	executionUpdatedAt := make(map[string]time.Time, len(candidates))
	for _, c := range candidates {
		workflowUpdatedAt[c.id] = now.Add(-c.workflowAge)
		executionUpdatedAt[c.id] = now.Add(-2 * time.Hour)
		require.NoError(t, ls.StoreWorkflowExecution(ctx, retryTestWorkflow(c.id, workflowUpdatedAt[c.id])))
		require.NoError(t, ls.CreateExecutionRecord(ctx, retryTestExecution(c.id, executionUpdatedAt[c.id])))
		backdateExecutionUpdatedAt(t, ls, "executions", c.id, executionUpdatedAt[c.id])
	}

	// The heartbeat lands between the retry's workflow UPDATE and its paired
	// execution UPDATE for the oldest and newest candidates only. The middle
	// candidate is silent and is the only pair that may move.
	db := ls.requireSQLDB()
	_, err := db.Exec(fmt.Sprintf(`
		CREATE TRIGGER retry_batch_heartbeat
		AFTER UPDATE ON workflow_executions
		FOR EACH ROW
		WHEN NEW.execution_id IN ('%s', '%s')
		BEGIN
			UPDATE executions
			SET updated_at = CURRENT_TIMESTAMP,
			    status_reason = 'heartbeat-between-statements'
			WHERE execution_id = NEW.execution_id;
		END`, losesOldest, losesNewest))
	require.NoError(t, err)
	t.Cleanup(func() {
		_, _ = db.Exec("DROP TRIGGER IF EXISTS retry_batch_heartbeat")
	})

	retried, err := ls.RetryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100)
	require.NoError(t, err)
	require.Equal(t, 1, len(retried),
		"the reported count must match the candidates whose pair moved")
	require.Equal(t, []string{moves}, retried,
		"the reported ids must match the candidates whose pair moved")

	for _, id := range []string{losesOldest, losesNewest} {
		workflowRecord, err := ls.GetWorkflowExecution(ctx, id)
		require.NoError(t, err)
		require.Equal(t, "running", workflowRecord.Status,
			"%s must not be dragged to pending", id)
		require.Equal(t, 0, workflowRecord.RetryCount,
			"%s must not have retry_count bumped", id)
		require.Nil(t, workflowRecord.CompletedAt)
		require.Nil(t, workflowRecord.ErrorMessage)
		require.True(t, workflowRecord.UpdatedAt.Equal(workflowUpdatedAt[id]),
			"%s workflow half must keep its original updated_at", id)

		executionRecord, err := ls.GetExecutionRecord(ctx, id)
		require.NoError(t, err)
		require.Equal(t, "running", executionRecord.Status,
			"%s execution must keep its live state, not be reported pending", id)
		require.True(t, executionRecord.UpdatedAt.Equal(executionUpdatedAt[id]),
			"%s execution half must be rolled back with its candidate", id)
	}

	workflowRecord, err := ls.GetWorkflowExecution(ctx, moves)
	require.NoError(t, err)
	require.Equal(t, "pending", workflowRecord.Status)
	require.Equal(t, 1, workflowRecord.RetryCount)
	require.Nil(t, workflowRecord.CompletedAt)

	executionRecord, err := ls.GetExecutionRecord(ctx, moves)
	require.NoError(t, err)
	require.Equal(t, "pending", executionRecord.Status)
	require.True(t, executionRecord.UpdatedAt.After(now),
		"the moved execution must carry the retry's timestamp")
}

// TestRetryStaleWorkflowExecutions_MidBatchFailureLeavesNothingCommitted covers
// window 7 of the interleaving inventory: a statement failure after an earlier
// candidate in the same batch has already moved both halves. An AFTER UPDATE
// trigger raises on the second candidate's workflow update, so the first
// candidate is staged before the batch aborts. The deferred rollback must
// discard the staged candidate too, and the returned error must be treated as
// authoritative over the partial id list.
func TestRetryStaleWorkflowExecutions_MidBatchFailureLeavesNothingCommitted(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)
	now := time.Now().UTC()

	const (
		stagedThenRolled = "exec-batch-abort-first"
		raises           = "exec-batch-abort-second"
	)
	workflowUpdatedAt := map[string]time.Time{
		stagedThenRolled: now.Add(-3 * time.Hour),
		raises:           now.Add(-2 * time.Hour),
	}
	executionUpdatedAt := now.Add(-2 * time.Hour)
	for id, updatedAt := range workflowUpdatedAt {
		require.NoError(t, ls.StoreWorkflowExecution(ctx, retryTestWorkflow(id, updatedAt)))
		require.NoError(t, ls.CreateExecutionRecord(ctx, retryTestExecution(id, executionUpdatedAt)))
		backdateExecutionUpdatedAt(t, ls, "executions", id, executionUpdatedAt)
	}

	db := ls.requireSQLDB()
	_, err := db.Exec(fmt.Sprintf(`
		CREATE TRIGGER retry_batch_abort
		AFTER UPDATE ON workflow_executions
		FOR EACH ROW
		WHEN NEW.execution_id = '%s'
		BEGIN
			SELECT RAISE(ABORT, 'simulated mid-batch failure');
		END`, raises))
	require.NoError(t, err)
	t.Cleanup(func() {
		_, _ = db.Exec("DROP TRIGGER IF EXISTS retry_batch_abort")
	})

	retried, err := ls.RetryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100)
	require.Error(t, err, "a failed statement must be reported")
	require.Contains(t, err.Error(), "simulated mid-batch failure")
	// The error is authoritative over the partial id list: the batch was
	// rolled back, so nothing in it may be treated as retried.
	t.Logf("partial retried list on error (not authoritative): %v", retried)

	for id := range workflowUpdatedAt {
		workflowRecord, err := ls.GetWorkflowExecution(ctx, id)
		require.NoError(t, err)
		require.Equal(t, "running", workflowRecord.Status,
			"%s must not commit after a mid-batch failure", id)
		require.Equal(t, 0, workflowRecord.RetryCount,
			"%s retry_count must stay put when the batch aborts", id)
		require.Nil(t, workflowRecord.ErrorMessage,
			"%s must not keep the retry error message", id)
		require.True(t, workflowRecord.UpdatedAt.Equal(workflowUpdatedAt[id]),
			"%s workflow updated_at must be the committed one", id)

		executionRecord, err := ls.GetExecutionRecord(ctx, id)
		require.NoError(t, err)
		require.Equal(t, "running", executionRecord.Status,
			"%s execution half must not be moved by a rolled-back batch", id)
		require.True(t, executionRecord.UpdatedAt.Equal(executionUpdatedAt),
			"%s execution updated_at must be the committed one", id)
	}
}

// TestStaleWorkflowReapersDiscriminateStaleFromFresh pins the stale-versus-fresh
// discrimination of both workflow reapers, so the paired-execution guard cannot
// hide a genuinely stale candidate (a false negative) or move a candidate whose
// own clock is fresh. The same four fixtures run through each reaper:
//
//   - stale workflow + stale execution  -> moved (timeout / pending)
//   - stale workflow + fresh execution  -> spared
//   - fresh workflow + stale execution  -> spared
//   - stale workflow + no execution row -> moved (workflow-only recovery path)
//
// The live PostgreSQL twins are
// TestPostgresMarkStaleWorkflowExecutionsReapsStaleAndSparesFreshExecution and
// TestPostgresRetryStaleWorkflowExecutionsRepassesStaleAndSparesFreshExecution.
func TestStaleWorkflowReapersDiscriminateStaleFromFresh(t *testing.T) {
	type reaper struct {
		name        string
		movedStatus string
		run         func(*LocalStorage, context.Context) (int, error)
	}

	reapers := []reaper{
		{
			name:        "MarkStaleWorkflowExecutions",
			movedStatus: string(types.ExecutionStatusTimeout),
			run: func(ls *LocalStorage, ctx context.Context) (int, error) {
				return ls.MarkStaleWorkflowExecutions(ctx, 30*time.Minute, 100)
			},
		},
		{
			name:        "RetryStaleWorkflowExecutions",
			movedStatus: string(types.ExecutionStatusPending),
			run: func(ls *LocalStorage, ctx context.Context) (int, error) {
				ids, err := ls.RetryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100)
				return len(ids), err
			},
		},
	}

	cases := []struct {
		name         string
		workflowAge  time.Duration
		executionAge time.Duration
		paired       bool
		wantMoved    bool
	}{
		{name: "stale on both clocks", workflowAge: 2 * time.Hour, executionAge: 2 * time.Hour, paired: true, wantMoved: true},
		{name: "stale workflow fresh execution", workflowAge: 2 * time.Hour, executionAge: 0, paired: true, wantMoved: false},
		{name: "fresh workflow stale execution", workflowAge: 0, executionAge: 2 * time.Hour, paired: true, wantMoved: false},
		{name: "stale workflow no execution", workflowAge: 2 * time.Hour, paired: false, wantMoved: true},
	}

	for _, rp := range reapers {
		t.Run(rp.name, func(t *testing.T) {
			for _, tc := range cases {
				t.Run(tc.name, func(t *testing.T) {
					ls, ctx := setupRetryTestStorage(t)
					now := time.Now().UTC()
					id := "exec-discriminate-" + strings.ReplaceAll(tc.name, " ", "-")

					require.NoError(t, ls.StoreWorkflowExecution(ctx, retryTestWorkflow(id, now.Add(-tc.workflowAge))))
					if tc.paired {
						require.NoError(t, ls.CreateExecutionRecord(ctx, retryTestExecution(id, now.Add(-2*time.Hour))))
						backdateExecutionUpdatedAt(t, ls, "executions", id, now.Add(-tc.executionAge))
					}

					moved, err := rp.run(ls, ctx)
					require.NoError(t, err)
					want := 0
					if tc.wantMoved {
						want = 1
					}
					require.Equal(t, want, moved,
						"the guard must move stale candidates without hiding them")

					workflowRecord, err := ls.GetWorkflowExecution(ctx, id)
					require.NoError(t, err)
					if tc.wantMoved {
						require.Equal(t, rp.movedStatus, workflowRecord.Status)
					} else {
						require.Equal(t, "running", workflowRecord.Status,
							"a fresh clock on either side must spare the candidate")
					}

					executionRecord, err := ls.GetExecutionRecord(ctx, id)
					require.NoError(t, err)
					if !tc.paired {
						require.Nil(t, executionRecord)
						return
					}
					require.Equal(t, workflowRecord.Status, executionRecord.Status,
						"the paired records must agree after the sweep")
				})
			}
		})
	}
}

func TestRetryStaleWorkflowExecutions_DisabledWithZeroMaxRetries(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)

	// maxRetries=0 should return nil without querying
	retriedIDs, err := ls.RetryStaleWorkflowExecutions(ctx, 1*time.Hour, 0, 100)
	require.NoError(t, err)
	assert.Nil(t, retriedIDs)
}

func TestRetryStaleWorkflowExecutions_NoStaleExecutions(t *testing.T) {
	ls, ctx := setupRetryTestStorage(t)

	// No executions at all — should return empty
	retriedIDs, err := ls.RetryStaleWorkflowExecutions(ctx, 1*time.Hour, 3, 100)
	require.NoError(t, err)
	assert.Nil(t, retriedIDs)
}
