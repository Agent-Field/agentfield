package storage

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// EVERY STALE SWEEP SKIPS A HELD ROW. The executions reaper, the workflow
// reaper and the stale-retry sweep each leave an execution held for its
// offline node exactly as it is, however long it has been quiet, while an
// unheld neighbour just as quiet is reaped.
func TestStaleSweepsSkipExecutionsHeldForAnOfflineNode(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()
	for _, id := range []string{"exec-held", "exec-quiet"} {
		require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
			ExecutionID: id, RunID: "run-" + id, AgentNodeID: "laptop", ReasonerID: "intake", NodeID: "laptop",
			Status: types.ExecutionStatusRunning, StartedAt: now.Add(-2 * time.Hour),
		}))
		require.NoError(t, ls.StoreWorkflowExecution(ctx, &types.WorkflowExecution{
			WorkflowID: "run-" + id, ExecutionID: id, AgentFieldRequestID: "req-" + id, AgentNodeID: "laptop",
			ReasonerID: "intake", Status: types.ExecutionStatusRunning, StartedAt: now.Add(-2 * time.Hour),
			CreatedAt: now.Add(-2 * time.Hour), UpdatedAt: now.Add(-2 * time.Hour), WorkflowTags: []string{},
			InputData: json.RawMessage("{}"), OutputData: json.RawMessage("{}"),
		}))
		backdateExecutionUpdatedAt(t, ls, "executions", id, now.Add(-time.Hour))
		backdateExecutionUpdatedAt(t, ls, "workflow_executions", id, now.Add(-time.Hour))
	}
	db := ls.requireSQLDB()
	_, err := db.Exec(`UPDATE executions SET status_reason = ? WHERE execution_id = 'exec-held'`, types.ExecutionReasonNodeOffline)
	require.NoError(t, err)
	_, err = db.Exec(`UPDATE workflow_executions SET status_reason = ? WHERE execution_id = 'exec-held'`, types.ExecutionReasonNodeOffline)
	require.NoError(t, err)

	retried, err := ls.RetryStaleWorkflowExecutions(ctx, time.Minute, 3, 100)
	require.NoError(t, err)
	require.Equal(t, []string{"exec-quiet"}, retried)
	backdateExecutionUpdatedAt(t, ls, "executions", "exec-quiet", now.Add(-time.Hour))
	backdateExecutionUpdatedAt(t, ls, "workflow_executions", "exec-quiet", now.Add(-time.Hour))

	reaped, err := ls.MarkStaleExecutions(ctx, time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped)
	wfReaped, err := ls.MarkStaleWorkflowExecutions(ctx, time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, wfReaped)

	held, err := ls.GetExecutionRecord(ctx, "exec-held")
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusRunning, held.Status)
	heldWF, err := ls.GetWorkflowExecution(ctx, "exec-held")
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusRunning, heldWF.Status)
	quiet, err := ls.GetExecutionRecord(ctx, "exec-quiet")
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusTimeout, quiet.Status)
}

// A HOLD COVERS ONLY THE QUIET WORK OF THE NAMED NODE, and never an execution
// already held through an agent restart.
func TestHoldOfflineNodeExecutionsTakesOnlyQuietRowsOfThatNode(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	now := time.Now().UTC()
	for id, node := range map[string]string{"exec-quiet": "laptop", "exec-fresh": "laptop", "exec-restart": "laptop", "exec-other": "server"} {
		require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
			ExecutionID: id, RunID: "run-" + id, AgentNodeID: node, ReasonerID: "intake", NodeID: node,
			Status: types.ExecutionStatusRunning, StartedAt: now.Add(-2 * time.Hour),
		}))
		if id != "exec-fresh" {
			backdateExecutionUpdatedAt(t, ls, "executions", id, now.Add(-time.Hour))
		}
	}
	_, err := ls.requireSQLDB().Exec(`UPDATE executions SET status_reason = ? WHERE execution_id = 'exec-restart'`, types.ExecutionReasonAwaitingAgentRestart)
	require.NoError(t, err)

	n, err := ls.HoldOfflineNodeExecutions(ctx, "laptop", time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, n)
	held, err := ls.ListHeldExecutions(ctx, "")
	require.NoError(t, err)
	require.Len(t, held, 1)
	require.Equal(t, "exec-quiet", held[0].ExecutionID)
	require.WithinDuration(t, time.Now(), held[0].HeldSince, time.Minute)

	// Holding again is a no-op: the row is already held.
	n, err = ls.HoldOfflineNodeExecutions(ctx, "laptop", time.Minute, 100)
	require.NoError(t, err)
	require.Zero(t, n)
}
