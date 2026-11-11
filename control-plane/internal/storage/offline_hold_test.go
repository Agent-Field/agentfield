package storage

import (
	"database/sql/driver"
	"encoding/json"
	"errors"
	"strings"
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

func seedQuietExecutionBothTables(t *testing.T, ls *LocalStorage, id, node, status, wfStatus string, quietSince time.Time) {
	t.Helper()
	ctx := t.Context()
	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID: id, RunID: "run-" + id, AgentNodeID: node, ReasonerID: "intake", NodeID: node,
		Status: status, StartedAt: quietSince.Add(-time.Hour),
	}))
	require.NoError(t, ls.StoreWorkflowExecution(ctx, &types.WorkflowExecution{
		WorkflowID: "run-" + id, ExecutionID: id, AgentFieldRequestID: "req-" + id, AgentNodeID: node,
		ReasonerID: "intake", Status: wfStatus, StartedAt: quietSince.Add(-time.Hour),
		CreatedAt: quietSince.Add(-time.Hour), UpdatedAt: quietSince.Add(-time.Hour), WorkflowTags: []string{},
		InputData: json.RawMessage("{}"), OutputData: json.RawMessage("{}"),
	}))
	backdateExecutionUpdatedAt(t, ls, "executions", id, quietSince)
	backdateExecutionUpdatedAt(t, ls, "workflow_executions", id, quietSince)
}

// HOLDING AND LISTING BY NODE UPDATES BOTH TABLES AND FILTERS BY AGENT NODE ID.
func TestHoldAndListOfflineNodeExecutionsUpdatesBothTablesAndFiltersByNode(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	quietSince := time.Now().UTC().Add(-time.Hour)
	seedQuietExecutionBothTables(t, ls, "exec-laptop-1", "laptop", types.ExecutionStatusRunning, types.ExecutionStatusRunning, quietSince)
	seedQuietExecutionBothTables(t, ls, "exec-laptop-2", "laptop", types.ExecutionStatusPending, types.ExecutionStatusWaiting, quietSince)
	seedQuietExecutionBothTables(t, ls, "exec-desktop-1", "desktop", types.ExecutionStatusQueued, types.ExecutionStatusQueued, quietSince)

	n, err := ls.HoldOfflineNodeExecutions(ctx, "laptop", time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 2, n)

	nDesktop, err := ls.HoldOfflineNodeExecutions(ctx, "desktop", time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, nDesktop)

	laptopHeld, err := ls.ListHeldExecutions(ctx, "laptop")
	require.NoError(t, err)
	require.Len(t, laptopHeld, 2)
	require.Equal(t, "exec-laptop-1", laptopHeld[0].ExecutionID)
	require.Equal(t, "laptop", laptopHeld[0].AgentNodeID)
	require.Equal(t, "exec-laptop-2", laptopHeld[1].ExecutionID)

	allHeld, err := ls.ListHeldExecutions(ctx, "")
	require.NoError(t, err)
	require.Len(t, allHeld, 3)

	for _, id := range []string{"exec-laptop-1", "exec-laptop-2", "exec-desktop-1"} {
		rec, err := ls.GetExecutionRecord(ctx, id)
		require.NoError(t, err)
		require.NotNil(t, rec.StatusReason)
		require.Equal(t, types.ExecutionReasonNodeOffline, *rec.StatusReason)

		wf, err := ls.GetWorkflowExecution(ctx, id)
		require.NoError(t, err)
		require.NotNil(t, wf.StatusReason)
		require.Equal(t, types.ExecutionReasonNodeOffline, *wf.StatusReason)
	}
}

// WHEN A NODE RETURNS WITH ITS EXECUTION, ReleaseHeldExecutions clears the
// hold reason on both tables while leaving unheld or other-node rows alone.
func TestReleaseHeldExecutionsRestoresRunningStateOnBothTablesForReturnedNode(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	quietSince := time.Now().UTC().Add(-time.Hour)
	seedQuietExecutionBothTables(t, ls, "exec-keep", "laptop", types.ExecutionStatusRunning, types.ExecutionStatusRunning, quietSince)
	seedQuietExecutionBothTables(t, ls, "exec-still-held", "laptop", types.ExecutionStatusRunning, types.ExecutionStatusRunning, quietSince)
	seedQuietExecutionBothTables(t, ls, "exec-other-node", "desktop", types.ExecutionStatusRunning, types.ExecutionStatusRunning, quietSince)

	_, err := ls.HoldOfflineNodeExecutions(ctx, "laptop", time.Minute, 100)
	require.NoError(t, err)
	_, err = ls.HoldOfflineNodeExecutions(ctx, "desktop", time.Minute, 100)
	require.NoError(t, err)

	// Empty slice is a no-op without starting a transaction.
	zeroReleased, err := ls.ReleaseHeldExecutions(ctx, "laptop", nil)
	require.NoError(t, err)
	require.Zero(t, zeroReleased)

	// Releasing exec-keep along with an unknown ID and another node's held ID
	// releases only exec-keep.
	released, err := ls.ReleaseHeldExecutions(ctx, "laptop", []string{"exec-keep", "exec-missing", "exec-other-node"})
	require.NoError(t, err)
	require.Equal(t, 1, released)

	rec, err := ls.GetExecutionRecord(ctx, "exec-keep")
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusRunning, rec.Status)
	require.Nil(t, rec.StatusReason)

	wf, err := ls.GetWorkflowExecution(ctx, "exec-keep")
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusRunning, wf.Status)
	require.Nil(t, wf.StatusReason)

	// Releasing again is a no-op because the hold was already cleared.
	releasedAgain, err := ls.ReleaseHeldExecutions(ctx, "laptop", []string{"exec-keep"})
	require.NoError(t, err)
	require.Zero(t, releasedAgain)

	remaining, err := ls.ListHeldExecutions(ctx, "laptop")
	require.NoError(t, err)
	require.Len(t, remaining, 1)
	require.Equal(t, "exec-still-held", remaining[0].ExecutionID)
}

// WHEN A HOLD ENDS IN FAILURE (lost on return, node removed, or ceiling
// exceeded), FailHeldExecutions marks both tables failed with the reason and
// message, and leaves rows that already left the hold alone.
func TestFailHeldExecutionsRecordsEachEndingOnBothTablesAndLeavesUnheldRowsAlone(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)
	quietSince := time.Now().UTC().Add(-time.Hour)
	seedQuietExecutionBothTables(t, ls, "exec-lost", "laptop", types.ExecutionStatusRunning, types.ExecutionStatusRunning, quietSince)
	seedQuietExecutionBothTables(t, ls, "exec-removed", "laptop", types.ExecutionStatusPending, types.ExecutionStatusWaiting, quietSince)
	seedQuietExecutionBothTables(t, ls, "exec-expired", "laptop", types.ExecutionStatusQueued, types.ExecutionStatusQueued, quietSince)
	seedQuietExecutionBothTables(t, ls, "exec-released", "laptop", types.ExecutionStatusRunning, types.ExecutionStatusRunning, quietSince)

	heldCount, err := ls.HoldOfflineNodeExecutions(ctx, "laptop", time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 4, heldCount)

	_, err = ls.ReleaseHeldExecutions(ctx, "laptop", []string{"exec-released"})
	require.NoError(t, err)

	zeroFailed, err := ls.FailHeldExecutions(ctx, nil, types.ExecutionReasonNodeLostExecution, "none")
	require.NoError(t, err)
	require.Zero(t, zeroFailed)

	for _, tc := range []struct {
		id      string
		reason  string
		message string
	}{
		{
			id:      "exec-lost",
			reason:  types.ExecutionReasonNodeLostExecution,
			message: "node laptop came back without this execution: it was lost while the node was offline",
		},
		{
			id:      "exec-removed",
			reason:  types.ExecutionReasonNodeRemoved,
			message: "the node holding this execution is no longer registered",
		},
		{
			id:      "exec-expired",
			reason:  types.ExecutionReasonNodeOfflineTooLong,
			message: "the node holding this execution stayed offline for more than 168h0m0s",
		},
	} {
		t.Run(tc.reason, func(t *testing.T) {
			failed, err := ls.FailHeldExecutions(ctx, []string{tc.id, "exec-released", "exec-nonexistent"}, tc.reason, tc.message)
			require.NoError(t, err)
			require.Equal(t, 1, failed)

			rec, err := ls.GetExecutionRecord(ctx, tc.id)
			require.NoError(t, err)
			require.Equal(t, types.ExecutionStatusFailed, rec.Status)
			require.NotNil(t, rec.StatusReason)
			require.Equal(t, tc.reason, *rec.StatusReason)
			require.NotNil(t, rec.ErrorMessage)
			require.Equal(t, tc.message, *rec.ErrorMessage)
			require.NotNil(t, rec.CompletedAt)

			wf, err := ls.GetWorkflowExecution(ctx, tc.id)
			require.NoError(t, err)
			require.Equal(t, types.ExecutionStatusFailed, wf.Status)
			require.NotNil(t, wf.StatusReason)
			require.Equal(t, tc.reason, *wf.StatusReason)
			require.NotNil(t, wf.ErrorMessage)
			require.Equal(t, tc.message, *wf.ErrorMessage)
			require.NotNil(t, wf.CompletedAt)
		})
	}

	// The released execution was not touched by any FailHeldExecutions call.
	releasedRec, err := ls.GetExecutionRecord(ctx, "exec-released")
	require.NoError(t, err)
	require.Equal(t, types.ExecutionStatusRunning, releasedRec.Status)
	require.Nil(t, releasedRec.StatusReason)
}

// ACTIVITY THAT LANDS BETWEEN SELECT AND UPDATE WINS OVER THE HOLD, and blank
// node IDs or non-positive limits are rejected or ignored up front.
func TestHoldOfflineNodeExecutionsSkipsRowFreshenedBeforeUpdateAndValidatesInputs(t *testing.T) {
	ctx := t.Context()
	_, conn, ls := newInterceptingSQLiteDB(t, "local")
	require.NoError(t, ls.autoMigrateSchema(ctx))

	_, err := ls.HoldOfflineNodeExecutions(ctx, "   ", time.Minute, 10)
	require.ErrorContains(t, err, "agent_node_id is required")

	n, err := ls.HoldOfflineNodeExecutions(ctx, "laptop", time.Minute, 0)
	require.NoError(t, err)
	require.Zero(t, n)

	quietSince := time.Now().UTC().Add(-time.Hour)
	seedQuietExecutionBothTables(t, ls, "exec-woke", "laptop", types.ExecutionStatusRunning, types.ExecutionStatusRunning, quietSince)
	seedQuietExecutionBothTables(t, ls, "exec-asleep", "laptop", types.ExecutionStatusRunning, types.ExecutionStatusRunning, quietSince)

	// Simulate the node touching exec-woke after SELECT but before the
	// conditional UPDATE inside HoldOfflineNodeExecutions runs.
	freshenedOnce := false
	conn.setHook(func(inner driver.Conn, query string) error {
		if !freshenedOnce && strings.HasPrefix(strings.TrimSpace(query), "UPDATE executions SET status_reason") {
			freshenedOnce = true
			stmt, prepErr := inner.Prepare(`UPDATE executions SET status = 'succeeded' WHERE execution_id = 'exec-woke'`)
			if prepErr != nil {
				return prepErr
			}
			_, execErr := stmt.Exec(nil)
			_ = stmt.Close()
			return execErr
		}
		return nil
	})

	held, err := ls.HoldOfflineNodeExecutions(ctx, "laptop", time.Minute, 10)
	require.NoError(t, err)
	require.Equal(t, 1, held)
	conn.setHook(nil)

	wokeWF, err := ls.GetWorkflowExecution(ctx, "exec-woke")
	require.NoError(t, err)
	require.Nil(t, wokeWF.StatusReason)

	asleepWF, err := ls.GetWorkflowExecution(ctx, "exec-asleep")
	require.NoError(t, err)
	require.NotNil(t, asleepWF.StatusReason)
	require.Equal(t, types.ExecutionReasonNodeOffline, *asleepWF.StatusReason)
}

// STORAGE ERRORS ON QUERIES, TRANSACTIONS, UPDATES AND TIMESTAMP PARSING
// ARE SURFACED WITHOUT CORRUPTING HELD STATE.
func TestOfflineHoldStoragePropagatesDatabaseErrors(t *testing.T) {
	ctx := t.Context()
	rawDB, conn, ls := newInterceptingSQLiteDB(t, "local")
	require.NoError(t, ls.autoMigrateSchema(ctx))

	quietSince := time.Now().UTC().Add(-time.Hour)
	seedQuietExecutionBothTables(t, ls, "exec-err", "laptop", types.ExecutionStatusRunning, types.ExecutionStatusRunning, quietSince)

	injected := errors.New("simulated storage failure")

	// 1. HoldOfflineNodeExecutions: SELECT error
	conn.setHook(func(_ driver.Conn, query string) error {
		if strings.Contains(query, "SELECT execution_id FROM executions") {
			return injected
		}
		return nil
	})
	_, err := ls.HoldOfflineNodeExecutions(ctx, "laptop", time.Minute, 10)
	require.ErrorIs(t, err, injected)

	// 2. HoldOfflineNodeExecutions: UPDATE executions error
	conn.setHook(func(_ driver.Conn, query string) error {
		if strings.HasPrefix(strings.TrimSpace(query), "UPDATE executions SET status_reason") {
			return injected
		}
		return nil
	})
	_, err = ls.HoldOfflineNodeExecutions(ctx, "laptop", time.Minute, 10)
	require.ErrorIs(t, err, injected)

	// 3. HoldOfflineNodeExecutions: UPDATE workflow_executions error (rolls back executions update)
	conn.setHook(func(_ driver.Conn, query string) error {
		if strings.HasPrefix(strings.TrimSpace(query), "UPDATE workflow_executions SET status_reason") {
			return injected
		}
		return nil
	})
	_, err = ls.HoldOfflineNodeExecutions(ctx, "laptop", time.Minute, 10)
	require.ErrorIs(t, err, injected)

	// Succeeds once hooks are cleared.
	conn.setHook(nil)
	n, err := ls.HoldOfflineNodeExecutions(ctx, "laptop", time.Minute, 10)
	require.NoError(t, err)
	require.Equal(t, 1, n)

	// 4. ListHeldExecutions: query error
	conn.setHook(func(_ driver.Conn, query string) error {
		if strings.Contains(query, "SELECT execution_id, agent_node_id, updated_at FROM executions") {
			return injected
		}
		return nil
	})
	_, err = ls.ListHeldExecutions(ctx, "laptop")
	require.ErrorIs(t, err, injected)
	conn.setHook(nil)

	// 5. ListHeldExecutions: malformed updated_at blob surfaces parseDBTime error
	_, err = rawDB.Exec(`UPDATE executions SET updated_at = X'626164' WHERE execution_id = 'exec-err'`)
	require.NoError(t, err)
	_, err = ls.ListHeldExecutions(ctx, "laptop")
	require.ErrorContains(t, err, "held execution exec-err updated_at")
	backdateExecutionUpdatedAt(t, ls, "executions", "exec-err", quietSince)

	// 6. ReleaseHeldExecutions & FailHeldExecutions: UPDATE executions / workflow_executions errors
	conn.setHook(func(_ driver.Conn, query string) error {
		if strings.HasPrefix(strings.TrimSpace(query), "UPDATE executions SET status_reason = NULL") {
			return injected
		}
		return nil
	})
	_, err = ls.ReleaseHeldExecutions(ctx, "laptop", []string{"exec-err"})
	require.ErrorIs(t, err, injected)

	conn.setHook(func(_ driver.Conn, query string) error {
		if strings.HasPrefix(strings.TrimSpace(query), "UPDATE workflow_executions SET status_reason = NULL") {
			return injected
		}
		return nil
	})
	_, err = ls.ReleaseHeldExecutions(ctx, "laptop", []string{"exec-err"})
	require.ErrorIs(t, err, injected)

	conn.setHook(func(_ driver.Conn, query string) error {
		if strings.HasPrefix(strings.TrimSpace(query), "UPDATE executions") && strings.Contains(query, "error_message") {
			return injected
		}
		return nil
	})
	_, err = ls.FailHeldExecutions(ctx, []string{"exec-err"}, types.ExecutionReasonNodeRemoved, "gone")
	require.ErrorIs(t, err, injected)

	conn.setHook(func(_ driver.Conn, query string) error {
		if strings.HasPrefix(strings.TrimSpace(query), "UPDATE workflow_executions") && strings.Contains(query, "error_message") {
			return injected
		}
		return nil
	})
	_, err = ls.FailHeldExecutions(ctx, []string{"exec-err"}, types.ExecutionReasonNodeRemoved, "gone")
	require.ErrorIs(t, err, injected)
	conn.setHook(nil)

	// 7. Closed database surfaces BeginTx error in HoldOfflineNodeExecutions and endHolds.
	require.NoError(t, rawDB.Close())
	_, err = ls.FailHeldExecutions(ctx, []string{"exec-err"}, types.ExecutionReasonNodeRemoved, "gone")
	require.ErrorContains(t, err, "begin end-hold transaction")
}

