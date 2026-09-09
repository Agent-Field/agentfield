package storage

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

// setExecutionTimestampsWithOffset overwrites created_at/updated_at/started_at
// for a row with a value carrying a fixed non-UTC zone offset. On a non-UTC
// host, time.Now() produces exactly this kind of value, so this reproduces
// what real deployments persist (#1040).
func setExecutionTimestampsWithOffset(t *testing.T, ls *LocalStorage, table, executionID string, ts time.Time) {
	t.Helper()
	db := ls.requireSQLDB()
	_, err := db.Exec(
		"UPDATE "+table+" SET created_at = ?, updated_at = ?, started_at = ? WHERE execution_id = ?",
		ts, ts, ts, executionID,
	)
	require.NoError(t, err)
}

// TestMarkStaleExecutions_FreshNonUTCTimestampNotReaped reproduces #1040: a
// fresh execution whose timestamps were persisted with a non-UTC offset must
// not be reaped, because its instant is well within the stale window. Before
// the fix SQLite compared timestamps lexically as text, so a "-05:00" fresh
// value sorted before a UTC cutoff and got wrongly timed out.
func TestMarkStaleExecutions_FreshNonUTCTimestampNotReaped(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)

	loc := time.FixedZone("CDT", -5*60*60)
	freshLocal := time.Now().In(loc) // fresh instant, but stored with -05:00 offset

	fresh := &types.Execution{
		ExecutionID: "exec-fresh-nonutc",
		RunID:       "run-1",
		AgentNodeID: "agent-1",
		ReasonerID:  "reasoner-1",
		NodeID:      "node-1",
		Status:      "running",
		StartedAt:   freshLocal,
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, fresh))
	setExecutionTimestampsWithOffset(t, ls, "executions", "exec-fresh-nonutc", freshLocal)

	reaped, err := ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 0, reaped, "fresh execution stored with non-UTC offset must not be reaped")

	rec, err := ls.GetExecutionRecord(ctx, "exec-fresh-nonutc")
	require.NoError(t, err)
	require.Equal(t, "running", rec.Status, "fresh execution must stay running")
}

// TestMarkStaleExecutions_StaleNonUTCTimestampReaped is the inverse guard: a
// genuinely stale execution stored with a non-UTC offset must still be reaped.
func TestMarkStaleExecutions_StaleNonUTCTimestampReaped(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)

	loc := time.FixedZone("CDT", -5*60*60)
	staleLocal := time.Now().In(loc).Add(-1 * time.Hour) // 1h old, past the 30m window

	stale := &types.Execution{
		ExecutionID: "exec-stale-nonutc",
		RunID:       "run-1",
		AgentNodeID: "agent-1",
		ReasonerID:  "reasoner-1",
		NodeID:      "node-1",
		Status:      "running",
		StartedAt:   staleLocal,
	}
	require.NoError(t, ls.CreateExecutionRecord(ctx, stale))
	setExecutionTimestampsWithOffset(t, ls, "executions", "exec-stale-nonutc", staleLocal)

	reaped, err := ls.MarkStaleExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 1, reaped, "genuinely stale non-UTC execution must be reaped")

	rec, err := ls.GetExecutionRecord(ctx, "exec-stale-nonutc")
	require.NoError(t, err)
	require.Equal(t, "timeout", rec.Status)
}

// TestMarkStaleWorkflowExecutions_FreshNonUTCTimestampNotReaped mirrors the
// reproduction for the workflow_executions selection path.
func TestMarkStaleWorkflowExecutions_FreshNonUTCTimestampNotReaped(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)

	loc := time.FixedZone("CDT", -5*60*60)
	freshLocal := time.Now().In(loc)

	fresh := &types.WorkflowExecution{
		WorkflowID:          "wf-fresh",
		ExecutionID:         "wf-fresh-nonutc",
		AgentFieldRequestID: "req-fresh",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reasoner-1",
		Status:              "running",
		StartedAt:           freshLocal,
		CreatedAt:           freshLocal,
		UpdatedAt:           freshLocal,
		WorkflowTags:        []string{},
		InputData:           json.RawMessage("{}"),
		OutputData:          json.RawMessage("{}"),
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, fresh))
	setExecutionTimestampsWithOffset(t, ls, "workflow_executions", "wf-fresh-nonutc", freshLocal)

	reaped, err := ls.MarkStaleWorkflowExecutions(ctx, 30*time.Minute, 100)
	require.NoError(t, err)
	require.Equal(t, 0, reaped, "fresh workflow execution stored with non-UTC offset must not be reaped")
}

// TestRetryStaleWorkflowExecutions_FreshNonUTCTimestampNotRetried mirrors the
// reproduction for the retry-selection path.
func TestRetryStaleWorkflowExecutions_FreshNonUTCTimestampNotRetried(t *testing.T) {
	ls, ctx := setupTestLocalStorage(t)

	loc := time.FixedZone("CDT", -5*60*60)
	freshLocal := time.Now().In(loc)

	fresh := &types.WorkflowExecution{
		WorkflowID:          "wf-retry-fresh",
		ExecutionID:         "wf-retry-fresh-nonutc",
		AgentFieldRequestID: "req-retry-fresh",
		AgentNodeID:         "agent-1",
		ReasonerID:          "reasoner-1",
		Status:              "running",
		StartedAt:           freshLocal,
		CreatedAt:           freshLocal,
		UpdatedAt:           freshLocal,
		WorkflowTags:        []string{},
		InputData:           json.RawMessage("{}"),
		OutputData:          json.RawMessage("{}"),
	}
	require.NoError(t, ls.StoreWorkflowExecution(ctx, fresh))
	setExecutionTimestampsWithOffset(t, ls, "workflow_executions", "wf-retry-fresh-nonutc", freshLocal)

	retried, err := ls.RetryStaleWorkflowExecutions(ctx, 30*time.Minute, 3, 100)
	require.NoError(t, err)
	require.Empty(t, retried, "fresh workflow execution stored with non-UTC offset must not be retried")
}
