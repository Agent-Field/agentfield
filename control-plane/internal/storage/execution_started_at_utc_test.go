package storage

import (
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

// SQLite compares started_at as the text the driver wrote, and the driver
// writes a time in the zone it carries. An execution started in a zone west of
// UTC therefore sorted, and filtered, as though it started hours earlier than
// it did: a "since" bound given in UTC dropped it, and it fell behind runs
// that really started before it. started_at is stored in UTC on insert and on
// update so every row compares by instant.
func TestExecutionStartedAtIsStoredInUTC(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	west := time.FixedZone("UTC-5", -5*60*60)
	started := time.Date(2025, 6, 1, 10, 0, 0, 0, west) // 15:00 UTC
	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID: "exec-west", RunID: "run-west", AgentNodeID: "agent", NodeID: "agent",
		ReasonerID: "r", Status: string(types.ExecutionStatusRunning), StartedAt: started,
	}))

	since := time.Date(2025, 6, 1, 14, 30, 0, 0, time.UTC)
	runs, total, err := ls.QueryRunSummaries(ctx, types.ExecutionFilter{StartTime: &since})
	require.NoError(t, err)
	require.Equal(t, 1, total, "an execution that started after the bound must pass it")
	require.Len(t, runs, 1)
	require.True(t, runs[0].EarliestStarted.Equal(started))

	_, err = ls.UpdateExecutionRecord(ctx, "exec-west", func(current *types.Execution) (*types.Execution, error) {
		current.Status = string(types.ExecutionStatusSucceeded)
		current.StartedAt = current.StartedAt.In(west)
		return current, nil
	})
	require.NoError(t, err)

	var stored string
	require.NoError(t, ls.requireSQLDB().QueryRowContext(ctx,
		"SELECT CAST(started_at AS TEXT) FROM executions WHERE execution_id = ?", "exec-west").Scan(&stored))
	require.Contains(t, stored, "15:00:00", "started_at should be written as its UTC instant, got %q", stored)

	runs, total, err = ls.QueryRunSummaries(ctx, types.ExecutionFilter{StartTime: &since})
	require.NoError(t, err)
	require.Equal(t, 1, total)
	require.Len(t, runs, 1)
}
