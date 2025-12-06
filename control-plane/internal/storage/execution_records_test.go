package storage

import (
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"

	"github.com/stretchr/testify/require"
)

func TestQueryRunSummariesParsesTextTimestamps(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	const runID = "run-test-aggregate"
	base := time.Date(2024, 1, 2, 15, 4, 5, 0, time.UTC)

	executions := []*types.Execution{
		{
			ExecutionID: "exec-a",
			RunID:       runID,
			AgentNodeID: "agent-1",
			ReasonerID:  "reasoner.a",
			NodeID:      "node-a",
			Status:      string(types.ExecutionStatusSucceeded),
			StartedAt:   base.Add(-3 * time.Minute),
			CompletedAt: pointerTime(base.Add(-2 * time.Minute)),
			CreatedAt:   base.Add(-3 * time.Minute),
			UpdatedAt:   base.Add(-2 * time.Minute),
		},
		{
			ExecutionID: "exec-b",
			RunID:       runID,
			AgentNodeID: "agent-1",
			ReasonerID:  "reasoner.b",
			NodeID:      "node-b",
			Status:      string(types.ExecutionStatusRunning),
			StartedAt:   base.Add(-1 * time.Minute),
			CreatedAt:   base.Add(-1 * time.Minute),
			UpdatedAt:   base.Add(-30 * time.Second),
		},
	}

	for _, exec := range executions {
		require.NoError(t, ls.CreateExecutionRecord(ctx, exec))
	}

	results, _, err := ls.QueryRunSummaries(ctx, types.ExecutionFilter{})
	require.NoError(t, err)
	require.Len(t, results, 1)

	summary := results[0]
	require.Equal(t, runID, summary.RunID)
	require.Equal(t, 2, summary.TotalExecutions)
	require.False(t, summary.EarliestStarted.IsZero(), "earliest started should be parsed from TEXT timestamps")
	require.False(t, summary.LatestStarted.IsZero(), "latest started should be parsed from TEXT timestamps")
	require.Equal(t, summary.EarliestStarted, base.Add(-3*time.Minute))
	require.Equal(t, summary.LatestStarted, base.Add(-1*time.Minute))
}

func pointerTime(t time.Time) *time.Time {
	return &t
}
