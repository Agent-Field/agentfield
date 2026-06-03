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
	// LatestStarted comes from MAX(COALESCE(updated_at, started_at)).
	// CreateExecutionRecord always overwrites updated_at with time.Now(),
	// so LatestStarted will be approximately now, not the test's started_at.
	require.True(t, summary.LatestStarted.After(base), "latest started should be after the test base time")
}

func TestQueryRunSummariesSearchFilter(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	base := time.Date(2024, 1, 2, 15, 4, 5, 0, time.UTC)

	// Three runs with distinguishable run_id, agent_node_id and reasoner_id
	// so we can target each column of the LIKE search independently.
	executions := []*types.Execution{
		{
			ExecutionID: "exec-alpha",
			RunID:       "run-alpha",
			AgentNodeID: "billing-agent",
			ReasonerID:  "reasoner.charge",
			NodeID:      "node-a",
			Status:      string(types.ExecutionStatusSucceeded),
			StartedAt:   base,
			CreatedAt:   base,
			UpdatedAt:   base,
		},
		{
			ExecutionID: "exec-beta",
			RunID:       "run-beta",
			AgentNodeID: "shipping-agent",
			ReasonerID:  "reasoner.dispatch",
			NodeID:      "node-b",
			Status:      string(types.ExecutionStatusRunning),
			StartedAt:   base.Add(time.Minute),
			CreatedAt:   base.Add(time.Minute),
			UpdatedAt:   base.Add(time.Minute),
		},
		{
			ExecutionID: "exec-gamma",
			RunID:       "run-gamma",
			AgentNodeID: "notify-agent",
			ReasonerID:  "reasoner.charge-refund",
			NodeID:      "node-c",
			Status:      string(types.ExecutionStatusSucceeded),
			StartedAt:   base.Add(2 * time.Minute),
			CreatedAt:   base.Add(2 * time.Minute),
			UpdatedAt:   base.Add(2 * time.Minute),
		},
	}
	for _, exec := range executions {
		require.NoError(t, ls.CreateExecutionRecord(ctx, exec))
	}

	// Sanity: no filter returns all three runs.
	all, _, err := ls.QueryRunSummaries(ctx, types.ExecutionFilter{})
	require.NoError(t, err)
	require.Len(t, all, 3)

	runIDs := func(rows []*RunSummaryAggregation) []string {
		out := make([]string, 0, len(rows))
		for _, r := range rows {
			out = append(out, r.RunID)
		}
		return out
	}

	// Match on run_id.
	term := "alpha"
	got, total, err := ls.QueryRunSummaries(ctx, types.ExecutionFilter{Search: &term})
	require.NoError(t, err)
	require.Equal(t, 1, total)
	require.ElementsMatch(t, []string{"run-alpha"}, runIDs(got))

	// Match on agent_node_id.
	term = "shipping"
	got, total, err = ls.QueryRunSummaries(ctx, types.ExecutionFilter{Search: &term})
	require.NoError(t, err)
	require.Equal(t, 1, total)
	require.ElementsMatch(t, []string{"run-beta"}, runIDs(got))

	// Match on reasoner_id — should return both "charge" and "charge-refund" runs.
	term = "charge"
	got, total, err = ls.QueryRunSummaries(ctx, types.ExecutionFilter{Search: &term})
	require.NoError(t, err)
	require.Equal(t, 2, total)
	require.ElementsMatch(t, []string{"run-alpha", "run-gamma"}, runIDs(got))

	// No match → empty result set, not an error.
	term = "nonexistent-needle"
	got, total, err = ls.QueryRunSummaries(ctx, types.ExecutionFilter{Search: &term})
	require.NoError(t, err)
	require.Equal(t, 0, total)
	require.Empty(t, got)
}

func TestQueryRunSummariesIncludesRootErrorFields(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	runID := "run-root-error-fields"
	base := time.Date(2024, 2, 10, 9, 0, 0, 0, time.UTC)
	rootCategory := "concurrency_limit"
	rootMessage := "agent test-slow has reached max concurrent executions (3)"
	rootExecutionID := "exec-root"

	root := &types.Execution{
		ExecutionID: rootExecutionID,
		RunID:       runID,
		AgentNodeID: "test-slow",
		ReasonerID:  "slow_task",
		NodeID:      "node-root",
		Status:      string(types.ExecutionStatusFailed),
		StatusReason: &rootCategory,
		ErrorMessage: &rootMessage,
		StartedAt:   base,
		CreatedAt:   base,
		UpdatedAt:   base,
	}
	child := &types.Execution{
		ExecutionID:       "exec-child",
		RunID:             runID,
		ParentExecutionID: &rootExecutionID,
		AgentNodeID:       "test-slow",
		ReasonerID:        "child_task",
		NodeID:            "node-child",
		Status:            string(types.ExecutionStatusFailed),
		StartedAt:         base.Add(1 * time.Second),
		CreatedAt:         base.Add(1 * time.Second),
		UpdatedAt:         base.Add(1 * time.Second),
	}

	require.NoError(t, ls.CreateExecutionRecord(ctx, root))
	require.NoError(t, ls.CreateExecutionRecord(ctx, child))

	results, total, err := ls.QueryRunSummaries(ctx, types.ExecutionFilter{})
	require.NoError(t, err)
	require.Equal(t, 1, total)
	require.Len(t, results, 1)
	require.Equal(t, runID, results[0].RunID)
	require.NotNil(t, results[0].RootErrorCategory)
	require.Equal(t, rootCategory, *results[0].RootErrorCategory)
	require.NotNil(t, results[0].RootErrorMessage)
	require.Equal(t, rootMessage, *results[0].RootErrorMessage)
}

func TestGetRunAggregationIncludesRootErrorFields(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	runID := "run-root-error-get-aggregation"
	base := time.Date(2024, 2, 10, 10, 0, 0, 0, time.UTC)
	rootCategory := "concurrency_limit"
	rootMessage := "agent test-slow has reached max concurrent executions (3)"

	root := &types.Execution{
		ExecutionID:  "exec-root",
		RunID:        runID,
		AgentNodeID:  "test-slow",
		ReasonerID:   "slow_task",
		NodeID:       "node-root",
		Status:       string(types.ExecutionStatusFailed),
		StatusReason: &rootCategory,
		ErrorMessage: &rootMessage,
		StartedAt:    base,
		CreatedAt:    base,
		UpdatedAt:    base,
	}

	require.NoError(t, ls.CreateExecutionRecord(ctx, root))

	agg, err := ls.getRunAggregation(ctx, runID)
	require.NoError(t, err)
	require.NotNil(t, agg)
	require.NotNil(t, agg.RootErrorCategory)
	require.Equal(t, rootCategory, *agg.RootErrorCategory)
	require.NotNil(t, agg.RootErrorMessage)
	require.Equal(t, rootMessage, *agg.RootErrorMessage)
}

func pointerTime(t time.Time) *time.Time {
	return &t
}
