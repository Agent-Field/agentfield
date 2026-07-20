package storage

import (
	"math"
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

func TestQueryRunSummariesActiveOnly(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	base := time.Date(2024, 3, 1, 8, 0, 0, 0, time.UTC)
	rootActive := "exec-active-root"

	executions := []*types.Execution{
		// Run with one running root and one succeeded child → active.
		{
			ExecutionID: rootActive,
			RunID:       "run-active",
			AgentNodeID: "agent-x",
			ReasonerID:  "review",
			NodeID:      "node-x",
			Status:      string(types.ExecutionStatusRunning),
			StartedAt:   base,
			CreatedAt:   base,
			UpdatedAt:   base,
		},
		{
			ExecutionID:       "exec-active-child",
			RunID:             "run-active",
			ParentExecutionID: &rootActive,
			AgentNodeID:       "agent-x",
			ReasonerID:        "intake_phase",
			NodeID:            "node-x",
			Status:            string(types.ExecutionStatusSucceeded),
			StartedAt:         base.Add(time.Second),
			CreatedAt:         base.Add(time.Second),
			UpdatedAt:         base.Add(time.Second),
		},
		// Fully terminal run → excluded.
		{
			ExecutionID: "exec-done",
			RunID:       "run-done",
			AgentNodeID: "agent-x",
			ReasonerID:  "review",
			NodeID:      "node-x",
			Status:      string(types.ExecutionStatusSucceeded),
			StartedAt:   base,
			CreatedAt:   base,
			UpdatedAt:   base,
		},
		// Queued-only run → active. A Status="running" pre-filter would miss
		// this one; ActiveOnly must not.
		{
			ExecutionID: "exec-queued",
			RunID:       "run-queued",
			AgentNodeID: "agent-y",
			ReasonerID:  "plan",
			NodeID:      "node-y",
			Status:      string(types.ExecutionStatusQueued),
			StartedAt:   base,
			CreatedAt:   base,
			UpdatedAt:   base,
		},
		// Paused-only run → active. Paused is non-terminal; a pause-wedged
		// run vanishing from the in-flight view was the original bug this
		// surface exists to expose.
		{
			ExecutionID: "exec-paused",
			RunID:       "run-paused",
			AgentNodeID: "agent-x",
			ReasonerID:  "plan",
			NodeID:      "node-x",
			Status:      string(types.ExecutionStatusPaused),
			StartedAt:   base,
			CreatedAt:   base,
			UpdatedAt:   base,
		},
	}
	crossRoot := "exec-cross-root"
	executions = append(executions,
		// Cross-agent run: running root on agent-x, terminal child on
		// agent-y. The agent-y filter must still return this run — with the
		// agent-x rows in its counts and the agent-x root fields intact.
		&types.Execution{
			ExecutionID: crossRoot,
			RunID:       "run-cross",
			AgentNodeID: "agent-x",
			ReasonerID:  "orchestrate",
			NodeID:      "node-x",
			Status:      string(types.ExecutionStatusRunning),
			StartedAt:   base,
			CreatedAt:   base,
			UpdatedAt:   base,
		},
		&types.Execution{
			ExecutionID:       "exec-cross-child",
			RunID:             "run-cross",
			ParentExecutionID: &crossRoot,
			AgentNodeID:       "agent-y",
			ReasonerID:        "summarize",
			NodeID:            "node-y",
			Status:            string(types.ExecutionStatusSucceeded),
			StartedAt:         base.Add(time.Second),
			CreatedAt:         base.Add(time.Second),
			UpdatedAt:         base.Add(time.Second),
		},
	)
	for _, exec := range executions {
		require.NoError(t, ls.CreateExecutionRecord(ctx, exec))
	}

	results, total, err := ls.QueryRunSummaries(ctx, types.ExecutionFilter{ActiveOnly: true})
	require.NoError(t, err)
	require.Equal(t, 4, total)
	require.Len(t, results, 4)

	byRunID := map[string]*RunSummaryAggregation{}
	for _, r := range results {
		byRunID[r.RunID] = r
	}
	require.Contains(t, byRunID, "run-active")
	require.Contains(t, byRunID, "run-queued")
	require.Contains(t, byRunID, "run-paused")
	require.Contains(t, byRunID, "run-cross")
	require.NotContains(t, byRunID, "run-done")

	// ActiveOnly filters whole runs, not rows: the surviving run's terminal
	// children must still be counted.
	active := byRunID["run-active"]
	require.Equal(t, 2, active.TotalExecutions)
	require.Equal(t, 1, active.StatusCounts[string(types.ExecutionStatusSucceeded)])
	require.Equal(t, 1, active.StatusCounts[string(types.ExecutionStatusRunning)])
	require.Equal(t, 1, active.ActiveExecutions)

	// AgentNodeID keeps whole runs that touched the agent — including
	// run-cross, whose only agent-y execution is already terminal, and whose
	// counts and root fields must reflect ALL rows, not just agent-y's.
	agent := "agent-y"
	results, total, err = ls.QueryRunSummaries(ctx, types.ExecutionFilter{ActiveOnly: true, AgentNodeID: &agent})
	require.NoError(t, err)
	require.Equal(t, 2, total)
	require.Len(t, results, 2)
	byRunID = map[string]*RunSummaryAggregation{}
	for _, r := range results {
		byRunID[r.RunID] = r
	}
	require.Contains(t, byRunID, "run-queued")
	cross := byRunID["run-cross"]
	require.NotNil(t, cross)
	require.Equal(t, 2, cross.TotalExecutions)
	require.Equal(t, 1, cross.StatusCounts[string(types.ExecutionStatusRunning)])
	require.Equal(t, 1, cross.StatusCounts[string(types.ExecutionStatusSucceeded)])
	require.Equal(t, crossRoot, derefOrEmptyTest(cross.RootExecutionID))
	require.Equal(t, "agent-x", derefOrEmptyTest(cross.RootAgentNodeID))
}

func derefOrEmptyTest(s *string) string {
	if s == nil {
		return ""
	}
	return *s
}

// TestQueryRunSummariesSessionRunMembership reproduces the live wedge-heuristic
// false alarm: a session-scoped /executions/active poll must count a run's
// in-process child executions even though only the root row carries session_id.
// Child records created through the workflow-execution-events path (SDK
// CallLocal) are persisted without a session_id, so a row-level session_id = ?
// filter collapsed the whole run to its root alone (total/active stuck at 1,
// latest_activity frozen at dispatch). The filter is run-level membership.
func TestQueryRunSummariesSessionRunMembership(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	base := time.Date(2024, 4, 1, 8, 0, 0, 0, time.UTC)
	session := "claude-subharness-test"
	rootID := "exec-session-root"

	sessionRun := []*types.Execution{
		// Root carries the session id (set by the dispatch path).
		{
			ExecutionID: rootID,
			RunID:       "run-session",
			AgentNodeID: "pr-af-go",
			ReasonerID:  "review",
			NodeID:      "pr-af-go",
			Status:      string(types.ExecutionStatusRunning),
			SessionID:   &session,
			StartedAt:   base,
			CreatedAt:   base,
			UpdatedAt:   base,
		},
	}
	// Two in-process child calls with NO session_id — the shape the
	// workflow-execution-events path persists. One still running, one done.
	childStatuses := []string{
		string(types.ExecutionStatusSucceeded),
		string(types.ExecutionStatusRunning),
	}
	for i, status := range childStatuses {
		child := &types.Execution{
			ExecutionID:       "exec-session-child-" + string(rune('a'+i)),
			RunID:             "run-session",
			ParentExecutionID: &rootID,
			AgentNodeID:       "pr-af-go",
			ReasonerID:        "verify_obligation",
			NodeID:            "pr-af-go",
			Status:            status,
			// SessionID deliberately nil.
			StartedAt: base.Add(time.Duration(i+1) * time.Minute),
			CreatedAt: base.Add(time.Duration(i+1) * time.Minute),
			UpdatedAt: base.Add(time.Duration(i+1) * time.Minute),
		}
		sessionRun = append(sessionRun, child)
	}
	// A second run in a different session that must NOT leak into the results.
	otherSession := "someone-else"
	sessionRun = append(sessionRun, &types.Execution{
		ExecutionID: "exec-other-root",
		RunID:       "run-other",
		AgentNodeID: "pr-af-go",
		ReasonerID:  "review",
		NodeID:      "pr-af-go",
		Status:      string(types.ExecutionStatusRunning),
		SessionID:   &otherSession,
		StartedAt:   base,
		CreatedAt:   base,
		UpdatedAt:   base,
	})
	for _, exec := range sessionRun {
		require.NoError(t, ls.CreateExecutionRecord(ctx, exec))
	}

	results, total, err := ls.QueryRunSummaries(ctx, types.ExecutionFilter{
		ActiveOnly: true,
		SessionID:  &session,
	})
	require.NoError(t, err)
	require.Equal(t, 1, total)
	require.Len(t, results, 1)

	run := results[0]
	require.Equal(t, "run-session", run.RunID)
	// Root + both children counted — not collapsed to the root alone.
	require.Equal(t, 3, run.TotalExecutions)
	require.Equal(t, 2, run.StatusCounts[string(types.ExecutionStatusRunning)])
	require.Equal(t, 1, run.StatusCounts[string(types.ExecutionStatusSucceeded)])
	require.Equal(t, 2, run.ActiveExecutions)
	// Run's session id is still reported from the root row.
	require.Equal(t, session, derefOrEmptyTest(run.SessionID))

	// The other session's run must not appear.
	other, otherTotal, err := ls.QueryRunSummaries(ctx, types.ExecutionFilter{
		ActiveOnly: true,
		SessionID:  &otherSession,
	})
	require.NoError(t, err)
	require.Equal(t, 1, otherTotal)
	require.Len(t, other, 1)
	require.Equal(t, "run-other", other[0].RunID)
	require.Equal(t, 1, other[0].TotalExecutions)
}

func TestQueryRunSummariesClampsLimit(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	base := time.Date(2024, 3, 1, 8, 0, 0, 0, time.UTC)
	require.NoError(t, ls.CreateExecutionRecord(ctx, &types.Execution{
		ExecutionID: "exec-clamp",
		RunID:       "run-clamp",
		AgentNodeID: "agent-x",
		ReasonerID:  "review",
		NodeID:      "node-x",
		Status:      string(types.ExecutionStatusSucceeded),
		StartedAt:   base,
		CreatedAt:   base,
		UpdatedAt:   base,
	}))

	// Limit pre-sizes the result slices; without the maxRunSummaryLimit clamp
	// this request would attempt a multi-gigabyte allocation before reading a
	// single row.
	results, total, err := ls.QueryRunSummaries(ctx, types.ExecutionFilter{Limit: math.MaxInt32})
	require.NoError(t, err)
	require.Equal(t, 1, total)
	require.Len(t, results, 1)
	require.Equal(t, "run-clamp", results[0].RunID)
}

func pointerTime(t time.Time) *time.Time {
	return &t
}
