package storage

import (
	"database/sql"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestExecutionRecordHelpers(t *testing.T) {
	t.Run("maps run summary sort columns", func(t *testing.T) {
		require.Equal(t, "earliest_started", mapRunSummarySortColumn("started_at"))
		require.Equal(t, "earliest_started", mapRunSummarySortColumn("created_at"))
		require.Equal(t, "status_rank", mapRunSummarySortColumn("status"))
		require.Equal(t, "total_executions", mapRunSummarySortColumn("nodes"))
		require.Equal(t, "failed_count", mapRunSummarySortColumn("failed"))
		require.Equal(t, "active_executions", mapRunSummarySortColumn("active"))
		require.Equal(t, "latest_activity", mapRunSummarySortColumn("latest"))
		require.Equal(t, "latest_activity", mapRunSummarySortColumn("unexpected"))
	})

	t.Run("computes max depth from parent-child relationships", func(t *testing.T) {
		require.Equal(t, 0, computeMaxDepth(nil))
		root := "root"
		child := "child"
		execInfos := []execDepthInfo{
			{executionID: root},
			{executionID: child, parentExecutionID: &root},
			{executionID: "grandchild", parentExecutionID: &child},
			{executionID: "sibling", parentExecutionID: &root},
		}
		require.Equal(t, 2, computeMaxDepth(execInfos))
	})

	t.Run("assigns and parses database time values", func(t *testing.T) {
		var assigned time.Time
		require.EqualError(t, assignTimeValue(nil, time.Now()), "nil destination provided for time assignment")
		require.NoError(t, assignTimeValue(&assigned, "2026-04-07T12:34:56"))
		require.Equal(t, time.Date(2026, 4, 7, 12, 34, 56, 0, time.UTC), assigned)

		parsed, err := parseDBTime(time.Date(2026, 4, 7, 12, 34, 56, 0, time.FixedZone("offset", 3600)))
		require.NoError(t, err)
		require.Equal(t, time.Date(2026, 4, 7, 11, 34, 56, 0, time.UTC), parsed)

		parsed, err = parseDBTime([]byte("2026-04-07 12:34:56+00:00"))
		require.NoError(t, err)
		require.Equal(t, time.Date(2026, 4, 7, 12, 34, 56, 0, time.UTC), parsed)

		parsed, err = parseDBTime(sql.NullString{String: "2026-04-07T12:34:56Z", Valid: true})
		require.NoError(t, err)
		require.Equal(t, time.Date(2026, 4, 7, 12, 34, 56, 0, time.UTC), parsed)

		parsed, err = parseDBTime(nil)
		require.NoError(t, err)
		require.True(t, parsed.IsZero())

		parsed, err = parseTimeString("2026-04-07 12:34:56.123456789")
		require.NoError(t, err)
		require.Equal(t, time.Date(2026, 4, 7, 12, 34, 56, 123456789, time.UTC), parsed)

		parsed, err = parseTimeString("2026-04-07T12:34:56")
		require.NoError(t, err)
		require.Equal(t, time.Date(2026, 4, 7, 12, 34, 56, 0, time.UTC), parsed)

		_, err = parseDBTime(123)
		require.EqualError(t, err, "unsupported time value type int")
		_, err = parseTimeString("not-a-time")
		require.EqualError(t, err, `unable to parse time value "not-a-time"`)
	})
}

func TestGetRunAggregation(t *testing.T) {
	ls, ctx := setupLocalStorage(t)

	rootID := "exec-root"
	started := time.Date(2026, 4, 7, 12, 0, 0, 0, time.UTC)
	completed := started.Add(4 * time.Minute)
	sessionID := "session-42"
	actorID := "actor-42"

	records := []*types.Execution{
		{
			ExecutionID: "exec-child-running",
			RunID:       "run-agg",
			ParentExecutionID: &rootID,
			AgentNodeID: "agent-beta",
			ReasonerID:  "review",
			Status:      string(types.ExecutionStatusRunning),
			StartedAt:   started.Add(2 * time.Minute),
		},
		{
			ExecutionID: "exec-root",
			RunID:       "run-agg",
			AgentNodeID: "agent-alpha",
			ReasonerID:  "planner",
			Status:      string(types.ExecutionStatusSucceeded),
			StartedAt:   started,
			CompletedAt: &completed,
			SessionID:   &sessionID,
			ActorID:     &actorID,
		},
		{
			ExecutionID: "exec-child-queued",
			RunID:       "run-agg",
			ParentExecutionID: &rootID,
			AgentNodeID: "agent-gamma",
			ReasonerID:  "draft",
			Status:      string(types.ExecutionStatusQueued),
			StartedAt:   started.Add(1 * time.Minute),
		},
	}

	for _, record := range records {
		require.NoError(t, ls.CreateExecutionRecord(ctx, record))
	}

	agg, err := ls.getRunAggregation(ctx, "run-agg")
	require.NoError(t, err)
	require.Equal(t, "run-agg", agg.RunID)
	require.Equal(t, 3, agg.TotalExecutions)
	require.Equal(t, started, agg.EarliestStarted)
	require.Equal(t, started.Add(2*time.Minute), agg.LatestStarted)
	require.Equal(t, 2, agg.ActiveExecutions)
	require.Equal(t, 1, agg.MaxDepth)
	require.NotNil(t, agg.RootExecutionID)
	require.Equal(t, "exec-root", *agg.RootExecutionID)
	require.NotNil(t, agg.RootAgentNodeID)
	require.Equal(t, "agent-alpha", *agg.RootAgentNodeID)
	require.NotNil(t, agg.RootReasonerID)
	require.Equal(t, "planner", *agg.RootReasonerID)
	require.NotNil(t, agg.SessionID)
	require.Equal(t, "session-42", *agg.SessionID)
	require.NotNil(t, agg.ActorID)
	require.Equal(t, "actor-42", *agg.ActorID)
	require.Equal(t, map[string]int{
		string(types.ExecutionStatusSucceeded): 1,
		string(types.ExecutionStatusRunning):   1,
		string(types.ExecutionStatusQueued):    1,
	}, agg.StatusCounts)
}
