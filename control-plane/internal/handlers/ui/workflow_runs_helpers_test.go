package ui

import (
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/handlers"
	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

func TestConvertAggregationToSummary(t *testing.T) {
	now := time.Date(2026, 4, 7, 12, 0, 0, 0, time.UTC)
	rootExecutionID := "exec-root"
	rootReasonerID := "planner"
	rootAgentID := "agent-alpha"
	sessionID := "session-1"
	actorID := "actor-1"

	agg := &storage.RunSummaryAggregation{
		RunID:            "run-1",
		RootExecutionID:  &rootExecutionID,
		RootReasonerID:   &rootReasonerID,
		RootAgentNodeID:  &rootAgentID,
		SessionID:        &sessionID,
		ActorID:          &actorID,
		StatusCounts:     map[string]int{string(types.ExecutionStatusSucceeded): 3},
		TotalExecutions:  3,
		MaxDepth:         2,
		ActiveExecutions: 0,
		EarliestStarted:  now.Add(-2 * time.Minute),
		LatestStarted:    now,
	}

	summary := convertAggregationToSummary(agg)
	require.Equal(t, "run-1", summary.WorkflowID)
	require.Equal(t, "run-1", summary.RunID)
	require.Equal(t, "exec-root", summary.RootExecutionID)
	require.Equal(t, "planner", summary.DisplayName)
	require.Equal(t, "planner", summary.RootReasoner)
	require.Equal(t, "planner", summary.CurrentTask)
	require.Equal(t, &rootAgentID, summary.AgentID)
	require.Equal(t, &sessionID, summary.SessionID)
	require.Equal(t, &actorID, summary.ActorID)
	require.Equal(t, string(types.ExecutionStatusSucceeded), summary.Status)
	require.True(t, summary.Terminal)
	require.NotNil(t, summary.CompletedAt)
	require.NotNil(t, summary.DurationMs)
	require.Equal(t, int64(120000), *summary.DurationMs)

	t.Run("falls back to run id when root reasoner is missing", func(t *testing.T) {
		agg := &storage.RunSummaryAggregation{
			RunID:            "run-2",
			StatusCounts:     map[string]int{string(types.ExecutionStatusFailed): 1},
			TotalExecutions:  1,
			ActiveExecutions: 0,
			EarliestStarted:  now.Add(-1 * time.Minute),
			LatestStarted:    now,
		}

		summary := convertAggregationToSummary(agg)
		require.Equal(t, "run-2", summary.DisplayName)
		require.Equal(t, "run-2", summary.RootReasoner)
		require.Equal(t, "run-2", summary.CurrentTask)
		require.Equal(t, string(types.ExecutionStatusFailed), summary.Status)
		require.True(t, summary.Terminal)
	})
}

func TestDeriveStatusFromCounts(t *testing.T) {
	require.Equal(t, string(types.ExecutionStatusRunning), deriveStatusFromCounts(nil, 1))
	require.Equal(t, string(types.ExecutionStatusFailed), deriveStatusFromCounts(map[string]int{
		string(types.ExecutionStatusFailed): 1,
	}, 0))
	require.Equal(t, string(types.ExecutionStatusTimeout), deriveStatusFromCounts(map[string]int{
		string(types.ExecutionStatusTimeout): 1,
	}, 0))
	require.Equal(t, string(types.ExecutionStatusCancelled), deriveStatusFromCounts(map[string]int{
		string(types.ExecutionStatusCancelled): 1,
	}, 0))
	require.Equal(t, string(types.ExecutionStatusSucceeded), deriveStatusFromCounts(map[string]int{}, 0))
}

func TestSummarizeRun(t *testing.T) {
	started := time.Date(2026, 4, 7, 11, 58, 0, 0, time.UTC)
	rootCompleted := started.Add(3 * time.Minute)
	childCompleted := started.Add(4 * time.Minute)
	rootID := "exec-root"
	sessionID := "session-7"
	actorID := "actor-7"

	executions := []*types.Execution{
		{
			ExecutionID: "exec-child",
			RunID:       "run-7",
			ParentExecutionID: &rootID,
			AgentNodeID: "agent-beta",
			ReasonerID:  "finalizer",
			Status:      string(types.ExecutionStatusSucceeded),
			StartedAt:   started.Add(2 * time.Minute),
			CompletedAt: &childCompleted,
			SessionID:   &sessionID,
			ActorID:     &actorID,
		},
		{
			ExecutionID: "exec-root",
			RunID:       "run-7",
			AgentNodeID: "agent-alpha",
			ReasonerID:  "planner",
			Status:      string(types.ExecutionStatusSucceeded),
			StartedAt:   started,
			CompletedAt: &rootCompleted,
			SessionID:   &sessionID,
			ActorID:     &actorID,
		},
	}

	summary := summarizeRun("run-7", executions)
	require.Equal(t, "run-7", summary.WorkflowID)
	require.Equal(t, "run-7", summary.RunID)
	require.Equal(t, "exec-root", summary.RootExecutionID)
	require.Equal(t, "planner", summary.RootReasoner)
	require.Equal(t, "finalizer", summary.CurrentTask)
	require.Equal(t, "planner", summary.DisplayName)
	require.Equal(t, string(types.ExecutionStatusSucceeded), summary.Status)
	require.Equal(t, 2, summary.TotalExecutions)
	require.Equal(t, 1, summary.MaxDepth)
	require.Equal(t, 0, summary.ActiveExecutions)
	require.Equal(t, started, summary.StartedAt)
	require.Equal(t, started.Add(2*time.Minute), summary.UpdatedAt)
	require.Equal(t, &sessionID, summary.SessionID)
	require.Equal(t, &actorID, summary.ActorID)
	require.True(t, summary.Terminal)
	require.NotNil(t, summary.DurationMs)
	require.Equal(t, int64(240000), *summary.DurationMs)
	require.Equal(t, map[string]int{string(types.ExecutionStatusSucceeded): 2}, summary.StatusCounts)

	empty := summarizeRun("run-empty", nil)
	require.Equal(t, "run-empty", empty.RunID)
	require.Zero(t, empty.TotalExecutions)
	require.Empty(t, empty.StatusCounts)
}

func TestWorkflowRunHelperUtilities(t *testing.T) {
	t.Run("clone status counts", func(t *testing.T) {
		require.Nil(t, cloneStatusCounts(nil))
		counts := map[string]int{"running": 2}
		cloned := cloneStatusCounts(counts)
		require.Equal(t, counts, cloned)
		cloned["running"] = 7
		require.Equal(t, 2, counts["running"])
	})

	t.Run("counts outcome steps", func(t *testing.T) {
		completed, failed := countOutcomeSteps([]*types.Execution{
			{Status: string(types.ExecutionStatusSucceeded)},
			{Status: string(types.ExecutionStatusFailed)},
			{Status: string(types.ExecutionStatusCancelled)},
			{Status: string(types.ExecutionStatusTimeout)},
			{Status: string(types.ExecutionStatusRunning)},
		})
		require.Equal(t, 1, completed)
		require.Equal(t, 3, failed)
	})

	t.Run("builds api executions with child counts", func(t *testing.T) {
		parentID := "exec-root"
		completedAt := "2026-04-07T12:05:00Z"
		reason := "waiting on approval"
		nodes := []handlers.WorkflowDAGNode{
			{
				WorkflowID:  "run-1",
				ExecutionID: parentID,
				AgentNodeID: "agent-alpha",
				ReasonerID:  "planner",
				Status:      string(types.ExecutionStatusRunning),
				StartedAt:   "2026-04-07T12:00:00Z",
			},
			{
				WorkflowID:        "run-1",
				ExecutionID:       "exec-waiting",
				ParentExecutionID: &parentID,
				AgentNodeID:       "agent-beta",
				ReasonerID:        "review",
				Status:            string(types.ExecutionStatusWaiting),
				StatusReason:      &reason,
				StartedAt:         "2026-04-07T12:01:00Z",
			},
			{
				WorkflowID:        "run-1",
				ExecutionID:       "exec-queued",
				ParentExecutionID: &parentID,
				AgentNodeID:       "agent-gamma",
				ReasonerID:        "draft",
				Status:            string(types.ExecutionStatusQueued),
				StartedAt:         "2026-04-07T12:02:00Z",
				CompletedAt:       &completedAt,
				WorkflowDepth:     1,
			},
		}

		apiExecutions := buildAPIExecutions(nodes)
		require.Len(t, apiExecutions, 3)
		require.Equal(t, 1, apiExecutions[0].ActiveChildren)
		require.Equal(t, 1, apiExecutions[0].PendingChildren)
		require.Nil(t, apiExecutions[0].ParentWorkflowID)
		require.NotNil(t, apiExecutions[1].ParentWorkflowID)
		require.Equal(t, "run-1", *apiExecutions[1].ParentWorkflowID)
		require.Equal(t, &completedAt, apiExecutions[2].CompletedAt)
		require.Equal(t, &reason, apiExecutions[1].StatusReason)
	})

	t.Run("parses pagination helpers", func(t *testing.T) {
		require.Equal(t, 10, parsePositiveInt(" 10 ", 2))
		require.Equal(t, 2, parsePositiveInt("0", 2))
		require.Equal(t, 4, parsePositiveIntWithin("0", 4, 1, 50))
		require.Equal(t, 50, parsePositiveIntWithin("99", 4, 1, 50))
		require.Equal(t, 7, parsePositiveIntWithin("7", 4, 1, 50))
	})

	t.Run("sanitizes run sort fields", func(t *testing.T) {
		require.Equal(t, "started_at", sanitizeRunSortField("created_at"))
		require.Equal(t, "status", sanitizeRunSortField("status"))
		require.Equal(t, "total_steps", sanitizeRunSortField("nodes"))
		require.Equal(t, "failed_steps", sanitizeRunSortField("failed"))
		require.Equal(t, "active_executions", sanitizeRunSortField("active"))
		require.Equal(t, "updated_at", sanitizeRunSortField("latest"))
		require.Equal(t, "updated_at", sanitizeRunSortField("unexpected"))
	})
}
