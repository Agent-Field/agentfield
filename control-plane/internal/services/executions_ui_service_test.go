package services

import (
	"context"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

type executionStorageStub struct {
	storage.StorageProvider
	executions  []*types.WorkflowExecution
	lastFilters types.WorkflowExecutionFilters
}

func (s *executionStorageStub) QueryWorkflowExecutions(_ context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error) {
	s.lastFilters = filters

	filtered := make([]*types.WorkflowExecution, 0, len(s.executions))
	for _, execution := range s.executions {
		if filters.AgentNodeID != nil && execution.AgentNodeID != *filters.AgentNodeID {
			continue
		}
		if filters.WorkflowID != nil && execution.WorkflowID != *filters.WorkflowID {
			continue
		}
		if filters.SessionID != nil {
			if execution.SessionID == nil || *execution.SessionID != *filters.SessionID {
				continue
			}
		}
		if filters.Status != nil && execution.Status != *filters.Status {
			continue
		}
		filtered = append(filtered, execution)
	}

	if filters.Offset >= len(filtered) {
		return []*types.WorkflowExecution{}, nil
	}
	if filters.Offset > 0 {
		filtered = filtered[filters.Offset:]
	}
	if filters.Limit > 0 && len(filtered) > filters.Limit {
		filtered = filtered[:filters.Limit]
	}

	return filtered, nil
}

func executionStringPtr(value string) *string {
	return &value
}

func executionInt64Ptr(value int64) *int64 {
	return &value
}

func executionByID(t *testing.T, executions []ExecutionSummaryForUI, executionIDs ...string) {
	t.Helper()

	require.Len(t, executions, len(executionIDs))

	got := make([]string, 0, len(executions))
	for _, execution := range executions {
		got = append(got, execution.ExecutionID)
	}
	require.ElementsMatch(t, executionIDs, got)
}

func groupedExecutionByKey(t *testing.T, groups []GroupedExecutionSummary, key string) GroupedExecutionSummary {
	t.Helper()

	for _, group := range groups {
		if group.GroupKey == key {
			return group
		}
	}
	t.Fatalf("group %q not found", key)
	return GroupedExecutionSummary{}
}

func TestExecutionsUIServiceGroupedExecutionSummary(t *testing.T) {
	service := &ExecutionsUIService{}
	now := time.Date(2026, 4, 7, 12, 0, 0, 0, time.UTC)
	workflowName := "Workflow One"
	sessionID := "session-a"

	groups := service.groupExecutions([]*types.WorkflowExecution{
		{
			WorkflowID:   "wf-1",
			WorkflowName: &workflowName,
			ExecutionID:  "exec-1",
			AgentNodeID:  "node-a",
			Status:       "running",
			StartedAt:    now.Add(-3 * time.Minute),
			DurationMS:   executionInt64Ptr(100),
			SessionID:    executionStringPtr(sessionID),
		},
		{
			WorkflowID:   "wf-1",
			WorkflowName: &workflowName,
			ExecutionID:  "exec-2",
			AgentNodeID:  "node-b",
			Status:       "completed",
			StartedAt:    now.Add(-1 * time.Minute),
			DurationMS:   executionInt64Ptr(300),
			SessionID:    executionStringPtr(sessionID),
		},
		{
			WorkflowID:  "wf-2",
			ExecutionID: "exec-3",
			AgentNodeID: "node-a",
			Status:      "queued",
			StartedAt:   now.Add(-2 * time.Minute),
		},
	}, "workflow")

	require.Len(t, groups, 2)

	wf1 := groupedExecutionByKey(t, groups, "wf-1")
	require.Equal(t, "Workflow One", wf1.GroupLabel)
	require.Equal(t, 2, wf1.Count)
	require.Equal(t, int64(400), wf1.TotalDurationMS)
	require.Equal(t, int64(200), wf1.AvgDurationMS)
	require.Equal(t, now.Add(-1*time.Minute), wf1.LatestExecution)
	require.Equal(t, map[string]int{"running": 1, "completed": 1}, wf1.StatusSummary)
	require.Len(t, wf1.Executions, 2)

	wf2 := groupedExecutionByKey(t, groups, "wf-2")
	require.Equal(t, 1, wf2.Count)
	require.Equal(t, int64(0), wf2.TotalDurationMS)
	require.Equal(t, int64(0), wf2.AvgDurationMS)
	require.Equal(t, map[string]int{"queued": 1}, wf2.StatusSummary)
}

func TestExecutionsUIServiceStatusSummaryIncludesAllEncounteredStatuses(t *testing.T) {
	service := &ExecutionsUIService{}
	now := time.Date(2026, 4, 7, 12, 0, 0, 0, time.UTC)

	groups := service.groupExecutions([]*types.WorkflowExecution{
		{WorkflowID: "wf-1", AgentNodeID: "node-a", ExecutionID: "exec-1", Status: "running", StartedAt: now},
		{WorkflowID: "wf-2", AgentNodeID: "node-a", ExecutionID: "exec-2", Status: "waiting_for_approval", StartedAt: now.Add(time.Minute)},
		{WorkflowID: "wf-3", AgentNodeID: "node-a", ExecutionID: "exec-3", Status: "custom_terminal_status", StartedAt: now.Add(2 * time.Minute)},
	}, "agent")

	require.Len(t, groups, 1)
	require.Equal(t, map[string]int{
		"running":                1,
		"waiting_for_approval":   1,
		"custom_terminal_status": 1,
	}, groups[0].StatusSummary)
}

func TestExecutionsUIServiceEmptyGroupingInput(t *testing.T) {
	service := &ExecutionsUIService{}

	groups := service.groupExecutions(nil, "workflow")
	require.Empty(t, groups)
}

func TestExecutionsUIServiceFiltersByAgentWorkflowSessionAndStatus(t *testing.T) {
	now := time.Date(2026, 4, 7, 12, 0, 0, 0, time.UTC)
	stub := &executionStorageStub{
		executions: []*types.WorkflowExecution{
			{ID: 1, WorkflowID: "wf-1", ExecutionID: "exec-1", AgentNodeID: "node-a", Status: "running", StartedAt: now, SessionID: executionStringPtr("session-a")},
			{ID: 2, WorkflowID: "wf-1", ExecutionID: "exec-2", AgentNodeID: "node-b", Status: "completed", StartedAt: now.Add(time.Minute), SessionID: executionStringPtr("session-b")},
			{ID: 3, WorkflowID: "wf-2", ExecutionID: "exec-3", AgentNodeID: "node-a", Status: "failed", StartedAt: now.Add(2 * time.Minute), SessionID: executionStringPtr("session-c")},
			{ID: 4, WorkflowID: "wf-3", ExecutionID: "exec-4", AgentNodeID: "node-c", Status: "completed", StartedAt: now.Add(3 * time.Minute), SessionID: executionStringPtr("session-b")},
		},
	}
	service := NewExecutionsUIService(stub)

	tests := []struct {
		name         string
		filters      ExecutionFiltersForUI
		executionIDs []string
	}{
		{
			name: "agent node filter",
			filters: ExecutionFiltersForUI{
				AgentNodeID: executionStringPtr("node-a"),
				Page:        1,
				PageSize:    10,
			},
			executionIDs: []string{"exec-1", "exec-3"},
		},
		{
			name: "workflow filter",
			filters: ExecutionFiltersForUI{
				WorkflowID: executionStringPtr("wf-1"),
				Page:       1,
				PageSize:   10,
			},
			executionIDs: []string{"exec-1", "exec-2"},
		},
		{
			name: "session filter",
			filters: ExecutionFiltersForUI{
				SessionID: executionStringPtr("session-b"),
				Page:      1,
				PageSize:  10,
			},
			executionIDs: []string{"exec-2", "exec-4"},
		},
		{
			name: "status filter",
			filters: ExecutionFiltersForUI{
				Status:   executionStringPtr("completed"),
				Page:     1,
				PageSize: 10,
			},
			executionIDs: []string{"exec-2", "exec-4"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := service.GetExecutionsSummary(context.Background(), tt.filters, ExecutionGroupingForUI{})
			require.NoError(t, err)
			executionByID(t, result.Executions, tt.executionIDs...)
		})
	}
}
