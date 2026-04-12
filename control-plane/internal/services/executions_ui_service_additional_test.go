package services

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/storage"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
	"github.com/stretchr/testify/require"
)

type executionStorageServiceStub struct {
	storage.StorageProvider
	executions  []*types.WorkflowExecution
	queryErr    error
	details     map[string]*types.WorkflowExecution
	detailsErr  error
	lastFilters types.WorkflowExecutionFilters
}

func (s *executionStorageServiceStub) QueryWorkflowExecutions(_ context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error) {
	s.lastFilters = filters
	if s.queryErr != nil {
		return nil, s.queryErr
	}

	filtered := make([]*types.WorkflowExecution, 0, len(s.executions))
	for _, execution := range s.executions {
		if filters.AgentNodeID != nil && execution.AgentNodeID != *filters.AgentNodeID {
			continue
		}
		if filters.WorkflowID != nil && execution.WorkflowID != *filters.WorkflowID {
			continue
		}
		if filters.SessionID != nil && (execution.SessionID == nil || *execution.SessionID != *filters.SessionID) {
			continue
		}
		if filters.ActorID != nil && (execution.ActorID == nil || *execution.ActorID != *filters.ActorID) {
			continue
		}
		if filters.Status != nil && execution.Status != *filters.Status {
			continue
		}
		filtered = append(filtered, execution)
	}

	return filtered, nil
}

func (s *executionStorageServiceStub) GetWorkflowExecution(_ context.Context, executionID string) (*types.WorkflowExecution, error) {
	if s.detailsErr != nil {
		return nil, s.detailsErr
	}
	if execution, ok := s.details[executionID]; ok {
		return execution, nil
	}
	return nil, errors.New("not found")
}

func TestExecutionsUIServiceSummaryGroupingAndDetails(t *testing.T) {
	now := time.Date(2026, 4, 7, 12, 0, 0, 0, time.UTC)
	workflowName := "Workflow One"
	sessionA := "session-a"
	sessionB := "session-b"
	actorA := "actor-a"
	actorB := "actor-b"
	input := int64(150)
	output := int64(300)
	completedAt := now.Add(2 * time.Minute)
	durationShort := int64(100)
	durationLong := int64(200)

	stub := &executionStorageServiceStub{
		executions: []*types.WorkflowExecution{
			{
				ID:           1,
				WorkflowID:   "wf-1",
				WorkflowName: &workflowName,
				ExecutionID:  "exec-1",
				SessionID:    &sessionA,
				ActorID:      &actorA,
				AgentNodeID:  "node-a",
				ReasonerID:   "reasoner-a",
				Status:       "completed",
				StartedAt:    now,
				CompletedAt:  &completedAt,
				DurationMS:   &durationLong,
				InputSize:    int(input),
				OutputSize:   int(output),
			},
			{
				ID:          2,
				WorkflowID:  "wf-2",
				ExecutionID: "exec-2",
				SessionID:   &sessionB,
				ActorID:     &actorB,
				AgentNodeID: "node-b",
				ReasonerID:  "reasoner-b",
				Status:      "running",
				StartedAt:   now.Add(time.Minute),
				DurationMS:  &durationShort,
			},
			{
				ID:          3,
				WorkflowID:  "wf-3",
				ExecutionID: "exec-3",
				AgentNodeID: "node-a",
				ReasonerID:  "reasoner-c",
				Status:      "failed",
				StartedAt:   now.Add(3 * time.Minute),
			},
		},
		details: map[string]*types.WorkflowExecution{
			"exec-2": {ExecutionID: "exec-2", WorkflowID: "wf-2"},
		},
	}
	service := NewExecutionsUIService(stub)

	summary, err := service.GetExecutionsSummary(context.Background(), ExecutionFiltersForUI{
		AgentNodeID: &[]string{"node-a"}[0],
		WorkflowID:  &[]string{"wf-1"}[0],
		SessionID:   &sessionA,
		ActorID:     &actorA,
		Status:      &[]string{"completed"}[0],
		StartTime:   &now,
		EndTime:     &completedAt,
		Search:      &[]string{"exec"}[0],
		Page:        1,
		PageSize:    10,
	}, ExecutionGroupingForUI{SortBy: "time", SortOrder: "asc"})
	require.NoError(t, err)
	require.Equal(t, int64(1), summary.TotalCount)
	require.Equal(t, 1, summary.TotalPages)
	require.False(t, summary.HasNext)
	require.False(t, summary.HasPrev)
	require.Len(t, summary.Executions, 1)
	require.NotNil(t, summary.Executions[0].InputSize)
	require.NotNil(t, summary.Executions[0].OutputSize)
	require.Equal(t, input, *summary.Executions[0].InputSize)
	require.Equal(t, output, *summary.Executions[0].OutputSize)
	require.Equal(t, "time", *stub.lastFilters.SortBy)
	require.Equal(t, "asc", *stub.lastFilters.SortOrder)
	require.Equal(t, 10, stub.lastFilters.Limit)
	require.Equal(t, 0, stub.lastFilters.Offset)
	require.NotNil(t, stub.lastFilters.StartTime)
	require.NotNil(t, stub.lastFilters.EndTime)
	require.NotNil(t, stub.lastFilters.Search)

	grouped, err := service.GetGroupedExecutions(context.Background(), ExecutionFiltersForUI{
		Page:     1,
		PageSize: 1,
	}, ExecutionGroupingForUI{GroupBy: "status", SortBy: "time", SortOrder: "desc"})
	require.NoError(t, err)
	require.Equal(t, int64(3), grouped.TotalCount)
	require.Len(t, grouped.Groups, 1)
	require.Equal(t, 3, grouped.TotalPages)
	require.True(t, grouped.HasNext)
	require.False(t, grouped.HasPrev)
	require.Equal(t, "failed", grouped.Groups[0].GroupKey)

	details, err := service.GetExecutionDetails(context.Background(), "exec-2")
	require.NoError(t, err)
	require.Equal(t, "wf-2", details.WorkflowID)
}

func TestExecutionsUIServiceErrorAndEventPaths(t *testing.T) {
	service := NewExecutionsUIService(&executionStorageServiceStub{queryErr: errors.New("boom"), detailsErr: errors.New("details boom")})

	_, err := service.GetExecutionsSummary(context.Background(), ExecutionFiltersForUI{Page: 1, PageSize: 10}, ExecutionGroupingForUI{})
	require.ErrorContains(t, err, "failed to query executions")

	_, err = service.GetGroupedExecutions(context.Background(), ExecutionFiltersForUI{Page: 1, PageSize: 10}, ExecutionGroupingForUI{GroupBy: "status"})
	require.ErrorContains(t, err, "failed to query executions")

	_, err = service.GetExecutionDetails(context.Background(), "missing")
	require.ErrorContains(t, err, "details boom")

	client := service.RegisterClient()
	blocked := service.RegisterClient()
	service.DeregisterClient(blocked)
	require.Equal(t, 1, service.countClients())

	execution := &types.WorkflowExecution{ExecutionID: "exec-1", WorkflowID: "wf-1", AgentNodeID: "node-a", Status: "running", StartedAt: time.Now().UTC()}
	service.BroadcastExecutionEvent("custom", execution)

	select {
	case event := <-client:
		require.Equal(t, "custom", event.Type)
		require.Equal(t, execution, event.Execution)
	default:
		t.Fatal("expected broadcast event")
	}

	callbackClient := service.RegisterClient()
	service.OnExecutionStarted(execution)
	service.OnExecutionCompleted(execution)
	service.OnExecutionFailed(execution)

	var eventTypes []string
	for i := 0; i < 3; i++ {
		select {
		case event := <-callbackClient:
			eventTypes = append(eventTypes, event.Type)
			require.IsType(t, ExecutionSummaryForUI{}, event.Execution)
		default:
			t.Fatal("expected callback event")
		}
	}
	require.ElementsMatch(t, []string{"execution_started", "execution_completed", "execution_failed"}, eventTypes)

	service.DeregisterClient(client)
	service.DeregisterClient(callbackClient)
	require.Equal(t, 0, service.countClients())
}

func TestExecutionsUIServiceSortGroupsCoversOrders(t *testing.T) {
	service := &ExecutionsUIService{}
	base := []GroupedExecutionSummary{
		{GroupKey: "c", GroupLabel: "Charlie", Count: 1, AvgDurationMS: 300, LatestExecution: time.Unix(300, 0)},
		{GroupKey: "a", GroupLabel: "Alpha", Count: 3, AvgDurationMS: 100, LatestExecution: time.Unix(100, 0)},
		{GroupKey: "b", GroupLabel: "Bravo", Count: 2, AvgDurationMS: 200, LatestExecution: time.Unix(200, 0)},
	}

	tests := []struct {
		name      string
		sortBy    string
		sortOrder string
		wantFirst string
	}{
		{name: "time asc", sortBy: "time", sortOrder: "asc", wantFirst: "a"},
		{name: "time desc", sortBy: "time", sortOrder: "desc", wantFirst: "c"},
		{name: "duration asc", sortBy: "duration", sortOrder: "asc", wantFirst: "a"},
		{name: "duration desc", sortBy: "duration", sortOrder: "desc", wantFirst: "c"},
		{name: "status asc", sortBy: "status", sortOrder: "asc", wantFirst: "a"},
		{name: "status desc", sortBy: "status", sortOrder: "desc", wantFirst: "c"},
		{name: "default count", sortBy: "other", sortOrder: "desc", wantFirst: "a"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			groups := append([]GroupedExecutionSummary(nil), base...)
			service.sortGroups(groups, tt.sortBy, tt.sortOrder)
			require.Equal(t, tt.wantFirst, groups[0].GroupKey)
		})
	}

	service.sortExecutions(nil, "time", "asc")
}
