package handlers

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/Agent-Field/agentfield/control-plane/internal/events"
	"github.com/Agent-Field/agentfield/control-plane/pkg/types"
)

type testExecutionStorage struct {
	mu                        sync.Mutex
	agent                     *types.AgentNode
	workflowExecutions        map[string]*types.WorkflowExecution
	executionRecords          map[string]*types.Execution
	executionLogs             map[string][]*types.ExecutionLogEntry
	runs                      map[string]*types.WorkflowRun
	steps                     map[string]*types.WorkflowStep
	webhooks                  map[string]*types.ExecutionWebhook
	eventBus                  *events.ExecutionEventBus
	workflowExecutionEventBus *events.EventBus[*types.WorkflowExecutionEvent]
	executionLogEventBus      *events.EventBus[*types.ExecutionLogEntry]
	workflowRunEventBus       *events.EventBus[*types.WorkflowRunEvent]
	updateCh                  chan string
}

func newTestExecutionStorage(agent *types.AgentNode) *testExecutionStorage {
	return &testExecutionStorage{
		agent:                     agent,
		workflowExecutions:        make(map[string]*types.WorkflowExecution),
		executionRecords:          make(map[string]*types.Execution),
		executionLogs:             make(map[string][]*types.ExecutionLogEntry),
		runs:                      make(map[string]*types.WorkflowRun),
		steps:                     make(map[string]*types.WorkflowStep),
		webhooks:                  make(map[string]*types.ExecutionWebhook),
		eventBus:                  events.NewExecutionEventBus(),
		workflowExecutionEventBus: events.NewEventBus[*types.WorkflowExecutionEvent](),
		executionLogEventBus:      events.NewEventBus[*types.ExecutionLogEntry](),
		workflowRunEventBus:       events.NewEventBus[*types.WorkflowRunEvent](),
		updateCh:                  make(chan string, 10),
	}
}

func (s *testExecutionStorage) GetAgent(ctx context.Context, id string) (*types.AgentNode, error) {
	if s.agent != nil && s.agent.ID == id {
		return s.agent, nil
	}
	return nil, nil
}

func (s *testExecutionStorage) ListAgentVersions(ctx context.Context, id string) ([]*types.AgentNode, error) {
	return nil, nil
}

func (s *testExecutionStorage) StoreWorkflowExecution(ctx context.Context, execution *types.WorkflowExecution) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if execution == nil {
		return fmt.Errorf("execution cannot be nil")
	}
	s.workflowExecutions[execution.ExecutionID] = execution
	select {
	case s.updateCh <- execution.ExecutionID:
	default:
	}
	return nil
}

func (s *testExecutionStorage) UpdateWorkflowExecution(ctx context.Context, executionID string, updateFunc func(*types.WorkflowExecution) (*types.WorkflowExecution, error)) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	existing, ok := s.workflowExecutions[executionID]
	if !ok {
		return fmt.Errorf("execution %s not found", executionID)
	}

	if updateFunc == nil {
		return fmt.Errorf("updateFunc cannot be nil")
	}

	updated, err := updateFunc(existing)
	if err != nil {
		return err
	}
	if updated != nil {
		s.workflowExecutions[executionID] = updated
	}
	select {
	case s.updateCh <- executionID:
	default:
	}
	return nil
}

func (s *testExecutionStorage) GetWorkflowExecution(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	execution, ok := s.workflowExecutions[executionID]
	if !ok {
		return nil, nil
	}
	return execution, nil
}

func (s *testExecutionStorage) StoreWorkflowRun(ctx context.Context, run *types.WorkflowRun) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if run == nil {
		return fmt.Errorf("run cannot be nil")
	}
	s.runs[run.RunID] = run
	return nil
}

func (s *testExecutionStorage) GetWorkflowRun(ctx context.Context, runID string) (*types.WorkflowRun, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	run, ok := s.runs[runID]
	if !ok {
		return nil, nil
	}
	return run, nil
}

func (s *testExecutionStorage) UpdateWorkflowRun(ctx context.Context, runID string, updateFunc func(*types.WorkflowRun) (*types.WorkflowRun, error)) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	run, ok := s.runs[runID]
	if !ok {
		return fmt.Errorf("run %s not found", runID)
	}
	updated, err := updateFunc(run)
	if err != nil {
		return err
	}
	if updated != nil {
		s.runs[runID] = updated
	}
	return nil
}

func (s *testExecutionStorage) QueryWorkflowRuns(ctx context.Context, filters types.WorkflowRunFilters) ([]*types.WorkflowRun, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var results []*types.WorkflowRun
	for _, run := range s.runs {
		if filters.RunID != nil && *filters.RunID != run.RunID {
			continue
		}
		results = append(results, run)
	}
	return results, nil
}

func (s *testExecutionStorage) CountWorkflowRuns(ctx context.Context, filters types.WorkflowRunFilters) (int, error) {
	runs, _ := s.QueryWorkflowRuns(ctx, filters)
	return len(runs), nil
}

func (s *testExecutionStorage) StoreWorkflowStep(ctx context.Context, step *types.WorkflowStep) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if step == nil {
		return fmt.Errorf("step cannot be nil")
	}
	s.steps[step.StepID] = step
	return nil
}

func (s *testExecutionStorage) StoreWorkflowRunAndStep(ctx context.Context, run *types.WorkflowRun, step *types.WorkflowStep) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if run == nil {
		return fmt.Errorf("run cannot be nil")
	}
	if step == nil {
		return fmt.Errorf("step cannot be nil")
	}
	s.runs[run.RunID] = run
	s.steps[step.StepID] = step
	return nil
}

func (s *testExecutionStorage) GetWorkflowStep(ctx context.Context, stepID string) (*types.WorkflowStep, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	step, ok := s.steps[stepID]
	if !ok {
		return nil, nil
	}
	return step, nil
}

func (s *testExecutionStorage) GetExecutionEventBus() *events.ExecutionEventBus {
	return s.eventBus
}

func (s *testExecutionStorage) GetWorkflowExecutionEventBus() *events.EventBus[*types.WorkflowExecutionEvent] {
	return s.workflowExecutionEventBus
}

func (s *testExecutionStorage) GetExecutionLogEventBus() *events.EventBus[*types.ExecutionLogEntry] {
	return s.executionLogEventBus
}

func (s *testExecutionStorage) GetWorkflowRunEventBus() *events.EventBus[*types.WorkflowRunEvent] {
	return s.workflowRunEventBus
}

func (s *testExecutionStorage) RegisterExecutionWebhook(ctx context.Context, webhook *types.ExecutionWebhook) error {
	if webhook == nil {
		return fmt.Errorf("webhook cannot be nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.webhooks[webhook.ExecutionID] = webhook
	return nil
}

func (s *testExecutionStorage) CreateExecutionRecord(ctx context.Context, execution *types.Execution) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if execution == nil {
		return fmt.Errorf("execution cannot be nil")
	}
	copy := *execution
	s.executionRecords[execution.ExecutionID] = &copy
	select {
	case s.updateCh <- execution.ExecutionID:
	default:
	}
	return nil
}

func (s *testExecutionStorage) GetExecutionRecord(ctx context.Context, executionID string) (*types.Execution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	execution, ok := s.executionRecords[executionID]
	if !ok {
		return nil, nil
	}
	copy := *execution
	return &copy, nil
}

func (s *testExecutionStorage) UpdateExecutionRecord(ctx context.Context, executionID string, update func(*types.Execution) (*types.Execution, error)) (*types.Execution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	current, ok := s.executionRecords[executionID]
	if !ok {
		return nil, fmt.Errorf("execution %s not found", executionID)
	}

	cloned := *current
	updated, err := update(&cloned)
	if err != nil {
		return nil, err
	}
	if updated != nil {
		cloned = *updated
	}
	s.executionRecords[executionID] = &cloned
	select {
	case s.updateCh <- executionID:
	default:
	}
	out := cloned
	return &out, nil
}

func (s *testExecutionStorage) QueryWorkflowExecutions(ctx context.Context, filters types.WorkflowExecutionFilters) ([]*types.WorkflowExecution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var results []*types.WorkflowExecution
	for _, wfExec := range s.workflowExecutions {
		if filters.ApprovalRequestID != nil && (wfExec.ApprovalRequestID == nil || *wfExec.ApprovalRequestID != *filters.ApprovalRequestID) {
			continue
		}
		results = append(results, wfExec)
	}
	return results, nil
}

func (s *testExecutionStorage) StoreWorkflowExecutionEvent(ctx context.Context, event *types.WorkflowExecutionEvent) error {
	return nil
}

func (s *testExecutionStorage) StoreExecutionLogEntry(ctx context.Context, entry *types.ExecutionLogEntry) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if entry == nil {
		return fmt.Errorf("execution log entry cannot be nil")
	}

	copy := *entry
	copy.Sequence = int64(len(s.executionLogs[copy.ExecutionID]) + 1)
	s.executionLogs[copy.ExecutionID] = append(s.executionLogs[copy.ExecutionID], &copy)
	s.executionLogEventBus.Publish(&copy)
	return nil
}

func (s *testExecutionStorage) ListExecutionLogEntries(ctx context.Context, executionID string, afterSeq *int64, limit int, levels []string, nodeIDs []string, sources []string, query string) ([]*types.ExecutionLogEntry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	matchesString := func(needle string, haystack []string) bool {
		if len(haystack) == 0 {
			return true
		}
		for _, value := range haystack {
			if value == needle {
				return true
			}
		}
		return false
	}

	out := make([]*types.ExecutionLogEntry, 0)
	for _, entry := range s.executionLogs[executionID] {
		if afterSeq != nil && entry.Sequence <= *afterSeq {
			continue
		}
		if !matchesString(entry.Level, levels) {
			continue
		}
		if !matchesString(entry.AgentNodeID, nodeIDs) {
			continue
		}
		if !matchesString(entry.Source, sources) {
			continue
		}
		if trimmed := strings.TrimSpace(query); trimmed != "" &&
			!strings.Contains(entry.Message, trimmed) &&
			!strings.Contains(string(entry.Attributes), trimmed) {
			continue
		}
		copy := *entry
		out = append(out, &copy)
	}
	if limit > 0 && len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

func (s *testExecutionStorage) PruneExecutionLogEntries(ctx context.Context, executionID string, maxEntries int, olderThan time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	current := s.executionLogs[executionID]
	filtered := make([]*types.ExecutionLogEntry, 0, len(current))
	for _, entry := range current {
		if !olderThan.IsZero() && entry.EmittedAt.Before(olderThan) {
			continue
		}
		filtered = append(filtered, entry)
	}
	if maxEntries > 0 && len(filtered) > maxEntries {
		filtered = filtered[len(filtered)-maxEntries:]
	}
	s.executionLogs[executionID] = filtered
	return nil
}

func (s *testExecutionStorage) QueryExecutionRecords(ctx context.Context, filter types.ExecutionFilter) ([]*types.Execution, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	results := make([]*types.Execution, 0, len(s.executionRecords))
	for _, exec := range s.executionRecords {
		if filter.ExecutionID != nil && *filter.ExecutionID != exec.ExecutionID {
			continue
		}
		if filter.RunID != nil && *filter.RunID != exec.RunID {
			continue
		}
		copy := *exec
		results = append(results, &copy)
	}
	return results, nil
}
