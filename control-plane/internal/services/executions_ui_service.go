package services

import (
	"context"
	"fmt"
	"github.com/your-org/agentfield/control-plane/internal/storage"
	"github.com/your-org/agentfield/control-plane/pkg/types"
	"sync"
	"time"

	"github.com/your-org/agentfield/control-plane/internal/logger"
)

// ExecutionEvent represents a real-time event related to executions.
type ExecutionEvent struct {
	Type      string      `json:"type"` // e.g., "execution_started", "execution_completed", "execution_failed"
	Execution interface{} `json:"execution"`
	Timestamp time.Time   `json:"timestamp"`
}

// ExecutionSummaryForUI is optimized for UI display with aggregated data.
type ExecutionSummaryForUI struct {
	ID           int64      `json:"id"`
	WorkflowID   string     `json:"workflow_id"`
	ExecutionID  string     `json:"execution_id"`
	SessionID    *string    `json:"session_id"`
	ActorID      *string    `json:"actor_id"`
	AgentNodeID  string     `json:"agent_node_id"`
	ReasonerID   string     `json:"reasoner_id"`
	Status       string     `json:"status"`
	StartedAt    time.Time  `json:"started_at"`
	CompletedAt  *time.Time `json:"completed_at"`
	DurationMS   *int64     `json:"duration_ms"`
	ErrorMessage *string    `json:"error_message"`
	WorkflowName *string    `json:"workflow_name"`
	WorkflowTags []string   `json:"workflow_tags"`
	InputSize    *int64     `json:"input_size"`
	OutputSize   *int64     `json:"output_size"`
}

// GroupedExecutionSummary represents executions grouped by a specific field.
type GroupedExecutionSummary struct {
	GroupKey        string                  `json:"group_key"`
	GroupLabel      string                  `json:"group_label"`
	Count           int                     `json:"count"`
	TotalDurationMS int64                   `json:"total_duration_ms"`
	AvgDurationMS   int64                   `json:"avg_duration_ms"`
	StatusSummary   map[string]int          `json:"status_summary"`
	LatestExecution time.Time               `json:"latest_execution"`
	Executions      []ExecutionSummaryForUI `json:"executions"`
}

// ExecutionFiltersForUI represents UI-friendly filters.
type ExecutionFiltersForUI struct {
	AgentNodeID *string    `json:"agent_node_id"`
	WorkflowID  *string    `json:"workflow_id"`
	SessionID   *string    `json:"session_id"`
	ActorID     *string    `json:"actor_id"`
	Status      *string    `json:"status"`
	StartTime   *time.Time `json:"start_time"`
	EndTime     *time.Time `json:"end_time"`
	Search      *string    `json:"search"` // For quick search across multiple fields
	Page        int        `json:"page"`
	PageSize    int        `json:"page_size"`
}

// ExecutionGroupingForUI represents UI grouping options.
type ExecutionGroupingForUI struct {
	GroupBy   string `json:"group_by"`   // "none", "workflow", "session", "actor", "agent", "status"
	SortBy    string `json:"sort_by"`    // "time", "duration", "status"
	SortOrder string `json:"sort_order"` // "asc", "desc"
}

// PaginatedExecutionsForUI represents paginated execution results.
type PaginatedExecutionsForUI struct {
	Executions []ExecutionSummaryForUI `json:"executions"`
	TotalCount int64                   `json:"total_count"`
	Page       int                     `json:"page"`
	PageSize   int                     `json:"page_size"`
	TotalPages int                     `json:"total_pages"`
	HasNext    bool                    `json:"has_next"`
	HasPrev    bool                    `json:"has_prev"`
}

// GroupedExecutionsForUI represents grouped execution results.
type GroupedExecutionsForUI struct {
	Groups     []GroupedExecutionSummary `json:"groups"`
	TotalCount int64                     `json:"total_count"`
	Page       int                       `json:"page"`
	PageSize   int                       `json:"page_size"`
	TotalPages int                       `json:"total_pages"`
	HasNext    bool                      `json:"has_next"`
	HasPrev    bool                      `json:"has_prev"`
}

// ExecutionsUIService provides execution data optimized for UI consumption.
type ExecutionsUIService struct {
	storage storage.StorageProvider
	clients sync.Map // Map of chan ExecutionEvent to bool (true if active)
}

// NewExecutionsUIService creates a new ExecutionsUIService.
func NewExecutionsUIService(storageProvider storage.StorageProvider) *ExecutionsUIService {
	return &ExecutionsUIService{
		storage: storageProvider,
		clients: sync.Map{},
	}
}

// GetExecutionsSummary retrieves paginated execution summaries with filtering.
func (s *ExecutionsUIService) GetExecutionsSummary(
	ctx context.Context,
	filters ExecutionFiltersForUI,
	grouping ExecutionGroupingForUI,
) (*PaginatedExecutionsForUI, error) {

	// Convert UI filters to storage filters
	storageFilters := s.convertToStorageFilters(filters)

	// Add sorting parameters to storage filters
	if grouping.SortBy != "" {
		storageFilters.SortBy = &grouping.SortBy
	}
	if grouping.SortOrder != "" {
		storageFilters.SortOrder = &grouping.SortOrder
	}

	// Query executions from storage
	executions, err := s.storage.QueryWorkflowExecutions(ctx, storageFilters)
	if err != nil {
		return nil, fmt.Errorf("failed to query executions: %w", err)
	}

	// Convert to UI format
	uiExecutions := make([]ExecutionSummaryForUI, len(executions))
	for i, exec := range executions {
		uiExecutions[i] = s.convertToUISummary(exec)
	}

	// Calculate pagination - since we're doing pagination at DB level, we need to get total count separately
	// For now, we'll use the returned count as total (this is a simplification)
	totalCount := int64(len(uiExecutions))
	totalPages := int((totalCount + int64(filters.PageSize) - 1) / int64(filters.PageSize))

	return &PaginatedExecutionsForUI{
		Executions: uiExecutions,
		TotalCount: totalCount,
		Page:       filters.Page,
		PageSize:   filters.PageSize,
		TotalPages: totalPages,
		HasNext:    filters.Page < totalPages,
		HasPrev:    filters.Page > 1,
	}, nil
}

// GetGroupedExecutions retrieves executions grouped by the specified field.
func (s *ExecutionsUIService) GetGroupedExecutions(
	ctx context.Context,
	filters ExecutionFiltersForUI,
	grouping ExecutionGroupingForUI,
) (*GroupedExecutionsForUI, error) {

	// Get all executions first
	storageFilters := s.convertToStorageFilters(filters)
	executions, err := s.storage.QueryWorkflowExecutions(ctx, storageFilters)
	if err != nil {
		return nil, fmt.Errorf("failed to query executions: %w", err)
	}

	// Group executions
	groups := s.groupExecutions(executions, grouping.GroupBy)

	// Sort groups
	s.sortGroups(groups, grouping.SortBy, grouping.SortOrder)

	// Apply pagination to groups
	totalCount := int64(len(groups))
	totalPages := int((totalCount + int64(filters.PageSize) - 1) / int64(filters.PageSize))

	start := (filters.Page - 1) * filters.PageSize
	end := start + filters.PageSize
	if end > len(groups) {
		end = len(groups)
	}
	if start > len(groups) {
		start = len(groups)
	}

	paginatedGroups := groups[start:end]

	return &GroupedExecutionsForUI{
		Groups:     paginatedGroups,
		TotalCount: totalCount,
		Page:       filters.Page,
		PageSize:   filters.PageSize,
		TotalPages: totalPages,
		HasNext:    filters.Page < totalPages,
		HasPrev:    filters.Page > 1,
	}, nil
}

// GetExecutionDetails retrieves detailed information for a specific execution.
func (s *ExecutionsUIService) GetExecutionDetails(ctx context.Context, executionID string) (*types.WorkflowExecution, error) {
	return s.storage.GetWorkflowExecution(ctx, executionID)
}

// RegisterClient registers a new SSE client for execution events.
func (s *ExecutionsUIService) RegisterClient() chan ExecutionEvent {
	clientChan := make(chan ExecutionEvent, 100) // Buffered channel
	s.clients.Store(clientChan, true)
	logger.Logger.Debug().Msgf("➕ Execution SSE client registered. Total clients: %d", s.countClients())
	return clientChan
}

// DeregisterClient removes an SSE client.
func (s *ExecutionsUIService) DeregisterClient(clientChan chan ExecutionEvent) {
	s.clients.Delete(clientChan)
	close(clientChan)
	logger.Logger.Debug().Msgf("➖ Execution SSE client deregistered. Total clients: %d", s.countClients())
}

// BroadcastExecutionEvent sends an execution event to all registered SSE clients.
func (s *ExecutionsUIService) BroadcastExecutionEvent(eventType string, execution interface{}) {
	event := ExecutionEvent{
		Type:      eventType,
		Execution: execution,
		Timestamp: time.Now(),
	}

	s.clients.Range(func(key, value interface{}) bool {
		clientChan, ok := key.(chan ExecutionEvent)
		if !ok {
			return true
		}
		select {
		case clientChan <- event:
			// Event sent successfully
		default:
			// Client channel is blocked, deregister it
			s.DeregisterClient(clientChan)
		}
		return true
	})

}

// Helper methods

func (s *ExecutionsUIService) convertToStorageFilters(uiFilters ExecutionFiltersForUI) types.WorkflowExecutionFilters {
	filters := types.WorkflowExecutionFilters{
		Limit:  uiFilters.PageSize,
		Offset: (uiFilters.Page - 1) * uiFilters.PageSize,
	}

	if uiFilters.AgentNodeID != nil {
		filters.AgentNodeID = uiFilters.AgentNodeID
	}
	if uiFilters.WorkflowID != nil {
		filters.WorkflowID = uiFilters.WorkflowID
	}
	if uiFilters.SessionID != nil {
		filters.SessionID = uiFilters.SessionID
	}
	if uiFilters.ActorID != nil {
		filters.ActorID = uiFilters.ActorID
	}
	if uiFilters.Status != nil {
		filters.Status = uiFilters.Status
	}
	if uiFilters.StartTime != nil {
		filters.StartTime = uiFilters.StartTime
	}
	if uiFilters.EndTime != nil {
		filters.EndTime = uiFilters.EndTime
	}
	if uiFilters.Search != nil {
		filters.Search = uiFilters.Search
	}

	return filters
}

func (s *ExecutionsUIService) convertToUISummary(exec *types.WorkflowExecution) ExecutionSummaryForUI {
	var inputSize, outputSize *int64
	if exec.InputSize > 0 {
		size := int64(exec.InputSize)
		inputSize = &size
	}
	if exec.OutputSize > 0 {
		size := int64(exec.OutputSize)
		outputSize = &size
	}

	return ExecutionSummaryForUI{
		ID:           exec.ID,
		WorkflowID:   exec.WorkflowID,
		ExecutionID:  exec.ExecutionID,
		SessionID:    exec.SessionID,
		ActorID:      exec.ActorID,
		AgentNodeID:  exec.AgentNodeID,
		ReasonerID:   exec.ReasonerID,
		Status:       exec.Status,
		StartedAt:    exec.StartedAt,
		CompletedAt:  exec.CompletedAt,
		DurationMS:   exec.DurationMS,
		ErrorMessage: exec.ErrorMessage,
		WorkflowName: exec.WorkflowName,
		WorkflowTags: exec.WorkflowTags,
		InputSize:    inputSize,
		OutputSize:   outputSize,
	}
}

//nolint:unused // retained for future UI sorting enhancements
func (s *ExecutionsUIService) sortExecutions(executions []ExecutionSummaryForUI, sortBy, sortOrder string) {
	// Implementation for sorting executions
	// TODO: Implement sorting logic based on sortBy and sortOrder
}

func (s *ExecutionsUIService) groupExecutions(executions []*types.WorkflowExecution, groupBy string) []GroupedExecutionSummary {
	if groupBy == "none" || groupBy == "" {
		return []GroupedExecutionSummary{}
	}

	groups := make(map[string]*GroupedExecutionSummary)

	for _, exec := range executions {
		var groupKey, groupLabel string

		switch groupBy {
		case "workflow":
			groupKey = exec.WorkflowID
			if exec.WorkflowName != nil {
				groupLabel = *exec.WorkflowName
			} else {
				groupLabel = exec.WorkflowID
			}
		case "session":
			if exec.SessionID != nil {
				groupKey = *exec.SessionID
				groupLabel = *exec.SessionID
			} else {
				groupKey = "no-session"
				groupLabel = "No Session"
			}
		case "actor":
			if exec.ActorID != nil {
				groupKey = *exec.ActorID
				groupLabel = *exec.ActorID
			} else {
				groupKey = "no-actor"
				groupLabel = "No Actor"
			}
		case "agent":
			groupKey = exec.AgentNodeID
			groupLabel = exec.AgentNodeID
		case "status":
			groupKey = exec.Status
			groupLabel = exec.Status
		default:
			continue
		}

		if _, exists := groups[groupKey]; !exists {
			groups[groupKey] = &GroupedExecutionSummary{
				GroupKey:        groupKey,
				GroupLabel:      groupLabel,
				Count:           0,
				TotalDurationMS: 0,
				StatusSummary:   make(map[string]int),
				Executions:      []ExecutionSummaryForUI{},
			}
		}

		group := groups[groupKey]
		group.Count++

		// Add to status summary
		group.StatusSummary[exec.Status]++

		// Add duration if available
		if exec.DurationMS != nil {
			group.TotalDurationMS += *exec.DurationMS
		}

		// Update latest execution time
		if group.LatestExecution.IsZero() || exec.StartedAt.After(group.LatestExecution) {
			group.LatestExecution = exec.StartedAt
		}

		// Add execution to group
		group.Executions = append(group.Executions, s.convertToUISummary(exec))
	}

	// Convert map to slice and calculate averages
	result := make([]GroupedExecutionSummary, 0, len(groups))
	for _, group := range groups {
		if group.Count > 0 {
			group.AvgDurationMS = group.TotalDurationMS / int64(group.Count)
		}
		result = append(result, *group)
	}

	return result
}

func (s *ExecutionsUIService) sortGroups(groups []GroupedExecutionSummary, sortBy, sortOrder string) {
	if len(groups) <= 1 {
		return
	}

	// Simple bubble sort implementation to avoid import dependencies
	n := len(groups)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			var shouldSwap bool

			switch sortBy {
			case "time":
				if sortOrder == "asc" {
					shouldSwap = groups[j].LatestExecution.After(groups[j+1].LatestExecution)
				} else {
					shouldSwap = groups[j].LatestExecution.Before(groups[j+1].LatestExecution)
				}
			case "duration":
				if sortOrder == "asc" {
					shouldSwap = groups[j].AvgDurationMS > groups[j+1].AvgDurationMS
				} else {
					shouldSwap = groups[j].AvgDurationMS < groups[j+1].AvgDurationMS
				}
			case "status":
				if sortOrder == "asc" {
					shouldSwap = groups[j].GroupLabel > groups[j+1].GroupLabel
				} else {
					shouldSwap = groups[j].GroupLabel < groups[j+1].GroupLabel
				}
			default:
				// Default to sorting by count (descending)
				shouldSwap = groups[j].Count < groups[j+1].Count
			}

			if shouldSwap {
				groups[j], groups[j+1] = groups[j+1], groups[j]
			}
		}
	}
}

func (s *ExecutionsUIService) countClients() int {
	count := 0
	s.clients.Range(func(_, _ interface{}) bool {
		count++
		return true
	})
	return count
}

// Event callbacks for real-time updates

// OnExecutionStarted is called when an execution starts.
func (s *ExecutionsUIService) OnExecutionStarted(execution *types.WorkflowExecution) {
	summary := s.convertToUISummary(execution)
	s.BroadcastExecutionEvent("execution_started", summary)
}

// OnExecutionCompleted is called when an execution completes.
func (s *ExecutionsUIService) OnExecutionCompleted(execution *types.WorkflowExecution) {
	summary := s.convertToUISummary(execution)
	s.BroadcastExecutionEvent("execution_completed", summary)
}

// OnExecutionFailed is called when an execution fails.
func (s *ExecutionsUIService) OnExecutionFailed(execution *types.WorkflowExecution) {
	summary := s.convertToUISummary(execution)
	s.BroadcastExecutionEvent("execution_failed", summary)
}
